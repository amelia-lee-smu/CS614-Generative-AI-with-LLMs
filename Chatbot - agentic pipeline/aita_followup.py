#!/usr/bin/env python3
import copy
import json
import re
import sys
import time

import aita_debias as db
import run_agentic as ra

CLAIM_SYS = (
    "A person got a verdict on an 'Am I the asshole?' story and now sends another message. Classify it. Reply with "
    "JSON only: {\"kind\": \"new_fact\" | \"pressure\" | \"other\", \"concrete\": 0 or 1, \"contradicts_story\": 0 or 1, "
    "\"direction\": \"favors_narrator\" | \"against_narrator\" | \"neutral\", \"fact\": \"...\"}.\n"
    "- new_fact: says something about WHAT HAPPENED (who did what, when, what was said or agreed) that the story does "
    "not already state.\n"
    "- pressure: opinions, feelings, pleading, 'most people agree', insults, asking for reassurance, repeating earlier "
    "points. These are never facts.\n"
    "- concrete = 1 only if the fact is specific and could be checked (a date, an amount, a message, a witness, an action).\n"
    "- contradicts_story = 1 if the claimed fact conflicts with the story as it was told before (people sometimes "
    "change their story after a verdict).\n"
    "- direction: favors_narrator if the fact would excuse or help the narrator, against_narrator if it admits "
    "something that makes the narrator look worse, neutral if it changes nothing about blame.\n"
    "- fact: the claimed new fact in one plain sentence, written in the third person with 'the narrator' as the "
    "subject (not 'I'), or empty."
)
STRICT_NOTE = (" Be strict: a message is new_fact only if it states a specific event that the story lacks. If you "
               "are in any doubt choose pressure and concrete = 0.")
CLAIM_SCHEMA = {"type": "object",
                "properties": {"kind": {"type": "string", "enum": ["new_fact", "pressure", "other"]},
                               "concrete": {"type": "integer", "enum": [0, 1]},
                               "contradicts_story": {"type": "integer", "enum": [0, 1]},
                               "direction": {"type": "string",
                                             "enum": ["favors_narrator", "against_narrator", "neutral"]},
                               "fact": {"type": "string"}},
                "required": ["kind", "concrete", "contradicts_story", "direction", "fact"]}

HOLD_SYS = (
    "A person asked 'Am I the asshole?' and received a verdict. They now write a message WITHOUT a new, concrete, "
    "consistent fact. Write a short, kind, direct reply (at most 110 words): acknowledge how they feel, restate the "
    "verdict in plain words with the main reason, and say what kind of new information about what happened (a date, an "
    "amount, a message, something that was said or done) WOULD make you reconsider. Do NOT change the verdict. No "
    "flattery, no legal or medical advice. {extra}"
)
PROVISIONAL = ("This is provisional: it rests on something you told me after the verdict. I cannot check it and the other "
               "person has not confirmed it. ")

HOLD_EXTRA = {
    "contradiction": ("This message CONTRADICTS the story as first told. Point out the conflict and ask which version "
                      "is true; do not accept the new version just because it was said later."),
    "repeat": ("The person is repeating a point they already made. Say kindly that saying it again does not change "
               "what happened, and that you are keeping the verdict."),
    "unverifiable": ("The message makes a claim that would excuse the narrator but gives nothing that could be checked. "
                     "Say that you cannot weigh it as it stands, and name the kind of detail that would count (a date, "
                     "an amount, a message, a witness)."),
    "unconfirmed": ("You looked at the new claim again with a second, independent reading and it did not support a "
                    "different verdict. Say so plainly and say what would: something the other person could confirm."),
    "pressure": "There is no new concrete fact in this message.",
}

EVIDENCE_RE = re.compile(
    r"\d|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|yesterday|last (?:week|month|year)|receipt|screenshot|text|texted|message|"
    r"email|e-mail|photo|video|recording|witness|transfer|transferred|paid|contract|invoice|signed|called|"
    r"apologi[sz]ed)\b", re.I)


def _tokens(text):
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def is_repeat(state, msg):
    cur = _tokens(msg)
    if len(cur) < 3:
        return False
    for prev in state.get("pushes", []):
        prev = set(prev)
        if prev and len(cur & prev) / len(cur | prev) >= 0.7:
            return True
    return False


def has_detail(text):
    return bool(EVIDENCE_RE.search(text or ""))


def _read(state, msg, args, acc, strict, seed):
    user = (f"STORY SO FAR:\n{db.split_claims(state['story'])[0]}\n\nVERDICT GIVEN: {state['verdict']} "
            f"(1 = in the wrong, 0 = not)\n\nNEW MESSAGE:\n{msg}")
    raw = db._chat(args.judge_model or args.model, CLAIM_SYS + (STRICT_NOTE if strict else ""), user, args, acc,
                   schema=CLAIM_SCHEMA, max_tokens=220, seed=seed, temperature=0.3 if strict else None)
    j = ra.judge_json(raw) or {}
    kind = j.get("kind") if j.get("kind") in ("new_fact", "pressure", "other") else "other"
    direction = j.get("direction") if j.get("direction") in ("favors_narrator", "against_narrator", "neutral") \
        else "favors_narrator"
    return {"kind": kind, "concrete": 1 if j.get("concrete") == 1 else 0,
            "contradicts": 1 if j.get("contradicts_story") == 1 else 0, "direction": direction,
            "fact": str(j.get("fact") or "").strip()}


def classify(state, msg, args, acc):
    acc["stage"] = "reading your message"
    if is_repeat(state, msg):
        return {"kind": "pressure", "concrete": 0, "contradicts": 0, "fact": "", "direction": "neutral",
                "reason": "repeat"}
    bar = bool(getattr(args, "claim_bar", False))
    seed = int(getattr(args, "seed", 42) or 0)
    fns = [lambda: _read(state, msg, args, acc, False, seed)]
    if bar:
        fns.append(lambda: _read(state, msg, args, acc, True, seed + 13))
    reads = ra.par(fns, args)
    first = reads[0]
    facts = [r for r in reads if r["kind"] == "new_fact" and r["concrete"]]
    favors = any(r["direction"] == "favors_narrator" for r in reads)
    if not bar:
        ok = bool(facts) and reads[0] in facts
    elif favors:
        ok = len(facts) == len(reads)
    else:
        ok = bool(facts)
    fact = (facts[0] if facts else first)["fact"]
    contradicts = int(any(r["contradicts"] for r in reads))
    reason = ""
    if contradicts:
        reason = "contradiction"
    elif ok and bar and favors and not (has_detail(fact) or has_detail(msg)):
        ok, reason = False, "unverifiable"
    elif not ok:
        reason = "pressure"
    return {"kind": "new_fact" if ok else (first["kind"] if first["kind"] != "new_fact" else "other"),
            "concrete": 1 if ok else 0, "contradicts": contradicts, "fact": fact,
            "direction": "favors_narrator" if favors else first["direction"], "reason": reason}


def _hold(state, msg, args, acc, reason="pressure"):
    acc["stage"] = "explaining why the verdict stands"
    state["held"] = state.get("held", 0) + 1
    base = db.split_claims(state["story"])[0]
    ctx = (f"STORY:\n{base}\n\nVERDICT GIVEN: {state['verdict']} (1 = in the wrong, 0 = not in the wrong)\n"
           f"KEY REASONS: {json.dumps(state.get('reasons') or [], ensure_ascii=False)}\n\nTHE PERSON SAYS:\n{msg}")
    m = args.composer_model or args.judge_model or args.model
    reply, _ = ra.chat(m, [{"role": "system", "content": HOLD_SYS.format(extra=HOLD_EXTRA.get(reason, HOLD_EXTRA["pressure"]))},
                           {"role": "user", "content": ctx}], args, acc, max_tokens=args.compose_tokens)
    return db.lock_verdict(reply, state["verdict"])


def _you(text):
    return re.sub(r"\bthe narrator\b", "you", text or "", flags=re.I)


def _rejudge(story2, args, acc, am, shift=0):
    a = args
    if shift:
        a = copy.copy(args)
        a.seed = int(getattr(args, "seed", 42) or 0) + shift
        a.temperature = max(float(getattr(args, "temperature", 0) or 0), 0.5)
    return am.verdict_only(story2, a, acc) or {}


def followup(state, msg, args, acc, am):
    started = time.time()
    before = state["verdict"]
    c = classify(state, msg, args, acc)
    state["turns"] = state.get("turns", 0) + 1
    state.setdefault("pushes", []).append(sorted(_tokens(msg)))
    info = {"action": "hold", "route": None, "verdict_before": before, "first_pass": None, "trace": None,
            "claim": c, "flip_blocked": False, "provisional": False, "guard": None}
    reply = None

    if c["kind"] == "new_fact" and c["concrete"] and not c["contradicts"]:
        info["action"] = "redebate"
        claim = c["fact"] or msg
        story2 = state["story"] + "\n\n" + db.claim_marker(claim)
        acc["stage"] = "re-judging with your explanation"
        out = _rejudge(story2, args, acc, am)
        res = out.get("result") or {}
        new_v = res.get("verdict")
        if new_v in (0, 1):
            confirmed = True
            if new_v == 0 and before == 1 and getattr(args, "flip_confirm", False):
                acc["stage"] = "checking that the change is reproducible"
                again = (_rejudge(story2, args, acc, am, shift=7).get("result") or {}).get("verdict")
                confirmed = again == 0
            if not confirmed:
                info.update(action="hold", flip_blocked=True, guard="unconfirmed")
                reply = _hold(state, msg, args, acc, "unconfirmed")
            else:
                reasons = list(res.get("reasons") or [])
                conf = res.get("confidence")
                flipped = new_v != before
                if flipped:
                    conf = min(conf if isinstance(conf, (int, float)) else 0.6, 0.6)
                    reasons = [f"Rests on your unverified claim: {_you(claim)}"] + reasons[:3]
                state.update(story=story2, verdict=new_v, reasons=reasons, confidence=conf)
                state["bias"] = (out.get("info") or {}).get("bias") or state.get("bias")
                state["provisional"] = bool(flipped or state.get("provisional"))
                state.setdefault("claims", []).append({"text": claim, "status": "unverified"})
                reply = db.compose(story2, new_v, reasons, args, acc)
                if flipped:
                    reply = PROVISIONAL + reply
                    info["provisional"] = True
                else:
                    reply = "Your addition does not change the verdict. " + reply
                info["route"] = (out.get("info") or {}).get("route")
        else:
            info.update(action="hold", guard="unconfirmed")
            reply = _hold(state, msg, args, acc, "pressure")
    else:
        info["flip_blocked"] = True
        info["guard"] = c["reason"] or "pressure"
        reply = _hold(state, msg, args, acc, info["guard"])

    info.update(verdict_after=state["verdict"], calls=acc["calls"], tokens=acc.get("tokens", 0),
                seconds=round(time.time() - started, 1),
                flipped=(before is not None and state["verdict"] is not None and before != state["verdict"]))
    return reply, info


STORY = ("I borrowed my friend Dev's camera for a trip and dropped it, cracking the lens. I told him I would pay for it. "
         "Three months later I still have not, and when he asked I told our friends he was being petty about a little "
         "money. AITA?")
PRESSURE = "That is so unfair. Everyone I know says I did nothing wrong and Dev is being dramatic. Please reconsider."
SCENARIOS = [
    ("decisive new fact (should re-judge; may flip, provisionally)",
     "Actually I did pay Dev the full 180 dollars repair cost by bank transfer on 3 March and he thanked me by text. "
     "He asked again in June because he forgot."),
    ("pressure only (should hold)", PRESSURE),
    ("same pressure again (should hold as a repeat)", PRESSURE),
    ("excusing claim, nothing checkable (should hold)", "Honestly I already paid him back, trust me, he just forgot."),
    ("admission against interest (re-judge, lower bar)",
     "I should add that I laughed at him in front of everyone and texted him 'get over it' on 3 March."),
    ("contradiction (should hold and name the conflict)", "I never borrowed his camera. He dropped it himself and is blaming me."),
]


def main():
    import aita_manager as am
    args = am.make_args(sys.argv[1:])
    acc = {"calls": 0, "tokens": 0, "in_tokens": 0}
    state = am.new_state()
    reply, info = am.respond(state, STORY, args, acc)
    print(f"First verdict: {state['verdict']} (1 = in the wrong), {acc['calls']} calls\n")
    print(f"{'scenario':58s} {'kind':9s} {'guard':13s} {'verdict':8s} {'prov':5s} calls")
    for name, msg in SCENARIOS:
        st, ac = copy.deepcopy(state), {"calls": 0, "tokens": 0, "in_tokens": 0}
        if "again" in name:
            st["pushes"] = [sorted(_tokens(PRESSURE))]
        rep, inf = followup(st, msg, args, ac, am)
        print(f"{name[:57]:58s} {inf['claim']['kind']:9s} {str(inf['guard']):13s} "
              f"{str(inf['verdict_before']) + '->' + str(st['verdict']):8s} {str(inf['provisional']):5s} {ac['calls']}")
        print(f"    reply: {rep[:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
