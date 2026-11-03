#!/usr/bin/env python3
import argparse
import json
import os
import time

import run_agentic as ra
import aita_debias as db

rm = ra.rm

MANAGER_SYS = (
    "You manage a chat in which a person tells an 'Am I the asshole?' story and gets a verdict. Decide the next "
    "action. Reply with JSON only: {\"action\": ..., \"question\": ..., \"reason\": ...}.\n"
    "Actions:\n"
    "- \"support\": the person may be in danger, being abused, or mentions self-harm or suicide.\n"
    "- \"ask\": ONLY for the first story, and ONLY if a key fact is missing so that a fair verdict is impossible "
    "(who did what, what was said). Put exactly ONE short question in \"question\". If the story is clear enough, "
    "do not ask. Never ask if a question was already asked.\n"
    "- \"debate\": the first story has enough facts for a verdict.\n"
    "- \"redebate\": a later message adds NEW FACTS about what happened. Opinions, feelings, pressure, "
    "appeals to what 'most people' think, and repeating earlier points are NOT new facts.\n"
    "- \"hold\": a later message only disagrees, pleads, asks for reassurance, or repeats points without new facts.\n"
    "Leave \"question\" empty unless the action is \"ask\"."
)
MANAGER_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["ask", "debate", "redebate", "hold", "support"]},
                   "question": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["action", "question", "reason"],
}
HOLD_SYS = (
    "A person asked 'Am I the asshole?' and received a verdict. They now push back WITHOUT giving new facts. "
    "Write a short, kind, direct reply (at most 120 words): acknowledge how they feel, restate the verdict in "
    "plain words and the main reasons, and say that you would reconsider if there were new facts about what "
    "happened. Do NOT change the verdict, do NOT say 'you may be right' or 'fair point' unless they gave a new "
    "fact, and do not apologise for the verdict. No flattery, no legal or medical advice."
)

FACT_SYS = getattr(ra, "FACT_SYS", (
    "You are a strict fact-checker. You get a STORY and an ARGUMENT written about it. Rewrite the argument so it "
    "keeps ONLY claims the story supports. Remove or replace with 'unknown' any claim about events, people, "
    "motives or details that the story does not state. Keep the same stance, plain words, same length or "
    "shorter. Output only the rewritten argument."
))


def _install_fact_check():
    if getattr(ra, "_aita_fact_check", False):
        return
    real_chat = ra.chat

    def chat(model, messages, args, acc, *a, **kw):
        out = real_chat(model, messages, args, acc, *a, **kw)
        if not getattr(args, "fact_check", False):
            return out
        system = messages[0]["content"] if messages else ""
        if system in (ra.ADV_WRONG_SYS, ra.ADV_RIGHT_SYS) and out and out[0].strip():
            story = messages[-1]["content"]
            checked = real_chat(args.judge_model or args.model,
                                [{"role": "system", "content": FACT_SYS},
                                 {"role": "user", "content": f"STORY:\n{story}\n\nARGUMENT:\n{out[0]}"}],
                                args, acc, max_tokens=args.agent_tokens)
            if checked and checked[0].strip():
                return checked
        return out

    ra.chat = chat
    ra._aita_fact_check = True


def new_state():
    return {"story": "", "verdict": None, "reasons": [], "confidence": None, "asked": False, "turns": 0, "held": 0,
            "bias": None}


def _add(parser, *names, **kw):
    try:
        parser.add_argument(*names, **kw)
    except argparse.ArgumentError:
        pass


def build_parser():
    p = ra.build_parser()
    _add(p, "--adaptive", action="store_true",
         help="first verdict: one single-call pass, full debate only when it looks risky (see --escalate-on). "
              "Ignored when a rubric option is on.")
    _add(p, "--fact-check", action="store_true",
         help="rewrite each advocate argument to keep only story-supported claims before the judge (+2 calls).")
    _add(p, "--manager-model", default=None,
         help="model that decides ask/debate/hold/redebate each turn (default: the judge model, else --model).")
    _add(p, "--escalate-on", default="either", choices=["not_wrong", "low_conf", "either"],
         help="adaptive: escalate to the debate when the first verdict is 'not in the wrong', confidence is low, or either.")
    _add(p, "--escalate-conf", type=float, default=0.7, help="adaptive: confidence below this counts as low.")
    db.add_flags(lambda *n, **kw: _add(p, *n, **kw))
    return p


def prepare_args(args):
    if hasattr(ra, "prepare_args"):
        return ra.prepare_args(args)
    args.schema = rm.make_schema(False)
    ds = json.loads(json.dumps(args.schema))
    ds.setdefault("properties", {}).update({
        "confidence": {"type": "number"},
        "key_reasons": {"type": "array", "items": {"type": "string"}},
    })
    ds["required"] = list(dict.fromkeys(list(ds.get("required", [])) + ["confidence", "key_reasons"]))
    args.debate_schema = ds
    return args


def load_calibration(args):
    path = getattr(args, "calibration", None)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cal = json.load(fh)
        args.threshold = float(cal.get("threshold", args.threshold))
        args.aggregate = cal.get("aggregate", args.aggregate)
    return args


def make_args(argv):
    args = build_parser().parse_args(["--pipeline", "debate", "--compose", "--safety"] + list(argv))
    if not hasattr(ra, "FACT_SYS"):
        _install_fact_check()
    db.apply_preset(args)
    load_calibration(args)
    return prepare_args(args)


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _verdict_and_reasons(raw):
    v = rm.parse_output(raw)[0]
    j = ra.judge_json(raw)
    return v, (j.get("key_reasons") or [j.get("explanation", "")]), _as_float(j.get("confidence"))


def _schema_for(allowed):
    sch = json.loads(json.dumps(MANAGER_SCHEMA))
    sch["properties"]["action"]["enum"] = list(allowed)
    return sch


def _decide(state, msg, args, acc):
    mgr_m = getattr(args, "manager_model", None) or args.judge_model or args.model
    followup = state["verdict"] is not None
    allowed = ("hold", "redebate", "support") if followup else ("ask", "debate", "support")
    ctx = (f"Question already asked: {state['asked']}\n"
           f"Verdict already given: {state['verdict'] if followup else 'none yet'}\n"
           f"Allowed actions this turn: {', '.join(allowed)}\n"
           f"Story so far:\n{state['story'] or '(nothing yet)'}\n\nLatest message from the person:\n{msg}")
    raw, _ = ra.chat(mgr_m, [{"role": "system", "content": MANAGER_SYS}, {"role": "user", "content": ctx}],
                     args, acc, schema=_schema_for(allowed), max_tokens=120)
    j = ra.judge_json(raw)
    chosen = j.get("action")
    action = chosen if chosen in allowed else ("hold" if followup else "debate")
    if action == "ask" and (state["asked"] or not (j.get("question") or "").strip()):
        action = "debate"
    return action, (j.get("question") or "").strip(), {"manager_raw": (raw or "")[:300], "manager_choice": chosen,
                                                       "manager_model": mgr_m}


def _merge_story(state, msg, action):
    if state["story"] and state["verdict"] is None:
        return state["story"] + f"\n\nAnswer to the clarifying question: {msg}"
    if state["story"] and action == "redebate":
        return state["story"] + f"\n\nNew information from the person (added after the first verdict, unverified): {msg}"
    return msg


def _first_pass(story, args, acc):
    raw, _ = ra.chat(args.model, [{"role": "system", "content": ra.JUDGE_NO_ARGS_SYS},
                                  {"role": "user", "content": story}],
                     args, acc, schema=args.debate_schema, max_tokens=args.judge_tokens)
    verdict = rm.parse_output(raw)[0]
    conf = _as_float(ra.judge_json(raw).get("confidence"))
    low_conf = conf is None or conf < args.escalate_conf
    not_wrong = verdict != 1
    escalate = {"not_wrong": not_wrong, "low_conf": low_conf, "either": not_wrong or low_conf}[args.escalate_on]
    return raw, {"verdict": verdict, "confidence": conf, "escalated": escalate}


def baseline(story, args, acc):
    acc["stage"] = "plain single call"
    raw, fp = _first_pass(story, args, acc)
    reply = str(ra.judge_json(raw).get("explanation") or "").strip()
    return {"verdict": fp["verdict"], "confidence": fp["confidence"], "reply": reply}


def _write_reply(story, raw, args, acc):
    if not args.compose:
        return ra.judge_json(raw).get("explanation") or ""
    acc["stage"] = "writing the reply"
    v, reasons, _ = _verdict_and_reasons(raw)
    msg = f"STORY:\n{story}\n\nVERDICT: {v}\nKEY REASONS: {json.dumps(reasons, ensure_ascii=False)}"
    com_m = args.composer_model or args.judge_model or args.model
    reply, _ = ra.chat(com_m, [{"role": "system", "content": ra.COMPOSER_SYS}, {"role": "user", "content": msg}],
                       args, acc, max_tokens=args.compose_tokens)
    return reply


def _from_raw(raw):
    v, reasons, conf = _verdict_and_reasons(raw)
    return {"verdict": v, "reasons": reasons, "confidence": conf}


def _verdict_turn(state, info, args, acc):
    story = state["story"]
    crisis = bool(args.safety and ra.CRISIS_RE.search(story))
    if db.enabled(args) and not crisis:
        res = db.run(story, args, acc)
        if res.get("routed"):
            info["action"] = "support"
            return None, res.get("reply")
        if not res.get("fallback"):
            info["route"] = "debiased"
            info["trace"] = res.get("trace")
            info["bias"] = res["bias"]
            reply = db.compose(story, res["verdict"], res["reasons"], args, acc)
            return res, reply
        info["bias"] = res["bias"]
        info["trace"] = res.get("trace")
        info["route"] = "debate"
        raw = res.get("debate_raw")
        if raw is not None:
            r = _from_raw(raw)
            return r, db.compose(story, r["verdict"], r["reasons"], args, acc)

    quick = (getattr(args, "adaptive", False) and state["verdict"] is None and not crisis and not db.enabled(args))
    if quick:
        acc["stage"] = "first pass"
        raw, fp = _first_pass(story, args, acc)
        info["first_pass"] = fp
        if not fp["escalated"]:
            info["route"] = "single_call"
            return _from_raw(raw), _write_reply(story, raw, args, acc)

    text = story
    if getattr(args, "neutralize", False) and not crisis:
        text = db.neutralize(story, args, acc) or story
        info["bias"] = {"neutral_story": text, "flags": [], "steps": ["neutral rewrite"]}
    acc["stage"] = "running the debate"
    raw, _, trace = ra.run_debate(args.model, text, args, acc)
    info["trace"] = trace
    if trace.get("routed"):
        info["action"] = "support"
        return None, trace.get("final_reply")
    info["route"] = "debate"
    return _from_raw(raw), trace.get("final_reply") or (ra.judge_json(raw).get("explanation") or "")


def verdict_only(story, args, acc):
    state = new_state()
    state["story"] = story
    info = {"action": "debate", "route": None, "bias": None, "trace": None}
    res, reply = _verdict_turn(state, info, args, acc)
    return {"result": res, "reply": reply, "info": info}


def respond(state, msg, args, acc=None):
    acc = acc if acc is not None else {"calls": 0, "tokens": 0, "in_tokens": 0}
    started = time.time()
    model = args.model
    acc["stage"] = "reading your message"
    action, question, details = _decide(state, msg, args, acc)
    info = {"action": action, "route": None, "verdict_before": state["verdict"], "confidence": None,
            "first_pass": None, "trace": None, "bias": None, "flip_blocked": False, **details}
    state["turns"] += 1

    if action == "support":
        acc["stage"] = "writing a supportive reply"
        reply, _ = ra.chat(args.composer_model or args.judge_model or model,
                           [{"role": "system", "content": ra.SUPPORT_SYS}, {"role": "user", "content": msg}],
                           args, acc, max_tokens=args.compose_tokens)
    elif action == "ask":
        state["story"] = msg
        state["asked"] = True
        reply = question
    elif action == "hold":
        acc["stage"] = "explaining the verdict"
        state["held"] += 1
        com_m = args.composer_model or args.judge_model or model
        ctx = (f"STORY:\n{state['story']}\n\nVERDICT GIVEN: {state['verdict']} (1 = in the wrong, 0 = not in the "
               f"wrong)\nKEY REASONS: {json.dumps(state['reasons'], ensure_ascii=False)}\n\nTHE PERSON SAYS:\n{msg}")
        reply, _ = ra.chat(com_m, [{"role": "system", "content": HOLD_SYS},
                                   {"role": "user", "content": ctx}], args, acc, max_tokens=args.compose_tokens)
    else:
        prior = state["verdict"]
        state["story"] = _merge_story(state, msg, action)
        res, reply = _verdict_turn(state, info, args, acc)
        if (res is not None and getattr(args, "flip_guard", False) and prior == 1 and res.get("verdict") == 0):
            acc["stage"] = "double-checking the flip"
            res2, reply2 = _verdict_turn(state, dict(info), args, acc)
            if res2 is not None and res2.get("verdict") != 0:
                res, reply, info["flip_blocked"] = res2, reply2, True
        if res is not None:
            v = res.get("verdict")
            if v is not None:
                state["verdict"], state["reasons"], state["confidence"] = v, res.get("reasons") or [], res.get("confidence")
            info["confidence"] = res.get("confidence")
            if info.get("bias"):
                state["bias"] = info["bias"]
    info.update(verdict_after=state["verdict"], calls=acc["calls"], tokens=acc["tokens"],
                seconds=round(time.time() - started, 1),
                flipped=(info["verdict_before"] is not None and state["verdict"] is not None
                         and info["verdict_before"] != state["verdict"]))
    return reply, info
