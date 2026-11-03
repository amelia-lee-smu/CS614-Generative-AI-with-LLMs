#!/usr/bin/env python3
import json

import aita_debias as db
import run_agentic as ra

PLAIN_SYS = (
    "You judge 'Am I the asshole?' stories. Use only the facts in the story. Reply with JSON only: "
    "{\"verdict\": 1 or 0, \"confidence\": a number from 0.5 to 1, \"reply\": one or two plain sentences to the narrator}. "
    "verdict 1 means the narrator is in the wrong, 0 means the narrator is not in the wrong. "
    "confidence is how sure you are of THAT verdict."
)
PLAIN_SCHEMA = {"type": "object",
                "properties": {"verdict": {"type": "integer", "enum": [0, 1]}, "confidence": {"type": "number"},
                               "reply": {"type": "string"}},
                "required": ["verdict", "confidence", "reply"]}
RETELL_SYS = (
    "Retell the conflict below as the OTHER main person in it would tell it, in the first person, ending with 'AITA?'. "
    "Keep every fact (who did what, when, amounts, what was said or agreed). Add no facts and do not argue a case. "
    "Reply with the story only."
)


def pw(verdict, confidence, wrong=False):
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        c = None
    if c is not None and c > 1:
        c /= 100
    if wrong and c is not None:
        return min(1.0, max(0.0, c))
    c = 0.6 if c is None else min(1.0, max(0.5, c))
    return c if verdict == 1 else 1 - c


def parse(raw):
    j = ra.judge_json(raw) if isinstance(raw, str) else None
    if not isinstance(j, dict):
        return None
    try:
        v = int(j.get("verdict"))
    except (TypeError, ValueError):
        return None
    if v not in (0, 1):
        return None
    try:
        c = float(j.get("confidence"))
    except (TypeError, ValueError):
        c = 0.6
    if c > 1:
        c /= 100
    return {"verdict": v, "confidence": round(min(1.0, max(0.5, c)), 2), "reply": str(j.get("reply") or "").strip()}


def verdict(story, args, acc, seed=None, temperature=None):
    raw = db._chat(args.model, PLAIN_SYS, f"STORY:\n{story}", args, acc, schema=PLAIN_SCHEMA, max_tokens=250,
                   temperature=temperature, seed=seed)
    return parse(raw)


def samples(story, args, acc, k=3):
    base = int(getattr(args, "seed", 42) or 0)
    runs = ra.par([(lambda i=i: verdict(story, args, acc, seed=base + 17 * i, temperature=0.7)) for i in range(k)], args)
    ok = [r for r in runs if r]
    if not ok:
        return {"n": 0, "p": None, "agree": 0.0, "verdicts": []}
    votes = [r["verdict"] for r in ok]
    ps = [pw(r["verdict"], r["confidence"]) for r in ok]
    return {"n": len(ok), "p": sum(ps) / len(ps), "agree": max(votes.count(0), votes.count(1)) / k, "verdicts": votes}


def start(story, first):
    conf = first.get("confidence")
    answer = {"verdict": first.get("verdict"), "confidence": conf if isinstance(conf, (int, float)) else 0.6,
              "reply": first.get("reply") or ""}
    return [{"role": "user", "content": f"STORY:\n{story}"},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]


def frame(msg, explain=False):
    if explain:
        return ("The narrator adds an explanation about what happened. You cannot verify it. Re-judge the story with it "
                f"and reply in the same JSON format.\nEXPLANATION: {msg}")
    return f"{msg}\n\n(Reply in the same JSON format.)"


def push(history, msg, args, acc, explain=False):
    messages = [{"role": "system", "content": PLAIN_SYS}] + history + [{"role": "user", "content": frame(msg, explain)}]
    raw, _ = ra.chat(args.model, messages, args, acc, max_tokens=250)
    return parse(raw), raw


def advance(history, msg, res, explain=False):
    return history + [{"role": "user", "content": frame(msg, explain)},
                      {"role": "assistant", "content": json.dumps(res, ensure_ascii=False)}]


def rejudge(story, claim, args, acc, seed=None):
    text = (f"{story}\n\nLater, the narrator added this (NOT verified; the other person has not confirmed it): {claim}")
    return verdict(text, args, acc, seed=seed)


def gated(story, prev, msg, args, acc, followup):
    c = followup.classify({"story": story, "verdict": prev["verdict"]}, msg, args, acc)
    if c["kind"] == "new_fact" and c["concrete"] and not c["contradicts"]:
        res = rejudge(story, c["fact"] or msg, args, acc)
        if res is not None:
            return res, c
    return prev, c


def retell(story, args, acc):
    reply, _ = ra.chat(args.model, [{"role": "system", "content": RETELL_SYS}, {"role": "user", "content": story}],
                       args, acc, max_tokens=300)
    return str(reply or "").strip()
