#!/usr/bin/env python3
import json
import sys

import aita_debias as db
import run_agentic as ra

VALUES = ("honesty", "fairness", "care", "responsibility", "respect", "trust")
VALUE_HELP = {
    "honesty": "told the truth, did not hide or twist facts",
    "fairness": "treated the other person even-handedly, kept to what was agreed",
    "care": "paid attention to the other person's wellbeing and feelings",
    "responsibility": "owned their part, kept commitments, repaired harm",
    "respect": "respected the other person's dignity, boundaries and choices",
    "trust": "acted in a way that keeps or rebuilds the other person's trust",
}

ETHICS_SYS = (
    "You are an ethics reflection aid, not a judge. You get a neutral account of a conflict between Person A and "
    "Person B. Use ONLY facts stated in the account. Judge conduct, not likability, and give nobody credit for being "
    "the one who tells the story.\n"
    "1) For EACH person rate each value from -2 (clearly violated) to +2 (clearly upheld). 0 = not at stake or unknown. "
    "Fields: a_<value> for Person A and b_<value> for Person B. Values: "
    + "; ".join(f"{k} = {v}" for k, v in VALUE_HELP.items()) + ".\\n"
    "2) disturbed: up to 3 items {value, person ('A', 'B' or 'both'), what}, naming the specific action that disturbed "
    "a value. Only include values that were actually violated.\\n"
    "3) lesson: at most 45 words, a concrete takeaway about conduct that anyone in this situation can use. Do not "
    "flatter or comfort. If one person clearly did something wrong, say what.\\n"
    "4) repair: at most 30 words, what a fair repair would look like.\\n"
    "5) society_scope: 'private' if this stays between the people involved, 'community' if it affects a group, "
    "family or workplace, 'wider' only if it touches shared rules, public trust or safety.\\n"
    "6) society: at most 50 words, what would happen to trust and cooperation if this behaviour became normal. "
    "Do not exaggerate; if the effect is small, say so.\\n"
    "Output JSON only."
).replace("\\n", "\n")


def schema():
    props = {f"{p}_{v}": {"type": "integer"} for p in ("a", "b") for v in VALUES}
    props["disturbed"] = {"type": "array", "items": {
        "type": "object",
        "properties": {"value": {"type": "string"}, "person": {"type": "string", "enum": ["A", "B", "both"]},
                       "what": {"type": "string"}},
        "required": ["value", "person", "what"]}}
    props["lesson"] = {"type": "string"}
    props["repair"] = {"type": "string"}
    props["society_scope"] = {"type": "string", "enum": ["private", "community", "wider"]}
    props["society"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


def _score(x):
    try:
        return max(-2, min(2, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def _one(text, swap, model, args, acc, seed):
    t = db.swap_roles(text) if swap else text
    raw = db._chat(model, ETHICS_SYS, f"ACCOUNT:\n{t}", args, acc, schema=schema(),
                   max_tokens=getattr(args, "judge_tokens", 400) + 250, temperature=0.4 if swap is not None else None,
                   seed=seed)
    j = ra.judge_json(raw) or {}
    a = {v: _score(j.get(f"a_{v}")) for v in VALUES}
    b = {v: _score(j.get(f"b_{v}")) for v in VALUES}
    if sum(x is not None for x in list(a.values()) + list(b.values())) < 8:
        return None
    a = {k: v or 0 for k, v in a.items()}
    b = {k: v or 0 for k, v in b.items()}
    asker, other = (b, a) if swap else (a, b)
    flip = {"A": "B", "B": "A", "both": "both"}
    disturbed = []
    for d in (j.get("disturbed") or [])[:3]:
        if not isinstance(d, dict):
            continue
        who = d.get("person")
        who = flip.get(who, who) if swap else who
        what = db.swap_roles(str(d.get("what") or "")) if swap else str(d.get("what") or "")
        value = str(d.get("value") or "").lower()
        if value in VALUES and who in ("A", "B", "both") and what.strip():
            disturbed.append({"value": value, "person": who, "what": what.strip()})
    text_fields = {k: (db.swap_roles(str(j.get(k) or "")) if swap else str(j.get(k) or "")).strip()
                   for k in ("lesson", "repair", "society")}
    return {"asker": asker, "other": other, "disturbed": disturbed, "scope": j.get("society_scope"), **text_fields}


def _cap(t):
    t = (t or "").strip()
    t = db.fix_grammar(t)
    return t[:1].upper() + t[1:]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def compass_text(text, args, acc):
    flags = []
    if "Person A" not in text or "Person B" not in text:
        text = "(Person A is the narrator. Person B is the main other person.)\n" + text
        flags.append("unlabelled")
    model = getattr(args, "judge_model", None) or args.model
    seed = int(getattr(args, "seed", 42) or 0)
    acc["stage"] = "reading the account against six values"
    runs = ra.par([lambda: _one(text, False, model, args, acc, seed),
                   lambda: _one(text, True, model, args, acc, seed + 101)], args)
    ok = [r for r in runs if r]
    if not ok:
        raise ValueError("The model could not produce a usable compass. Try again or use a larger model.")
    if len(ok) == 1:
        flags.append("single_run")
    asker = {v: round(_mean([r["asker"][v] for r in ok]), 2) for v in VALUES}
    other = {v: round(_mean([r["other"][v] for r in ok]), 2) for v in VALUES}
    agree = None
    if len(ok) == 2:
        diffs = [abs(ok[0]["asker"][v] - ok[1]["asker"][v]) + abs(ok[0]["other"][v] - ok[1]["other"][v]) for v in VALUES]
        agree = round(1 - sum(diffs) / (len(VALUES) * 8), 2)
        if agree < 0.75:
            flags.append("runs_disagree")
    first = ok[0]
    who = {"A": "you", "B": "the other person", "both": "both of you"}
    disturbed = [{"value": d["value"], "who": who[d["person"]], "what": _cap(db.humanize(d["what"]))} for d in first["disturbed"]]
    if not disturbed and len(ok) == 2:
        disturbed = [{"value": d["value"], "who": who[d["person"]], "what": _cap(db.humanize(d["what"]))} for d in ok[1]["disturbed"]]
    scope = first["scope"] if first["scope"] in ("private", "community", "wider") else "private"
    return {"asker": asker, "other": other, "disturbed": disturbed, "scope": scope,
            "lesson": _cap(db.humanize(first["lesson"] or ok[-1]["lesson"])),
            "repair": _cap(db.humanize(first["repair"] or ok[-1]["repair"])),
            "society": _cap(db.humanize(first["society"] or ok[-1]["society"])),
            "agreement": agree, "flags": flags, "runs": len(ok)}


def compass(state, args, acc):
    if state.get("verdict") is None or not state.get("story"):
        raise ValueError("Get a verdict first, then open the Ethics tab.")
    bias = state.get("bias") or {}
    cached = bias.get("ethics")
    if cached and cached.get("asker") and cached.get("lesson") is not None:
        res = {k: cached[k] for k in ("asker", "other", "disturbed", "scope", "lesson", "repair", "society",
                                      "agreement", "flags", "runs") if k in cached}
        res["flags"] = list(res.get("flags") or [])
        res["from_verdict"] = True
    else:
        text = bias.get("neutral_story") or db.neutralize(state["story"], args, acc)
        res = compass_text(text, args, acc)
    gap = _mean(list(res["asker"].values())) - _mean(list(res["other"].values()))
    verdict = state["verdict"]
    if (verdict == 1 and gap > 0.5) or (verdict == 0 and gap < -0.5):
        res["flags"].append("compass_vs_verdict")
    res["verdict"] = verdict
    return res


def main():
    import aita_manager as am
    args = am.make_args(sys.argv[1:])
    story = ("I borrowed my friend Dev's camera for a trip and dropped it, cracking the lens. I told him I would pay "
             "for it. Three months later I still have not, and when he asked I told our friends he was being petty "
             "about a little money. AITA?")
    state = {"story": story, "verdict": 1, "bias": {}}
    acc = {"calls": 0, "tokens": 0, "in_tokens": 0}
    print(json.dumps(compass(state, args, acc), indent=2, ensure_ascii=False))
    print(f"\n{acc['calls']} model calls")


if __name__ == "__main__":
    main()
