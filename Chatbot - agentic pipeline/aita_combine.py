#!/usr/bin/env python3
import json

import aita_debias as db
import aita_ethics as ethics
import run_agentic as ra

CAP = 0.60
PRESSURE = ("pressure", "repeat", "contradiction")
BASE_W = {"pipeline": 0.5, "plain": 0.3, "ethics": 0.2}
NAME = {"plain": "single call", "pipeline": "pipeline"}

SYS = (
    "You are the final adjudicator in an 'Am I the asshole?' case. Two independent reviewers (X and Y) gave verdicts and "
    "an ethics reflection is attached. Decide the case yourself from the facts in the account.\n"
    "- Judge conduct from facts stated in the account. Do not decide by which reviewer sounds more sure, which comes "
    "first, or how the narrator feels.\n"
    "- Reviewers can be wrong. A reviewer who changed its answer under pressure with no new concrete fact (see "
    "STABILITY) is less reliable.\n"
    "- A claim marked unverified counts only partly and never as proof. It cannot erase facts the account states.\n"
    "- The ethics reflection says which values were disturbed and by whom. Use it as evidence about conduct, not as a "
    "vote that overrides the facts.\n"
    "- Do not flatter the narrator and do not be harsher than the facts justify.\n"
    "Reply with JSON only: {\"verdict\": 1 or 0 (1 = the narrator is in the wrong), \"confidence\": number from 0 to 1, "
    "\"reasons\": up to 3 short strings, each naming a concrete fact from the account, \"trusted\": \"X\" | \"Y\" | "
    "\"neither\", \"why\": at most 35 words}."
)
SCHEMA = {"type": "object",
          "properties": {"verdict": {"type": "integer", "enum": [0, 1]}, "confidence": {"type": "number"},
                         "reasons": {"type": "array", "items": {"type": "string"}},
                         "trusted": {"type": "string", "enum": ["X", "Y", "neither"]}, "why": {"type": "string"}},
          "required": ["verdict", "confidence", "reasons", "trusted", "why"]}


def p_of(verdict, conf):
    c = min(1.0, max(0.0, float(conf) if isinstance(conf, (int, float)) else 0.6))
    return round(c if verdict == 1 else 1 - c, 3)


def conf_of(verdict, p):
    return round(p if verdict == 1 else 1 - p, 3)


def ethics_p(res):
    if not res or not res.get("asker") or not res.get("other"):
        return None
    gap = sum(res["asker"].values()) / len(res["asker"]) - sum(res["other"].values()) / len(res["other"])
    return round(min(1.0, max(0.0, 0.5 - gap / 4)), 3)


def account_text(state):
    base, claims = db.split_claims(state["story"])
    block = db.claim_block(claims) if claims else ""
    return base + ("\n\n" + block if block else "")


def ethics_reading(state, args, acc):
    bias = state.get("bias") or {}
    cached = bias.get("ethics")
    if cached and cached.get("asker") and cached.get("other"):
        return cached
    try:
        return ethics.compass_text(bias.get("neutral_story") or account_text(state), args, acc)
    except ValueError:
        return None


def stability(rows, who, kind_now):
    gave_way, prev = 0, None
    for r in rows:
        cur = r.get(who)
        if cur is None or cur.get("verdict") not in (0, 1):
            continue
        if prev is not None and r["kind"] in PRESSURE and cur["verdict"] != prev["verdict"]:
            gave_way += 1
        prev = cur
    return gave_way


def _stability_line(name, gave_way, last_move):
    parts = []
    if gave_way:
        parts.append(f"it changed its verdict {gave_way} time(s) on a message that carried no concrete fact")
    else:
        parts.append("it has not changed its verdict on a message that carried no concrete fact")
    if last_move:
        parts.append(last_move)
    return f"STABILITY of {name}: " + "; ".join(parts) + "."


def _last_move(row_prev, cur, kind):
    if kind != "fact" or not row_prev or row_prev.get("verdict") == cur.get("verdict"):
        return ""
    return (f"on the latest message, a concrete but unchecked claim, it moved from "
            f"{'in the wrong' if row_prev['verdict'] == 1 else 'not in the wrong'} to "
            f"{'in the wrong' if cur['verdict'] == 1 else 'not in the wrong'} at confidence "
            f"{conf_of(cur['verdict'], cur['p_wrong']):.2f}")


def _reasons_text(rs):
    rs = [str(x).strip()[:400] for x in (rs or []) if str(x).strip()][:4]
    return "; ".join(rs) if rs else "(none given)"


def _agent_run(order, account, sides, ethics_res, claim, args, acc, seed):
    lab = dict(zip(("X", "Y"), order))
    parts = [f"ACCOUNT (Person A is the narrator):\n{account}"]
    if claim:
        parts.append("UNVERIFIED CLAIM just made by the narrator, not part of the account (nobody has checked it): " + claim)
    for letter in ("X", "Y"):
        s = sides[lab[letter]]
        parts.append(f"VERDICT {letter}: {'the narrator is in the wrong' if s['verdict'] == 1 else 'the narrator is not in the wrong'} "
                     f"(confidence {conf_of(s['verdict'], s['p_wrong']):.2f}).\nReasons: {s['reasons']}\n{s['stability']}")
    if ethics_res:
        asker = sum(ethics_res["asker"].values()) / len(ethics_res["asker"])
        other = sum(ethics_res["other"].values()) / len(ethics_res["other"])
        dist = "; ".join(f"{d['value']} ({d['who']}): {d['what']}" for d in (ethics_res.get("disturbed") or [])[:3]) or "none"
        parts.append("ETHICS REFLECTION (values scored from -2 violated to +2 upheld): narrator average "
                     f"{asker:+.2f}, other person average {other:+.2f}. Disturbed values: {dist}. "
                     f"Lesson: {ethics_res.get('lesson') or ''}")
    raw = db._chat(args.judge_model or args.model, SYS, "\n\n".join(parts), args, acc, schema=SCHEMA, max_tokens=420,
                   seed=seed)
    j = ra.judge_json(raw) or {}
    if j.get("verdict") not in (0, 1):
        return None
    conf = j.get("confidence")
    conf = min(1.0, max(0.5, float(conf))) if isinstance(conf, (int, float)) else 0.6
    trusted = lab.get(j.get("trusted"))
    return {"verdict": j["verdict"], "confidence": conf,
            "reasons": [str(x).strip() for x in (j.get("reasons") or []) if str(x).strip()][:3],
            "trusted": trusted, "why": str(j.get("why") or "").strip()}


def weighted(sides, ep):
    w = dict(BASE_W)
    for who in ("plain", "pipeline"):
        if sides[who]["gave_way"]:
            w[who] *= 0.3
        if sides[who].get("credulous"):
            w[who] *= 0.6
    vals = {"plain": sides["plain"]["p_wrong"], "pipeline": sides["pipeline"]["p_wrong"]}
    if ep is not None:
        vals["ethics"] = ep
    total = sum(w[k] for k in vals)
    p = sum(w[k] * vals[k] for k in vals) / total
    return round(p, 3), {k: round(w[k] / total, 2) for k in vals}


def combine(state, plain, info, kind, history, prev, args, acc):
    calls0 = acc["calls"]
    if kind in PRESSURE and prev and prev.get("verdict") in (0, 1):
        out = dict(prev)
        out.update(source="held", calls=0,
                   note="No concrete new fact in this message, so the combined verdict was not re-decided.")
        return out

    pipe_v = state["verdict"]
    pipe_p = p_of(pipe_v, state.get("confidence"))
    plain_v = plain["verdict"]
    plain_p = p_of(plain_v, plain.get("confidence"))
    last = history[-1] if history else {}
    sides = {
        "plain": {"verdict": plain_v, "p_wrong": plain_p, "reasons": _reasons_text([plain.get("reply")] if plain.get("reply") else []),
                  "gave_way": stability(history, "plain", kind)},
        "pipeline": {"verdict": pipe_v, "p_wrong": pipe_p, "reasons": _reasons_text(state.get("reasons")),
                     "gave_way": stability(history, "pipeline", kind)},
    }
    for who in sides:
        cur = {"verdict": sides[who]["verdict"], "p_wrong": sides[who]["p_wrong"]}
        move = _last_move(last.get(who), cur, kind)
        sides[who]["credulous"] = bool(move) and conf_of(cur["verdict"], cur["p_wrong"]) > CAP
        sides[who]["stability"] = _stability_line(NAME[who], sides[who]["gave_way"], move)

    acc["stage"] = "reading the ethics judge"
    eres = ethics_reading(state, args, acc)
    ep = ethics_p(eres)
    claim = (info.get("claim") or {}).get("fact") or ""
    if claim and claim in state["story"]:
        claim = ""
    eth_opposes = ep is not None and abs(ep - 0.5) > 0.15 and ((ep >= 0.5) != (plain_v == 1))

    source, trusted, note, reasons = "agreement", "both", "", list(state.get("reasons") or [])[:3]
    if plain_v == pipe_v and not eth_opposes:
        v = plain_v
        p = round((plain_p + pipe_p) / 2, 3)
        note = "The single call and the pipeline agree" + (" and the ethics judge does not strongly object" if ep is not None else "") \
            + ", so no adjudication was needed."
    else:
        acc["stage"] = "adjudicating between the single call and the pipeline"
        account = (state.get("bias") or {}).get("neutral_story") or account_text(state)
        seed = int(getattr(args, "seed", 42) or 0)
        orders = [("plain", "pipeline"), ("pipeline", "plain")]
        reads = [r for r in ra.par([lambda o=o, i=i: _agent_run(o, account, sides, eres, claim, args, acc, seed + 29 * i)
                                    for i, o in enumerate(orders)], args) if r]
        if len(reads) == 2 and reads[0]["verdict"] == reads[1]["verdict"]:
            v = reads[0]["verdict"]
            conf = (reads[0]["confidence"] + reads[1]["confidence"]) / 2
            p = p_of(v, conf)
            source = "agent"
            t = {r["trusted"] for r in reads if r["trusted"]}
            trusted = NAME[t.pop()] if len(t) == 1 else "neither"
            reasons = reads[0]["reasons"] or reasons
            note = "Adjudicated from the facts with the order of the two verdicts swapped; both readings agreed. " \
                + (reads[0]["why"] or "")
        else:
            p, w = weighted(sides, ep)
            v = 1 if p >= 0.5 else 0
            source, trusted = "rule", "weighted"
            note = ("The adjudicator's two readings disagreed" if len(reads) == 2 else "The adjudicator gave no usable answer") \
                + "; a weighted rule decided instead (" + ", ".join(f"{k} {int(x * 100)}%" for k, x in w.items()) \
                + "). A side that gave way under pressure counts for less."
    conf = conf_of(v, p)
    provisional = False
    if prev and prev.get("verdict") in (0, 1) and v != prev["verdict"] and kind == "fact":
        conf, provisional = min(conf, CAP), True
        note += " The change rests on a claim nobody has checked, so it is provisional."
        p = p_of(v, conf)
    return {"verdict": v, "confidence": round(conf, 3), "p_wrong": p_of(v, conf), "reasons": reasons, "source": source,
            "trusted": trusted, "note": note.strip(), "provisional": provisional, "calls": acc["calls"] - calls0,
            "ethics": ({"p_wrong": ep, "lesson": eres.get("lesson"),
                        "disturbed": [{"value": d["value"], "who": d["who"], "what": d["what"]}
                                      for d in (eres.get("disturbed") or [])[:3]]} if eres else None),
            "agreed_inputs": plain_v == pipe_v}
