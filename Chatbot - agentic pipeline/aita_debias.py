#!/usr/bin/env python3
import copy
import random
import json
import re

import run_agentic as ra

rm = ra.rm

ITEMS = ("broke_commitment", "caused_harm", "had_better_options", "escalated", "acted_in_good_faith")
ITEM_LABELS = {
    "broke_commitment": "Broke a promise or duty",
    "caused_harm": "Caused harm",
    "had_better_options": "Had better options",
    "escalated": "Escalated",
    "acted_in_good_faith": "Acted in good faith",
}

NEUTRAL_SYS = (
    "Rewrite the story below as a short, neutral, third-person account. Call the narrator 'Person A' and the main "
    "other person 'Person B' (anyone else: 'Person C', 'Person D'). Keep every concrete fact, action, quote, number "
    "and timing exactly. Remove: the question 'am I the asshole' or 'AITA', the narrator's opinion about who is right, "
    "emotional or loaded wording, self-justification, and any judging adjectives. Do not add facts and do not give "
    "a verdict. Output only the account."
)
NEUTRAL_SYS_B = NEUTRAL_SYS.replace(
    "Call the narrator 'Person A' and the main other person 'Person B'",
    "Call the narrator 'Person B' and the main other person 'Person A'")
assert NEUTRAL_SYS_B != NEUTRAL_SYS

STEELMAN_SYS = (
    "You get a neutral account of a conflict between Person A and Person B. Write the strongest HONEST case, in at "
    "most 90 words, that {who} the one who bears more of the blame. Use only facts in the account; write "
    "'unknown' rather than invent anything. Output only the case."
)

FRAMING_RE = re.compile(r"\b(AITA|AITAH|WIBTA|am i the (asshole|a-?hole)|asshole)\b", re.I)
FIRST_PERSON_RE = re.compile(r"\b(I|me|my|mine|myself|I'm|I've|I'd|I'll)\b")
QUOTED_RE = re.compile(r"[\"\u201c][^\"\u201c\u201d]*[\"\u201d]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

CLAIM_OPEN = "[[UNVERIFIED CLAIM ADDED AFTER THE VERDICT: "
CLAIM_CLOSE = "]]"
CLAIM_RE = re.compile(re.escape(CLAIM_OPEN) + r"(.*?)" + re.escape(CLAIM_CLOSE), re.S)
UNVERIFIED_LABEL = "Unverified later claim by Person A"
UNVERIFIED_NOTE = (
    "\nA sentence that begins 'Unverified later claim' was added by the narrator AFTER seeing a verdict. Nobody has "
    "checked it and the other person has not confirmed it. Count it only as far as it is specific and plausible, never "
    "as proof. It cannot erase facts the rest of the account states, and a claim that excuses the narrator counts for "
    "less than one that admits fault."
)


def claim_marker(claim):
    return f"{CLAIM_OPEN}{claim.strip()}{CLAIM_CLOSE}"


def split_claims(story):
    claims = [c.strip() for c in CLAIM_RE.findall(story or "") if c.strip()]
    return CLAIM_RE.sub("", story or "").strip(), claims


def claim_block(claims):
    lines = []
    for c in claims:
        c = re.sub(r"\bthe narrator\b", "Person A", c, flags=re.I)
        lines.append(f"{UNVERIFIED_LABEL} (no evidence given; Person B has not confirmed it): {c}")
    return "\n".join(lines)


RUBRIC_HEAD = (
    "You are a careful, impartial judge. You get an account of a conflict between Person A and Person B. Score EACH "
    "person from 0 to 3 on the five items below, using ONLY facts stated in the account (unknown = 0). Fill in all "
    "ten numbers. Do not pick a winner and do not pick a side. Judge conduct, not likability, writing style, or who "
    "gives more detail. The first person in the account is not more likely to be right.\n"
    "Items (0 = none, 3 = serious or repeated):\n"
    "- broke_commitment: broke a promise, agreement, rule or clear duty.\n"
    "- caused_harm: caused real harm or cost to the other person (money, safety, trust, feelings).\n"
    "- had_better_options: had a clearly better, reasonable option and did not use it.\n"
    "- escalated: made things worse (insults, public shaming, retaliation, ultimatums).\n"
    "- acted_in_good_faith: made a real effort to be fair, communicate honestly or fix things (this lowers the "
    "person's total).\n"
    "Names: use the fields a_<item> for Person A and b_<item> for Person B. Then add up to 3 key_reasons: short plain "
    "sentences that name the person and the specific action. Output JSON only."
)

FEWSHOT = (
    "\nExamples (account -> scores):\n"
    "1) Person A did not pay half the rent for two months, Person B covered it, and when asked Person A called B "
    "controlling. -> a: broke_commitment 3, caused_harm 2, had_better_options 2, escalated 1, acted_in_good_faith 0; "
    "b: all 0 except acted_in_good_faith 2.\n"
    "2) Person A refused to lend a car to Person B who had crashed it twice and not paid for repairs. -> a: all 0 "
    "except acted_in_good_faith 2; b: broke_commitment 2, caused_harm 2, had_better_options 1, escalated 0, "
    "acted_in_good_faith 0.\n"
    "3) Person A cancelled on Person B the day before a wedding because of a minor argument, then posted about it "
    "publicly. -> a: broke_commitment 3, caused_harm 3, had_better_options 3, escalated 3, acted_in_good_faith 0; "
    "b: broke_commitment 0, caused_harm 0, had_better_options 1, escalated 1, acted_in_good_faith 1.\n"
    "Most stories have at least one person with a clear problem. 'Everyone is fine' is rare: score honestly, "
    "including when the narrator is the one at fault.\n"
)

FORCE_SYS = (
    "You get an account of a conflict between Person A and Person B, plus a score table. Name the single most "
    "blameworthy action in the account and who did it. Reply with JSON only: {\"party\": \"A\" | \"B\" | \"both\" | "
    "\"none\", \"action\": \"short description\"}. Use \"none\" ONLY if nobody did anything a reasonable person would "
    "call wrong. Do not favour a person because they tell the story."
)

COMPOSER_ADDENDUM = (
    "\n\nExtra rules: keep the verdict exactly as given. If the verdict is 1 (the person is in the wrong), say so "
    "plainly in the first sentence and name the specific action; do not soften it into reassurance, do not say "
    "'you are valid' or 'it is understandable' as a substitute for the verdict, and do not add praise that the "
    "reasons do not support. If the verdict is 0, say so without gushing. Be kind in tone, not in substance."
)

FORCE_SCHEMA = {"type": "object",
                "properties": {"party": {"type": "string", "enum": ["A", "B", "both", "none"]},
                               "action": {"type": "string"}},
                "required": ["party", "action"]}


def rubric_system(order=None, few_shot=False):
    lines = RUBRIC_HEAD.split("\n")
    idx = [k for k, ln in enumerate(lines) if ln.startswith("- ") and ln[2:].split(":")[0] in ITEMS]
    bullets = {lines[k][2:].split(":")[0]: lines[k] for k in idx}
    for pos, item in zip(idx, order or ITEMS):
        lines[pos] = bullets[item]
    return "\n".join(lines) + (FEWSHOT if few_shot else "")


def rubric_schema(order=None, b_first=False):
    people = ("b", "a") if b_first else ("a", "b")
    props = {f"{p}_{i}": {"type": "integer"} for p in people for i in (order or ITEMS)}
    props["key_reasons"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props)}


def add_flags(add):
    add("--debias", action="store_true",
        help="preset: neutralize + rubric + steelman + consistency + few-shot + flip-guard.")
    add("--neutralize", action="store_true", help="rewrite the story neutrally in third person first.")
    add("--rubric", action="store_true", help="rubric judge; the verdict is computed from its scores.")
    add("--steelman", action="store_true", help="add the strongest honest case against the asker for the judge.")
    add("--consistency", action="store_true", help="role-swap check plus forced pass when runs disagree.")
    add("--few-shot", action="store_true", help="put examples of clearly-in-the-wrong askers in the rubric prompt.")
    add("--jury", type=int, default=1, help="number of rubric judgments (consistency rounds it up to an even number).")
    add("--jury-models", default="", help="comma list of models for the jury, e.g. llama3.1:8b,qwen:7b,gemma2:9b.")
    add("--jury-temp", type=float, default=None, help="sampling temperature for jury calls (only if chat supports it).")
    add("--threshold", type=float, default=0.45, help="asker share at or above this = in the wrong (calibrate it).")
    add("--min-fault", type=int, default=2, help="asker needs at least this rubric total to be called in the wrong.")
    add("--aggregate", default="share", choices=["share", "vote"], help="compare mean share or vote fraction to the threshold.")
    add("--calibration", default=None, help="calibration.json written by calibrate_threshold.py.")
    add("--no-debate", action="store_true", help="skip the advocates; judge the neutral story directly.")
    add("--dual-neutral", action="store_true",
        help="second independent neutral rewrite (narrator = Person B) used for the role-swapped votes.")
    add("--clean-rewrite", action="store_true",
        help="check the neutral rewrite in code (leaked first person / AITA, dropped numbers, missing labels); retry once.")
    add("--tie-band", type=float, default=0.0,
        help="asker share within this distance of 0.5 goes to the forced pass (0 = off, 0.10 is a good start).")
    add("--sym-force", action="store_true",
        help="forced pass is asked twice with the labels swapped; only an answer both runs agree on is used.")
    add("--ethics-weight", type=float, default=0.0,
        help="0-0.5: how far the ethics agent (six values, role-swapped) may move the blame share. 0 = off.")
    add("--blindspots", action="store_true",
        help="list facts the one-sided account cannot show; cap confidence when they could flip the verdict.")
    add("--identity-blind", action="store_true",
        help="rewrite without gender, race, religion, nationality or status unless essential to what happened.")
    add("--audit", action="store_true",
        help="after the verdict, a second model checks it for lean toward the narrator (flag + confidence cap only).")
    add("--audit-model", default=None, help="model for --audit (default: last of --jury-models, else the judge model).")
    add("--openai-api-key", default=None,
        help="OpenAI key for models given as openai:<name> (else $OPENAI_API_KEY, else config.yaml openai_api_key).")
    add("--openai-config", default="config.yaml", help="yaml file holding openai_api_key (and optional pricing:).")
    add("--openai-spend-limit", type=float, default=0.50,
        help="stop calling OpenAI after this many USD in the current process (0 = no limit).")
    add("--openai-reasoning", default="low",
        help="reasoning_effort sent to OpenAI models (low keeps answers fast and cheap; dropped if the model refuses it).")
    add("--claim-bar", action="store_true",
        help="follow-up claims: two independent readings must agree, a claim that excuses the narrator needs a checkable detail.")
    add("--flip-confirm", action="store_true",
        help="a follow-up that would move the verdict toward 'not in the wrong' must be reproduced by a second, sampled re-judge.")
    add("--flip-guard", action="store_true",
        help="a verdict that flips to 'not in the wrong' after follow-up claims must be reproduced by a second run.")


def apply_preset(args):
    if getattr(args, "debias", False):
        for k in ("neutralize", "rubric", "steelman", "consistency", "few_shot", "flip_guard",
                  "dual_neutral", "clean_rewrite", "sym_force", "audit", "blindspots", "identity_blind",
                  "claim_bar", "flip_confirm"):
            setattr(args, k, True)
        if not getattr(args, "ethics_weight", 0.0):
            args.ethics_weight = 0.25
        if not getattr(args, "tie_band", 0.0):
            args.tie_band = 0.10
    return args


def enabled(args):
    return bool(getattr(args, "rubric", False) or getattr(args, "steelman", False)
                or getattr(args, "consistency", False) or getattr(args, "few_shot", False)
                or int(getattr(args, "jury", 1) or 1) > 1)


def _chat(model, system, user, args, acc, schema=None, max_tokens=300, temperature=None, seed=None):
    kw = {"max_tokens": max_tokens}
    if schema is not None:
        kw["schema"] = schema
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    extra = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if seed is not None:
        extra["seed"] = seed
    if extra:
        try:
            return ra.chat(model, msgs, args, acc, **extra, **kw)[0]
        except TypeError:
            pass
    return ra.chat(model, msgs, args, acc, **kw)[0]


def swap_roles(text):
    return re.sub(r"Person ([AB])\b", lambda m: "Person B" if m.group(1) == "A" else "Person A", text or "")


def humanize(text):
    t = re.sub(r"Person A's", "your", text or "")
    t = re.sub(r"Person A\b", "you", t)
    return re.sub(r"Person B\b", "the other person", t)


def rewrite_problems(original, rewritten):
    problems = []
    if FIRST_PERSON_RE.search(QUOTED_RE.sub(" ", rewritten)):
        problems.append("it still uses first-person words (I, me, my); the narrator must be called Person A or Person B")
    if FRAMING_RE.search(rewritten):
        problems.append("it still contains the 'am I the asshole' question")
    missing = sorted({n for n in NUMBER_RE.findall(original) if len(n) >= 2 and n not in rewritten})
    if missing:
        problems.append("it dropped these numbers: " + ", ".join(missing[:6]))
    if "Person A" not in rewritten or "Person B" not in rewritten:
        problems.append("it did not use the labels Person A and Person B")
    return problems


def _scrub_framing(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [x for x in parts if not FRAMING_RE.search(x)]
    return " ".join(kept) if kept else text


def neutralize(story, args, acc, narrator="A"):
    acc["stage"] = "rewriting the story neutrally" + (" (second, independent rewrite)" if narrator == "B" else "")
    m = args.judge_model or args.model
    system = NEUTRAL_SYS_B if narrator == "B" else NEUTRAL_SYS
    if getattr(args, "identity_blind", False):
        system += (" Do not mention anyone's gender, race, ethnicity, religion, nationality, wealth or job prestige unless "
                   "it is essential to understand what happened; use role words (friend, roommate, boss, parent) and 'they'.")
    tokens = getattr(args, "agent_tokens", 300) * 2
    out = _chat(m, system, story, args, acc, max_tokens=tokens).strip()
    if getattr(args, "clean_rewrite", False):
        problems = rewrite_problems(story, out)
        if problems:
            fix = "Your previous rewrite had problems: " + "; ".join(problems) + ". Rewrite the story again, fixing them."
            retry = _chat(m, system, f"{story}\n\n{fix}", args, acc, max_tokens=tokens).strip()
            if retry and len(rewrite_problems(story, retry)) <= len(problems):
                out = retry
        out = _scrub_framing(out)
    return out


def steelman(text, args, acc):
    acc["stage"] = "building the strongest case against each person"
    m = args.advocate_model or args.model
    tokens = getattr(args, "agent_tokens", 300)

    def case(who):
        return lambda: _chat(m, STEELMAN_SYS.format(who=who), text, args, acc, max_tokens=tokens).strip()
    a, b = ra.par([case("Person A is"), case("Person B is")], args)
    return {"A": a, "B": b}


def steel_text(steel):
    if not steel:
        return ""
    return ("Case that you bear more of the blame: " + humanize(steel["A"])
            + "\n\nCase that the other person does: " + humanize(steel["B"]))


def _int(x):
    try:
        return max(0, min(3, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def party_score(fields):
    s = fields["broke_commitment"] + fields["caused_harm"] + fields["had_better_options"] + fields["escalated"]
    return max(0, s - fields["acted_in_good_faith"])


def decide(asker, other, thr, min_fault):
    total = asker + other
    share = asker / total if total else 0.5
    return share, int(share >= thr and asker >= min_fault)


def judge_once(text, steel, args, acc, model, swap, claims="", text_swapped=None, seed=None, temperature=None,
               variant=None):
    t = (text_swapped or swap_roles(text)) if swap else text
    user = f"ACCOUNT:\n{t}"
    if steel:
        case_a, case_b = (swap_roles(steel["B"]), swap_roles(steel["A"])) if swap else (steel["A"], steel["B"])
        user += (f"\n\nCASE THAT PERSON A BEARS MORE OF THE BLAME (check every claim against the account):\n{case_a}"
                 f"\n\nCASE THAT PERSON B BEARS MORE OF THE BLAME (check every claim against the account):\n{case_b}")
    if claims:
        user += f"\n\nCLAIMS MADE BY TWO ADVOCATES (may be exaggerated; use only what the account supports):\n{swap_roles(claims) if swap else claims}"
    order, b_first = list(ITEMS), False
    if variant:
        random.Random(1000 + int(variant)).shuffle(order)
        b_first = int(variant) % 3 == 1
    system = rubric_system(order, getattr(args, "few_shot", False))
    if "Unverified later claim" in t:
        system += UNVERIFIED_NOTE
    raw = _chat(model, system, user, args, acc, schema=rubric_schema(order, b_first), max_tokens=getattr(args, "judge_tokens", 400),
                temperature=temperature, seed=seed)
    j = ra.judge_json(raw) or {}
    parsed = {p: {i: _int(j.get(f"{p}_{i}")) for i in ITEMS} for p in ("a", "b")}
    n_ok = sum(v is not None for p in parsed.values() for v in p.values())
    out = {"model": model, "swap": swap, "ok": n_ok >= 6}
    if not out["ok"]:
        return out
    for p in parsed.values():
        for k, v in p.items():
            p[k] = v or 0
    a, b = parsed["a"], parsed["b"]
    asker, other = (b, a) if swap else (a, b)
    reasons = [r.strip() for r in (j.get("key_reasons") or []) if isinstance(r, str) and r.strip()]
    if swap:
        reasons = [swap_roles(r) for r in reasons]
    out.update(asker=asker, other=other, asker_score=party_score(asker), other_score=party_score(other),
               reasons=reasons)
    out["share"], out["verdict"] = decide(out["asker_score"], out["other_score"], args.threshold, args.min_fault)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _mean_fields(judgments, who):
    return {i: round(_mean([j[who][i] for j in judgments]), 2) for i in ITEMS}


def force_pass(text, summary, args, acc):
    acc["stage"] = "forcing a decision on the most blameworthy action"
    raw = _chat(args.judge_model or args.model, FORCE_SYS, f"ACCOUNT:\n{text}\n\nSCORES:\n{summary}", args, acc,
                schema=FORCE_SCHEMA, max_tokens=150)
    j = ra.judge_json(raw) or {}
    party = str(j.get("party", "")).strip().upper()
    party = {"BOTH": "both", "NONE": "none"}.get(party, party)
    return party if party in ("A", "B", "both", "none") else None, (j.get("action") or "").strip()


AUDIT_SYS = (
    "You audit a verdict on a conflict for SYCOPHANCY toward the narrator. Person A is the narrator and the one who "
    "asked. You get a neutral account, the verdict (1 = Person A is in the wrong, 0 = Person A is not in the wrong) "
    "and the reasons given. Check only against facts in the account. Reply with JSON only: "
    "{\"supported\": 0 or 1, \"leans_toward_narrator\": 0 or 1, \"note\": \"one short sentence\"}. "
    "supported=1 only if the facts in the account justify the verdict. leans_toward_narrator=1 if the verdict or reasons "
    "excuse Person A for something the account shows, or reward them for being the one who tells the story."
)
AUDIT_SCHEMA = {"type": "object",
                "properties": {"supported": {"type": "integer", "enum": [0, 1]},
                               "leans_toward_narrator": {"type": "integer", "enum": [0, 1]},
                               "note": {"type": "string"}},
                "required": ["supported", "leans_toward_narrator", "note"]}


def fix_grammar(t):
    for a, b in ((r"\byou is\b", "you are"), (r"\bYou is\b", "You are"), (r"\byou was\b", "you were"),
                 (r"\bYou was\b", "You were"), (r"\byou has\b", "you have"), (r"\bYou has\b", "You have"),
                 (r"\byou does\b", "you do"), (r"\bYou does\b", "You do")):
        t = re.sub(a, b, t)
    return t


def audit(text, verdict, reasons, args, acc, model):
    acc["stage"] = "auditing the verdict for lean toward the asker"
    user = (f"ACCOUNT:\n{text}\n\nVERDICT: {verdict}\nREASONS: {json.dumps(reasons, ensure_ascii=False)}")
    raw = _chat(model, AUDIT_SYS, user, args, acc, schema=AUDIT_SCHEMA, max_tokens=150)
    j = ra.judge_json(raw) or {}
    if j.get("supported") not in (0, 1):
        return None
    return {"model": model, "supported": int(j["supported"]), "leans": int(j.get("leans_toward_narrator") or 0),
            "note": fix_grammar(humanize(str(j.get("note") or "")))}


SOFT_PHRASES = [
    r"your feelings are (?:completely |totally |entirely )?valid", r"(?:completely|totally|entirely|perfectly) understandable",
    r"you did nothing wrong", r"you(?:'re| are) not the (?:bad guy|villain)", r"you did (?:the )?best you could",
    r"anyone would have (?:done|reacted)", r"don't be so hard on yourself", r"you(?:'re| are) (?:so )?(?:brave|amazing|wonderful)",
]
GUSH_PHRASES = [r"(?:absolutely|totally|completely|100%) (?:right|justified|correct)", r"you(?:'re| are) a (?:saint|hero)"]


def lint_reply(reply, verdict):
    import re as _re
    flags = []
    low = (reply or "").lower()
    if verdict == 1:
        flags += [f"reassurance: \u201c{m.group(0)}\u201d" for pat in SOFT_PHRASES for m in [_re.search(pat, low)] if m]
    if verdict == 0:
        flags += [f"gushing: \u201c{m.group(0)}\u201d" for pat in GUSH_PHRASES for m in [_re.search(pat, low)] if m]
    if verdict in (0, 1) and reply and not _re.search(r"other (?:person|side)|they might|their (?:side|view|point)", low):
        flags.append("the reply never mentions the other person's side")
    return flags


def ethics_signal(text, args, acc):
    import aita_ethics as et
    try:
        c = et.compass_text(text, args, acc)
    except ValueError:
        return None
    bad_a = sum(max(0.0, -x) for x in c["asker"].values())
    bad_b = sum(max(0.0, -x) for x in c["other"].values())
    c["harm"] = {"asker": round(bad_a, 2), "other": round(bad_b, 2)}
    c["share"] = bad_a / (bad_a + bad_b) if (bad_a + bad_b) > 0 else None
    return c


BLIND_SYS = (
    "You read ONE side's account of a conflict. Person A is the narrator; the other side has not been heard. List up "
    "to 3 specific facts that are NOT in the account, that the other person would likely add or dispute, and that "
    "could change who is mostly at fault. Reply with JSON only: {\"missing\": [{\"question\": \"short question\", "
    "\"matters\": \"short phrase: why it could change the verdict\"}], \"stakes\": \"low\" | \"medium\" | \"high\"}. "
    "stakes = how plausibly the missing facts could flip the verdict (high = it could easily flip). Do not guess the answers."
)
BLIND_SCHEMA = {"type": "object",
                "properties": {"missing": {"type": "array", "items": {
                    "type": "object", "properties": {"question": {"type": "string"}, "matters": {"type": "string"}},
                    "required": ["question", "matters"]}},
                               "stakes": {"type": "string", "enum": ["low", "medium", "high"]}},
                "required": ["missing", "stakes"]}


def blindspots(text, verdict, args, acc):
    acc["stage"] = "looking for what a one-sided account cannot show"
    raw = _chat(args.judge_model or args.model, BLIND_SYS, f"ACCOUNT:\n{text}\n\nVERDICT GIVEN: {verdict}", args, acc,
                schema=BLIND_SCHEMA, max_tokens=250)
    j = ra.judge_json(raw) or {}
    if j.get("stakes") not in ("low", "medium", "high"):
        return None
    items = []
    for m in (j.get("missing") or [])[:3]:
        if isinstance(m, dict) and str(m.get("question") or "").strip():
            items.append({"question": fix_grammar(humanize(str(m["question"]).strip())),
                          "matters": fix_grammar(humanize(str(m.get("matters") or "").strip()))})
    return {"stakes": j["stakes"], "missing": items}


def force_pass_sym(text, text_b, totals, args, acc):
    a_tot, b_tot, share = totals
    s1 = f"Person A total {a_tot:.1f}, Person B total {b_tot:.1f} (A share {share:.2f})."
    s2 = f"Person A total {b_tot:.1f}, Person B total {a_tot:.1f} (A share {1 - share:.2f})."
    (p1, act1), (p2, _act2) = ra.par([lambda: force_pass(text, s1, args, acc),
                                      lambda: force_pass(text_b or swap_roles(text), s2, args, acc)], args)
    p2 = {"A": "B", "B": "A"}.get(p2, p2)
    if p1 is not None and p1 == p2:
        return p1, act1, True
    if p1 is None and p2 is None:
        return None, "", True
    return None, "", False


def run(story, args, acc):
    bias = {"flags": [], "steps": []}
    base_story, later_claims = split_claims(story)
    story = base_story or story
    text = story
    acc["stage"] = "rewriting the story neutrally"
    text = neutralize(story, args, acc) or story
    bias["steps"].append("neutral rewrite")
    labelled = "Person A" in text and "Person B" in text
    if not labelled:
        text = "(Person A is the narrator. Person B is the main other person.)\n" + text
        bias["flags"].append("unlabelled")
    block = claim_block(later_claims) if later_claims else ""
    if block:
        text = text.rstrip() + "\n\n" + block
        bias["claims"] = later_claims
        bias["steps"].append("unverified claim kept separate")
    bias["neutral_story"] = text
    can_swap = labelled
    text_b = None
    if labelled and getattr(args, "dual_neutral", False) and getattr(args, "consistency", False):
        cand = neutralize(story, args, acc, narrator="B")
        if "Person A" in cand and "Person B" in cand:
            text_b = cand.rstrip() + ("\n\n" + swap_roles(block) if block else "")
            bias["neutral_story_b"] = cand
            bias["steps"].append("second neutral rewrite")
        else:
            bias["flags"].append("dual_rewrite_unlabelled")

    trace, claims, debate_raw = None, "", None
    if not getattr(args, "no_debate", False):
        acc["stage"] = "running the debate"
        a2 = copy.copy(args)
        a2.compose = False
        debate_raw, _, trace = ra.run_debate(args.model, text, a2, acc)
        if trace.get("routed"):
            return {"routed": True, "reply": trace.get("final_reply"), "trace": trace}
        wa, ra_ = trace.get("argument_in_the_wrong"), trace.get("argument_not_in_the_wrong")
        pieces = []
        if wa:
            pieces.append("Case that Person A is in the wrong: " + wa)
        if ra_:
            pieces.append("Case that Person A is not in the wrong: " + ra_)
        claims = "\n".join(pieces)
        bias["steps"].append("debate")

    steel = steelman(text, args, acc) if getattr(args, "steelman", False) else None
    if steel:
        bias["steps"].append("steelman (both sides)")

    models = [m.strip() for m in (getattr(args, "jury_models", "") or "").split(",") if m.strip()] \
        or [args.judge_model or args.model]
    n = max(1, int(getattr(args, "jury", 1) or 1))
    consistency = bool(getattr(args, "consistency", False)) and can_swap
    if getattr(args, "consistency", False) and not can_swap:
        bias["flags"].append("no_swap")
    if consistency and n % 2:
        n += 1
    temp = getattr(args, "jury_temp", None)
    if temp is None and n > 1:
        temp = 0.7
    base_seed = int(getattr(args, "seed", 42) or 0)

    def vote(i):
        if consistency:
            model, swap = models[(i // 2) % len(models)], bool(i % 2)
        else:
            model, swap = models[i % len(models)], False
        return lambda: judge_once(text, steel, args, acc, model, swap, claims,
                                  text_swapped=text_b if swap else None,
                                  seed=base_seed + 101 * i if n > 1 else None, temperature=temp,
                                  variant=i if n > 1 else None)
    acc["stage"] = f"judging on the rubric ({n} vote{'s' if n > 1 else ''})"
    judgments = ra.par([vote(i) for i in range(n)], args)
    ok = [j for j in judgments if j["ok"]]
    bias["steps"].append(f"rubric x{len(judgments)}")
    if not ok:
        bias["flags"].append("rubric_failed")
        return {"fallback": True, "debate_raw": debate_raw, "trace": trace, "bias": bias}

    thr, min_fault = args.threshold, args.min_fault
    share = _mean([j["share"] for j in ok])
    vote_frac = _mean([j["verdict"] for j in ok])
    asker_mean, other_mean = _mean([j["asker_score"] for j in ok]), _mean([j["other_score"] for j in ok])
    ew = max(0.0, min(0.5, float(getattr(args, "ethics_weight", 0.0) or 0.0)))
    eth = None
    if ew > 0:
        acc["stage"] = "the ethics agent is reading the account"
        eth = ethics_signal(text, args, acc)
        if eth is None:
            bias["flags"].append("ethics_unparsed")
        elif eth["share"] is None:
            bias["flags"].append("ethics_no_signal")
            eth = None
        else:
            bias["rubric_share"] = round(share, 3)
            share = (1 - ew) * share + ew * eth["share"]
            vote_frac = (1 - ew) * vote_frac + ew * eth["share"]
            bias["steps"].append("ethics agent")
    if args.aggregate == "vote":
        score, verdict = vote_frac, int(vote_frac >= thr)
    else:
        score, verdict = share, int(share >= thr and asker_mean >= min_fault)

    orig, swp = [j for j in ok if not j["swap"]], [j for j in ok if j["swap"]]
    if orig and swp:
        v_o = decide(_mean([j["asker_score"] for j in orig]), _mean([j["other_score"] for j in orig]), thr, min_fault)[1]
        v_s = decide(_mean([j["asker_score"] for j in swp]), _mean([j["other_score"] for j in swp]), thr, min_fault)[1]
        bias["original_verdict"], bias["swapped_verdict"] = v_o, v_s
        if v_o != v_s:
            bias["flags"].append("role_swap_disagree")
    if verdict == 0 and asker_mean < min_fault and other_mean < min_fault:
        bias["flags"].append("no_fault_found")
    tie_band = float(getattr(args, "tie_band", 0.0) or 0.0)
    if tie_band > 0 and abs(share - 0.5) <= tie_band and (asker_mean + other_mean) > 0:
        bias["flags"].append("near_tie")

    forced = None
    triggers = [f for f in ("role_swap_disagree", "no_fault_found", "near_tie") if f in bias["flags"]]
    if getattr(args, "consistency", False) and triggers:
        if getattr(args, "sym_force", False):
            party, action, agreed = force_pass_sym(text, text_b, (asker_mean, other_mean, share), args, acc)
            if not agreed:
                bias["flags"].append("force_disagree")
        else:
            summary = (f"Person A total {asker_mean:.1f}, Person B total {other_mean:.1f} "
                       f"(A share {share:.2f}).")
            party, action = force_pass(text, summary, args, acc)
        if party is not None:
            if party == "both" and triggers == ["near_tie"]:
                new_v = verdict
            else:
                new_v = 1 if party in ("A", "both") else 0
            forced = {"party": party, "action": humanize(action), "changed": new_v != verdict}
            verdict = new_v
            bias["steps"].append("forced pass")
    bias["forced"] = forced

    agree = _mean([int(j["verdict"] == verdict) for j in ok])
    margin = min(1.0, abs(score - thr) / max(thr, 1 - thr, 1e-6))
    conf = round(0.5 * margin + 0.5 * agree, 2)
    if forced and forced["changed"]:
        conf = min(conf, 0.6)
    if "force_disagree" in bias["flags"]:
        conf = min(conf, 0.5)

    reasons = []
    for j in ok:
        if j["verdict"] == verdict:
            reasons += j["reasons"]
    if forced and forced["action"]:
        reasons.insert(0, forced["action"])
    seen, uniq = set(), []
    for r in reasons:
        r = humanize(r)
        r = r[:1].upper() + r[1:]
        if r.lower() not in seen:
            seen.add(r.lower())
            uniq.append(r)
    reasons = uniq[:4] or ["The rubric scores favour this verdict."]

    if getattr(args, "audit", False):
        audit_model = getattr(args, "audit_model", None) or (models[-1] if len(models) > 1 else (args.judge_model or args.model))
        res = audit(text, verdict, reasons, args, acc, audit_model)
        bias["audit"] = res
        if res is None:
            bias["flags"].append("audit_unparsed")
        elif not res["supported"] or res["leans"]:
            bias["flags"].append("audit_disagrees")
            conf = min(conf, 0.5)
        bias["steps"].append("audit")

    if getattr(args, "blindspots", False):
        blind = blindspots(text, verdict, args, acc)
        bias["blindspots"] = blind
        if blind is None:
            bias["flags"].append("blindspots_unparsed")
        else:
            bias["steps"].append("blind spots")
            if blind["stakes"] == "high":
                bias["flags"].append("blindspots_high")
                conf = min(conf, 0.6)
    if eth is not None:
        bias["ethics"] = {**eth, "weight": ew, "blended_share": round(share, 3)}

    bias.update(
        share=round(share, 3), vote_frac=round(vote_frac, 3), threshold=thr, aggregate=args.aggregate,
        asker=_mean_fields(ok, "asker"), other=_mean_fields(ok, "other"),
        asker_score=round(asker_mean, 2), other_score=round(other_mean, 2),
        votes=[{"model": j["model"], "swap": j["swap"], "verdict": j["verdict"], "share": round(j["share"], 2),
                "none": (j["asker_score"] + j["other_score"]) == 0}
               for j in ok],
        n_failed=len(judgments) - len(ok), steelman=steel_text(steel))
    return {"verdict": verdict, "confidence": conf, "reasons": reasons, "bias": bias, "trace": trace,
            "debate_raw": debate_raw}


VERDICT_LINE = {1: "Verdict: you are in the wrong here.", 0: "Verdict: you are not in the wrong here."}
_SAYS_NOT_WRONG = re.compile(r"\b(not in the wrong|not the asshole|not at fault|did nothing wrong|nta)\b", re.I)
_SAYS_WRONG = re.compile(r"\b(in the wrong|at fault|were wrong|are wrong|the asshole|your fault|yta)\b", re.I)
_YOU = re.compile(r"\b(you|your|you're)\b", re.I)


def _sentences(par):
    return [x for x in re.split(r"(?<=[.!?])\s+", par.strip()) if x]


def _opposes(sentence, verdict):
    if not _YOU.search(sentence):
        return False
    if verdict == 1:
        return bool(_SAYS_NOT_WRONG.search(sentence))
    return bool(_SAYS_WRONG.search(_SAYS_NOT_WRONG.sub("", sentence)))


def contradicts_verdict(reply, verdict):
    return verdict in (0, 1) and any(_opposes(x, verdict) for x in _sentences(reply or ""))


def lock_verdict(reply, verdict):
    if verdict not in (0, 1) or not reply:
        return reply
    pars = []
    for par in re.split(r"\n\s*\n", reply.strip()):
        kept = [x for x in _sentences(par) if not _opposes(x, verdict)]
        if kept:
            pars.append(" ".join(kept))
    reply = "\n\n".join(pars)
    if not reply:
        return VERDICT_LINE[verdict]
    first = _sentences(reply)[0]
    says_not = bool(_SAYS_NOT_WRONG.search(first))
    says_wrong = bool(_SAYS_WRONG.search(_SAYS_NOT_WRONG.sub("", first)))
    ok = says_wrong if verdict == 1 else says_not
    if ok and not (says_not if verdict == 1 else says_wrong):
        return reply
    return VERDICT_LINE[verdict] + "\n\n" + reply


def reply_problems(reply, verdict):
    out = []
    if contradicts_verdict(reply, verdict):
        out.append("it states the opposite of the verdict (the verdict is "
                   + ("'you are in the wrong'" if verdict == 1 else "'you are not in the wrong'") + ")")
    out += [f for f in lint_reply(reply, verdict) if f.startswith(("reassurance", "gushing"))]
    return out


def compose(story, verdict, reasons, args, acc):
    acc["stage"] = "writing the reply"
    story, claims = split_claims(story)
    note = ("\n\nThe person added claims after the verdict that nobody has checked: " + "; ".join(claims)
            + ". Mention them only as unverified claims, never as established facts.") if claims else ""
    msg = (f"STORY:\n{story}{note}\n\nVERDICT: {verdict} (1 = the person is in the wrong, 0 = not in the wrong)\n"
           f"KEY REASONS: {json.dumps(reasons, ensure_ascii=False)}")
    m = args.composer_model or args.judge_model or args.model
    system = ra.COMPOSER_SYS + COMPOSER_ADDENDUM
    reply, _ = ra.chat(m, [{"role": "system", "content": system}, {"role": "user", "content": msg}], args, acc,
                       max_tokens=args.compose_tokens)
    problems = reply_problems(reply, verdict)
    if problems:
        fix = ("Your previous reply had problems: " + "; ".join(problems)
               + ". Write the reply again. Keep the verdict exactly as given.")
        retry, _ = ra.chat(m, [{"role": "system", "content": system},
                               {"role": "user", "content": msg + "\n\n" + fix}], args, acc,
                           max_tokens=args.compose_tokens)
        if len(reply_problems(retry, verdict)) <= len(problems):
            reply = retry
    return lock_verdict(reply, verdict)
