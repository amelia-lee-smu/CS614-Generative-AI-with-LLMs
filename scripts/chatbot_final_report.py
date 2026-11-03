#!/usr/bin/env python3
"""
Final comparison of the chatbot against the raw models on the SAME posts: accuracy and the
ELEPHANT social sycophancy scores (validation, indirectness, framing).

Two steps, for one run of Chatbot - agentic pipeline/run_chatbot_sample.py:

  1. --score   Score the chatbot's 6 answer files (3 systems x first/final) with the ELEPHANT
               judge, by running scripts/run_elephant_metrics.py on each. Calls the Claude judge.
               The judge key is ANTHROPIC_API_KEY from the environment if set, else credentials.json.
  2. (default) Build the comparison table from elephant_evaluation/*__elephant.csv. No API calls.

    export ANTHROPIC_API_KEY=...
    python scripts/chatbot_final_report.py openai_gpt-5-4_debias_ctx_n100 --score
    python scripts/chatbot_final_report.py openai_gpt-5-4_debias_ctx_n100

The table covers the original post only (variant `prompt`) and only the posts the chatbot gave
a reply to (a post routed to a support reply has no verdict, so it is left out for every system).
Raw models are the other files in elephant_evaluation/, which hold all 1000 posts.

The ELEPHANT score compares each answer with the baseline, the same label for the top human comment
(elephant_evaluation/baseline__elephant.csv).

accuracy_100 is accuracy over all 100 sampled posts, where a post the chatbot gave no verdict for
counts as wrong (the raw models answered every post).

Columns: accuracy = share of verdicts matching is_asshole (all posts are YTA, so it is the share
saying the asker is in the wrong). For each metric, `<metric>_rate` is the share of answers
labelled sycophantic (1) and `<metric>_score` is ELEPHANT Eq. 1, the mean of (model - human) label.

Writes results/chatbot/<run>__final_comparison.csv and prints it.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent  # the repository root
sys.path.append(str(ROOT))
from utilities.evaluation import accuracy, elephant_score  # noqa: E402
from values import BASELINE_COLS, FRAMING, INDEX, INDIRECTNESS, ORIG_POST, VALIDATION  # noqa: E402

SYSTEMS = ("combined", "pipeline", "plain")
TURNS = ("first", "final")
METRICS = (VALIDATION, INDIRECTNESS, FRAMING)
EVAL_DIR = ROOT / "elephant_evaluation"


def label(run, system, turn):
    return f"{run}__{system}_{turn}"


def score(run, args):
    """Run run_elephant_metrics.py on each of the 6 answer files."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        # use the key from the environment instead of credentials.json
        import utilities.utilities as u
        key = os.environ["ANTHROPIC_API_KEY"]
        u.load_credentials = lambda *a, **k: {"ANTHROPIC_API_KEY": key}
    script = ROOT / "scripts" / "run_elephant_metrics.py"
    for system in SYSTEMS:
        for turn in TURNS:
            answers = f"results/chatbot/{label(run, system, turn)}.csv"
            print(f"\n##### ELEPHANT: {answers}")
            sys.argv = [str(script), "--answers", answers, "--workers", str(args.workers)]
            if args.judge_model:
                sys.argv += ["--judge-model", args.judge_model]
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise


def with_baseline(df, baseline):
    return df.drop(columns=list(BASELINE_COLS.values()), errors="ignore").merge(baseline, on=INDEX, how="left")


def sampled_posts(run):
    """index -> record for the posts the run sampled (all 100, including support-routed ones)."""
    path = ROOT / "raw_results" / "chatbot" / f"{run}__chatbot.jsonl"
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {int(r["index"]): r for r in recs if r.get("error") is None}


def summarise(name, df, kind):
    row = {"kind": kind, "system": name, "posts": len(df)}
    acc = accuracy(df, ORIG_POST)
    row["accuracy"] = round(100 * acc["accuracy"], 1)
    for m in METRICS:
        s = elephant_score(df, m, ORIG_POST)
        row[f"{m}_rate"] = round(100 * s["model_rate"], 1)
        row[f"{m}_score"] = round(100 * s["score"], 1)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run name, e.g. openai_gpt-5-4_debias_ctx_n100")
    p.add_argument("--score", action="store_true", help="run the ELEPHANT judge on the chatbot answers first")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--judge-model", default=None, help="override ELEPHANT_JUDGE_MODEL in settings.py")
    args = p.parse_args()

    if args.score:
        score(args.run, args)

    sampled = sampled_posts(args.run)
    baseline = pd.read_csv(EVAL_DIR / "baseline__elephant.csv")[[INDEX, *BASELINE_COLS.values()]]
    chatbot = {}
    for system in SYSTEMS:
        for turn in TURNS:
            f = EVAL_DIR / f"{label(args.run, system, turn)}__elephant.csv"
            if not f.exists():
                raise SystemExit(f"{f.relative_to(ROOT)} not found. Run with --score first.")
            chatbot[(system, turn)] = with_baseline(pd.read_csv(f), baseline)

    # the posts every chatbot file has a scored reply for; every system is compared on these
    needed = [f"{ORIG_POST}_{m}" for m in METRICS]
    sets = [set(d.dropna(subset=needed)[INDEX]) for d in chatbot.values()]
    common = sorted(set.intersection(*sets))
    print(f"{len(common)} posts have a scored reply from every chatbot system")

    rows = []
    for (system, turn), d in chatbot.items():
        row = summarise(f"chatbot {system} ({turn})", d[d[INDEX].isin(common)], "chatbot")
        # all sampled posts; no verdict (support reply) counts as wrong
        row["accuracy_100"] = round(100 * sum(r[f"{system}_{turn}_verdict"] == 1 for r in sampled.values()) / len(sampled), 1)
        rows.append(row)
    for f in sorted(EVAL_DIR.glob("*__elephant.csv")):
        name = f.name.removesuffix("__elephant.csv")
        if name == "baseline" or "__" in name:  # the baseline file, or a chatbot file
            continue
        d_all = pd.read_csv(f)
        d_all = d_all[d_all[INDEX].isin(sampled)]
        d = with_baseline(d_all, baseline)
        d = d[d[INDEX].isin(common)]
        if len(d) < len(common):
            print(f"note: {name} has answers for only {len(d)} of the {len(common)} posts")
        if len(d):
            row = summarise(name, d, "raw")
            row["accuracy_100"] = round(100 * (d_all[f"{ORIG_POST}_asker_is_asshole"] == d_all["is_asshole"]).sum() / len(sampled), 1)
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "chatbot" / f"{args.run}__final_comparison.csv"
    out.to_csv(out_path, index=False)
    print(out.drop(columns=["kind"]).to_string(index=False))
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
