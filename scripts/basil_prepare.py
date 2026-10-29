#!/usr/bin/env python3
"""
BASIL steps 1 and 2 (docs/basil_implementation_plan.md): for each selected post,
extract the poster's action phrase, then generate one piece of synthetic
evidence E (paper Appendix B.2). Both are made from the ORIGINAL post and shared
by every variant, so variants differ only in how the story is told.

    python scripts/basil_prepare.py --step action   --chunk-size 5 --chunk 0   # smoke test
    python scripts/basil_prepare.py --step action
    python scripts/basil_prepare.py --step evidence                            # needs actions.csv

Resumable: each finished post is appended to <out>.jsonl and skipped on re-run;
the CSV is rebuilt from the JSONL at the end of every run.

Default generator: GPT-5 on Azure, the paper's evidence generator (paper §4.1).
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from run_models_closed import build_client, call_model  # noqa: E402
from utilities.basil_prompts import ACTION_EXTRACTION_PROMPT, EVIDENCE_PROMPT  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_posts(csv_path, select_path):
    df = pd.read_csv(csv_path)
    if "index" not in df.columns:
        df["index"] = range(len(df))
    keep = set(pd.read_csv(select_path)["post_index"])
    return df[df["index"].isin(keep)].sort_values("index").reset_index(drop=True)


def load_done(path):
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None and rec.get("text"):
                done[rec["post_index"]] = rec
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, choices=["action", "evidence"])
    p.add_argument("--provider", default="azure", choices=["azure", "gpteal", "gemini"])
    p.add_argument("--model", default="gpt-5-2025-08-07")
    p.add_argument("--csv", default=str(ROOT / "dataset/dataset_generation_combined.csv"))
    p.add_argument("--select", default=str(ROOT / "selected_1000_posts.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "dataset/basil"))
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="Reasoning models count thinking tokens against this cap.")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / ("actions.jsonl" if args.step == "action" else "evidence.jsonl")

    posts = load_posts(args.csv, args.select)
    if args.step == "evidence":
        actions = pd.read_csv(out_dir / "actions.csv").set_index("post_index")["action"]
        posts = posts[posts["index"].isin(actions.index)]
    if args.chunk is not None:
        posts = posts.iloc[args.chunk * args.chunk_size:(args.chunk + 1) * args.chunk_size]

    done = load_done(out_jsonl)
    todo = [row for _, row in posts.iterrows() if int(row["index"]) not in done]
    print(f"{args.step}: {len(posts)} posts, {len(posts) - len(todo)} done, {len(todo)} to run")

    call_args = SimpleNamespace(provider=args.provider, max_tokens=args.max_tokens,
                                temperature=None, reasoning_effort=None, effort=None,
                                schema=None, timeout=args.timeout)
    lock = threading.Lock()

    def work(row):
        idx = int(row["index"])
        if args.step == "action":
            prompt = ACTION_EXTRACTION_PROMPT.format(post=row["prompt"])
        else:
            prompt = EVIDENCE_PROMPT.format(situation=row["prompt"], action=actions[idx])
        last_err = None
        for attempt in range(args.retries):
            try:
                data = call_model(query_apis, args.model, prompt, call_args, None)
                text = (data["text"] or "").strip().strip('"').strip()
                if text:
                    return {"post_index": idx, "text": text, "model": args.model,
                            "stop_reason": data.get("stop_reason"), "error": None}
                last_err = f"empty reply (stop_reason={data.get('stop_reason')})"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        return {"post_index": idx, "text": None, "model": args.model, "error": last_err}

    if todo:
        query_apis = build_client(args.provider)
        with open(out_jsonl, "a", encoding="utf-8") as fout, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            it = as_completed([ex.submit(work, r) for r in todo])
            for fut in (tqdm(it, total=len(todo)) if tqdm else it):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                if rec["error"] is None:
                    done[rec["post_index"]] = rec

    col = "action" if args.step == "action" else "evidence"
    out_csv = out_jsonl.with_suffix(".csv")
    pd.DataFrame([{"post_index": k, col: v["text"], f"{col}_model": v["model"]}
                  for k, v in sorted(done.items())]).to_csv(out_csv, index=False)
    print(f"{len(done)} posts -> {out_csv}")


if __name__ == "__main__":
    main()
