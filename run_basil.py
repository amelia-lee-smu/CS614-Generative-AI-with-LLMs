#!/usr/bin/env python3
"""
BASIL step 3 (docs/basil_implementation_plan.md): elicit the 7 probabilities
(paper §3.1, §3.2.1; Appendix D.2.1) for every selected post, in each of the 8
texts (original + 7 variants), from ONE model.

    python run_basil.py --provider azure --model gpt-4o-2024-08-06 --chunk-size 5 --chunk 0     # smoke test
    python run_basil.py --provider azure --model gpt-4o-2024-08-06                              # direct probing
    python run_basil.py --provider azure --model gpt-5-2025-08-07 --no-temperature              # model rejects temperature
    python run_basil.py --provider gpteal --model claude-sonnet-5 --method samples --no-temperature

Methods (paper §4.2):
    direct    one call per quantity at temperature 0 (greedy)
    samples   k calls per quantity at temperature 1.0 ([Our choice]; the paper says "above zero")

--no-temperature sends no temperature for models that reject one (the GPT-5
family's reasoning models, newer Claude models). Records carry "greedy": false,
and every results table must flag those models (plan, step 3).

Needs dataset/basil/actions.csv and evidence.csv from scripts/basil_prepare.py.
Each prompt is a single-turn plain-text request with no system prompt and no
JSON schema, as in the paper. Output, resumable, one line per
(post, text, quantity, sample):
    raw_results/basil/<model>__<method>.jsonl
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

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from run_models import safe_name  # noqa: E402
from run_models_closed import build_client, call_model  # noqa: E402
from utilities.basil_prompts import (  # noqa: E402
    BASIL_TEXTS,
    QUANTITIES,
    build_elicitation_prompt,
    parse_probability,
    pick_agent,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def build_tasks(df, actions, evidence, texts, k, seed):
    tasks = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        if idx not in actions.index or idx not in evidence.index:
            continue
        agent = pick_agent(idx, [row.get(t) for t in BASIL_TEXTS], seed=seed)
        for t in texts:
            situation = row.get(t)
            if not isinstance(situation, str) or not situation.strip():
                continue  # e.g. the 3 posts with no original_rewritten
            for q in QUANTITIES:
                prompt = build_elicitation_prompt(q, situation, actions[idx], evidence[idx], agent)
                for s in range(k):
                    tasks.append({"post_index": idx, "text": t, "quantity": q,
                                  "sample": s, "agent": agent, "prompt": prompt})
    return tasks


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add((r["post_index"], r["text"], r["quantity"], r["sample"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["azure", "gpteal", "gemini"])
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="direct", choices=["direct", "samples"])
    p.add_argument("--k", type=int, default=5, help="Samples per quantity for --method samples (paper: 5).")
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--no-temperature", action="store_true",
                   help="Send no temperature (for models that reject one). Marks records greedy=false.")
    p.add_argument("--texts", nargs="+", default=BASIL_TEXTS, choices=BASIL_TEXTS)
    p.add_argument("--csv", default=str(ROOT / "dataset/dataset_generation_combined.csv"))
    p.add_argument("--select", default=str(ROOT / "selected_1000_posts.csv"))
    p.add_argument("--basil-dir", default=str(ROOT / "dataset/basil"))
    p.add_argument("--out-dir", default=str(ROOT / "raw_results/basil"))
    p.add_argument("--chunk", type=int, default=None, help="Chunk of posts (0 = first). Omit for all.")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42, help="Agent-name seed (paper: seed 42, Appendix C).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Reasoning models count thinking tokens against this cap; raise it for them.")
    p.add_argument("--reasoning-effort", default=None, help="Azure reasoning models only.")
    p.add_argument("--effort", default=None, help="GPTeal Claude only.")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    args = p.parse_args()

    k = 1 if args.method == "direct" else args.k
    if args.no_temperature:
        temperature = None
    else:
        temperature = 0.0 if args.method == "direct" else args.sample_temperature
    greedy = args.method == "direct" and temperature == 0.0

    df = pd.read_csv(args.csv)
    if "index" not in df.columns:
        df["index"] = range(len(df))
    keep = set(pd.read_csv(args.select)["post_index"])
    df = df[df["index"].isin(keep)].sort_values("index").reset_index(drop=True)
    if args.chunk is not None:
        df = df.iloc[args.chunk * args.chunk_size:(args.chunk + 1) * args.chunk_size]

    basil_dir = Path(args.basil_dir)
    actions = pd.read_csv(basil_dir / "actions.csv").set_index("post_index")["action"]
    evidence = pd.read_csv(basil_dir / "evidence.csv").set_index("post_index")["evidence"]

    tasks = build_tasks(df, actions, evidence, args.texts, k, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(args.model)}__{args.method}.jsonl"
    done = load_done(out_path)
    todo = [t for t in tasks if (t["post_index"], t["text"], t["quantity"], t["sample"]) not in done]
    print(f"{len(df)} posts -> {len(tasks)} calls ({len(tasks) - len(todo)} done, {len(todo)} to run)"
          f" -> {out_path}\nmethod={args.method} k={k} temperature={temperature} greedy={greedy}")
    if not todo:
        return

    call_args = SimpleNamespace(provider=args.provider, max_tokens=args.max_tokens,
                                temperature=temperature, reasoning_effort=args.reasoning_effort,
                                effort=args.effort, schema=None, timeout=args.timeout)
    query_apis = build_client(args.provider)

    def work(task):
        start, last_err, raw, stop = time.time(), None, None, None
        for attempt in range(args.retries):
            try:
                data = call_model(query_apis, args.model, task["prompt"], call_args, None)
                raw, stop = data["text"], data.get("stop_reason")
                value = parse_probability(raw)
                if value is not None or attempt == args.retries - 1:
                    # [Our choice]: an unparseable reply is retried, then kept as missing.
                    return {**{k_: task[k_] for k_ in ("post_index", "text", "quantity", "sample", "agent")},
                            "model": args.model, "method": args.method, "greedy": greedy,
                            "temperature": temperature, "value": value, "parse_ok": value is not None,
                            "raw_response": raw, "stop_reason": stop,
                            "seconds": round(time.time() - start, 2), "error": None}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return {**{k_: task[k_] for k_ in ("post_index", "text", "quantity", "sample", "agent")},
                "model": args.model, "method": args.method, "greedy": greedy,
                "temperature": temperature, "value": None, "parse_ok": False,
                "raw_response": raw, "stop_reason": stop,
                "seconds": round(time.time() - start, 2), "error": last_err}

    lock = threading.Lock()
    n_err = n_unparsed = 0
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        it = as_completed([ex.submit(work, t) for t in todo])
        for fut in (tqdm(it, total=len(todo), desc=args.model) if tqdm else it):
            rec = fut.result()
            n_err += rec["error"] is not None
            n_unparsed += rec["error"] is None and not rec["parse_ok"]
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
    print(f"done. request errors: {n_err} (re-run to retry), unparseable replies: {n_unparsed}")


if __name__ == "__main__":
    main()
