"""
Generate the synthetic rewrites for AITA-YTA rows.

Verdicts are produced separately by judge_verdicts.py, which sees only the
original post.

Runs rows concurrently and appends each finished row to a JSONL checkpoint, so
an interrupted run can be restarted without repeating completed work.

    python scripts/generate_dataset.py --rows 500 --workers 20
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent  # the repository root
sys.path.append(str(ROOT))
from settings import *
from utilities.calling_apis import QueryAPIs
from utilities.dataset_generation_functions import *
from utilities.dataset_generation_prompts import *
from utilities.utilities import load_credentials

GENERATED_COLUMNS = list(DATASET_GENERATION_REWRITE_SCHEMA["properties"])


def generate_perspectives(query_apis, original_post, params):
    return query_apis.call_claude_for_json(
        prompt=build_dataset_generation_rewrite_prompt(post=original_post),
        system=DATASET_GENERATION_REWRITE_SYSTEM_PROMPT,
        json_schema=DATASET_GENERATION_REWRITE_SCHEMA,
        **params,
    )


def load_checkpoint(path):
    """Return {row index: generated fields} for rows already completed."""
    if not path.exists():
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            done[record["index"]] = record["result"]
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--out", default="dataset/dataset_generation_perspectives.csv")
    args = parser.parse_args()

    out_path = ROOT / args.out
    checkpoint_path = out_path.with_suffix(".jsonl")

    credentials = load_credentials()
    query_apis = QueryAPIs()
    query_apis.set_keys(
        gpteal_key=credentials["GPTEAL_KEY"],
        gpteal_base_url=credentials["GPTEAL_BASE_URL"],
    )
    params = {
        "model": DATASET_GENERATION_MODEL,
        "effort": DATASET_GENERATION_EFFORT,
        "provider": "gpteal",
    }

    dataset_df = pd.read_csv(ROOT / "dataset/AITA-YTA.csv").drop(["Unnamed: 0"], axis=1)
    dataset_df = dataset_df[args.start : args.start + args.rows]

    results = load_checkpoint(checkpoint_path)
    todo = [i for i in dataset_df.index if i not in results]
    print(f"{len(results)} rows already done, {len(todo)} to generate", flush=True)

    write_lock = threading.Lock()
    failures = {}
    start_time = time.time()

    def worker(index):
        post = dataset_df.loc[index, "prompt"]
        try:
            result = generate_perspectives(query_apis, post, params)
        except Exception as exc:
            with write_lock:
                failures[index] = f"{type(exc).__name__}: {exc}"
            return
        with write_lock:
            results[index] = result
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps({"index": int(index), "result": result}) + "\n")

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, index) for index in todo]
            progress = tqdm(as_completed(futures), total=len(futures), unit="row")
            for _ in progress:
                progress.set_postfix(ok=len(results), failed=len(failures))

    results_df = pd.DataFrame.from_dict(results, orient="index").reindex(
        columns=GENERATED_COLUMNS
    )
    out_df = dataset_df.join(results_df, how="inner")
    out_df.to_csv(out_path)

    print(f"\nsaved {out_path} {out_df.shape} in {time.time() - start_time:.0f}s")
    if failures:
        print(f"{len(failures)} rows failed (re-run to retry):")
        for index, message in list(failures.items())[:10]:
            print(f"  row {index}: {message[:120]}")


if __name__ == "__main__":
    main()
