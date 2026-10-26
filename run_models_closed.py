#!/usr/bin/env python3
"""
Run query variants from dataset_generation_combined.csv through ONE closed model.

Same task, prompt, row selection, resumability and output format as
run_models.py (which runs local Ollama models), so check_results.py reads both.
Only the model call differs:

    --provider azure     Azure OpenAI, called by deployment name
    --provider gpteal    Claude through the GPTeal gateway
    --provider gemini    Gemini through the GPTeal gateway (same GPTEAL_KEY / GPTEAL_BASE_URL)

Every model is forced (structured output) to return JSON:
    {"asker_is_asshole": 0 or 1, "explanation": "<short free text>"}

Keys come from credentials.json (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT,
or GPTEAL_KEY and GPTEAL_BASE_URL). Only the provider you run needs its keys.

Results are appended to results/<model>.jsonl; finished (index, variant) pairs
are skipped on re-run.

Usage:
    python run_models_closed.py --provider azure --model gpt-4o-mini-2024-07-18 --chunk-size 5 --chunk 0   # smoke test
    python run_models_closed.py --provider gpteal --model claude-sonnet-4-6 --only-clear --balance --chunk 0
    python run_models_closed.py --provider gpteal --model claude-sonnet-4-6                                 # ALL rows

Azure deployment names need the version suffix (gpt-4o-mini-2024-07-18, not
gpt-4o-mini). Verified GPTeal models are listed in the README.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from run_models import (  # noqa: E402  (shared task definition, so both scripts stay identical)
    ALL_VARIANTS,
    BASE_SYSTEM_PROMPT,
    DEFAULT_VARIANTS,
    OUTPUT_SPEC,
    OUTPUT_SPEC_SHORT,
    load_done,
    parse_output,
    safe_name,
    select_rows,
)
from utilities.calling_apis import QueryAPIs  # noqa: E402
from utilities.utilities import load_credentials  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


def make_schema(short: bool):
    """Strict structured-output schema (needs additionalProperties false, no maxLength)."""
    return {
        "type": "object",
        "properties": {
            "asker_is_asshole": {"type": "integer", "enum": [0, 1]},
            "explanation": {"type": "string"},
        },
        "required": ["asker_is_asshole", "explanation"],
        "additionalProperties": False,
    }


def build_client(provider):
    credentials = load_credentials()
    query_apis = QueryAPIs()
    if provider == "azure":
        query_apis.set_keys(
            azure_openai_api_key=credentials["AZURE_OPENAI_API_KEY"],
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
        )
    else:
        query_apis.set_keys(
            gpteal_key=credentials["GPTEAL_KEY"],
            gpteal_base_url=credentials["GPTEAL_BASE_URL"],
        )
    return query_apis


def call_model(query_apis, model, text, args, system_prompt):
    """One request. Returns the API result dict: {text, model, stop_reason}."""
    if args.provider == "azure":
        return query_apis.call_azure_openai(
            text,
            deployment=model,
            system=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            json_schema=args.schema,
        )
    if args.provider == "gemini":
        return query_apis.call_gemini(
            text,
            model=model,
            system=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_schema=args.schema,
            timeout=args.timeout,
        )
    return query_apis.call_claude(
        text,
        model=model,
        system=system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        effort=args.effort,
        json_schema=args.schema,
        provider="gpteal",
        timeout=args.timeout,
    )


def run_task(query_apis, model, idx, variant, text, args, system_prompt):
    start = time.time()
    last_err = None
    for attempt in range(args.retries):
        try:
            data = call_model(query_apis, model, text, args, system_prompt)
            raw = data["text"]
            verdict, expl, json_valid = parse_output(raw)
            return {
                "index": idx,
                "variant": variant,
                "model": model,
                "asker_is_asshole": verdict,
                "explanation": expl,
                "parse_ok": verdict is not None,
                "json_valid": json_valid,
                "raw_response": raw,
                "thinking_chars": 0,
                "done_reason": data.get("stop_reason"),
                "prompt_tokens": None,
                "output_tokens": None,
                "seconds": round(time.time() - start, 2),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return {
        "index": idx,
        "variant": variant,
        "model": model,
        "asker_is_asshole": None,
        "explanation": None,
        "parse_ok": False,
        "json_valid": False,
        "raw_response": None,
        "seconds": round(time.time() - start, 2),
        "error": last_err,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["azure", "gpteal", "gemini"])
    p.add_argument("--model", required=True,
                   help="Azure deployment name (with version suffix), GPTeal Claude model name, "
                        "or GPTeal Gemini deployment name (e.g. gemini-2-5-flash).")
    p.add_argument("--csv", default="dataset_generation_combined.csv")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=ALL_VARIANTS)
    p.add_argument("--chunk", type=int, default=None,
                   help="Which chunk of rows to run (0 = first). Omit to run ALL rows.")
    p.add_argument("--chunk-size", type=int, default=100,
                   help="Rows per chunk. Chunks never overlap.")
    p.add_argument("--only-clear", action="store_true",
                   help="Keep only rows whose asker_verdict is clearly_wrong or clearly_not_wrong.")
    p.add_argument("--balance", action="store_true",
                   help="With --only-clear: 50/50 clearly_wrong / clearly_not_wrong in every chunk.")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="Fixed seed for the one-time shuffle. Do not change between runs.")
    p.add_argument("--workers", type=int, default=8, help="Concurrent requests.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Leave unset for reasoning models and newer Claude models, which reject it.")
    p.add_argument("--reasoning-effort", default=None,
                   help="Azure reasoning models only (low, medium, high).")
    p.add_argument("--effort", default=None,
                   help="GPTeal Claude only (low, medium, high, xhigh, max); unset uses the model default.")
    p.add_argument("--short-explanations", action="store_true",
                   help="Stricter 'ONE or TWO sentences, 40 words' prompt + 160-token cap. "
                        "Do NOT mix with results produced without this flag.")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Output cap (default 256, or 160 with --short-explanations). Reasoning "
                        "models count thinking tokens too, so raise it for those.")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--system-file", default=None,
                   help="Optional text file replacing the base task prompt (e.g. a mitigation prompt). "
                        "The JSON output instructions are always appended.")
    p.add_argument("--tag", default="",
                   help="Suffix for output file, e.g. --tag mitigation1 -> results/<model>__mitigation1.jsonl")
    args = p.parse_args()

    if args.balance and not args.only_clear:
        raise SystemExit("--balance requires --only-clear")
    if args.max_tokens is None:
        args.max_tokens = 160 if args.short_explanations else 256
    args.schema = make_schema(args.short_explanations)

    base = BASE_SYSTEM_PROMPT
    if args.system_file:
        base = Path(args.system_file).read_text(encoding="utf-8").strip()
    system_prompt = base + (OUTPUT_SPEC_SHORT if args.short_explanations else OUTPUT_SPEC)

    df = pd.read_csv(args.csv)
    if "index" not in df.columns:
        df["index"] = range(len(df))
    df = select_rows(df, args.only_clear, args.balance, args.shuffle_seed)
    print(f"Pool: {len(df)} rows"
          f"{' (clear only' + (', balanced' if args.balance else '') + ')' if args.only_clear else ''}")

    if args.chunk is not None:
        lo = args.chunk * args.chunk_size
        if lo >= len(df):
            raise SystemExit(f"Chunk {args.chunk} is past the end ({len(df)} rows in the pool).")
        df = df.iloc[lo:lo + args.chunk_size]
        print(f"Chunk {args.chunk}: pool rows {lo}-{lo + len(df) - 1}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_all = []
    for _, row in df.iterrows():
        for v in args.variants:
            text = row.get(v)
            if isinstance(text, str) and text.strip():
                tasks_all.append((str(row["index"]), v, text))
    print(f"{len(df)} rows x {len(args.variants)} variants -> {len(tasks_all)} queries")

    model = args.model
    suffix = f"__{args.tag}" if args.tag else ""
    out_path = out_dir / f"{safe_name(model)}{suffix}.jsonl"
    done = load_done(out_path)
    todo = [t for t in tasks_all if (t[0], t[1]) not in done]
    print(f"=== {model} ({args.provider}): {len(tasks_all) - len(todo)} of these already done, "
          f"{len(todo)} to run -> {out_path}")
    if not todo:
        return

    query_apis = build_client(args.provider)

    lock = threading.Lock()
    n_err = n_unparsed = 0
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_task, query_apis, model, i, v, t, args, system_prompt)
                   for i, v, t in todo]
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc=model)
        for fut in iterator:
            rec = fut.result()
            n_err += bool(rec["error"])
            n_unparsed += (not rec["parse_ok"]) and not rec["error"]
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"{model} finished. Request errors: {n_err} (re-run to retry). "
          f"Responses with no usable verdict: {n_unparsed}")


if __name__ == "__main__":
    main()
