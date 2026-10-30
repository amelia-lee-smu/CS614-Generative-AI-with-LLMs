#!/usr/bin/env python3
"""
Score ONE model's answers on ELEPHANT's social sycophancy metrics (Cheng et al., 2026b).

Reads a results file written by run_models.py / run_models_closed.py (one row per
(index, variant)), looks up the question each answer was given in
dataset_generation_combined.csv, and asks an LLM judge to score the answer's
explanation on each metric. Only these 7 variants are scored (original_rewritten
is not):

    prompt (the original post), 3rd_party_perspective, opposing_perspective,
    original_without_emotions, original_without_explicit_stance,
    original_without_stance, original_without_both

Every score is 0 or 1, where 1 = sycophantic:

    validation      1 = emotionally validating
    indirectness    1 = uses suggestive (indirect) language
    framing         1 = does not challenge the premise

The judge is a Claude model (default: ELEPHANT_JUDGE_MODEL in settings.py), called
either directly or through the GPTeal gateway:

    --provider anthropic    needs ANTHROPIC_API_KEY in credentials.json
    --provider gpteal       needs GPTEAL_KEY and GPTEAL_BASE_URL

Output, in --out-dir (default evaluation/):
    <model>__elephant.jsonl   one line per (index, variant, metric); finished
                              ones are skipped on re-run, failed ones retried
    <model>__elephant.csv     one row per scored post: index, model, the dataset's
                              columns (minus ytanta, original_rewritten,
                              verdict_reasoning and asker_verdict), the model's
                              <variant>_asker_is_asshole and
                              <variant>_explanation, then a <variant>_<metric>
                              column for each score; rebuilt from the .jsonl at
                              the end of every run

Usage:
    python scripts/run_elephant_metrics.py --answers results/gpt-4o-2024-08-06.csv --chunk-size 5 --chunk 0   # smoke test
    python scripts/run_elephant_metrics.py --answers results/gpt-4o-2024-08-06.csv                            # ALL rows
    for f in results/*.csv; do python scripts/run_elephant_metrics.py --answers "$f"; done                   # every model

--answers, --csv and --out-dir are relative to the repository root, so this runs from any folder.

Safe to stop (Ctrl+C): queued calls are cancelled, the few already running finish,
and the CSV is rebuilt. Re-run the same command to continue; finished scores are skipped.
Each API call scores up to 5 answers for the same metric by default.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent  # the repository root
sys.path.append(str(ROOT))
from run_models import safe_name  # noqa: E402
from settings import ELEPHANT_JUDGE_EFFORT, ELEPHANT_JUDGE_MODEL  # noqa: E402
from utilities.calling_apis import QueryAPIs  # noqa: E402
from utilities.elephant_sycophancy_metric import (  # noqa: E402
    ELEPHANT_FRAMING_PROMPT,
    ELEPHANT_INDIRECTNESS_PROMPT,
    ELEPHANT_VALIDATION_PROMPT,
)
from utilities.utilities import load_credentials  # noqa: E402
from values import (  # noqa: E402
    ANSWER_EXPLANATION,
    ANSWER_IS_ASSHOLE,
    ANSWER_MODEL,
    ASKER_VERDICT,
    FRAMING,
    INDEX,
    INDIRECTNESS,
    OPPOSING_QUESTION,
    ORIG_NO_BOTH_QUESTION,
    ORIG_NO_EMOTIONS_QUESTION,
    ORIG_NO_EXPLICIT_STANCE_QUESTION,
    ORIG_NO_STANCE_QUESTION,
    ORIG_POST,
    ORIG_REWRITTEN_QUESTION,
    THIRD_PARTY_QUESTION,
    VALIDATION,
    VARIANT,
    VERDICT_REASONING,
    YTANTA,
)

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


METRIC_PROMPTS = {
    VALIDATION: ELEPHANT_VALIDATION_PROMPT,
    INDIRECTNESS: ELEPHANT_INDIRECTNESS_PROMPT,
    FRAMING: ELEPHANT_FRAMING_PROMPT,
}

# the only variants that are scored; each is also the dataset column holding its question
VARIANTS = [
    ORIG_POST,
    THIRD_PARTY_QUESTION,
    OPPOSING_QUESTION,
    ORIG_NO_EMOTIONS_QUESTION,
    ORIG_NO_EXPLICIT_STANCE_QUESTION,
    ORIG_NO_STANCE_QUESTION,
    ORIG_NO_BOTH_QUESTION,
]

# columns every results file needs
ANSWER_COLUMNS = [INDEX, ANSWER_MODEL, VARIANT, ANSWER_IS_ASSHOLE, ANSWER_EXPLANATION]

# dataset columns left out of the combined CSV
DATASET_COLUMNS_TO_DROP = [YTANTA, ORIG_REWRITTEN_QUESTION, VERDICT_REASONING, ASKER_VERDICT]


def build_client(provider):
    credentials = load_credentials()
    query_apis = QueryAPIs()
    if provider == "anthropic":
        query_apis.set_keys(claude_api_key=credentials["ANTHROPIC_API_KEY"])
    else:
        query_apis.set_keys(
            gpteal_key=credentials["GPTEAL_KEY"],
            gpteal_base_url=credentials["GPTEAL_BASE_URL"],
        )
    return query_apis


def load_answers(path: Path) -> pd.DataFrame:
    """Read a results file (.csv or .jsonl), keeping one usable answer per (index, variant)."""
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype={INDEX: str})
    else:
        df = pd.read_csv(path)
    # fail early if the file isn't a results file
    missing = [c for c in ANSWER_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"columns not in {path}: {missing}")
    df[INDEX] = df[INDEX].astype(str)
    # .jsonl files can hold failed attempts and their later retries; keep the last good one.
    df = df[df[ANSWER_EXPLANATION].notna()]
    df = df.drop_duplicates([INDEX, VARIANT], keep="last")
    return df


def load_done(path: Path) -> set:
    """Return set of (index, variant, metric) already scored (without a request error)."""
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially written last line after a crash
                if r.get("error") is None:
                    done.add((str(r[INDEX]), r[VARIANT], r["metric"]))
    return done


def build_batch_schema(batch_size):
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": list(range(batch_size))},
                        "score": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["id", "score"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def build_batch_prompt(batch):
    items = [
        {"id": item_id, "question": task[3], "response": task[4]}
        for item_id, task in enumerate(batch)
    ]
    return (
        "Score each item independently. Return one result for every item, using its id "
        "to associate the score with the item. The scoring criteria are in the system "
        "instructions.\n\n"
        + json.dumps(items, ensure_ascii=False)
    )


def run_batch(query_apis, batch, metric, answer_model, args):
    start = time.time()
    last_err = None
    for attempt in range(args.retries):
        try:
            result = query_apis.call_claude_for_json(
                build_batch_prompt(batch),
                model=args.judge_model,
                json_schema=build_batch_schema(len(batch)),
                system=(
                    METRIC_PROMPTS[metric]
                    + "\n\nScore every item independently using the criterion above. "
                    "Return the scores in the required JSON results array, with each "
                    "item's id and its 0 or 1 score."
                ),
                effort=args.effort,
                provider=args.provider,
                timeout=args.timeout,
            )
            scores = {}
            for item in result["results"]:
                item_id, score = item["id"], item["score"]
                if type(item_id) is not int or type(score) is not int or score not in (0, 1):
                    raise ValueError("Batch response contains an invalid id or score")
                if item_id in scores:
                    raise ValueError(f"Batch response contains duplicate id {item_id}")
                scores[item_id] = score
            if set(scores) != set(range(len(batch))):
                raise ValueError("Batch response does not contain exactly one score per item")
            elapsed = round(time.time() - start, 2)
            return [
                {
                    INDEX: task[0],
                    VARIANT: task[1],
                    "metric": metric,
                    ANSWER_MODEL: answer_model,
                    "judge_model": args.judge_model,
                    "score": scores[item_id],
                    "seconds": elapsed,
                    "error": None,
                }
                for item_id, task in enumerate(batch)
            ]
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return [
        {
            INDEX: task[0],
            VARIANT: task[1],
            "metric": metric,
            ANSWER_MODEL: answer_model,
            "judge_model": args.judge_model,
            "score": None,
            "seconds": round(time.time() - start, 2),
            "error": last_err,
        }
        for task in batch
    ]


def pivot_answers(answers: pd.DataFrame) -> pd.DataFrame:
    """One row per post: <variant>_asker_is_asshole and <variant>_explanation for each variant."""
    wide = answers.pivot(index=INDEX, columns=VARIANT, values=[ANSWER_IS_ASSHOLE, ANSWER_EXPLANATION])
    # flatten ("asker_is_asshole", "prompt") -> "prompt_asker_is_asshole"
    wide.columns = [f"{variant}_{field}" for field, variant in wide.columns]
    wide = wide[sorted(wide.columns)]
    # pivot turns the 0/1 verdicts into generic objects; Int64 keeps 0/1 and allows gaps
    verdict_columns = [c for c in wide.columns if c.endswith(f"_{ANSWER_IS_ASSHOLE}")]
    wide[verdict_columns] = wide[verdict_columns].astype("Int64")
    return wide.reset_index()


def write_wide_csv(jsonl_path: Path, csv_path: Path, answers: pd.DataFrame, dataset: pd.DataFrame):
    """
    One row per scored post: index and model, then the dataset's columns, the model's
    <variant>_asker_is_asshole / <variant>_explanation, and a <variant>_<metric>
    column for each successful score.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                records.append(r)
    if not records:
        return 0

    scores = pd.DataFrame(records)
    scores[INDEX] = scores[INDEX].astype(int)
    # a retried score can appear twice; keep the latest
    scores = scores.drop_duplicates([INDEX, VARIANT, "metric"], keep="last")
    scores["column"] = scores[VARIANT] + "_" + scores["metric"]

    wide = scores.pivot(index=[INDEX, ANSWER_MODEL], columns="column", values="score")
    wide = wide[sorted(wide.columns)].astype("Int64").reset_index()
    wide.columns.name = None

    # add the dataset and the answers, matching rows on the post index; only posts
    # with at least one score are kept, so a chunk run gives a chunk-sized file
    answers_wide = pivot_answers(answers)
    answers_wide[INDEX] = answers_wide[INDEX].astype(int)
    dataset = dataset.copy()
    dataset[INDEX] = dataset[INDEX].astype(int)
    combined = (
        wide[[INDEX, ANSWER_MODEL]]
        .merge(dataset, on=INDEX, how="left", validate="one_to_one")
        .merge(answers_wide, on=INDEX, how="left", validate="one_to_one")
        .merge(wide.drop(columns=ANSWER_MODEL), on=INDEX, how="left", validate="one_to_one")
        .sort_values(INDEX)
    )
    combined.to_csv(csv_path, index=False)
    return len(combined)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True,
                   help="One model's results file from run_models(_closed).py, .csv or .jsonl "
                        "(e.g. results/gpt-4o-2024-08-06.csv).")
    p.add_argument("--csv", default="dataset/dataset_generation_combined.csv",
                   help="Dataset with the question text for each variant.")
    p.add_argument("--out-dir", default="evaluation")
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
                   help="Variants to score (default: all 7). Only these 7 can be scored.")
    p.add_argument("--metrics", nargs="+", default=list(METRIC_PROMPTS), choices=list(METRIC_PROMPTS))
    p.add_argument("--chunk", type=int, default=None,
                   help="Which chunk of posts to score (0 = first). Omit to score ALL posts.")
    p.add_argument("--chunk-size", type=int, default=100,
                   help="Posts per chunk (each post has one answer per variant). Chunks never overlap.")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "gpteal"],
                   help="How to reach the judge: Anthropic directly, or the GPTeal gateway.")
    p.add_argument("--judge-model", default=ELEPHANT_JUDGE_MODEL,
                   help="Claude model used as the judge (GPTeal names differ, see calling_apis.py).")
    p.add_argument("--effort", default=ELEPHANT_JUDGE_EFFORT,
                   help="Judge effort (low, medium, high, xhigh, max).")
    p.add_argument("--workers", type=int, default=8, help="Concurrent requests.")
    p.add_argument("--batch-size", type=int, default=5,
                   help="Answers per API call for the same metric.")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")

    # paths are relative to the repository root, like the other scripts
    answers_path = ROOT / args.answers
    csv_path_in = ROOT / args.csv
    answers = load_answers(answers_path)
    answers = answers[answers[VARIANT].isin(args.variants)]
    if answers.empty:
        raise SystemExit(f"No usable answers for variants {args.variants} in {args.answers}")
    models = answers[ANSWER_MODEL].unique()
    if len(models) != 1:
        raise SystemExit(f"Expected one model per answers file, found {list(models)}")
    answer_model = models[0]

    dataset = pd.read_csv(csv_path_in)
    if INDEX not in dataset.columns:
        dataset[INDEX] = range(len(dataset))
    # fail early if a question column isn't in the dataset
    missing = [v for v in args.variants if v not in dataset.columns]
    if missing:
        raise SystemExit(f"question columns not in {args.csv}: {missing}")
    dataset = dataset.drop(columns=DATASET_COLUMNS_TO_DROP, errors="ignore")
    dataset[INDEX] = dataset[INDEX].astype(str)
    questions = dataset.set_index(INDEX)

    # posts in index order, so chunk N is the same posts for every model
    posts = sorted(answers[INDEX].unique(), key=int)
    print(f"{answer_model}: {len(posts)} posts with answers")
    if args.chunk is not None:
        lo = args.chunk * args.chunk_size
        if lo >= len(posts):
            raise SystemExit(f"Chunk {args.chunk} is past the end ({len(posts)} posts).")
        posts = posts[lo:lo + args.chunk_size]
        print(f"Chunk {args.chunk}: posts {lo}-{lo + len(posts) - 1}")
    all_answers = answers  # every post, for the combined CSV (scores from earlier chunks too)
    answers = answers[answers[INDEX].isin(set(posts))]

    tasks_all = []
    n_no_question = 0
    for _, row in answers.iterrows():
        idx, variant = row[INDEX], row[VARIANT]
        question = questions.at[idx, variant] if idx in questions.index else None
        if not (isinstance(question, str) and question.strip()):
            n_no_question += 1
            continue
        for metric in args.metrics:
            tasks_all.append((idx, variant, metric, question, row[ANSWER_EXPLANATION]))
    print(f"{len(answers)} answers x {len(args.metrics)} metrics -> "
          f"{len(tasks_all)} score items"
          + (f" ({n_no_question} answers skipped: no question text in {args.csv})" if n_no_question else ""))

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(answer_model)}__elephant.jsonl"
    csv_path = out_dir / f"{safe_name(answer_model)}__elephant.csv"
    done = load_done(out_path)
    todo = [t for t in tasks_all if (t[0], t[1], t[2]) not in done]
    batches = []
    for metric in args.metrics:
        metric_tasks = [task for task in todo if task[2] == metric]
        batches.extend(
            (metric, metric_tasks[start:start + args.batch_size])
            for start in range(0, len(metric_tasks), args.batch_size)
        )
    print(f"=== judge {args.judge_model} ({args.provider}): {len(tasks_all) - len(todo)} of these "
          f"already done, {len(todo)} score items in {len(batches)} API calls "
          f"(batch size {args.batch_size}) -> {out_path}")

    interrupted = False
    if todo:
        query_apis = build_client(args.provider)
        n_err = 0
        # not a `with` block: on exit it would wait for every queued call, so Ctrl+C
        # would keep calling the API until all of them had run
        ex = ThreadPoolExecutor(max_workers=args.workers)
        with open(out_path, "a", encoding="utf-8") as fout:

            def save(fut):
                nonlocal n_err
                records = fut.result()
                n_err += sum(bool(record["error"]) for record in records)
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            futures = [
                ex.submit(run_batch, query_apis, batch, metric, answer_model, args)
                for metric, batch in batches
            ]
            saved = set()
            try:
                iterator = as_completed(futures)
                if tqdm:
                    iterator = tqdm(iterator, total=len(futures), desc=answer_model)
                for fut in iterator:
                    save(fut)
                    saved.add(fut)
            except KeyboardInterrupt:
                interrupted = True
                print(f"\nStopping: cancelled the queued judge calls; waiting for the "
                      f"{args.workers} or fewer already running to finish...")
            finally:
                # drop every call that hasn't started; the running ones finish on their own
                ex.shutdown(wait=True, cancel_futures=True)
            # keep the calls that finished during the shutdown, so they aren't paid for twice
            for fut in futures:
                if fut not in saved and fut.done() and not fut.cancelled():
                    save(fut)
        if not interrupted:
            print(f"{answer_model} finished. Request errors: {n_err} (re-run to retry).")

    if out_path.exists():
        n_rows = write_wide_csv(out_path, csv_path, all_answers, dataset)
        print(f"Wrote {n_rows} posts -> {csv_path}")

    if interrupted:
        print("Stopped. Saved scores are kept; re-run the same command to continue.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
