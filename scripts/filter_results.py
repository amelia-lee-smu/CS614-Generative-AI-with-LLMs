#!/usr/bin/env python3
"""
Build results/ (CSV only) from raw_results/ (everything), restricted to the posts
in a selection CSV.

For each model in raw_results/:
  - if <model>.csv exists, keep its rows whose `index` is in the selection;
  - otherwise convert <model>.jsonl to the same CSV columns, keeping one record
    per (index, variant): the last one without a request error.
raw_results/ is never modified.

    python scripts/filter_results.py --select selected_1000_posts.csv
"""
import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(2**31 - 1)
COLUMNS = ["model", "index", "variant", "asker_is_asshole", "explanation",
           "json_valid", "done_reason", "seconds", "raw_response"]


def from_jsonl(path, keep):
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(rec["index"]), rec["variant"])
            if str(rec["index"]) in keep and (rec.get("error") is None or key not in latest):
                latest[key] = rec
    return [{c: ("" if rec.get(c) is None else rec.get(c)) for c in COLUMNS} | {"index": rec["index"]}
            for rec in latest.values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", default="selected_1000_posts.csv")
    p.add_argument("--raw-dir", default="raw_results")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    with open(args.select, newline="", encoding="utf-8") as f:
        keep = {row["post_index"] for row in csv.DictReader(f)}
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = sorted({f.stem for f in raw.iterdir() if f.suffix in (".csv", ".jsonl")})
    for m in models:
        if (raw / f"{m}.csv").exists():
            with open(raw / f"{m}.csv", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows = [r for r in reader if r["index"] in keep]
        else:
            fields, rows = COLUMNS, from_jsonl(raw / f"{m}.jsonl", keep)
        with open(out / f"{m}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"{m}: {len(rows)} rows, {len({r['index'] for r in rows})} posts")


if __name__ == "__main__":
    main()
