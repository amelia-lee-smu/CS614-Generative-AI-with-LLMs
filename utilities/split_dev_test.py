#!/usr/bin/env python3
"""
Split the 1,000 posts ONCE with a fixed seed into a development set (~300) and a test set (the rest).
Tune prompts / pick the best pipeline on dev only; report test-set numbers only.

    python3 split_dev_test.py --posts selected_1000_posts.csv --dev-size 300 --seed 0

Writes splits/dev_posts.csv and splits/test_posts.csv (column post_index, same format as --posts in
run_agentic.py and --only-posts in compare_methods.py). Refuses to overwrite an existing split, so the
split cannot be changed by accident after you have started tuning.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--posts", default="selected_1000_posts.csv")
p.add_argument("--dev-size", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out-dir", default="splits")
p.add_argument("--force", action="store_true")
a = p.parse_args()

out = Path(a.out_dir)
if (out / "test_posts.csv").exists() and not a.force:
    raise SystemExit(f"{out}/test_posts.csv already exists; the split is frozen (use --force to override).")
ids = sorted(pd.read_csv(a.posts)["post_index"].astype(str).unique())
dev = sorted(random.Random(a.seed).sample(ids, min(a.dev_size, len(ids))))
dev_set = set(dev)
test = [i for i in ids if i not in dev_set]
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"post_index": dev}).to_csv(out / "dev_posts.csv", index=False)
pd.DataFrame({"post_index": test}).to_csv(out / "test_posts.csv", index=False)
print(f"{len(ids)} posts -> dev {len(dev)}, test {len(test)} (seed {a.seed}) in {out}/")
