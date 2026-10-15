"""Stratified train/val split of the bridging dataset, per task type, before
tokenization -- kept at the text level so both splits can still be inspected/
diffed as plain jsonl and so the precompute step (precompute_bridging_dataset.py)
can be run on each independently.

95/5 split per type (~750 val / ~14,250 train per task, 5 types).

Usage:
    python src/utils/split_bridging_dataset.py
"""

import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/bridging/bridging_dataset.jsonl")
    parser.add_argument("--train-out", default="data/bridging/bridging_train.jsonl")
    parser.add_argument("--val-out", default="data/bridging/bridging_val.jsonl")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    by_type = defaultdict(list)
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_type[rec["type"]].append(rec)

    rng = random.Random(args.seed)
    train_recs, val_recs = [], []
    for ttype, recs in sorted(by_type.items()):
        recs = recs[:]
        rng.shuffle(recs)
        n_val = round(len(recs) * args.val_frac)
        val_recs.extend(recs[:n_val])
        train_recs.extend(recs[n_val:])
        print(f"  {ttype:18} total {len(recs):6,}  ->  train {len(recs) - n_val:6,}  val {n_val:4,}")

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for rec in train_recs:
            f.write(json.dumps(rec) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        for rec in val_recs:
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {len(train_recs):,} -> {args.train_out}")
    print(f"wrote {len(val_recs):,} -> {args.val_out}")


if __name__ == "__main__":
    main()
