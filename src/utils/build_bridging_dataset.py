"""Build the balanced bridging dataset from the pretrain curriculum corpus.

Runs all 5 generators in src/data_set/bridging_corruption.py over every
snippet in the full corpus (data/curriculum/tier1..5.jsonl, 194,627
snippets, shuffled once so acceptance order isn't tier-biased) and keeps a
running reservoir sample (Algorithm R) of exactly --cap accepted records
per task. This gives each task's output an exact uniform-random draw from
its FULL accepted population, not just whichever records were accepted
first before that task's cap was hit -- important because acceptance rates
vary hugely across tasks (rename/remove_dead_code ~86%, extract_method
~69%, deduplicate ~29%, simplify ~8%), so a naive "stop scanning once you
hit --cap" approach would only ever draw the high-acceptance tasks from a
small early slice of the corpus instead of the whole thing (caught and
fixed after the first version did exactly that).

--cap default 15000 matches simplify's real full-corpus yield (~15,444,
the lowest of the 5 tasks): capping every task there gives a balanced
dataset without inflating any task above what the corpus can actually supply.

Output: one combined JSONL, one record per line, fields
{type, before, after, spans (rename only)} -- same record shape as
data/finetune_data/supervised_finetuning_generated.jsonl, so it can be
consumed the same way (rename via make_bridge_sr, others via make_sft_sr;
see src/data_loaders/bridging_dataset.py).

Usage:
    python src/utils/build_bridging_dataset.py
    python src/utils/build_bridging_dataset.py --cap 15000 --seed 2024
"""

import sys
import os
import json
import argparse
import random
import collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer
from src.model_config import ModelConfig
from src.data_set.bridging_corruption import (
    make_rename_bridge, make_remove_dead_code_bridge, make_simplify_bridge,
    make_extract_method_bridge, make_deduplicate_bridge,
)

GENERATORS = {
    "rename":           make_rename_bridge,
    "remove_dead_code": make_remove_dead_code_bridge,
    "simplify":         make_simplify_bridge,
    "extract_method":   make_extract_method_bridge,
    "deduplicate":       make_deduplicate_bridge,
}

DEFAULT_TIER_FILES = [f"data/curriculum/tier{t}.jsonl" for t in range(1, 6)]


class Reservoir:
    """Algorithm R: after seeing n items, self.items is a uniform-random
    sample of min(n, cap) of them, built in one pass with O(cap) memory."""

    def __init__(self, cap: int, rng: random.Random):
        self.cap = cap
        self.rng = rng
        self.items = []
        self.n_seen = 0

    def offer(self, item):
        self.n_seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n_seen)
            if j < self.cap:
                self.items[j] = item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=15000)
    parser.add_argument("--tokenizer", default="src/tokenizer/tokenizer_tlm_small.json")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out", default="data/bridging/bridging_dataset.jsonl")
    args = parser.parse_args()

    cfg = ModelConfig()
    tok = TLM_Tokenizer()
    tok.load(args.tokenizer)
    rng = random.Random(args.seed)

    codes = []
    for path in DEFAULT_TIER_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                codes.append(json.loads(line)["code"])
    rng.shuffle(codes)
    print(f"loaded {len(codes)} snippets from {len(DEFAULT_TIER_FILES)} tiers, shuffled (seed={args.seed})")

    reservoirs = {name: Reservoir(args.cap, rng) for name in GENERATORS}

    for idx, code in enumerate(codes):
        for name, gen_fn in GENERATORS.items():
            rec = gen_fn(code, tok, cfg, rng)
            if rec is not None:
                reservoirs[name].offer(rec)
        if (idx + 1) % 20000 == 0:
            print(f"  scanned {idx + 1}/{len(codes)}  "
                  + "  ".join(f"{n}:{len(r.items)}/{r.n_seen}" for n, r in reservoirs.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for name in GENERATORS:
            for rec in reservoirs[name].items:
                f.write(json.dumps(rec) + "\n")

    print(f"\nwrote {args.out}")
    total = 0
    for name in GENERATORS:
        r = reservoirs[name]
        total += len(r.items)
        flag = "OK" if len(r.items) >= args.cap else f"SHORT (only {r.n_seen} accepted in whole corpus, wanted {args.cap})"
        print(f"  {name:18s} sampled {len(r.items):6d} / {r.n_seen:6d} total accepted  {flag}")
    print(f"  {'total':18s} {total:6d}")


if __name__ == "__main__":
    main()
