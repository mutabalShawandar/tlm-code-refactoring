"""Round-trip validation for the bridging data generators.

For a sample of real code drawn from the pretrain curriculum corpus, runs
every generator in `src/data_set/bridging_corruption.py`, then independently
rebuilds each accepted record's training sample (via make_bridge_sr for
rename, make_sft_sr for the rest) and re-checks it through apply_sr a second
time, exactly like the generators' own internal self-validation does. This
is a *second, independent* check on top of that internal self-validation --
same "don't trust it just because it built without an exception" discipline
as round_trip_check.py.

Bugs found this way while building the bridging generators: bare-word
tok.encode() injecting a spurious newline; before/after built from
inconsistent normalization baselines.

Usage:
    python src/utils/bridging_round_trip_check.py
    python src/utils/bridging_round_trip_check.py --n-per-tier 5000
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
from src.data_set.sft_corruption import apply_sr, split_lines_by_newline, join_lines
from src.data_set.bridging_corruption import (
    make_rename_bridge, make_remove_dead_code_bridge, make_simplify_bridge,
    make_extract_method_bridge, make_deduplicate_bridge, make_bridge_sr, make_sft_sr,
)

GENERATORS = {
    "rename":           (make_rename_bridge, make_bridge_sr),
    "remove_dead_code": (make_remove_dead_code_bridge, make_sft_sr),
    "simplify":         (make_simplify_bridge, make_sft_sr),
    "extract_method":   (make_extract_method_bridge, make_sft_sr),
    "deduplicate":       (make_deduplicate_bridge, make_sft_sr),
}

DEFAULT_TIER_FILES = [f"data/curriculum/tier{t}.jsonl" for t in range(1, 6)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-tier", type=int, default=3000)
    parser.add_argument("--tokenizer", default="src/tokenizer/tokenizer_tlm_small.json")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    cfg = ModelConfig()
    tok = TLM_Tokenizer()
    tok.load(args.tokenizer)
    rng = random.Random(args.seed)
    NL = cfg.special_token_ids["newline"]

    accepted = collections.Counter()
    rejected = collections.Counter()
    post_build_mismatch = collections.Counter()

    for path in DEFAULT_TIER_FILES:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.n_per_tier:
                    break
                rec = json.loads(line)
                code = rec["code"]

                for name, (gen_fn, build_fn) in GENERATORS.items():
                    result = gen_fn(code, tok, cfg, rng)
                    if result is None:
                        rejected[name] += 1
                        continue
                    accepted[name] += 1

                    res2 = build_fn(result, tok, cfg)
                    if res2 is None:
                        post_build_mismatch[name] += 1
                        continue
                    _, dec_ids, _ = res2
                    before_ids = tok.encode(result["before"])
                    after_ids = tok.encode(result["after"])
                    pred = apply_sr(before_ids, dec_ids, cfg)
                    gold = join_lines(split_lines_by_newline(after_ids, NL), NL)
                    if pred != gold:
                        post_build_mismatch[name] += 1

    print(f"tried {args.n_per_tier} snippets per tier x {len(DEFAULT_TIER_FILES)} tiers, "
          f"all {len(GENERATORS)} generators run against every snippet\n")
    total_bad = 0
    for name in GENERATORS:
        a, r = accepted[name], rejected[name]
        mm = post_build_mismatch[name]
        total_bad += mm
        rate = 100 * a / (a + r) if (a + r) else 0.0
        print(f"  {name:18s} accepted {a:6d}/{a + r:6d} ({rate:5.1f}%)  post-build mismatches: {mm}")

    print("\nPASS: no post-build round-trip mismatches" if total_bad == 0
          else f"\nFAIL: {total_bad} post-build round-trip mismatches -- see above")


if __name__ == "__main__":
    main()
