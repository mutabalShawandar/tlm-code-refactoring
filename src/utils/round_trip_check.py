"""Round-trip validation for SR SFT datasets.

For every {before, after, type} record, builds the decoder target via
`make_sft_sr` (which internally calls `diff_hunks`), then applies that same
target back to `before` via `apply_sr` and checks the result equals `after`.
No model is involved -- this validates the data/format pipeline itself: a
mismatch means `make_sft_sr`/`diff_hunks`/`apply_sr` produced or reconstructed
a target that doesn't actually encode the intended edit, not a model failure.

Usage:
    python notebooks/utils/round_trip_check.py
    python notebooks/utils/round_trip_check.py --file path/to/data.jsonl [--file ...]
    python notebooks/utils/round_trip_check.py --show 3   # print N mismatch examples per type
"""

import sys
import os
import json
import argparse
import collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer
from src.model_config import ModelConfig
from src.data_set.sft_corruption import make_sft_sr, apply_sr, join_lines, split_lines_by_newline

DEFAULT_FILES = [
    "data/finetune_data/supervised_finetuning_generated.jsonl",
]


def check_file(path, tok, cfg, show=0):
    NL = cfg.special_token_ids["newline"]
    kept = collections.Counter()
    dropped = collections.Counter()
    mismatches = collections.Counter()
    unlocatable = collections.Counter()
    wrong_content = collections.Counter()
    shown = collections.Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ttype = record.get("type", "?")

            result = make_sft_sr(record, tok, cfg)
            if result is None:
                dropped[ttype] += 1
                continue
            kept[ttype] += 1

            enc_ids, dec_ids, task_id = result
            before_ids = tok.encode(record["before"])
            after_ids = tok.encode(record["after"])

            pred = apply_sr(before_ids, dec_ids, cfg)
            gold = join_lines(split_lines_by_newline(after_ids, NL), NL)

            if pred == gold:
                continue

            mismatches[ttype] += 1
            if pred is None:
                unlocatable[ttype] += 1
            else:
                wrong_content[ttype] += 1

            if shown[ttype] < show:
                shown[ttype] += 1
                print(f"=== {os.path.basename(path)} | {ttype} | mismatch "
                      f"({'unlocatable' if pred is None else 'wrong content'}) ===")
                print("BEFORE:")
                print(record["before"])
                print("AFTER:")
                print(record["after"])
                if pred is not None:
                    print("PRED:")
                    print(tok.decode(pred))
                print()

    return kept, dropped, mismatches, unlocatable, wrong_content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", dest="files",
                         help="jsonl file to check (repeatable); default: the two finetune_data files")
    parser.add_argument("--tokenizer", default="src/tokenizer/tokenizer_tlm_small.json")
    parser.add_argument("--show", type=int, default=0,
                         help="print up to N mismatch examples per type per file")
    args = parser.parse_args()

    files = args.files or DEFAULT_FILES
    cfg = ModelConfig()
    tok = TLM_Tokenizer()
    tok.load(args.tokenizer)

    for path in files:
        kept, dropped, mismatches, unlocatable, wrong_content = check_file(path, tok, cfg, show=args.show)
        print(f"\n{path}")
        print(" kept:       ", dict(kept))
        if dropped:
            print(" dropped:    ", dict(dropped))
        print(" mismatches: ", dict(mismatches))
        if mismatches:
            print("   unlocatable (SEARCH not found):", dict(unlocatable))
            print("   located, wrong content:         ", dict(wrong_content))
        total_kept = sum(kept.values())
        total_mismatch = sum(mismatches.values())
        rate = (total_mismatch / total_kept * 100) if total_kept else 0.0
        print(f" total: {total_mismatch}/{total_kept} mismatches ({rate:.2f}%)")


if __name__ == "__main__":
    main()
