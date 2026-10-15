"""Precompute the bridging dataset's (enc_ids, dec_ids, task_id) tuples to
disk, so a Colab training session doesn't have to repeat BridgingDataset's
eager tokenization build (BPE-encode + diff_hunks over 75,000 text records,
measured at ~141s locally -- not huge, but real friction on every session
restart). No variant-cycling needed (unlike pretrain's precompute) since
bridging targets are deterministic -- this is a one-time, one-file dump.

Usage:
    python src/utils/precompute_bridging_dataset.py
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer
from src.model_config import ModelConfig
from src.data_loaders import BridgingDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/bridging/bridging_dataset.jsonl")
    parser.add_argument("--out", default="data/bridging/bridging_dataset_ids.jsonl")
    parser.add_argument("--tokenizer", default="src/tokenizer/tokenizer_tlm_small.json")
    args = parser.parse_args()

    cfg = ModelConfig()
    tok = TLM_Tokenizer()
    tok.load(args.tokenizer)

    ds = BridgingDataset(args.src, tok, cfg)
    print(f"built {len(ds)} samples, dropped {ds.dropped} ({ds.dropped_by_type})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for enc_ids, dec_ids, task_id in ds.samples:
            f.write(json.dumps({"enc_ids": enc_ids, "dec_ids": dec_ids, "task_id": task_id}) + "\n")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
