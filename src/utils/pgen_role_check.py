"""p_gen role-mask round-trip check for SR SFT data.

Verifies `src/training/training.py::sr_role_masks` (the per-token
in_search/in_replace/structural classification used by `compute_loss`'s
role-aware p_gen penalty) against an independent, plain-sequential reference
implementation, over every real dec_ids sequence in the SFT dataset -- not
just a handful of hand-picked examples.

This exists because both real p_gen bugs found so far (the v2/v3 target
polarity flip, and <eos> silently falling into the in_replace role mask)
were found by reading code, not by testing. `sr_role_masks` is vectorized
(cumsum-based), which is exactly the
kind of code where an off-by-one at a hunk boundary is easy to introduce and
easy to miss by inspection. This script builds the role masks the same way
compute_loss does (production code path, real data) and compares them
position-by-position against a deliberately-independent reference that
shares no code with sr_role_masks, so agreement is a real check.

Usage:
    python src/utils/pgen_role_check.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer
from src.model_config import ModelConfig
from src.data_loaders import SFTDataset, collate_sft
from src.training import sr_role_masks
from src.data_set.sft_corruption import TASK_ID, TASK_TOKEN_NAME

DEFAULT_FILES = [
    "data/finetune_data/supervised_finetuning_generated.jsonl",
]

# Project spec canonical mapping -- task_id (consumed by PGEN_FA_BY_TASK and the
# model's PLE task embedding) and the encoder task token (consumed by the
# model's encoder input) must both trace back to the same intent for every
# type, or task-conditioning silently breaks for whichever one drifted.
EXPECTED_TASK_ID = {
    "rename": 1, "simplify": 2, "deduplicate": 3,
    "remove_dead_code": 4, "extract_method": 5,
}
EXPECTED_TOKEN_ID = {
    "rename": 9, "simplify": 10, "deduplicate": 11,
    "remove_dead_code": 12, "extract_method": 13,
}


def check_task_mapping(cfg):
    """Static consistency check: TASK_ID / TASK_TOKEN_NAME / cfg.special_token_ids
    must all agree with the project spec's documented task_id/token_id per type. Not
    data-dependent (doesn't need real samples) -- catches a hardcoded-mapping
    typo/drift the same class of bug the p_gen role-mask check catches for
    per-token classification."""
    ok = True
    for ttype, expected_tid in EXPECTED_TASK_ID.items():
        tid = TASK_ID.get(ttype)
        tok_name = TASK_TOKEN_NAME.get(ttype)
        tok_id = cfg.special_token_ids.get(tok_name) if tok_name else None
        expected_tok = EXPECTED_TOKEN_ID[ttype]
        good = (tid == expected_tid and tok_id == expected_tok)
        ok = ok and good
        status = "OK" if good else "MISMATCH"
        print(f"  {ttype:18s} task_id={tid} (expect {expected_tid})  "
              f"token_name={tok_name!r} token_id={tok_id} (expect {expected_tok})  {status}")
    return ok


def reference_roles(seq, search_id, replace_id, eos_id):
    """Ground-truth per-token role via a plain sequential scan. Deliberately
    does not share any code with sr_role_masks -- this is the point."""
    roles = []
    seen_search = 0
    seen_replace = 0
    for tok in seq:
        if tok in (search_id, replace_id, eos_id):
            roles.append("structural")
        elif seen_search > seen_replace:
            roles.append("in_search")
        elif seen_search == seen_replace and seen_search > 0:
            roles.append("in_replace")
        else:
            roles.append("other")          # BOS / anything before the first hunk
        if tok == search_id:
            seen_search += 1
        elif tok == replace_id:
            seen_replace += 1
    return roles


def main():
    cfg = ModelConfig()

    print("=== task_id / task-token mapping check (static, project spec canonical values) ===")
    mapping_ok = check_task_mapping(cfg)
    print("PASS" if mapping_ok else "FAIL", "\n")

    tok = TLM_Tokenizer()
    tok.load("src/tokenizer/tokenizer_tlm_small.json")
    PAD_ID = cfg.special_token_ids["pad"]
    SEARCH_ID = cfg.special_token_ids["search"]
    REPLACE_ID = cfg.special_token_ids["replace"]
    EOS_ID = cfg.special_token_ids["eos"]

    total = 0
    mismatched = 0
    other_role_positions = 0
    shown = 0

    for path in DEFAULT_FILES:
        ds = SFTDataset(path, tok, cfg)
        batch = collate_sft(ds.samples, PAD_ID)
        dec_target = batch["dec_ids"]

        in_search, in_replace, is_structural, _content_mask = sr_role_masks(dec_target, PAD_ID)

        for i in range(dec_target.size(0)):
            seq_full = dec_target[i].tolist()
            real_len = len(seq_full)
            while real_len > 0 and seq_full[real_len - 1] == PAD_ID:
                real_len -= 1
            seq = seq_full[:real_len]

            ref = reference_roles(seq, SEARCH_ID, REPLACE_ID, EOS_ID)
            total += 1
            row_bad = False

            for t in range(real_len):
                prod = (
                    "in_search" if bool(in_search[i, t]) else
                    "in_replace" if bool(in_replace[i, t]) else
                    "structural" if bool(is_structural[i, t]) else
                    "other"
                )
                if ref[t] == "other":
                    other_role_positions += 1
                if prod != ref[t]:
                    row_bad = True
                    if shown < 5:
                        shown += 1
                        print(f"MISMATCH  file={path}  sample={i}  pos={t}")
                        print(f"  token_id={seq[t]}  reference={ref[t]}  production={prod}")
                        print(f"  full seq: {seq}")
                        print()

            if row_bad:
                mismatched += 1

        print(f"{path}: {len(ds.samples)} samples checked")

    print(f"\nTotal sequences checked: {total}")
    print(f"Sequences with any role mismatch: {mismatched}")
    print(
        f"'other' (pre-first-hunk / <bos>) positions seen: {other_role_positions} "
        f"-- these get PGEN_NEUTRAL target and ARE counted in the p_gen loss "
        f"(is_structural only covers <search>/<replace>/<eos>, not <bos>). "
        f"Flagging for confirmation, not asserting as a bug -- see comment in "
        f"main() docstring."
    )
    if mismatched == 0:
        print("\nPASS: sr_role_masks agrees with the independent reference on every position.")
    else:
        print("\nFAIL: see mismatches above.")


if __name__ == "__main__":
    main()
