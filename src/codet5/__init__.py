# HF-T5-specific pipeline (official CodeT5-small weights/tokenizer) --
# deliberately separate from the rest of src/ (TLM's own architecture,
# tokenizer, and training code). No imports either direction. See
# sr_text.py's module docstring for the full rationale.

from .sr_text import (
    TASK_ID,
    TASK_TOKEN_NAME,
    split_lines,
    join_lines,
    diff_hunks_text,
    apply_sr_text,
    parse_sr_pairs_text,
    build_sr_target_text,
    make_sft_sr_text,
)
from .data import SRTextDataset, collate_sr
from .training import (
    MAX_NEW_TOKENS,
    hf_step_batch,
    hf_validate,
    hf_run_epoch,
    hf_greedy_decode,
    decode_target_text,
)

__all__ = [
    "TASK_ID", "TASK_TOKEN_NAME", "split_lines", "join_lines",
    "diff_hunks_text", "apply_sr_text", "parse_sr_pairs_text",
    "build_sr_target_text", "make_sft_sr_text",
    "SRTextDataset", "collate_sr",
    "MAX_NEW_TOKENS", "hf_step_batch", "hf_validate", "hf_run_epoch",
    "hf_greedy_decode", "decode_target_text",
]
