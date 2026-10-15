# -- Dataset/collate for the HF-T5 SEARCH/REPLACE pipeline --------------------
# See sr_text.py's module docstring: everything under src/codet5/ is
# HF-T5-specific, kept separate from src/data_set, src/data_loaders (TLM's own
# pipeline) -- no imports either direction.

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset

from .sr_text import make_sft_sr_text


class SRTextDataset(Dataset):
    """Eager SEARCH/REPLACE dataset for a real HF tokenizer: reads
    {before, after, type} records (same JSONL shape as TLM's SFTDataset/
    BridgingDataset -- data/finetune_data/supervised_finetuning_generated.jsonl
    or data/bridging/bridging_{train,val}.jsonl), builds text-level SEARCH/
    REPLACE targets via make_sft_sr_text, then tokenizes both sides with the
    given HF tokenizer. One class serves both SFT and bridging -- unlike
    TLM's BridgingDataset, no separate tracked-spans path is needed for
    `rename`: diff_hunks_text round-trips rename records correctly (verified
    1000/1000 against both the curated SFT data and bridging_train.jsonl's
    synthetic rename records), so every task type goes through the same
    diff-based construction here."""

    def __init__(self, path: Union[str, Path], tokenizer, task_marker_fmt: str,
                 search_marker: str, replace_marker: str,
                 max_src_len: int = 512, max_tgt_len: int = 512):
        self.samples: List[Tuple[List[int], List[int], int]] = []
        self.dropped = 0
        self.kept_by_type: Dict[str, int] = {}
        self.dropped_by_type: Dict[str, int] = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ttype = rec.get("type")
                res = make_sft_sr_text(rec, task_marker_fmt, search_marker, replace_marker)
                if res is None:
                    self.dropped += 1
                    self.dropped_by_type[ttype] = self.dropped_by_type.get(ttype, 0) + 1
                    continue
                enc_text, dec_text, task_id = res
                enc_ids = tokenizer(enc_text, truncation=True, max_length=max_src_len).input_ids
                dec_ids = tokenizer(dec_text, truncation=True, max_length=max_tgt_len).input_ids
                if not enc_ids or not dec_ids:
                    self.dropped += 1
                    self.dropped_by_type[ttype] = self.dropped_by_type.get(ttype, 0) + 1
                    continue
                self.samples.append((enc_ids, dec_ids, task_id))
                self.kept_by_type[ttype] = self.kept_by_type.get(ttype, 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (enc_ids, dec_ids, task_id)


def collate_sr(batch, pad_id: int):
    """Pad encoder/decoder id sequences to the batch max; build the encoder
    attention mask. `task_id` is carried through but unused by the plain HF
    model (no PLE/task-embedding mechanism here) -- kept only so this batch
    dict shape matches TLM's collate_sft for any shared eval/plotting code."""
    enc_lists, dec_lists, task_ids = zip(*batch)
    max_se = max(len(x) for x in enc_lists)
    max_sd = max(len(x) for x in dec_lists)
    B = len(batch)

    enc_ids = torch.full((B, max_se), pad_id, dtype=torch.long)
    dec_ids = torch.full((B, max_sd), pad_id, dtype=torch.long)
    enc_attn = torch.zeros((B, max_se), dtype=torch.long)

    for i, (e, d) in enumerate(zip(enc_lists, dec_lists)):
        enc_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        dec_ids[i, :len(d)] = torch.tensor(d, dtype=torch.long)
        enc_attn[i, :len(e)] = 1

    return {"enc_ids": enc_ids, "enc_attn": enc_attn, "dec_ids": dec_ids,
            "task_ids": torch.tensor(task_ids, dtype=torch.long)}
