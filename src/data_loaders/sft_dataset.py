import json
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset

from src.data_set.sft_corruption import make_sft_sr, make_full_after


class SFTDataset(Dataset):
    """Eager SFT dataset: reads {before, after, type} records and builds
    (enc_ids, dec_ids, task_id) SEARCH/REPLACE samples once at construction.

    Unlike PretrainDataset (random on-the-fly corruption), SFT targets are
    deterministic, so we build them up front. Records that yield no edit, an
    unknown task type, or exceed the length budget are dropped and counted in
    `self.dropped` (per-reason in `self.drop_reasons`).
    """

    def __init__(self, path: Union[str, Path], tok, cfg,
                 max_src_len: int = 512, max_tgt_len: int = 512):
        self.samples: List[Tuple[List[int], List[int], int]] = []
        self.dropped = 0
        self.kept_by_type: dict = {}
        self.dropped_by_type: dict = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ttype = rec.get("type")
                res = make_sft_sr(rec, tok, cfg, max_src_len, max_tgt_len)
                if res is None:
                    self.dropped += 1
                    self.dropped_by_type[ttype] = self.dropped_by_type.get(ttype, 0) + 1
                    continue
                self.samples.append(res)
                self.kept_by_type[ttype] = self.kept_by_type.get(ttype, 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (enc_ids, dec_ids, task_id)


class FullAfterSFTDataset(Dataset):
    """Eager SFT dataset for full-after generation.

    Decoder target is the complete refactored function (<bos> [after] <eos>)
    rather than SEARCH/REPLACE hunks. Simpler format, works for all task types
    including extract_method where the after contains entirely new code.
    """

    def __init__(self, path: Union[str, Path], tok, cfg,
                 max_src_len: int = 512, max_tgt_len: int = 512):
        self.samples: List[Tuple[List[int], List[int], int]] = []
        self.dropped = 0
        self.kept_by_type: dict = {}
        self.dropped_by_type: dict = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ttype = rec.get("type")
                res = make_full_after(rec, tok, cfg, max_src_len, max_tgt_len)
                if res is None:
                    self.dropped += 1
                    self.dropped_by_type[ttype] = self.dropped_by_type.get(ttype, 0) + 1
                    continue
                self.samples.append(res)
                self.kept_by_type[ttype] = self.kept_by_type.get(ttype, 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (enc_ids, dec_ids, task_id)


def collate_sft(batch, pad_id: int):
    """Pad encoder/decoder sequences and carry the per-sample task_id.

    Mirrors collate_fn but consumes 3-tuples and emits real task_ids (1..5)
    instead of hardcoding zeros -- required for SFT, or PLE task conditioning
    is silently lost (the pretrain collate_fn hardcodes task_ids=0)."""
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

    return {
        "enc_ids": enc_ids,
        "enc_attn": enc_attn,
        "dec_ids": dec_ids,
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
    }
