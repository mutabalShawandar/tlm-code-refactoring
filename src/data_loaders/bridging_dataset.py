import json
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset

from src.data_set.sft_corruption import make_sft_sr
from src.data_set.bridging_corruption import make_bridge_sr


class BridgingDataset(Dataset):
    """Eager bridging dataset: reads {type, before, after, spans?} records
    (built by src/utils/build_bridging_dataset.py) and builds
    (enc_ids, dec_ids, task_id) SEARCH/REPLACE samples once at construction.

    Same eager-build contract as SFTDataset, but dispatches per-record: type
    "rename" records carry their own tracked `spans` and go through
    make_bridge_sr (no SequenceMatcher/diff ambiguity risk); every other
    type goes through make_sft_sr (diff_hunks over before/after), same as
    the curated SFT dataset -- see make_rename_bridge's docstring in
    bridging_corruption.py for why rename alone needs its own path."""

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
                build_fn = make_bridge_sr if ttype == "rename" else make_sft_sr
                res = build_fn(rec, tok, cfg, max_src_len, max_tgt_len)
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


class PrecomputedBridgingDataset(Dataset):
    """Loads (enc_ids, dec_ids, task_id) triples already tokenized by
    src/utils/precompute_bridging_dataset.py, instead of re-running
    BridgingDataset's eager build (BPE-encode + diff_hunks over 75,000
    text records measured at ~141s locally -- non-trivial to repeat on
    every Colab session restart). No variant-cycling like
    PrecomputedPretrainDataset -- bridging targets are deterministic
    (unlike pretrain's random span-corruption), so there's only ever one
    version to load."""

    def __init__(self, path: Union[str, Path]):
        self.samples: List[Tuple[List[int], List[int], int]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append((rec["enc_ids"], rec["dec_ids"], rec["task_id"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (enc_ids, dec_ids, task_id)
