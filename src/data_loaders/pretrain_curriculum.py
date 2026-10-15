# -- Curriculum (tier sampling) helpers -- ported from the XS notebook ---------

import json
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset, ConcatDataset


def load_curriculum_config(path: Path) -> dict:
    """Load a curriculum config JSON file, with defaults for missing fields."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("schedule", [])
    cfg.setdefault("total_epochs", 20)
    return cfg

def tiers_for_epoch(epoch_1based: int, curriculum_cfg: dict) -> List[int]:
    """Active tier numbers for a 1-based epoch; falls back to all tiers if unlisted."""
    for phase in curriculum_cfg.get("schedule", []):
        if epoch_1based in phase.get("epochs", []):
            return [int(t) for t in phase.get("tiers", [])]
    return [1, 2, 3, 4, 5]


def max_epoch_from_curriculum(curriculum_cfg: dict) -> int:
    epochs = []
    for phase in curriculum_cfg.get("schedule", []):
        epochs.extend(int(e) for e in phase.get("epochs", []))
    return max(epochs) if epochs else curriculum_cfg.get("total_epochs", 20)


def make_concat_dataset(datasets_by_tier: Dict[int, Dataset], tiers: List[int]) -> Dataset:
    """Concatenate the PretrainDatasets for the requested tiers into one dataset."""
    selected = [datasets_by_tier[int(t)] for t in tiers if int(t) in datasets_by_tier]
    if not selected:
        raise ValueError(f"No datasets available for requested tiers: {tiers}")
    return selected[0] if len(selected) == 1 else ConcatDataset(selected)


def curriculum_total_optimizer_steps(curriculum_cfg, datasets_by_tier, batch_size) -> int:
    """Sum of per-epoch batch counts across the whole curriculum (for the LR schedule)."""
    total = 0
    for ep in range(1, max_epoch_from_curriculum(curriculum_cfg) + 1):
        active = tiers_for_epoch(ep, curriculum_cfg)
        n = sum(len(datasets_by_tier[t]) for t in active if t in datasets_by_tier)
        total += n // batch_size            # drop_last=True on the train loader
    return max(1, total)
