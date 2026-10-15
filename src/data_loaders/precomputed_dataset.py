from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset

try:
    import orjson as _json
    _JSON_LOADS = _json.loads          # accepts bytes, ~3x faster than stdlib json
    _READ_MODE = "rb"
except ImportError:
    import json as _json
    _JSON_LOADS = _json.loads
    _READ_MODE = "r"


def _read_jsonl_pairs(path: Path) -> List[Tuple[List[int], List[int]]]:
    pairs = []
    kwargs = {} if _READ_MODE == "rb" else {"encoding": "utf-8"}
    with path.open(_READ_MODE, **kwargs) as f:
        for line in f:
            rec = _JSON_LOADS(line)
            pairs.append((rec["enc_ids"], rec["dec_ids"]))
    return pairs


class PrecomputedPretrainDataset(Dataset):
    """
    Drop-in replacement for PretrainDataset: same (enc_ids, dec_ids) tuple contract,
    so it works unchanged with collate_fn / make_concat_dataset /
    curriculum_total_optimizer_steps. Loads N precomputed corruption variants for one
    tier (produced by notebooks/precompute_curriculum_corruption.ipynb) instead of
    corrupting on the fly. Call `.set_variant(epoch)` once per epoch in the training
    loop to cycle through variants (e.g. `ds.set_variant(epoch - 1)`), preserving most
    of the on-the-fly version's per-epoch augmentation diversity at a fraction of the
    per-batch CPU cost.
    """

    def __init__(self, variant_dir: Path, n_variants: int):
        self.variant_dir = Path(variant_dir)
        self.n_variants = n_variants
        self._variants = [
            _read_jsonl_pairs(self.variant_dir / f"variant_{v}.jsonl")
            for v in range(n_variants)
        ]
        n = len(self._variants[0])
        assert all(len(v) == n for v in self._variants), \
            f"variant length mismatch in {self.variant_dir}"
        self.variant = 0

    def set_variant(self, epoch_0based: int):
        self.variant = epoch_0based % self.n_variants

    def __len__(self):
        return len(self._variants[0])

    def __getitem__(self, idx):
        enc_ids, dec_ids = self._variants[self.variant][idx]
        return enc_ids, dec_ids


def load_precomputed_tiers(root: Path, tiers: List[int], n_variants: int) -> dict:
    """Build {tier_no: PrecomputedPretrainDataset} for every tier under `root`
    (root/tierN/variant_V.jsonl)."""
    return {
        t: PrecomputedPretrainDataset(root / f"tier{t}", n_variants)
        for t in tiers
    }
