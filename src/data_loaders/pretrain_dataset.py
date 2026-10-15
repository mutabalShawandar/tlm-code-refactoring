from torch.utils.data import Dataset
from typing import List
import random
from src.data_set import make_sr_corruption

class PretrainDataset(Dataset):
    """
    Holds raw (untokenized) code strings.  Corruption is applied on-the-fly in
    __getitem__ so every epoch sees a different SEARCH/REPLACE corruption of the
    same code (free augmentation).  If a given code yields no maskable spans,
    we retry on it, then fall back to other random indices so __getitem__ always
    returns a valid (enc_ids, dec_ids) pair.
    """
    def __init__(self, codes: List[str], tok, cfg, max_retries: int = 4):
        self.codes = codes
        self.tok = tok
        self.cfg = cfg
        self.max_retries = max_retries

    def __len__(self):
        return len(self.codes)

    def _try(self, idx):
        for _ in range(self.max_retries):
            res = make_sr_corruption(self.codes[idx], self.tok, self.cfg)
            if res is not None:
                return res
        return None

    def __getitem__(self, idx):
        res = self._try(idx)
        # Fall back to other random codes until one corrupts successfully.
        guard = 0
        while res is None and guard < 32:
            idx = random.randrange(len(self.codes))
            res = self._try(idx)
            guard += 1
        if res is None:
            # Degenerate guarantee: a minimal valid pair (never expected to hit).
            bos, eos = self.cfg.special_token_ids["bos"], self.cfg.special_token_ids["eos"]
            return [bos, eos], [bos, eos]
        return res