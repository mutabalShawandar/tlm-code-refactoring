    
from collections import defaultdict
# -- Metrics log (mirrors the XS notebook) ------------------------------------

class TrainingMetrics:
    """Append-only per-epoch metrics history, saved alongside checkpoints."""
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def get_last(self, key):
        return self.history[key][-1] if self.history.get(key) else None
