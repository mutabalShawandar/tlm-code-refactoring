import math
import torch


class ScheduledSampler:
    """
    p_sample schedule over epochs (linear example):
        epoch 0           -> 0.00  (pure teacher forcing -- safe early training)
        epoch total/2     -> 0.50  (50/50 gold/model mix)
        epoch total       -> 1.00  (pure model sampling -- matches inference)
    """
    def __init__(self, total_epochs: int, decay_type: str = "linear"):
        self.total_epochs = total_epochs
        self.decay_type   = decay_type

    def get_sampling_prob(self, epoch: int) -> float:
        progress = min(epoch / max(self.total_epochs, 1), 1.0)
        if self.decay_type == "linear":
            return progress
        if self.decay_type == "exponential":
            return 1.0 - math.exp(-3.0 * progress)
        if self.decay_type == "inverse_sigmoid":
            return 1.0 / (1.0 + math.exp(-(progress - 0.5) * 6.0))
        raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def sample_or_teach(self, model_pred: torch.Tensor, gold_token: torch.Tensor,
                        p_sample: float, pad_id: int) -> torch.Tensor:
        """Per-token choice: model prediction with prob p_sample, else gold (pad-safe)."""
        use_model = (torch.rand_like(gold_token, dtype=torch.float) < p_sample) & (gold_token != pad_id)
        return torch.where(use_model, model_pred, gold_token)
