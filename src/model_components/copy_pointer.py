import torch
import torch.nn as nn
import torch.nn.functional as F

class CopyPointerHead(nn.Module):
    """Pointer-generator gate: P = p_gen * P_vocab + (1 - p_gen) * P_copy."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    def forward(self, dec_hidden, vocab_logits, cross_attn_w, encoder_ids,
                pad_id: int, vocab_size: int):
        B, Sd, V = vocab_logits.shape
        Se = encoder_ids.size(1)
        p_gen = torch.sigmoid(self.gate(dec_hidden))            # (B, Sd, 1)
        P_vocab = F.softmax(vocab_logits, dim=-1)                # (B, Sd, V)
        pad_mask = (encoder_ids != pad_id).float().unsqueeze(1)  # (B, 1, Se)
        ca = cross_attn_w * pad_mask
        ca_sum = ca.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        ca = ca / ca_sum
        P_copy = torch.zeros_like(P_vocab)
        idx = encoder_ids.unsqueeze(1).expand(B, Sd, Se)
        P_copy.scatter_add_(2, idx, ca)
        P_final = p_gen * P_vocab + (1.0 - p_gen) * P_copy
        return torch.log(P_final.clamp(min=1e-12)), p_gen.squeeze(-1)
