import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, c):
        super().__init__()
        self.d_model = c.d_model; self.n_heads = c.n_heads
        self.d_head = c.d_model // c.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dropout = nn.Dropout(c.dropout)
        
    def forward(self, q_in, k_in, v_in, attention_mask=None, rope=None, return_weights=False,
                past_kv=None, use_cache=False, position_offset=0):
        """
        past_kv / use_cache: incremental-decode KV cache.
        - k_in/v_in is None: reuse past_kv verbatim (steady-state cross-attn after the
          first decode step, since encoder keys/values never change across steps).
        - k_in/v_in given + past_kv given: k_in/v_in hold only the *new* tokens; their
          projected k/v are concatenated onto past_kv (self-attn incremental step).
        - k_in/v_in given + past_kv None: normal full forward (training / first step).
        position_offset: absolute sequence position of the first row of q_in/k_in, used
        so RoPE rotates newly-computed k/q at their true position before concatenation
        with already-rotated cached k (cached k must never be re-rotated).
        """
        B, Sq, _ = q_in.size()
        q = self.q_proj(q_in).view(B, Sq, self.n_heads, self.d_head).transpose(1, 2)

        if k_in is None:
            assert past_kv is not None, "k_in=None requires a past_kv to reuse"
            k, v = past_kv
        else:
            _, Sk_new, _ = k_in.size()
            k_new = self.k_proj(k_in).view(B, Sk_new, self.n_heads, self.d_head).transpose(1, 2)
            v_new = self.v_proj(v_in).view(B, Sk_new, self.n_heads, self.d_head).transpose(1, 2)
            if rope is not None:
                cos, sin = rope(position_offset + Sq, q.device)
                cos_q = cos[position_offset:position_offset + Sq].unsqueeze(0).unsqueeze(0)
                sin_q = sin[position_offset:position_offset + Sq].unsqueeze(0).unsqueeze(0)
                q, k_new = apply_rope(q, k_new, cos_q, sin_q)
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k_new], dim=2)
                v = torch.cat([pv, v_new], dim=2)
            else:
                k, v = k_new, v_new

        present = (k, v) if use_cache else None
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        dropped = self.dropout(weights)
        out = torch.matmul(dropped, v).transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        out = self.out_proj(out)
        if return_weights and use_cache:
            return out, weights.max(dim=1).values, present
        if return_weights:
            return out, weights.max(dim=1).values  # (B, Sq, Sk)
        if use_cache:
            return out, present
        return out
