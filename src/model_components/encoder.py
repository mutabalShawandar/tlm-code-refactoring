import torch.nn as nn

from .feed_forward import FeedForwardLayer
from .multi_head_attention import MultiHeadAttention
from .rope import RotaryEmbedding

class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = MultiHeadAttention(c); self.ffn = FeedForwardLayer(c)
        self.norm1 = nn.LayerNorm(c.d_model); self.norm2 = nn.LayerNorm(c.d_model)
        # Per-layer task embedding: separate table per block so each layer learns
        # its own task-specific residual shift. Zero-init = identity at pretrain start.
        self.task_emb = nn.Embedding(c.n_task_types, c.d_model)
        nn.init.zeros_(self.task_emb.weight)
    def forward(self, x, rope, task_ids=None, attention_mask=None):
        if task_ids is not None:
            x = x + self.task_emb(task_ids).unsqueeze(1)  # (B,1,d) broadcast over seq
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attention_mask=attention_mask, rope=rope)
        x = x + self.ffn(self.norm2(x))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(c) for _ in range(c.n_encoder_layers)])
        self.rope = RotaryEmbedding(c.d_model // c.n_heads, c.max_seq_len)
        self.final_norm = nn.LayerNorm(c.d_model)
    def forward(self, x, attention_mask=None, task_ids=None):
        for b in self.blocks:
            x = b(x, self.rope, task_ids=task_ids, attention_mask=attention_mask)
        return self.final_norm(x)
