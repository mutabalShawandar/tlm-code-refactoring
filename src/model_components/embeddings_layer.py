import torch.nn as nn
import torch

class EmbeddingLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[c.special_token_ids["pad"]].zero_()
        self.dropout = nn.Dropout(c.dropout)
    def forward(self, ids): return self.dropout(self.token_embedding(ids))
