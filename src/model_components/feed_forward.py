import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.d_model, c.d_ff)
        self.fc2 = nn.Linear(c.d_ff, c.d_model)
        self.act = nn.GELU(); self.drop = nn.Dropout(c.dropout)
    def forward(self, x): return self.fc2(self.drop(self.act(self.fc1(x))))
