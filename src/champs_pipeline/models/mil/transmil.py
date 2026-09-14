"""TransMIL (Shao et al., NeurIPS 2021) with Nyström self-attention over the tiles.

Tiles are projected, given a sinusoidal position code in scan order,
passed through transformer blocks with Nyström attention, pooled by a
learned attention, and classified.
"""

import math

import torch
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from torch import nn


def positional_encoding(n_positions, d_model, device):
    """Sinusoidal codes of shape (1, n_positions, d_model)."""
    position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
    step = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device)
                     * (-math.log(10000.0) / d_model))
    codes = torch.zeros(n_positions, d_model, device=device)
    codes[:, 0::2] = torch.sin(position * step)
    codes[:, 1::2] = torch.cos(position * step)
    return codes.unsqueeze(0)


class TransMILBlock(nn.Module):
    """Nyström self-attention and a feed-forward layer, each with a residual and a norm."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attention = NystromAttention(dim=d_model, dim_head=d_model // nhead, heads=nhead,
                                          num_landmarks=d_model // 2, pinv_iterations=6,
                                          residual=True, dropout=dropout)
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
                                         nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x = self.norm1(x + self.dropout(self.attention(x, mask=mask)))
        return self.norm2(x + self.feedforward(x))


class TransMIL(nn.Module):
    """Transformer blocks over the tiles, attention pooling, a two-layer classifier."""

    def __init__(self, input_dim, n_classes, d_model=512, nhead=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.2, use_positional_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.blocks = nn.ModuleList([TransMILBlock(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.attention_pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(),
                                            nn.Linear(d_model // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes))

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns logits and the pooling attention."""
        x = self.projection(bag).unsqueeze(0)
        if self.use_positional_encoding:
            x = x + positional_encoding(x.shape[1], self.d_model, x.device)
        for block in self.blocks:
            x = block(x)
        attention = F.softmax(self.attention_pool(x), dim=1)
        pooled = torch.sum(attention * x, dim=1)
        return {"logits": self.classifier(pooled), "attention": attention.squeeze(-1)}
