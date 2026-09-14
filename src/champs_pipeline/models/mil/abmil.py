"""ABMIL: attention-based multiple instance learning (Ilse et al., ICML 2018).

One gated attention shared by all findings weights the tiles; the weighted
mean of the projected tiles is classified.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ABMIL(nn.Module):
    """Gated attention pooling with a two-layer classifier."""

    def __init__(self, input_dim, n_classes, hidden_dim=256, attention_dim=128, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.attention_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim // 2, n_classes))

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns logits and the attention over tiles."""
        h = self.projection(bag)
        scores = self.attention_w(self.attention_v(h) * self.attention_u(h)).transpose(1, 0)
        attention = self.attention_dropout(F.softmax(scores, dim=1))
        pooled = torch.mm(attention, h)
        return {"logits": self.classifier(pooled), "attention": attention}
