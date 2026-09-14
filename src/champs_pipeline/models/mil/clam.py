"""CLAM, multi-branch (Lu et al., Nature BME 2021): one attention per finding.

Each finding has its own gated attention over the tiles and its own linear
classifier on its attention-weighted mean.
"""

import torch
import torch.nn.functional as F
from torch import nn


class GatedAttention(nn.Module):
    """Tanh and sigmoid branches multiplied, then one score per finding."""

    def __init__(self, hidden_dim, attention_dim, n_classes, dropout):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.attention_c = nn.Linear(attention_dim, n_classes)

    def forward(self, h):
        return self.attention_c(self.dropout(self.attention_a(h) * self.attention_b(h)))


class CLAM_MB(nn.Module):
    """Per-finding gated attention pooling and per-finding linear classifiers."""

    def __init__(self, input_dim, n_classes, hidden_dim=512, attention_dim=256, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout))
        self.attention = GatedAttention(hidden_dim, attention_dim, n_classes, dropout)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_classes)])

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns logits and the per-finding attention."""
        h = self.projection(bag)
        attention = F.softmax(self.attention(h).transpose(1, 0), dim=1)
        pooled = torch.mm(attention, h)
        logits = torch.cat([classifier(pooled[i].unsqueeze(0))
                            for i, classifier in enumerate(self.classifiers)], dim=1)
        return {"logits": logits, "attention": attention}
