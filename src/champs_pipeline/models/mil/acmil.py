"""ACMIL, gated-attention variant (Zhang et al., ECCV 2024), one branch per finding.

The multiple-branch attention shares one gated representation and
projects it to one attention score per finding; each branch classifies its
own attention-weighted mean. The two regularisers of the paper are
parameters: stochastic top-k instance masking (``n_masked_patch`` tiles
per branch, of which a ``mask_drop`` fraction is hidden while training)
and the diversity loss (mean squared cosine similarity between the
branches' attention rows). The reported configuration runs both off.
"""

import torch
import torch.nn.functional as F
from torch import nn


def mask_top_tiles(scores, n_masked_patch, mask_drop):
    """Hide a random ``mask_drop`` share of each branch's ``n_masked_patch`` top tiles."""
    n_branches, n_tiles = scores.shape
    n_top = min(n_masked_patch, n_tiles)
    n_drop = int(n_top * mask_drop)
    if n_top <= 0 or n_drop <= 0:
        return scores
    _, top = torch.topk(scores, n_top, dim=-1)
    order = torch.argsort(torch.rand_like(top, dtype=torch.float), dim=-1)
    dropped = torch.gather(top, -1, order[:, :n_drop])
    keep = torch.ones(n_branches, n_tiles, device=scores.device)
    keep.scatter_(-1, dropped, 0)
    return scores.masked_fill(keep == 0, torch.finfo(scores.dtype).min)


def diversity_loss(attention):
    """Mean squared cosine similarity between distinct attention rows."""
    unit = attention / attention.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cosine = unit @ unit.T
    off_diagonal = ~torch.eye(attention.shape[0], dtype=torch.bool, device=attention.device)
    return cosine[off_diagonal].pow(2).mean()


class ACMIL(nn.Module):
    """Multiple-branch gated attention with per-branch classifiers."""

    def __init__(self, input_dim, n_classes, hidden_dim=512, attention_dim=256, dropout=0.25,
                 n_masked_patch=0, mask_drop=0.6, diversity_weight=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.diversity_weight = diversity_weight
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, n_classes)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_classes)])

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns logits, attention and the diversity term."""
        h = self.projection(bag)
        scores = self.attention_w(self.attention_v(h) * self.attention_u(h)).transpose(1, 0)
        if self.training and self.n_masked_patch > 0:
            scores = mask_top_tiles(scores, self.n_masked_patch, self.mask_drop)
        attention = F.softmax(scores, dim=1)
        pooled = torch.mm(attention, h)
        logits = torch.cat([classifier(pooled[i].unsqueeze(0))
                            for i, classifier in enumerate(self.classifiers)], dim=1)
        out = {"logits": logits, "attention": attention}
        if self.diversity_weight > 0 and self.n_classes > 1:
            out["extra_loss"] = self.diversity_weight * diversity_loss(attention)
        return out
