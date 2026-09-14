"""MamMIL (Yang et al., IEEE BIBM 2024): bidirectional state-space blocks over the tiles.

Needs the CUDA kernel of ``mamba-ssm`` (``make env-gpu``); there is no
fallback, so a run without the kernel fails at construction.
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - exercised only without the kernel
    Mamba = None


class MambaBlock(nn.Module):
    """Norm, Mamba, dropout, residual."""

    def __init__(self, dim, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class BidirectionalMamba(nn.Module):
    """One block along the scan order, one against it, fused by a linear layer."""

    def __init__(self, dim, d_state, d_conv, expand, dropout):
        super().__init__()
        self.forward_block = MambaBlock(dim, d_state, d_conv, expand, dropout)
        self.backward_block = MambaBlock(dim, d_state, d_conv, expand, dropout)
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x):
        ahead = self.forward_block(x)
        behind = torch.flip(self.backward_block(torch.flip(x, dims=[1])), dims=[1])
        return self.fusion(torch.cat([ahead, behind], dim=-1))


class InstanceAggregation(nn.Module):
    """Self-attention and a feed-forward layer between the state-space blocks."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(dim * 2, dim), nn.Dropout(dropout))

    def forward(self, x):
        normed = self.norm1(x)
        # Without the weight matrix the fused kernel runs; with it, bags of 30,000 tiles need
        # tens of gigabytes for the (heads, tiles, tiles) weights alone.
        x = x + self.attention(normed, normed, normed, need_weights=False)[0]
        return x + self.feedforward(self.norm2(x))


class MamMIL(nn.Module):
    """Projection, bidirectional Mamba blocks, attention pooling, classifier."""

    def __init__(self, input_dim, n_classes, hidden_dim=512, num_mamba_layers=2, d_state=16,
                 d_conv=4, expand=2, dropout=0.25, use_instance_aggregation=True):
        super().__init__()
        if Mamba is None:
            raise ImportError("MamMIL needs the mamba-ssm CUDA kernel: run `make env-gpu`")
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout))
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(hidden_dim, d_state, d_conv, expand, dropout)
            for _ in range(num_mamba_layers)])
        n_aggregations = num_mamba_layers - 1 if use_instance_aggregation else 0
        self.aggregations = nn.ModuleList([InstanceAggregation(hidden_dim, dropout)
                                           for _ in range(n_aggregations)])
        self.attention_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                                            nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim // 2, n_classes))

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns logits and the pooling attention."""
        x = self.projection(bag.unsqueeze(0))
        for i, block in enumerate(self.mamba_layers):
            x = block(x)
            if i < len(self.aggregations):
                x = self.aggregations[i](x)
        attention = F.softmax(self.attention_pool(x).transpose(2, 1), dim=2)
        pooled = torch.bmm(attention, x).squeeze(1)
        return {"logits": self.classifier(pooled), "attention": attention.squeeze(1)}
