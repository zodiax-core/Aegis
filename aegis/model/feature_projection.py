from __future__ import annotations

import torch.nn as nn


class FeatureProjection(nn.Module):
    """Learned projection so every modality (any input dim) lands in the shared d_model space."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
