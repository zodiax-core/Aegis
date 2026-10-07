from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Transformer over per-timestep tokens with a learned CLS token. Returns (sequence, cls)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor):
        b, t, _ = tokens.shape
        x = tokens + self.pe[:, :t]
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = self.norm(self.encoder(x))
        return x[:, 1:], x[:, 0]
