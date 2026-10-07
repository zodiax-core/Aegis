from __future__ import annotations

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Attention fusion over modality tokens [FUSE, temporal, visual, object, laya].
    Laya may be absent: it is masked out of attention, so the model works with and without it."""

    N_TOK = 5

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.fuse = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.fuse, std=0.02)
        self.type_emb = nn.Parameter(torch.zeros(1, self.N_TOK, d_model))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, temporal, visual, objects, laya, laya_mask):
        """each token (B, d); laya_mask (B,) bool: True = Laya present."""
        b = temporal.shape[0]
        x = torch.cat([self.fuse.expand(b, -1, -1), temporal[:, None], visual[:, None],
                       objects[:, None], laya[:, None]], dim=1) + self.type_emb
        pad = torch.zeros(b, self.N_TOK, dtype=torch.bool, device=x.device)
        pad[:, 4] = ~laya_mask.bool()                       # True = ignore this token
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.norm(x[:, 0])
