from __future__ import annotations

import torch.nn as nn


def _mlp(d: int, out: int, dropout: float):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, out))


class EventHeads(nn.Module):
    """Independent heads on the fused representation. Add future heads (severity, interaction,
    temporal localisation) with add_head(name, out_dim) without touching the trunk."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.d_model, self.dropout = d_model, dropout
        self.heads = nn.ModuleDict({
            "event": _mlp(d_model, num_classes, dropout),      # closed-set classification logits
            "anomaly": _mlp(d_model, 1, dropout),              # anomalous-vs-normal logit
            "confidence": _mlp(d_model, 1, dropout),           # predicted P(event prediction is correct)
        })

    def add_head(self, name: str, out_dim: int) -> None:
        self.heads[name] = _mlp(self.d_model, out_dim, self.dropout)

    def forward(self, z):
        out = {}
        for k, h in self.heads.items():
            o = h(z)
            out[k] = o.squeeze(-1) if o.shape[-1] == 1 else o
        return out
