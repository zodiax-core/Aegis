from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..perception.features import OBJ_FEAT_DIM
from ..semantic.laya_questions import LAYA_FEAT_DIM
from .event_head import EventHeads
from .feature_projection import FeatureProjection
from .fusion import FusionModule
from .temporal_encoder import TemporalEncoder


class AegisModel(nn.Module):
    """visual/object sequences -> temporal encoder -> fusion(+optional Laya) -> heads.

    forward(vis (B,T,Dv), obj (B,T,Do), laya (B,Dl)|None, laya_mask (B,)|None) -> dict of tensors:
      event_logits (B,C), anomaly_logit (B,), confidence_logit (B,)
    """

    def __init__(self, cfg: ModelConfig, vis_dim: int, obj_dim: int = OBJ_FEAT_DIM, laya_dim: int = LAYA_FEAT_DIM,
                 class_names: Optional[List[str]] = None):
        super().__init__()
        self.cfg, self.vis_dim, self.obj_dim, self.laya_dim = cfg, vis_dim, obj_dim, laya_dim
        self.class_names = list(class_names or cfg.class_names)
        d = cfg.d_model
        self.vis_proj = FeatureProjection(vis_dim, d, cfg.dropout)
        self.obj_proj = FeatureProjection(obj_dim, d, cfg.dropout)
        self.laya_proj = FeatureProjection(laya_dim, d, cfg.dropout)
        self.temporal_encoder = TemporalEncoder(d, cfg.n_heads, cfg.temporal_layers, cfg.ff_dim, cfg.dropout)
        self.fusion = FusionModule(d, cfg.n_heads, cfg.fusion_layers, cfg.ff_dim, cfg.dropout)
        self.heads = EventHeads(d, len(self.class_names), cfg.dropout)

    def forward(self, vis, obj, laya=None, laya_mask=None) -> Dict[str, torch.Tensor]:
        b = vis.shape[0]
        if laya is None:
            laya = torch.zeros(b, self.laya_dim, device=vis.device, dtype=vis.dtype)
            laya_mask = torch.zeros(b, dtype=torch.bool, device=vis.device)
        elif laya_mask is None:
            laya_mask = torch.ones(b, dtype=torch.bool, device=vis.device)
        v, o = self.vis_proj(vis), self.obj_proj(obj)
        _, temporal = self.temporal_encoder(v + o)
        z = self.fusion(temporal, v.mean(1), o.mean(1), self.laya_proj(laya.to(v.dtype)), laya_mask)
        h = self.heads(z)
        return {"event_logits": h["event"], "anomaly_logit": h["anomaly"], "confidence_logit": h["confidence"],
                **{f"{k}_out": v_ for k, v_ in h.items() if k not in ("event", "anomaly", "confidence")}}

    # ---- trainability switches (train.py) ---------------------------------------------------
    def set_trainable(self, temporal_encoder=True, fusion=True, event_head=True):
        for p in self.parameters():
            p.requires_grad = False
        groups = []
        if temporal_encoder:
            groups += [self.temporal_encoder, self.vis_proj, self.obj_proj]
        if fusion:
            groups += [self.fusion, self.laya_proj]
        if event_head:
            groups += [self.heads]
        for g in groups:
            for p in g.parameters():
                p.requires_grad = True

    def metadata(self) -> dict:
        return {"vis_dim": self.vis_dim, "obj_dim": self.obj_dim, "laya_dim": self.laya_dim,
                "class_names": self.class_names, "n_params": sum(p.numel() for p in self.parameters())}
