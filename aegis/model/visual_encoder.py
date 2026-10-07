from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm


class VisualEncoder(nn.Module):
    """Frozen pretrained 2-D backbone -> one embedding per frame. Each frame is encoded ONCE and the
    embedding is reused by every overlapping temporal window (and cached on disk for training)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, image_size: int = 224):
        super().__init__()
        self.backbone_name, self.pretrained, self.image_size = backbone, pretrained, image_size
        m = tvm.get_model(backbone, weights="DEFAULT" if pretrained else None)
        if hasattr(m, "fc"):
            self.out_dim = m.fc.in_features
            m.fc = nn.Identity()
        else:
            first = next(l for l in m.classifier if isinstance(l, nn.Linear))
            self.out_dim = first.in_features
            m.classifier = nn.Identity()
        self.net = m
        for p in self.net.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)          # frozen: BatchNorm stats must not update

    @property
    def tag(self) -> str:
        return f"{self.backbone_name}-{'in1k' if self.pretrained else 'rand'}-{self.image_size}"

    def preprocess(self, frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        """Resize on CPU (cheap), normalise on the model's device. Returns (B,3,H,W) float."""
        s = self.image_size
        arr = np.stack([cv2.cvtColor(cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                        for f in frames_bgr])
        x = torch.from_numpy(arr).to(self.mean.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def encode(self, frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        return self.net(self.preprocess(frames_bgr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
