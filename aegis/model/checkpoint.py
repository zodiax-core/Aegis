from __future__ import annotations

import json
from typing import Optional, Tuple

import torch

from ..config import AegisConfig, ModelConfig, _build
from .aegis_model import AegisModel


def _plain(x):
    return json.loads(json.dumps(x, default=float))


def save_checkpoint(path: str, model: AegisModel, cfg: AegisConfig, metrics: dict, epoch: int,
                    optimizer=None, visual_tag: str = "", extra: Optional[dict] = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_metadata": model.metadata(),
        "class_names": model.class_names,
        "config": _plain(cfg.to_dict()),
        "metrics": _plain(metrics),
        "epoch": int(epoch),
        "visual_encoder": {"backbone": cfg.model.visual_backbone, "pretrained": cfg.model.visual_pretrained,
                           "image_size": cfg.sampling.image_size, "tag": visual_tag},
        **(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_model_checkpoint(path: str, device: str = "cpu") -> Tuple[AegisModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = ck["model_metadata"]
    mcfg = _build(ModelConfig, ck["config"]["model"])
    model = AegisModel(mcfg, meta["vis_dim"], meta["obj_dim"], meta["laya_dim"], ck["class_names"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck
