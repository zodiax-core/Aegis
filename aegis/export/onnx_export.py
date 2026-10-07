"""PyTorch -> ONNX (temporal model) and optional TensorRT build. Training stays PyTorch."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import torch

from ..model.aegis_model import AegisModel
from ..utils import log


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, vis, obj, laya, laya_mask):
        o = self.m(vis, obj, laya, laya_mask)
        return o["event_logits"], o["anomaly_logit"], o["confidence_logit"]


def export_onnx(model: AegisModel, path: str, clip_length: int, opset: int = 17, verify: bool = True) -> str:
    model = model.eval().cpu()
    w = _Wrap(model).eval()      # wrapper must be eval too: export restores its train-mode afterwards
    vis = torch.randn(1, clip_length, model.vis_dim)
    obj = torch.randn(1, clip_length, model.obj_dim)
    laya = torch.zeros(1, model.laya_dim)
    mask = torch.zeros(1, dtype=torch.bool)
    torch.onnx.export(w, (vis, obj, laya, mask), path, opset_version=opset, dynamo=False,
                      input_names=["vis", "obj", "laya", "laya_mask"],
                      output_names=["event_logits", "anomaly_logit", "confidence_logit"],
                      dynamic_axes={k: {0: "batch"} for k in ["vis", "obj", "laya", "laya_mask",
                                                              "event_logits", "anomaly_logit", "confidence_logit"]})
    model.eval()
    if verify:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        vv, oo = torch.randn(3, clip_length, model.vis_dim), torch.randn(3, clip_length, model.obj_dim)  # new data, batch 3
        ll, mm = torch.rand(3, model.laya_dim), torch.tensor([True, False, True])
        out = s.run(None, {"vis": vv.numpy(), "obj": oo.numpy(), "laya": ll.numpy(), "laya_mask": mm.numpy()})
        with torch.no_grad():
            ref = w(vv, oo, ll, mm)[0].numpy()
        err = float(np.abs(out[0] - ref).max())
        log.info("ONNX parity max|diff| = %.2e", err)
        if err > 1e-3:
            raise RuntimeError(f"ONNX output differs from PyTorch by {err}")
    return path


def build_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True, int8: bool = False) -> str:
    """Requires NVIDIA TensorRT (`trtexec`). INT8 needs a calibration cache and an accuracy check;
    it is refused here rather than silently quantising."""
    if int8:
        raise NotImplementedError("INT8 is intentionally not automated: provide a calibration set and validate "
                                  "accuracy first (see README, 'Precision').")
    exe = shutil.which("trtexec")
    if exe is None:
        raise RuntimeError("trtexec not found: install TensorRT on the target NVIDIA machine.")
    cmd = [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"] + (["--fp16"] if fp16 else [])
    subprocess.run(cmd, check=True)
    return engine_path


def export_yolo_engine(weights: str = "yolo11n.pt", half: bool = True, imgsz: int = 640) -> str:
    from ultralytics import YOLO
    return str(YOLO(weights).export(format="engine", half=half, imgsz=imgsz))
