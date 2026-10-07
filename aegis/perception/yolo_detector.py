"""The ONLY module that touches the Ultralytics API. Everything else sees Detection objects."""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..config import PerceptionConfig
from ..utils import log, resolve_device
from .detector_base import Detector
from .object_schema import Detection


class YoloDetector(Detector):
    def __init__(self, cfg: PerceptionConfig, device: str = "auto"):
        from ultralytics import YOLO  # lazy: keeps the rest of AEGIS importable without it
        self.cfg = cfg
        self.device = resolve_device(device)
        self.model = YOLO(cfg.weights)
        self.name = f"yolo:{cfg.weights}"
        # ultralytics 8.4+ deprecated half= in .predict(); quantize= takes a string ('fp16') or is omitted
        self.quantize = "fp16" if (cfg.half and self.device == "cuda") else None
        names = self.model.names  # {id: name}
        wanted = {n.lower() for n in cfg.relevant_classes}
        self.class_ids = [i for i, n in names.items() if n.lower() in wanted]
        missing = wanted - {names[i].lower() for i in self.class_ids}
        if missing:
            log.info("Detector weights do not know classes %s (COCO has no gun/pistol/rifle; "
                     "use a weapon-trained checkpoint for those)", sorted(missing))
        if not self.class_ids:
            self.class_ids = None  # keep everything rather than silently detecting nothing

    def _convert(self, result) -> List[Detection]:
        out: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            out.append(Detection(int(k), str(result.names[int(k)]), float(c), tuple(float(v) for v in b)))
        return out

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> List[List[Detection]]:
        if not len(frames_bgr):
            return []
        kw = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                  device=self.device, classes=self.class_ids, verbose=False)
        if self.quantize:          # only set when 'fp16' — never pass the kwarg as False/None
            kw["quantize"] = self.quantize
        results = self.model.predict(list(frames_bgr), **kw)
        return [self._convert(r) for r in results]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.detect_batch([frame_bgr])[0]


class PoseDetector(YoloDetector):
    """YOLO11-pose: bounding boxes + 17 COCO body keypoints (x, y, conf) per person.

    Exposes ``detect_with_pose()`` which returns ``(List[Detection], List[dict|None])``
    where each dict has keys ``"xy": ndarray(17,2)`` and ``"conf": ndarray(17,)``.
    Code outside this module must use duck-typing (``hasattr(det, 'detect_with_pose')``).
    """

    def _convert_with_pose(self, result):
        dets = self._convert(result)
        keypoints: list = []
        if result.keypoints is not None and len(result.keypoints) > 0:
            kp_xy   = result.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            kp_conf = result.keypoints.conf.cpu().numpy()  # (N, 17)
            for xy, cf in zip(kp_xy, kp_conf):
                keypoints.append({"xy": xy, "conf": cf})
        else:
            keypoints = [None] * len(dets)
        return dets, keypoints

    def detect_with_pose(self, frame_bgr: np.ndarray):
        """Returns (List[Detection], List[dict|None]) for a single frame."""
        kw = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                  device=self.device, classes=self.class_ids, verbose=False)
        if self.quantize:
            kw["quantize"] = self.quantize
        results = self.model.predict([frame_bgr], **kw)
        return self._convert_with_pose(results[0])

    def detect_batch(self, frames_bgr):
        """Batch detect without pose (used by generic pipeline paths)."""
        dets_batch = []
        for frame in frames_bgr:
            dets, _ = self.detect_with_pose(frame)
            dets_batch.append(dets)
        return dets_batch


class MotionBlobDetector(Detector):
    """Weight-free fallback: reports moving blobs as 'person'. NOT a real detector.
    Exists so the pipeline, tests and demos run without downloading weights."""

    name = "motion-blobs(fallback)"

    def __init__(self, min_area: int = 600, history: int = 60):
        self.min_area = min_area
        self._bs = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=32, detectShadows=False)
        self._k = np.ones((5, 5), np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        small = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5)
        mask = self._bs.apply(small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k)
        mask = cv2.dilate(mask, self._k, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) < self.min_area / 4:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(Detection(0, "person", 0.5, (2.0 * x, 2.0 * y, 2.0 * (x + w), 2.0 * (y + h))))
        return dets


def build_detector(cfg: PerceptionConfig, device: str = "auto") -> Detector:
    backend = cfg.backend.lower()
    if backend in ("auto", "yolo"):
        try:
            weights = cfg.weights or ""
            # Route pose-capable weights to PoseDetector so callers can call detect_with_pose()
            if weights.endswith("-pose.pt") or "-pose" in weights:
                log.info("Pose weights detected (%s) — using PoseDetector.", weights)
                return PoseDetector(cfg, device)
            return YoloDetector(cfg, device)
        except Exception as e:  # missing package, offline weight download, ...
            if backend == "yolo":
                raise
            log.warning("YOLO unavailable (%s: %s). Falling back to MotionBlobDetector - "
                        "detections are NOT real object detections.", type(e).__name__, e)
    return MotionBlobDetector()
