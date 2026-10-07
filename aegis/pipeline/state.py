"""Per-camera state. The SAME class is used by the live engine and by offline feature extraction,
so training features and serving features cannot drift apart."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..camera.frame import Frame
from ..config import AegisConfig
from ..perception import features as F
from ..perception.object_schema import Detection
from ..perception.tracker import TrackManager, make_tracker
from .sampler import BufferEntry, FrameSampler, SamplingDecision, TemporalBuffer


class CameraState:
    def __init__(self, camera_id: str, cfg: AegisConfig, source_fps: float):
        self.camera_id = camera_id
        self.cfg = cfg
        self.sampler = FrameSampler(cfg.sampling, source_fps)
        self.tracker = make_tracker(cfg.perception, fps=cfg.sampling.detection_fps)
        self.tracks = TrackManager()
        self.buffer = TemporalBuffer(cfg.buffer_size, cfg.sampling.clip_length)
        self.obj_vec = np.zeros(F.OBJ_FEAT_DIM, np.float32)
        self.summary = {n: 0.0 for n in F.OBJ_FEAT_NAMES}
        self.summary["unusual_objects"] = []
        self.dets: List[Detection] = []
        self.last_event: Optional[dict] = None
        self.recent_top: List[str] = []
        self.n_frames = 0
        # Pose state — only populated when a PoseDetector is in use
        self._cur_keypoints: Optional[list] = None
        self._prev_keypoints: Optional[list] = None
        self._last_ts: float = 0.0

    def decide(self, frame: Frame) -> SamplingDecision:
        self.n_frames += 1
        return self.sampler.decide(frame.frame_id, frame.timestamp)

    def apply_detections(self, frame: Frame, dets: List[Detection],
                         detector=None) -> None:
        """Update tracker, obj_vec, and (when detector supports it) pose features.

        Args:
            frame    : current Frame.
            dets     : detections for this frame — pass pre-computed dets when the
                       caller already ran detection (e.g. the engine's async loop).
                       If ``detector`` is a PoseDetector we re-run detection here
                       to get keypoints; otherwise we skip the extra inference.
            detector : optional detector instance.  When it has ``detect_with_pose``
                       we use that to obtain keypoints alongside dets.
        """
        # --- pose branch (PoseDetector available) ---
        keypoints: Optional[list] = None
        if detector is not None and hasattr(detector, "detect_with_pose"):
            new_dets, keypoints = detector.detect_with_pose(frame.image)
            dets = new_dets   # use pose-detector's own dets (consistent bboxes)

        tracked = self.tracker.update(dets, frame.image.shape)
        self.tracks.update(tracked, frame.timestamp)
        self.dets = tracked

        # Compute dt for wrist velocity
        dt = max(float(frame.timestamp - self._last_ts), 1e-3) if self._last_ts else 0.1
        self._last_ts = float(frame.timestamp)

        self.obj_vec, self.summary = F.extract(
            tracked, self.tracks, frame.image.shape, frame.timestamp,
            self.cfg.laya.close_pair_distance,
            keypoints=keypoints,
            prev_keypoints=self._prev_keypoints,
            dt=dt,
        )

        # Rotate keypoints buffer
        self._prev_keypoints = self._cur_keypoints
        self._cur_keypoints = keypoints

    def thumb(self, frame: Frame) -> np.ndarray:
        h, w = frame.image.shape[:2]
        s = self.cfg.sampling.thumb_max_side / max(h, w)
        img = frame.image if s >= 1 else cv2.resize(frame.image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def push_temporal(self, frame: Frame, vis_emb: np.ndarray, obj_vec: Optional[np.ndarray] = None) -> BufferEntry:
        """obj_vec lets async pipelines pass a snapshot taken by the detection thread."""
        ov = self.obj_vec if obj_vec is None else obj_vec
        e = BufferEntry(frame.timestamp, frame.frame_id, vis_emb.astype(np.float32), ov.copy(), self.thumb(frame))
        self.buffer.push(e)
        return e
