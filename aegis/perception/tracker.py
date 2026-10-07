"""Tracking. Default = Ultralytics' BYTETracker (reused, not reimplemented). A tiny IoU
tracker exists only as a fallback for when ultralytics is not installed / for tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from ..config import PerceptionConfig
from ..utils import log
from .object_schema import Detection, Track


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker(ABC):
    @abstractmethod
    def update(self, dets: List[Detection], frame_shape) -> List[Detection]:
        """Return detections with track_id assigned (untracked detections are dropped)."""


class IoUTracker(Tracker):
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh, self.max_age = iou_thresh, max_age
        self._next = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, dets: List[Detection], frame_shape=None) -> List[Detection]:
        unmatched = list(range(len(dets)))
        pairs = []
        for tid, t in self._tracks.items():
            for di in unmatched:
                if dets[di].class_name == t["cls"]:
                    v = _iou(t["bbox"], dets[di].bbox)
                    if v >= self.iou_thresh:
                        pairs.append((v, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d, out = set(), set(), []
        for v, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid); used_d.add(di)
            self._tracks[tid].update(bbox=dets[di].bbox, miss=0)
            dets[di].track_id = tid
            out.append(dets[di])
        for di in range(len(dets)):
            if di in used_d:
                continue
            tid = self._next; self._next += 1
            self._tracks[tid] = dict(bbox=dets[di].bbox, cls=dets[di].class_name, miss=0)
            dets[di].track_id = tid
            out.append(dets[di])
        for tid in list(self._tracks):
            if tid not in used_t and self._tracks[tid]["miss"] >= 0 and tid not in [d.track_id for d in out]:
                self._tracks[tid]["miss"] += 1
                if self._tracks[tid]["miss"] > self.max_age:
                    del self._tracks[tid]
        return out


class _ByteInput:
    """Minimal object exposing the attributes BYTETracker.update() reads."""

    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _ByteInput(self.xywh[idx], self.conf[idx], self.cls[idx])


class ByteTrackAdapter(Tracker):
    def __init__(self, fps: float = 30.0, track_buffer: int = 30):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                               track_buffer=track_buffer, match_thresh=0.8, fuse_score=True,
                               frame_rate=max(1, int(round(fps))))
        try:
            self._t = BYTETracker(args, frame_rate=max(1, int(round(fps))))
        except TypeError:
            self._t = BYTETracker(args)

    def update(self, dets: List[Detection], frame_shape=None) -> List[Detection]:
        if not dets:
            self._t.update(_ByteInput(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                      np.zeros(0, np.float32)))
            return []
        xyxy = np.array([d.bbox for d in dets], np.float32)
        xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        res = self._t.update(_ByteInput(xywh, np.array([d.confidence for d in dets], np.float32),
                                        np.array([d.class_id for d in dets], np.float32)))
        out = []
        for row in res:  # x1,y1,x2,y2,track_id,score,cls,idx
            src = dets[int(row[-1])]
            src.track_id = int(row[4])
            out.append(src)
        return out


def make_tracker(cfg: PerceptionConfig, fps: float = 30.0) -> Tracker:
    if cfg.tracker.lower() == "bytetrack":
        try:
            return ByteTrackAdapter(fps, cfg.track_buffer)
        except Exception as e:
            log.warning("BYTETracker unavailable (%s); using IoUTracker fallback", e)
    return IoUTracker(max_age=cfg.track_buffer)


class TrackManager:
    """Keeps per-object trajectories from tracked detections (for motion / interaction features)."""

    def __init__(self, max_missing_sec: float = 3.0):
        self.tracks: Dict[int, Track] = {}
        self._last_seen: Dict[int, float] = {}
        self.max_missing = max_missing_sec

    def update(self, dets: List[Detection], timestamp: float) -> None:
        for d in dets:
            if d.track_id is None:
                continue
            tr = self.tracks.get(d.track_id)
            if tr is None:
                tr = self.tracks[d.track_id] = Track(d.track_id, d.class_name)
            tr.age += 1
            cx, cy = d.center
            tr.history.append((timestamp, cx, cy))
            self._last_seen[d.track_id] = timestamp
        for tid in [t for t, s in self._last_seen.items() if timestamp - s > self.max_missing]:
            self.tracks.pop(tid, None)
            self._last_seen.pop(tid, None)

    def active(self, timestamp: float, window: float = 1.0) -> List[Track]:
        return [t for tid, t in self.tracks.items() if timestamp - self._last_seen.get(tid, -1e9) <= window]
