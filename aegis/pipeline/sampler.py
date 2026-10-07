"""Rate control (capture vs detection vs temporal analysis) and the rolling temporal buffer."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..config import SamplingConfig


class _RateGate:
    def __init__(self, hz: float, tol: float):
        self.period = 1.0 / hz if hz > 0 else float("inf")
        self.tol = tol
        self.next: Optional[float] = None

    def fire(self, ts: float) -> bool:
        if self.next is None or ts + self.tol >= self.next:
            base = ts if self.next is None else self.next
            self.next = base + self.period
            if self.next <= ts:            # fell behind: don't burst-catch-up
                self.next = ts + self.period
            return True
        return False


@dataclass
class SamplingDecision:
    detect: bool
    temporal: bool


class FrameSampler:
    """Decides per frame whether YOLO and/or the temporal path should run.
    Uses frame timestamps (media time for files), so results are identical for a file and its live replay."""

    def __init__(self, cfg: SamplingConfig, source_fps: Optional[float] = None):
        fps = float(source_fps or cfg.capture_fps)
        tol = 0.5 / fps
        self.stride = cfg.frame_stride
        self._det = _RateGate(cfg.detection_fps, tol)
        self._tmp = _RateGate(cfg.temporal_fps, tol)
        self.source_fps = fps

    def decide(self, frame_id: int, timestamp: float) -> SamplingDecision:
        temporal = (frame_id % self.stride == 0) if self.stride > 0 else self._tmp.fire(timestamp)
        detect = self._det.fire(timestamp)
        # the temporal path always wants fresh object context on the frame it samples
        return SamplingDecision(detect=detect or temporal, temporal=temporal)


@dataclass
class BufferEntry:
    timestamp: float
    frame_id: int
    vis: np.ndarray        # (D_vis,) float32
    obj: np.ndarray        # (D_obj,) float32
    thumb: np.ndarray      # small RGB uint8 image kept for semantic verification


class TemporalBuffer:
    def __init__(self, buffer_size: int, clip_length: int):
        self.clip_length = clip_length
        self._d: Deque[BufferEntry] = deque(maxlen=max(buffer_size, clip_length))

    def push(self, e: BufferEntry) -> None:
        self._d.append(e)

    def __len__(self) -> int:
        return len(self._d)

    @property
    def ready(self) -> bool:
        return len(self._d) >= self.clip_length

    def window(self) -> Tuple[np.ndarray, np.ndarray]:
        items = list(self._d)[-self.clip_length:]
        return np.stack([i.vis for i in items]), np.stack([i.obj for i in items])

    def span(self) -> Tuple[float, float]:
        items = list(self._d)[-self.clip_length:]
        return items[0].timestamp, items[-1].timestamp

    def keyframes(self, k: int) -> List[np.ndarray]:
        items = list(self._d)[-self.clip_length:]
        if k >= len(items):
            return [i.thumb for i in items]
        idx = np.linspace(len(items) - 1, 0, k).round().astype(int)   # newest first
        return [items[i].thumb for i in sorted(set(idx), reverse=True)]

    def clear(self) -> None:
        self._d.clear()
