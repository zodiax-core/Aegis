from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    image: np.ndarray        # BGR uint8, HxWx3
    camera_id: str
    frame_id: int
    timestamp: float         # seconds: media time for files, monotonic-since-open for live
    capture_time: float = 0.0  # time.monotonic() at capture, used to measure queue latency
