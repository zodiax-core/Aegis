from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .camera_source import CameraSource
from .frame import Frame


class MockSource(CameraSource):
    """Deterministic synthetic scene (moving coloured blobs on a textured background).
    Used for tests, benchmarks and as a stand-in when no RTSP camera is available.
    `pace=True` sleeps so frames arrive at real fps (simulates a live camera)."""

    def __init__(self, camera_id: str = "CAM-001", n_frames: int = 300, fps: float = 30.0,
                 size=(640, 480), n_blobs: int = 3, seed: int = 0, pace: bool = False):
        super().__init__(camera_id)
        self.n_frames, self._fps, self.size, self.pace = n_frames, float(fps), size, pace
        rng = np.random.RandomState(seed)
        w, h = size
        self._bg = (rng.rand(h, w, 3) * 60 + 60).astype(np.uint8)
        self._blobs = [dict(x=rng.uniform(50, w - 50), y=rng.uniform(80, h - 80),
                            vx=rng.uniform(-90, 90), vy=rng.uniform(-40, 40),
                            color=tuple(int(c) for c in rng.randint(120, 255, 3))) for _ in range(n_blobs)]
        self._idx = 0
        self.is_live = pace
        self._t_start = time.monotonic()

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Optional[Frame]:
        if self._idx >= self.n_frames:
            return None
        if self.pace:
            due = self._t_start + self._idx / self._fps
            time.sleep(max(0.0, due - time.monotonic()))
        t = self._idx / self._fps
        w, h = self.size
        img = self._bg.copy()
        for b in self._blobs:
            cx = int(abs((b["x"] + b["vx"] * t) % (2 * (w - 60)) - (w - 60)) + 30)
            cy = int(abs((b["y"] + b["vy"] * t) % (2 * (h - 100)) - (h - 100)) + 50)
            cv2.rectangle(img, (cx - 20, cy - 45), (cx + 20, cy + 45), b["color"], -1)
        f = Frame(img, self.camera_id, self._idx, t, time.monotonic())
        self._idx += 1
        return f

    def release(self) -> None:
        self._idx = self.n_frames
