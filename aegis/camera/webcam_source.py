from __future__ import annotations

import time
from typing import Optional

import cv2

from .camera_source import CameraSource
from .frame import Frame


class WebcamSource(CameraSource):
    is_live = True

    def __init__(self, index: int = 0, camera_id: str = "CAM-001", width: int = 0, height: int = 0):
        super().__init__(camera_id)
        self._cap = cv2.VideoCapture(int(index))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open webcam index {index}")
        if width and height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # avoid stale buffered frames
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 1e-3 else 30.0
        self._t0 = time.monotonic()
        self._idx = 0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        now = time.monotonic()
        f = Frame(img, self.camera_id, self._idx, now - self._t0, now)
        self._idx += 1
        return f

    def release(self) -> None:
        self._cap.release()
