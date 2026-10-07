from __future__ import annotations

import os
import time
from typing import Optional

import cv2

from .camera_source import CameraSource
from .frame import Frame


class VideoFileSource(CameraSource):
    is_live = False

    def __init__(self, path: str, camera_id: str = "CAM-001"):
        super().__init__(camera_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video (unsupported codec or corrupt file): {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 1e-3 else 25.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._idx = 0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        f = Frame(img, self.camera_id, self._idx, self._idx / self._fps, time.monotonic())
        self._idx += 1
        return f

    def release(self) -> None:
        self._cap.release()
