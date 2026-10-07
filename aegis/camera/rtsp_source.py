from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from ..utils import log
from .camera_source import CameraSource
from .frame import Frame


class RTSPSource(CameraSource):
    """RTSP/HTTP stream. A background thread always drains the socket and keeps only the
    NEWEST frame, so a slow consumer sees fresh frames instead of a growing backlog.
    Reconnects with exponential backoff. NOT verified against a real camera in development:
    use MockSource for CI and test on real hardware before relying on it."""

    is_live = True

    def __init__(self, url: str, camera_id: str = "CAM-001", reconnect_max_sec: float = 10.0):
        super().__init__(camera_id)
        self.url = url
        self._reconnect_max = reconnect_max_sec
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._last_returned = -1
        self._stop = threading.Event()
        self._fps = 25.0
        self._idx = 0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"rtsp-{camera_id}")
        self._thread.start()

    @property
    def fps(self) -> float:
        return self._fps

    def _connect(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 1e-3:
                self._fps = float(f)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        return None

    def _run(self) -> None:
        backoff = 0.5
        cap = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._connect()
                if cap is None:
                    log.warning("%s: RTSP connect failed, retry in %.1fs", self.camera_id, backoff)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self._reconnect_max)
                    continue
                backoff = 0.5
            ok, img = cap.read()
            if not ok:
                cap.release()
                cap = None
                continue
            now = time.monotonic()
            fr = Frame(img, self.camera_id, self._idx, now - self._t0, now)
            self._idx += 1
            with self._lock:
                self._latest = fr
        if cap is not None:
            cap.release()

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                fr = self._latest
            if fr is not None and fr.frame_id != self._last_returned:
                self._last_returned = fr.frame_id
                return fr
            time.sleep(0.002)
        return None

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
