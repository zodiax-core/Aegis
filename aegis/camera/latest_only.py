from __future__ import annotations

import threading
import time
from typing import Optional

from .camera_source import CameraSource
from .frame import Frame


class LatestOnlySource(CameraSource):
    """Wraps any live source: a thread keeps reading so the driver buffer never backs up, and read()
    returns only the newest frame. Use for webcams when inference is slower than capture."""

    is_live = True

    def __init__(self, inner: CameraSource):
        super().__init__(inner.camera_id)
        self.inner = inner
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._seen = -1
        self._done = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @property
    def fps(self) -> float:
        return self.inner.fps

    def _run(self):
        while not self._stop.is_set():
            f = self.inner.read()
            if f is None:
                self._done = True
                return
            with self._lock:
                self._latest = f

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                f = self._latest
            if f is not None and f.frame_id != self._seen:
                self._seen = f.frame_id
                return f
            if self._done:
                return None
            time.sleep(0.002)
        return None

    def release(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)
        self.inner.release()
