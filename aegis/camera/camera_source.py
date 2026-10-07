from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .frame import Frame


class CameraSource(ABC):
    """Every source (file, webcam, RTSP, mock) exposes exactly this interface."""

    is_live: bool = False

    def __init__(self, camera_id: str = "CAM-001"):
        self.camera_id = camera_id

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Next frame, or None if the stream ended / no frame is available right now."""

    @abstractmethod
    def release(self) -> None: ...

    def __iter__(self) -> Iterator[Frame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
