from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from .object_schema import Detection


class Detector(ABC):
    """Detection only (batchable). Association across frames is the Tracker's job."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(f) for f in frames_bgr]
