from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle"}
WEAPON = {"knife", "gun", "pistol", "rifle", "firearm", "weapon"}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "class_name": self.class_name,
                "confidence": round(float(self.confidence), 4),
                "bbox": [round(float(v), 1) for v in self.bbox], "track_id": self.track_id}


@dataclass
class Track:
    track_id: int
    class_name: str
    history: Deque = field(default_factory=lambda: deque(maxlen=64))  # (t, cx, cy)
    age: int = 0

    def speed(self, diag: float) -> float:
        """Normalised speed (fraction of frame diagonal per second) over the last two samples."""
        if len(self.history) < 2 or diag <= 0:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = self.history[-2], self.history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / diag / dt)

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "class_name": self.class_name, "age": self.age,
                "last_center": [round(v, 1) for v in self.history[-1][1:]] if self.history else None}
