from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..config import EventConfig


@dataclass
class EventPolicy:
    """Decides which model outputs are alert-worthy. Per-event overrides are optional."""
    threshold: float = 0.70
    min_consecutive: int = 3
    per_event_threshold: Dict[str, float] = field(default_factory=dict)
    ignore: tuple = ("NORMAL",)

    @classmethod
    def from_config(cls, c: EventConfig) -> "EventPolicy":
        return cls(threshold=c.alert_threshold, min_consecutive=c.min_consecutive)

    def is_positive(self, event: dict) -> bool:
        """Known event over its threshold, or an 'unknown' anomaly over the threshold."""
        status, name = event["status"], event["event"]
        if status == "normal" or (status == "known" and name in self.ignore):
            return False
        score = event["anomaly_score"] if status == "unknown" else event["probability"]
        return score >= self.per_event_threshold.get(name, self.threshold)

    @staticmethod
    def severity_level(x: float) -> str:
        return "critical" if x >= 0.85 else "high" if x >= 0.65 else "medium" if x >= 0.4 else "low"
