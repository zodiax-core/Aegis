"""Known / normal / unknown assessment. This is a heuristic uncertainty gate, NOT validated open-set
recognition; evaluate it on held-out unseen event types before trusting it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import EventConfig
from ..semantic.laya_policy import normalized_entropy


@dataclass
class Assessment:
    status: str            # "normal" | "known" | "unknown"
    reasons: List[str]


class UnknownEventDetector:
    def __init__(self, cfg: EventConfig, normal_index: int = 0):
        self.cfg, self.normal_index = cfg, normal_index

    def assess(self, probs: np.ndarray, anomaly: float, laya_severity: Optional[float] = None) -> Assessment:
        c = self.cfg
        top = int(np.argmax(probs))
        top_p = float(probs[top])
        ent = normalized_entropy(probs)
        anomalous = anomaly >= c.anomaly_threshold
        if top == self.normal_index and not anomalous:
            return Assessment("normal", ["classifier_normal", "anomaly_low"])
        if top != self.normal_index and anomalous and top_p >= c.known_min_prob and ent < c.unknown_entropy:
            return Assessment("known", ["confident_known_class", "anomaly_high"])
        reasons = []
        if anomalous and top == self.normal_index:
            reasons.append("anomaly_high_but_classifier_says_normal")
        if top != self.normal_index and top_p < c.known_min_prob:
            reasons.append("low_class_confidence")
        if ent >= c.unknown_entropy:
            reasons.append("high_entropy")
        if not anomalous and top != self.normal_index:
            reasons.append("class_without_anomaly_support")
        if laya_severity is not None and laya_severity >= 0.66:
            reasons.append("laya_severity_high")
        if not reasons:
            return Assessment("normal", ["default_normal"])
        # only escalate to "unknown" when there is real anomaly evidence
        if anomalous or (laya_severity or 0) >= 0.66:
            return Assessment("unknown", reasons)
        return Assessment("normal", reasons)
