"""When is Laya worth calling? Thresholds are configuration, NOT calibrated values."""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import LayaConfig


@dataclass
class PolicyContext:
    camera_id: str
    timestamp: float
    anomaly_score: float
    class_probs: np.ndarray
    top_class: str
    top_prob: float
    recent_top: List[str] = field(default_factory=list)
    scene: Dict[str, float] = field(default_factory=dict)


@dataclass
class LayaDecision:
    call: bool
    reasons: List[str]


def normalized_entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(p))) if len(p) > 1 else 0.0


class LayaPolicy:
    def __init__(self, cfg: LayaConfig):
        self.cfg = cfg
        self._last: Dict[str, float] = {}
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)

    def _rate_ok(self, cam: str, ts: float) -> bool:
        if ts - self._last.get(cam, -1e9) < self.cfg.min_interval_sec:
            return False
        q = self._calls[cam]
        while q and ts - q[0] > 60.0:
            q.popleft()
        return len(q) < self.cfg.max_calls_per_minute

    def decide(self, c: PolicyContext) -> LayaDecision:
        cfg = self.cfg
        if not cfg.enabled:
            return LayaDecision(False, ["laya_disabled"])
        if c.anomaly_score < min(cfg.skip_below_anomaly, cfg.trigger_min_anomaly):
            return LayaDecision(False, ["anomaly_below_floor"])

        reasons: List[str] = []
        if c.anomaly_score < cfg.skip_below_anomaly:
            pass                                           # only extra triggers may fire below the skip line
        elif c.anomaly_score >= cfg.confirm_above_anomaly:
            if cfg.high_anomaly_action == "confirm":
                reasons.append("high_anomaly_confirm")
        else:
            reasons.append("anomaly_uncertain_zone")

        if c.anomaly_score >= cfg.trigger_min_anomaly:
            if c.top_prob < cfg.low_confidence:
                reasons.append("low_classifier_confidence")
            if normalized_entropy(c.class_probs) > cfg.entropy_trigger:
                reasons.append("temporal_uncertainty_entropy")
            flips = sum(1 for a, b in zip(c.recent_top, c.recent_top[1:]) if a != b)
            if flips >= cfg.flip_trigger:
                reasons.append("prediction_flipping")
            if cfg.enable_object_triggers:
                s = c.scene
                if s.get("n_weapon", 0) > 0:
                    reasons.append("weapon_like_object")
                if s.get("n_person", 0) >= 2 and s.get("n_close_pairs", 0) >= 1:
                    reasons.append("person_interaction")
                if s.get("person_speed_max", 0) > cfg.motion_trigger:
                    reasons.append("fast_motion")
                if s.get("unusual_objects"):
                    reasons.append("unusual_object")

        if not reasons:
            return LayaDecision(False, ["no_trigger"])
        if not self._rate_ok(c.camera_id, c.timestamp):
            return LayaDecision(False, ["rate_limited"] + reasons)
        return LayaDecision(True, reasons)

    def record_call(self, camera_id: str, ts: float) -> None:
        self._last[camera_id] = ts
        self._calls[camera_id].append(ts)
