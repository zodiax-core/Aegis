"""One continuous event -> ONE incident (OPEN, a few throttled UPDATEs, CLOSE), not a stream of alerts."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..config import EventConfig
from ..utils import log
from .cooldown import Cooldown
from .event_policy import EventPolicy


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    event: str
    start_time: float
    last_update: float
    peak_confidence: float
    peak_anomaly: float
    n_updates: int = 0
    state: str = "OPEN"
    end_time: Optional[float] = None
    last_emit: float = 0.0
    severity: str = "low"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    kind: str                # OPEN | UPDATE | CLOSE
    incident: Incident

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.incident.to_dict()}


class AlertManager:
    def __init__(self, cfg: EventConfig, policy: Optional[EventPolicy] = None):
        self.cfg = cfg
        self.policy = policy or EventPolicy.from_config(cfg)
        self.cooldown = Cooldown(cfg.cooldown_sec)
        self._lock = threading.Lock()
        self._streak: Dict[Tuple[str, str], int] = {}
        self._open: Dict[Tuple[str, str], Incident] = {}
        self._recent_closed: Dict[Tuple[str, str], Incident] = {}
        self._sinks: List[Callable[[Alert], None]] = []
        self.history: List[Alert] = []
        self._n = 0

    def add_sink(self, fn: Callable[[Alert], None]) -> None:
        self._sinks.append(fn)

    def jsonl_sink(self, path: str) -> None:
        def _w(a: Alert):
            with open(path, "a") as f:
                f.write(json.dumps(a.to_dict()) + "\n")
        self.add_sink(_w)

    def _emit(self, kind: str, inc: Incident, out: List[Alert]) -> None:
        a = Alert(kind, inc)
        out.append(a)
        self.history.append(a)
        for s in self._sinks:
            try:
                s(a)
            except Exception as e:  # a broken sink must never kill the pipeline
                log.warning("alert sink failed: %s", e)

    def update(self, camera_id: str, event: dict, ts: float) -> List[Alert]:
        out: List[Alert] = []
        with self._lock:
            key = (camera_id, event["event"])
            positive = self.policy.is_positive(event)
            score = event["probability"] if event["status"] != "unknown" else event["anomaly_score"]
            sev = self.policy.severity_level(event.get("severity") or score)
            if positive:
                self._streak[key] = self._streak.get(key, 0) + 1
                inc = self._open.get(key)
                if inc is not None:
                    inc.last_update = ts
                    inc.peak_confidence = max(inc.peak_confidence, score)
                    inc.peak_anomaly = max(inc.peak_anomaly, event["anomaly_score"])
                    inc.severity = max(inc.severity, sev, key=["low", "medium", "high", "critical"].index)
                    if ts - inc.last_emit >= self.cfg.update_interval_sec:
                        inc.n_updates += 1
                        inc.last_emit = ts
                        self._emit("UPDATE", inc, out)
                elif self._streak[key] >= self.policy.min_consecutive:
                    prev = self._recent_closed.get(key)
                    if prev is not None and not self.cooldown.ready(key, ts):
                        prev.state, prev.end_time, prev.last_update = "OPEN", None, ts   # same event: merge
                        self._open[key] = prev
                        del self._recent_closed[key]
                        prev.n_updates += 1
                        prev.last_emit = ts
                        self._emit("UPDATE", prev, out)
                    else:
                        self._n += 1
                        inc = Incident(f"{camera_id}-INC-{self._n:04d}", camera_id, event["event"], ts, ts,
                                       score, event["anomaly_score"], last_emit=ts, severity=sev)
                        self._open[key] = inc
                        self._emit("OPEN", inc, out)
            else:
                self._streak[key] = 0
            out += self._close_stale(camera_id, ts)
        return out

    def _close_stale(self, camera_id: str, ts: float) -> List[Alert]:
        out: List[Alert] = []
        for key, inc in list(self._open.items()):
            if key[0] == camera_id and ts - inc.last_update >= self.cfg.close_after_sec:
                inc.state, inc.end_time = "CLOSED", inc.last_update
                del self._open[key]
                self._recent_closed[key] = inc
                self.cooldown.mark(key, ts)
                self._emit("CLOSE", inc, out)
        return out

    def flush(self, ts: float) -> List[Alert]:
        """Close everything (end of stream / video)."""
        out: List[Alert] = []
        with self._lock:
            for key, inc in list(self._open.items()):
                inc.state, inc.end_time = "CLOSED", ts
                del self._open[key]
                self._emit("CLOSE", inc, out)
        return out

    @property
    def open_incidents(self) -> List[Incident]:
        return list(self._open.values())
