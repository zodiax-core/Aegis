"""Small shared helpers: device resolution, thread-safe latency profiler, logging."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

import numpy as np

log = logging.getLogger("aegis")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def resolve_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if pref == "cuda" and not has_cuda:
        log.warning("device=cuda requested but CUDA is unavailable; falling back to CPU")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


class Profiler:
    """Per-stage latency recorder (ms). Thread-safe; cheap enough to leave on."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._t: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            self._t[name].append(ms)

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out = {}
            for k, v in self._t.items():
                a = np.asarray(v, dtype=np.float64)
                out[k] = {"n": int(a.size), "mean_ms": float(a.mean()),
                          "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                          "total_ms": float(a.sum())}
            return out
