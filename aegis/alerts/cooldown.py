from __future__ import annotations

from typing import Dict, Hashable


class Cooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: Dict[Hashable, float] = {}

    def ready(self, key: Hashable, ts: float) -> bool:
        return ts - self._last.get(key, -1e18) >= self.seconds

    def mark(self, key: Hashable, ts: float) -> None:
        self._last[key] = ts
