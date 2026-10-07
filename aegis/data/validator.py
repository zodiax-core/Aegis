from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import os

from .manifest import VideoRecord


class LeakageError(RuntimeError):
    pass


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_leakage(records: Sequence[VideoRecord], split: Dict[str, List[str]], strict: bool = True) -> Report:
    """Detects: a video/group/path/content-duplicate appearing in more than one split."""
    rep = Report()
    by_id = {r.video_id: r for r in records}
    where: Dict[str, str] = {}
    for s, ids in split.items():
        for v in ids:
            if v not in by_id:
                rep.errors.append(f"split '{s}' references unknown video {v}")
            elif v in where and where[v] != s:
                rep.errors.append(f"video {v} in both '{where[v]}' and '{s}'")
            where[v] = s
    for attr, what in (("group_id", "source group"), ("fingerprint", "identical file content"), ("path", "path")):
        seen: Dict[str, str] = {}
        for v, s in where.items():
            r = by_id.get(v)
            if r is None:
                continue
            key = getattr(r, attr)
            if not key:
                continue
            if key in seen and seen[key] != s:
                rep.errors.append(f"leakage: {what} shared across splits '{seen[key]}' and '{s}' ({v})")
            seen.setdefault(key, s)
    missing = [r.video_id for r in records if r.video_id not in where]
    if missing:
        rep.warnings.append(f"{len(missing)} videos are in no split")
    if strict and rep.errors:
        raise LeakageError("; ".join(rep.errors[:5]))
    return rep


def validate_manifest(records: Sequence[VideoRecord], min_seconds: float = 1.0) -> Report:
    rep = Report()
    for r in records:
        if not os.path.isfile(r.path):
            rep.errors.append(f"missing file: {r.path}")
        elif r.frames <= 0 or r.fps <= 0:
            rep.errors.append(f"unreadable/empty video: {r.video_id}")
        elif r.duration < min_seconds:
            rep.warnings.append(f"too short ({r.duration:.2f}s): {r.video_id}")
    return rep
