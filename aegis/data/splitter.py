"""Video/source-level splitting. Windows/clips are generated AFTER this step, never before."""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .manifest import VideoRecord


def split_videos(records: Sequence[VideoRecord], fractions=(0.7, 0.15, 0.15), seed: int = 1337,
                 official_test: bool = True) -> Dict[str, List[str]]:
    """Returns {"train": [video_id..], "val": [...], "test": [...]}.
    Whole groups (group_id) move together; stratified per canonical label."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    rng = random.Random(seed)
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    groups: Dict[str, List[VideoRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_id].append(r)
    by_label: Dict[str, List[str]] = defaultdict(list)
    forced_test = set()
    for gid, rs in groups.items():
        if official_test and any(r.official_split == "test" for r in rs):
            forced_test.add(gid)
            continue
        by_label[Counter(r.label for r in rs).most_common(1)[0][0]].append(gid)
    for gid in forced_test:
        out["test"] += [r.video_id for r in groups[gid]]
    ftr, fva, fte = fractions
    for label, gids in sorted(by_label.items()):
        gids = sorted(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_val = max(1, round(n * fva)) if n >= 3 and fva > 0 else 0
        n_test = 0 if forced_test else (max(1, round(n * fte)) if n >= 3 and fte > 0 else 0)
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = n, 0, 0
        for split, chunk in (("train", gids[:n_train]), ("val", gids[n_train:n_train + n_val]),
                             ("test", gids[n_train + n_val:])):
            for gid in chunk:
                out[split] += [r.video_id for r in groups[gid]]
    for k in out:
        out[k].sort()
    return out


def save_split_manifest(split: Dict[str, List[str]], records: Sequence[VideoRecord], path: str, seed: int,
                        fractions, extra: dict = None) -> None:
    lab = {r.video_id: r.label for r in records}
    dist = {s: dict(Counter(lab[v] for v in ids)) for s, ids in split.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "fractions": list(fractions), "counts": {s: len(v) for s, v in split.items()},
                   "class_distribution": dist, "splits": split, **(extra or {})}, f, indent=1)
