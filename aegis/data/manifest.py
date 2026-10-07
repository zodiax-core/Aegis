"""Dataset manifests: what videos exist, their source + canonical labels, licence and a content hash."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2

from ..utils import log
from .label_mapper import LabelMapper, UnmappedLabelError

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm")
OFFICIAL_SPLIT_DIRS = {"train": "train", "training": "train", "val": "test", "validation": "test",
                       "test": "test", "testing": "test"}


@dataclass
class VideoRecord:
    video_id: str            # "<dataset>/<relative posix path>"
    dataset: str
    path: str
    source_label: str
    label: str               # canonical AEGIS label
    group_id: str            # source identity used for splitting (default = video_id)
    official_split: Optional[str] = None
    fingerprint: str = ""    # content fingerprint for cross-dataset duplicate detection
    fps: float = 0.0
    frames: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0


@dataclass
class DatasetManifest:
    dataset: str
    version: str
    source: str
    license: str
    classes: List[str]
    num_videos: int
    hash: str
    videos: List[VideoRecord] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    resolutions: Dict[str, int] = field(default_factory=dict)

    def to_dict(self, with_videos: bool = True) -> dict:
        d = asdict(self)
        if not with_videos:
            d.pop("videos")
        return d


def fingerprint(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return dict(fps=0.0, frames=0, width=0, height=0, duration=0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return dict(fps=fps, frames=n, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                    duration=(n / fps) if fps > 0 else 0.0)
    finally:
        cap.release()


def scan_dataset(name: str, root: str, mapper: LabelMapper, license: str = "unknown - check source terms",
                 version: str = "1", source: str = "", probe: bool = True) -> DatasetManifest:
    """Walk `root`; the label of a video is the nearest path component that has an explicit mapping."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    known = {k.lower() for k in mapper.source_labels(name)}
    if not mapper.has_dataset(name):
        raise UnmappedLabelError(f"Dataset '{name}' missing from label_mapping.yaml (add it explicitly)")
    records: List[VideoRecord] = []
    skipped = Counter()
    for dp, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(VIDEO_EXT):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            parts = rel.split("/")[:-1]
            src_label = next((p for p in reversed(parts) if p.lower() in known), None)
            if src_label is None:
                if mapper.on_unmapped == "skip":
                    skipped["<no mapped folder>"] += 1
                    continue
                raise UnmappedLabelError(f"{name}: no mapped label folder in path '{rel}' "
                                         f"(known: {mapper.source_labels(name)})")
            canon = mapper.map(name, src_label)
            split = next((OFFICIAL_SPLIT_DIRS[p.lower()] for p in parts if p.lower() in OFFICIAL_SPLIT_DIRS), None)
            vid = f"{name}/{rel}"
            rec = VideoRecord(vid, name, full, src_label, canon, vid, split, fingerprint(full))
            if probe:
                for k, v in probe_video(full).items():
                    setattr(rec, k, v)
                if rec.frames <= 0 or rec.fps <= 0:
                    skipped["<corrupt/unreadable video>"] += 1
                    log.warning("%s: skipping corrupt/unreadable video: %s", name, rel)
                    continue
            records.append(rec)
    records.sort(key=lambda r: r.video_id)
    if skipped:
        log.warning("%s: skipped %s", name, dict(skipped))
    dist = Counter(r.label for r in records)
    res = Counter(f"{r.width}x{r.height}" for r in records)
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.video_id}|{os.path.getsize(r.path)}|{r.label}\n".encode())
    return DatasetManifest(name, version, source, license, sorted(dist), len(records), h.hexdigest(),
                           records, dict(dist), dict(res))


def merge_manifests(manifests: List[DatasetManifest], dedup_fingerprint: bool = True) -> List[VideoRecord]:
    ids = Counter(r.video_id for m in manifests for r in m.videos)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise ValueError(f"Duplicate video ids across datasets: {dup[:3]}")
    all_records = [r for m in manifests for r in m.videos]
    if not dedup_fingerprint:
        return all_records
    seen_fp: Dict[str, str] = {}
    unique_records: List[VideoRecord] = []
    dup_count = 0
    for r in all_records:
        if r.fingerprint and r.fingerprint in seen_fp:
            dup_count += 1
            log.warning("Dropping duplicate video content: %s (identical content to %s)", r.video_id, seen_fp[r.fingerprint])
            continue
        if r.fingerprint:
            seen_fp[r.fingerprint] = r.video_id
        unique_records.append(r)
    if dup_count > 0:
        log.info("Deduplicated dataset: dropped %d duplicate video files by content fingerprint", dup_count)
    return unique_records


def save_manifests(manifests: List[DatasetManifest], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w") as f:
        json.dump([m.to_dict() for m in manifests], f, indent=1)
    with open(os.path.join(out_dir, "dataset_hash.json"), "w") as f:
        json.dump({m.dataset: {"hash": m.hash, "version": m.version, "num_videos": m.num_videos}
                   for m in manifests}, f, indent=1)


def load_records(path: str) -> List[VideoRecord]:
    with open(path) as f:
        data = json.load(f)
    return [VideoRecord(**v) for m in data for v in m["videos"]]
