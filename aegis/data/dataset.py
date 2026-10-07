"""Training datasets over CACHED features (windows of temporal samples)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..semantic.laya_questions import LAYA_FEAT_DIM
from .cache import FeatureCache
from .manifest import VideoRecord


class FeatureWindowDataset(Dataset):
    """Each item: a window of `clip_length` consecutive temporal samples from ONE video.
    Labels are weak (video-level) unless `annotations` = {video_id: [[start_s, end_s], ...]} is given,
    in which case windows overlapping <50% with an annotated event are labelled NORMAL.

    Used for EVALUATION (it yields every window of every video, which `training.trainer.evaluate`
    aggregates into one prediction per video) and by the smoke-test scripts. For training on weak
    labels prefer `VideoBagDataset` - see the note there on why window-level training underfits."""

    def __init__(self, records: Sequence[VideoRecord], keys: Dict[str, str], cache: FeatureCache,
                 class_names: List[str], clip_length: int, window_stride: int, normal_label: str = "NORMAL",
                 annotations: Optional[Dict[str, list]] = None, laya_cache: Optional[Dict[str, np.ndarray]] = None,
                 laya_dropout: float = 0.0, training: bool = False, seed: int = 0):
        self.cache, self.T, self.class_names = cache, clip_length, class_names
        self.idx = {c: i for i, c in enumerate(class_names)}
        self.normal = normal_label
        self.laya_cache, self.laya_dropout, self.training = laya_cache or {}, laya_dropout, training
        self.rng = np.random.RandomState(seed)
        self.items = []          # (video_id, key, start, label_idx)
        self._mem: Dict[str, dict] = {}
        for r in records:
            if r.video_id not in keys or r.label not in self.idx:
                continue
            z = cache.get(keys[r.video_id])
            if z is None:
                continue
            n = len(z["ts"])
            if n < clip_length:
                continue
            ann = (annotations or {}).get(r.video_id)
            for s in range(0, n - clip_length + 1, max(1, window_stride)):
                lab = r.label
                if ann is not None and r.label != normal_label:
                    t0, t1 = float(z["ts"][s]), float(z["ts"][s + clip_length - 1])
                    ov = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in ann)
                    lab = r.label if ov >= 0.5 * max(t1 - t0, 1e-6) else normal_label
                self.items.append((r.video_id, keys[r.video_id], s, self.idx[lab]))

    def __len__(self):
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.array([i[3] for i in self.items])

    def _load(self, key: str) -> dict:
        z = self._mem.get(key)
        if z is None:
            z = self.cache.get(key)
            if len(self._mem) > 64:
                self._mem.clear()
            self._mem[key] = z
        return z

    def __getitem__(self, i):
        vid, key, s, lab = self.items[i]
        z = self._load(key)
        vis = torch.from_numpy(z["vis"][s:s + self.T].astype(np.float32))
        obj = torch.from_numpy(z["obj"][s:s + self.T].astype(np.float32))
        laya = self.laya_cache.get(f"{vid}#{s}")
        present = laya is not None and not (self.training and self.rng.rand() < self.laya_dropout)
        laya_t = torch.from_numpy(laya.astype(np.float32)) if present else torch.zeros(LAYA_FEAT_DIM)
        return {"vis": vis, "obj": obj, "laya": laya_t, "laya_mask": torch.tensor(bool(present)),
                "event": torch.tensor(lab), "anomaly": torch.tensor(float(self.class_names[lab] != self.normal)),
                "video_id": vid, "start": s}


class VideoBagDataset(Dataset):
    """Each item: ONE video, carrying `bag_size` windows -> vis/obj of shape (bag_size, clip_length, D).

    This is the multiple-instance-learning (MIL) view of the data and it exists because a
    video-level label is wrong for most windows: every one of the ~590 windows of the 949-second
    Abuse042 is labelled ABUSE, so a window-level classifier learns scene identity (low train loss)
    rather than events (low val macro-F1).

    Two problems are fixed by the same change:
      * weak labels - the loss (see `training.trainer.compute_mil_losses`) only insists that the
        most anomalous `mil_k` windows of a positive video match the label, while *every* window of
        a NORMAL video is supervised as normal (negative bags are clean, positive bags are weak).
      * length imbalance - every video contributes exactly `bag_size` windows whether it is 949
        seconds or 10, instead of a ~120:1 ratio. As a side effect the existing class-balanced
        WeightedRandomSampler in `training.trainer.train` starts balancing *videos* by class, which
        is what it should always have been doing.

    `annotations` = {video_id: [[start_s, end_s], ...]} is the drop-in point for the official
    UCF-Crime `Temporal_Anomaly_Annotation.txt`. When supplied, a positive video's candidate windows
    are restricted to those overlapping an annotated event by >=50%, which makes positive bags
    strongly labelled; nothing else in the pipeline changes.
    """

    is_bag = True

    def __init__(self, records: Sequence[VideoRecord], keys: Dict[str, str], cache: FeatureCache,
                 class_names: List[str], clip_length: int, window_stride: int, normal_label: str = "NORMAL",
                 bag_size: int = 12, annotations: Optional[Dict[str, list]] = None,
                 laya_cache: Optional[Dict[str, np.ndarray]] = None, laya_dropout: float = 0.0,
                 training: bool = False, seed: int = 0):
        self.cache, self.T, self.class_names = cache, clip_length, class_names
        self.idx = {c: i for i, c in enumerate(class_names)}
        self.normal, self.bag_size = normal_label, max(1, bag_size)
        self.laya_cache, self.laya_dropout, self.training = laya_cache or {}, laya_dropout, training
        self.rng = np.random.RandomState(seed)
        self.videos = []         # (video_id, key, starts, label_idx)
        self._mem: Dict[str, dict] = {}
        for r in records:
            if r.video_id not in keys or r.label not in self.idx:
                continue
            z = cache.get(keys[r.video_id])
            if z is None:
                continue
            n = len(z["ts"])
            if n < clip_length:
                continue
            starts = list(range(0, n - clip_length + 1, max(1, window_stride)))
            ann = (annotations or {}).get(r.video_id)
            if ann and r.label != normal_label:
                keep = []
                for s in starts:
                    t0, t1 = float(z["ts"][s]), float(z["ts"][s + clip_length - 1])
                    ov = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in ann)
                    if ov >= 0.5 * max(t1 - t0, 1e-6):
                        keep.append(s)
                starts = keep or starts          # never let a bad annotation empty a bag
            if starts:
                self.videos.append((r.video_id, keys[r.video_id], starts, self.idx[r.label]))

    def __len__(self):
        return len(self.videos)

    def labels(self) -> np.ndarray:
        """One label per VIDEO - this is what makes the class-balanced sampler correct."""
        return np.array([v[3] for v in self.videos])

    def _load(self, key: str) -> dict:
        z = self._mem.get(key)
        if z is None:
            z = self.cache.get(key)
            if len(self._mem) > 16:
                self._mem.clear()
            self._mem[key] = z
        return z

    def _pick(self, starts: List[int]) -> np.ndarray:
        """Random windows while training (free augmentation), evenly spaced when not (deterministic
        eval). Always returned in temporal order so the smoothness regulariser is meaningful."""
        k, n = self.bag_size, len(starts)
        if self.training:
            sel = self.rng.choice(n, size=k, replace=n < k)
        else:
            sel = np.unique(np.linspace(0, n - 1, num=min(k, n)).round().astype(int))
            if len(sel) < k:
                sel = np.resize(sel, k)
        return np.asarray(starts)[np.sort(sel)]

    def __getitem__(self, i):
        vid, key, starts, lab = self.videos[i]
        z = self._load(key)
        picked = self._pick(starts)
        vis = torch.from_numpy(np.stack([z["vis"][s:s + self.T] for s in picked]).astype(np.float32))
        obj = torch.from_numpy(np.stack([z["obj"][s:s + self.T] for s in picked]).astype(np.float32))
        # One Laya vector per bag (stage-1 masks Laya out entirely; the fusion learns to ignore it).
        laya = self.laya_cache.get(f"{vid}#{int(picked[0])}")
        present = laya is not None and not (self.training and self.rng.rand() < self.laya_dropout)
        laya_t = torch.from_numpy(laya.astype(np.float32)) if present else torch.zeros(LAYA_FEAT_DIM)
        return {"vis": vis, "obj": obj, "laya": laya_t, "laya_mask": torch.tensor(bool(present)),
                "event": torch.tensor(lab), "anomaly": torch.tensor(float(self.class_names[lab] != self.normal)),
                "video_id": vid, "start": int(picked[0]), "starts": torch.from_numpy(picked.astype(np.int64))}


class SyntheticFeatureDataset(Dataset):
    """Random features with a learnable signal. For dry-runs/tests ONLY - never for reporting metrics."""

    def __init__(self, n: int, T: int, vis_dim: int, obj_dim: int, num_classes: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        self.vis = torch.randn(n, T, vis_dim, generator=g) + self.y[:, None, None].float() * 0.5
        self.obj = torch.randn(n, T, obj_dim, generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"vis": self.vis[i], "obj": self.obj[i], "laya": torch.zeros(LAYA_FEAT_DIM),
                "laya_mask": torch.tensor(False), "event": self.y[i], "anomaly": torch.tensor(float(self.y[i] != 0)),
                "video_id": f"synthetic/{i}", "start": 0}
