"""Deterministic on-disk cache of per-video features (fp16). The key covers everything that changes the
numbers: file content, sampling config, visual encoder, detector, tracker."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np


def cache_key(fingerprint: str, sampling: dict, encoder_tag: str, detector_name: str, tracker: str) -> str:
    blob = json.dumps({"fp": fingerprint, "s": sampling, "enc": encoder_tag, "det": detector_name,
                       "trk": tracker}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


class FeatureCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".npz")

    def has(self, key: str) -> bool:
        return os.path.isfile(self.path(key))

    def get(self, key: str) -> Optional[dict]:
        p = self.path(key)
        if not os.path.isfile(p):
            return None
        try:
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            return None          # corrupt cache entry: recompute

    def put(self, key: str, vis: np.ndarray, obj: np.ndarray, ts: np.ndarray) -> None:
        p = self.path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp.npz"
        np.savez(tmp, vis=vis.astype(np.float16), obj=obj.astype(np.float16), ts=ts.astype(np.float32))
        os.replace(tmp, p)
