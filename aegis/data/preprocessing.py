"""video -> cached per-frame features (visual embedding + object-context vector), using the SAME
CameraState code path as the live engine. Torch is required only for the visual encoder."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..camera.video_file_source import VideoFileSource
from ..config import AegisConfig
from ..pipeline.state import CameraState
from ..utils import log
from .cache import FeatureCache, cache_key
from .manifest import VideoRecord


def sampling_signature(cfg: AegisConfig) -> dict:
    s = cfg.sampling
    return {"det": s.detection_fps, "tmp": s.temporal_fps, "stride": s.frame_stride, "size": s.image_size,
            "obj_close": cfg.laya.close_pair_distance}


def extract_video_features(path: str, cfg: AegisConfig, detector, visual_encoder, batch: int = 32,
                           camera_id: str = "PREP") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns vis (N,Dv), obj (N,Do), timestamps (N,) for every temporal sample of the video."""
    src = VideoFileSource(path, camera_id)
    state = CameraState(camera_id, cfg, src.fps)
    pend_frames: List[np.ndarray] = []
    vis_chunks: List[np.ndarray] = []
    objs: List[np.ndarray] = []
    ts: List[float] = []

    def flush():
        if pend_frames:
            vis_chunks.append(visual_encoder.encode(pend_frames).float().cpu().numpy())
            pend_frames.clear()

    try:
        for fr in src:
            d = state.decide(fr)
            if d.detect:
                state.apply_detections(fr, detector.detect(fr.image))
            if d.temporal:
                pend_frames.append(fr.image)
                objs.append(state.obj_vec.copy())
                ts.append(fr.timestamp)
                if len(pend_frames) >= batch:
                    flush()
        flush()
    finally:
        src.release()
    if not ts:
        raise ValueError(f"no temporal samples extracted from {path} (empty or too short)")
    return np.concatenate(vis_chunks), np.stack(objs), np.asarray(ts, np.float32)


def build_feature_cache(records: List[VideoRecord], cfg: AegisConfig, detector, visual_encoder,
                        cache: FeatureCache, progress: Optional[Callable] = None) -> dict:
    """Extract + cache features for every record; skips ones already cached. Returns {video_id: key}."""
    keys, failed = {}, []
    sig = sampling_signature(cfg)
    for i, r in enumerate(records):
        k = cache_key(r.fingerprint, sig, visual_encoder.tag, detector.name, cfg.perception.tracker)
        keys[r.video_id] = k
        if cache.has(k):
            continue
        try:
            vis, obj, ts = extract_video_features(r.path, cfg, detector, visual_encoder)
            cache.put(k, vis, obj, ts)
        except Exception as e:                       # corrupt video / codec: skip, never crash the build
            log.warning("feature extraction failed for %s: %s", r.video_id, e)
            failed.append(r.video_id)
        if progress:
            progress(i + 1, len(records))
    for v in failed:
        keys.pop(v, None)
    return keys
