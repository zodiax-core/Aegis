"""Turns tracked detections into a fixed-size context vector (no torch).

The vector has two parts:
  * 16-float bounding-box context (person counts, speeds, proximity, etc.)
  * 34-float biomechanical pose context (wrist velocity, crouch, restraint, etc.)
  => OBJ_FEAT_DIM = 50 for Orion and later models.

Vesper (16-dim) checkpoints are NOT compatible with Orion (50-dim).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .object_schema import PERSON, VEHICLE, WEAPON, Detection
from .pose_features import POSE_FEAT_DIM, POSE_FEAT_NAMES, extract_pose_features
from .tracker import TrackManager, _iou

_BBOX_FEAT_NAMES = [
    "n_person", "n_vehicle", "n_weapon", "n_other", "person_area_mean", "person_area_max",
    "person_speed_mean", "person_speed_max", "pp_dist_min", "pp_dist_mean", "pp_iou_max",
    "n_close_pairs", "vehicle_speed_mean", "vehicle_person_iou_max", "conf_mean", "n_tracks",
]
_BBOX_FEAT_DIM = len(_BBOX_FEAT_NAMES)   # 16

OBJ_FEAT_DIM   = _BBOX_FEAT_DIM + POSE_FEAT_DIM   # 16 + 34 = 50
OBJ_FEAT_NAMES = _BBOX_FEAT_NAMES + POSE_FEAT_NAMES

COMMON_CLASSES = PERSON | VEHICLE | WEAPON


def summarize(dets: List[Detection], tm: TrackManager, frame_shape: Tuple[int, int], timestamp: float,
              close_dist: float = 0.15) -> Dict[str, float]:
    """Human-readable scene summary; also consumed by LayaPolicy."""
    h, w = frame_shape[:2]
    diag = math.hypot(w, h) or 1.0
    area = float(w * h) or 1.0
    persons = [d for d in dets if d.class_name in PERSON]
    vehicles = [d for d in dets if d.class_name in VEHICLE]
    weapons = [d for d in dets if d.class_name in WEAPON]
    unusual = sorted({d.class_name for d in dets if d.class_name not in COMMON_CLASSES})

    dists, ious = [], []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            (x1, y1), (x2, y2) = persons[i].center, persons[j].center
            dists.append(math.hypot(x1 - x2, y1 - y2) / diag)
            ious.append(_iou(persons[i].bbox, persons[j].bbox))
    v_p = [_iou(v.bbox, p.bbox) for v in vehicles for p in persons]

    active = tm.active(timestamp)
    ps = [t.speed(diag) for t in active if t.class_name in PERSON]
    vs = [t.speed(diag) for t in active if t.class_name in VEHICLE]
    return {
        "n_person": len(persons), "n_vehicle": len(vehicles), "n_weapon": len(weapons),
        "n_other": len(dets) - len(persons) - len(vehicles) - len(weapons),
        "person_area_mean": float(np.mean([p.area / area for p in persons])) if persons else 0.0,
        "person_area_max": float(max([p.area / area for p in persons])) if persons else 0.0,
        "person_speed_mean": float(np.mean(ps)) if ps else 0.0,
        "person_speed_max": float(max(ps)) if ps else 0.0,
        "pp_dist_min": float(min(dists)) if dists else 1.0,
        "pp_dist_mean": float(np.mean(dists)) if dists else 1.0,
        "pp_iou_max": float(max(ious)) if ious else 0.0,
        "n_close_pairs": sum(1 for d in dists if d < close_dist),
        "vehicle_speed_mean": float(np.mean(vs)) if vs else 0.0,
        "vehicle_person_iou_max": float(max(v_p)) if v_p else 0.0,
        "conf_mean": float(np.mean([d.confidence for d in dets])) if dets else 0.0,
        "n_tracks": len(active),
        "unusual_objects": unusual,
    }


def to_vector(s: Dict[str, float]) -> np.ndarray:
    v = np.zeros(_BBOX_FEAT_DIM, np.float32)
    for i, name in enumerate(_BBOX_FEAT_NAMES):
        x = float(s[name])
        if name.startswith("n_"):
            x = math.log1p(x)                 # counts -> compressed
        elif "speed" in name:
            x = min(x, 3.0) / 3.0             # clip outliers, ~[0,1]
        v[i] = x
    return v


def extract(
    dets,
    tm,
    frame_shape,
    timestamp,
    close_dist: float = 0.15,
    keypoints: Optional[list] = None,
    prev_keypoints: Optional[list] = None,
    dt: float = 0.1,
):
    """Return (obj_vec [50-float], summary dict).

    Args:
        keypoints      : list of {"xy":(17,2), "conf":(17,)} dicts from PoseDetector,
                         one per tracked person (or None per person with no pose).
                         Pass None or empty list when using the standard YoloDetector.
        prev_keypoints : same for the previous frame; used to compute wrist velocity.
        dt             : time between current and previous frame in seconds.
    """
    s = summarize(dets, tm, frame_shape, timestamp, close_dist)
    bbox_vec = to_vector(s)
    pose_vec = extract_pose_features(keypoints or [], frame_shape, prev_keypoints, dt)
    return np.concatenate([bbox_vec, pose_vec]), s
