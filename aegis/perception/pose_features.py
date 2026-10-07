"""Converts raw YOLO11-pose keypoints -> a 34-float biomechanical feature vector.

Pure NumPy, no Torch.  Called by ``aegis.perception.features.extract()`` and
concatenated to the existing 16-float bounding-box context vector to produce
the 50-float ``obj_vec`` used by AegisModel (Orion and later).

COCO 17-keypoint layout:
  0  nose          1  left_eye      2  right_eye
  3  left_ear      4  right_ear     5  left_shoulder
  6  right_shoulder 7 left_elbow   8  right_elbow
  9  left_wrist    10 right_wrist  11  left_hip
  12 right_hip     13 left_knee   14  right_knee
  15 left_ankle    16 right_ankle
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

# ------------------------------------------------------------------ keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

POSE_FEAT_DIM = 34
POSE_FEAT_NAMES = [
    # ---- whole-scene counts ----
    "pose_n_persons",           # 0   normalised person count
    # ---- wrist dynamics (punching / throwing) ----
    "pose_wrist_vel_max",       # 1   max wrist speed across all persons
    "pose_wrist_vel_mean",      # 2   mean wrist speed
    # ---- arm extension (cocking vs. throwing a punch) ----
    "pose_elbow_ang_mean",      # 3   mean elbow angle  (0=fully bent, 1=fully extended)
    "pose_elbow_ang_min",       # 4   most-extended arm across all persons
    # ---- wrist proximity (grappling / restraint) ----
    "pose_wrist_wrist_dist",    # 5   min bilateral wrist-to-wrist (own two wrists)
    # ---- restraint posture (ARREST) ----
    "pose_wrists_behind_back",  # 6   fraction of persons whose wrists are near hips/behind
    # ---- ground contact (tackled) ----
    "pose_head_ground_norm",    # 7   highest normalised y of nose across persons (1=bottom)
    # ---- crouching (BURGLARY / sneaking) ----
    "pose_crouch_mean",         # 8   mean knee-bend score
    "pose_crouch_max",          # 9   most crouched person
    # ---- bilateral facing (symmetric fight vs. one-sided assault) ----
    "pose_bilateral_sym",       # 10  shoulder-facing symmetry score
    # ---- inter-person keypoint distances (contact / proximity) ----
    # 23 slots: 5 keypoints × up to 4 pair-distances, sorted ascending
    "pose_pair_kp_00",          # 11
    "pose_pair_kp_01",          # 12
    "pose_pair_kp_02",          # 13
    "pose_pair_kp_03",          # 14
    "pose_pair_kp_04",          # 15
    "pose_pair_kp_05",          # 16
    "pose_pair_kp_06",          # 17
    "pose_pair_kp_07",          # 18
    "pose_pair_kp_08",          # 19
    "pose_pair_kp_09",          # 20
    "pose_pair_kp_10",          # 21
    "pose_pair_kp_11",          # 22
    "pose_pair_kp_12",          # 23
    "pose_pair_kp_13",          # 24
    "pose_pair_kp_14",          # 25
    "pose_pair_kp_15",          # 26
    "pose_pair_kp_16",          # 27
    "pose_pair_kp_17",          # 28
    "pose_pair_kp_18",          # 29
    "pose_pair_kp_19",          # 30
    "pose_pair_kp_20",          # 31
    "pose_pair_kp_21",          # 32
    "pose_pair_kp_22",          # 33
]

assert len(POSE_FEAT_NAMES) == POSE_FEAT_DIM


# ------------------------------------------------------------------ helpers
def _dist_norm(p1, p2, diag: float) -> float:
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])) / (diag + 1e-6)


def _angle_deg(a, b, c) -> float:
    """Angle in degrees at vertex b formed by segments b->a and b->c."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    na, nb_ = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nb_ < 1e-6:
        return 90.0          # unknown -> neutral
    cos = float(np.dot(ba, bc) / (na * nb_))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _good(kp: Dict, idx: int, thresh: float = 0.3) -> bool:
    """True if keypoint confidence is above threshold."""
    return float(kp["conf"][idx]) > thresh


def _pt(kp: Dict, idx: int):
    return kp["xy"][idx]   # (x, y) ndarray or tuple


# ------------------------------------------------------------------ main
def extract_pose_features(
    keypoints_list: Optional[List[Optional[Dict]]],
    frame_shape,
    prev_kp_list: Optional[List[Optional[Dict]]] = None,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Args:
        keypoints_list : list of dicts ``{"xy": (17,2), "conf": (17,)}``, one per
                         detected person in the current frame.  ``None`` entries are
                         ignored (no pose for that detection).
        frame_shape    : (H, W[, C]) of the source frame.
        prev_kp_list   : same format for the *previous* frame; used for velocity.
        dt             : time between the two frames in seconds.

    Returns:
        float32 ndarray of shape (POSE_FEAT_DIM,) — all values normalised to [0, 1].
    """
    h, w = frame_shape[:2]
    diag = math.hypot(w, h) or 1.0
    valid = [kp for kp in (keypoints_list or []) if kp is not None]
    n = len(valid)
    v = np.zeros(POSE_FEAT_DIM, np.float32)

    # ---- feature 0: n_persons_with_pose
    v[0] = min(n / 5.0, 1.0)

    if n == 0:
        v[5] = 1.0      # max wrist-wrist dist when no persons = 1 (neutral)
        return v

    # ---- features 1-2: wrist velocities
    wrist_vels: List[float] = []
    if prev_kp_list:
        prev_valid = [kp for kp in prev_kp_list if kp is not None]
        for i in range(min(n, len(prev_valid))):
            cur, prv = valid[i], prev_valid[i]
            for w_idx in (L_WRIST, R_WRIST):
                if _good(cur, w_idx) and _good(prv, w_idx):
                    vel = _dist_norm(_pt(cur, w_idx), _pt(prv, w_idx), diag) / (dt + 1e-6)
                    wrist_vels.append(vel)

    MAX_WRIST_VEL = 5.0     # clips normalisation ceiling
    v[1] = min(max(wrist_vels, default=0.0), MAX_WRIST_VEL) / MAX_WRIST_VEL
    v[2] = min(float(np.mean(wrist_vels)) if wrist_vels else 0.0, MAX_WRIST_VEL) / MAX_WRIST_VEL

    # ---- features 3-4: elbow angles
    elbow_angles: List[float] = []
    for kp in valid:
        for s_idx, e_idx, w_idx in [(L_SHOULDER, L_ELBOW, L_WRIST),
                                     (R_SHOULDER, R_ELBOW, R_WRIST)]:
            if _good(kp, s_idx) and _good(kp, e_idx) and _good(kp, w_idx):
                elbow_angles.append(_angle_deg(_pt(kp, s_idx), _pt(kp, e_idx), _pt(kp, w_idx)))

    if elbow_angles:
        v[3] = float(np.mean(elbow_angles)) / 180.0
        v[4] = max(elbow_angles) / 180.0   # most-extended arm
    else:
        v[3] = 0.5
        v[4] = 0.5

    # ---- feature 5: min bilateral wrist-wrist distance (own L/R wrists)
    ww_dists = [
        _dist_norm(_pt(kp, L_WRIST), _pt(kp, R_WRIST), diag)
        for kp in valid
        if _good(kp, L_WRIST) and _good(kp, R_WRIST)
    ]
    v[5] = min(ww_dists) if ww_dists else 1.0

    # ---- feature 6: wrists-behind-back score (ARREST: handcuffed posture)
    behind_scores: List[float] = []
    for kp in valid:
        # heuristic: both wrists near same y as hips AND centred between hips
        score = 0.0
        if all(_good(kp, i) for i in (L_WRIST, R_WRIST, L_HIP, R_HIP)):
            hip_cx = (float(_pt(kp, L_HIP)[0]) + float(_pt(kp, R_HIP)[0])) / 2.0
            hip_y  = (float(_pt(kp, L_HIP)[1]) + float(_pt(kp, R_HIP)[1])) / 2.0
            wrist_cx = (float(_pt(kp, L_WRIST)[0]) + float(_pt(kp, R_WRIST)[0])) / 2.0
            wrist_y  = (float(_pt(kp, L_WRIST)[1]) + float(_pt(kp, R_WRIST)[1])) / 2.0
            horiz_ok = abs(wrist_cx - hip_cx) / (w + 1e-6) < 0.15
            vert_ok  = abs(wrist_y  - hip_y)  / (h + 1e-6) < 0.20
            score = 1.0 if (horiz_ok and vert_ok) else 0.0
        behind_scores.append(score)
    v[6] = float(np.mean(behind_scores)) if behind_scores else 0.0

    # ---- feature 7: head-ground proximity (high y in frame = on the ground)
    nose_ys = [
        float(_pt(kp, NOSE)[1]) / (h + 1e-6)
        for kp in valid if _good(kp, NOSE)
    ]
    v[7] = max(nose_ys) if nose_ys else 0.5

    # ---- features 8-9: crouch score (knee-angle compression)
    crouch: List[float] = []
    for kp in valid:
        for hip_i, knee_i, ank_i in [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)]:
            if _good(kp, hip_i) and _good(kp, knee_i) and _good(kp, ank_i):
                ang = _angle_deg(_pt(kp, hip_i), _pt(kp, knee_i), _pt(kp, ank_i))
                # 180° = fully straight, 0° = fully bent.  Invert so 1=crouched.
                crouch.append(1.0 - ang / 180.0)
    v[8] = float(np.mean(crouch)) if crouch else 0.0
    v[9] = max(crouch) if crouch else 0.0

    # ---- feature 10: bilateral symmetry (persons facing each other = FIGHT)
    if len(valid) >= 2:
        sym_scores: List[float] = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                kpa, kpb = valid[i], valid[j]
                if all(_good(kpa, k) and _good(kpb, k)
                       for k in (L_SHOULDER, R_SHOULDER)):
                    # close inter-person shoulder distance = facing each other
                    d = _dist_norm(_pt(kpa, L_SHOULDER), _pt(kpb, R_SHOULDER), diag)
                    sym_scores.append(1.0 - min(d, 1.0))
        v[10] = float(np.mean(sym_scores)) if sym_scores else 0.0
    else:
        v[10] = 0.0

    # ---- features 11-33: inter-person pairwise keypoint proximity
    PAIR_KP = (NOSE, L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER)
    pair_dists: List[float] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            kpa, kpb = valid[i], valid[j]
            for kp_idx in PAIR_KP:
                if _good(kpa, kp_idx) and _good(kpb, kp_idx):
                    pair_dists.append(_dist_norm(_pt(kpa, kp_idx), _pt(kpb, kp_idx), diag))

    pair_dists.sort()
    n_slots = POSE_FEAT_DIM - 11   # 23 slots
    for k in range(min(n_slots, len(pair_dists))):
        v[11 + k] = float(pair_dists[k])
    # remaining slots stay 0.0 (no more pairs detected)

    return v
