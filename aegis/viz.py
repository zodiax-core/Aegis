"""Overlay drawing shared by aegis.live and aegis.infer (OpenCV only)."""
from __future__ import annotations

import cv2
import numpy as np

_COL = {"normal": (80, 200, 80), "known": (40, 40, 255), "unknown": (0, 165, 255)}


def draw_overlay(img: np.ndarray, result: dict, fps: float = 0.0) -> np.ndarray:
    out = img.copy()
    for d in result.get("detections", []):
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
        tag = f"{d['class_name']}" + (f" #{d['track_id']}" if d.get("track_id") is not None else "")
        cv2.putText(out, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    ev = result["events"][0] if result.get("events") else None
    status = ev["status"] if ev else "normal"
    col = _COL[status]
    persons = sum(1 for d in result.get("detections", []) if d["class_name"] == "person")
    lines = [
        (f"{result.get('camera_id', '')}   FPS: {fps:4.1f}   Latency: {result.get('latency_ms', 0):4.0f}ms", (255, 255, 255)),
        (f"EVENT: {ev['event'] if ev else 'WARMING UP'}", col),
        (f"Confidence: {ev['probability']:.2f}   Anomaly: {ev['anomaly_score']:.2f}" if ev else "Confidence: -", (255, 255, 255)),
        (f"Persons: {persons}   Status: {'ALERT' if (ev and status != 'normal') else 'NORMAL'}", col),
    ]
    if ev and not ev.get("model_trained", True):
        lines.append(("UNTRAINED MODEL - outputs are meaningless", (0, 0, 255)))
    box_h = 8 + 20 * len(lines)
    cv2.rectangle(out, (0, 0), (430, box_h), (20, 20, 20), -1)
    for i, (t, c) in enumerate(lines):
        cv2.putText(out, t, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out
