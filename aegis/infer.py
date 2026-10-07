"""python -m aegis.infer --source video.mp4|rtsp://...|0|mock [--checkpoint aegis_best.pt] --out outputs/run1"""
from __future__ import annotations

import argparse
import json

from .camera import open_source
from .config import load_config
from .pipeline.engine import AegisEngine
from .utils import setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="outputs/infer")
    ap.add_argument("--camera-id", default="CAM-001")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop live sources after N seconds")
    ap.add_argument("--no-annotate", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config, a.policy)
    eng = AegisEngine(cfg, checkpoint=a.checkpoint)
    res = eng.run_source(open_source(a.source, a.camera_id), output_dir=a.out, annotate=not a.no_annotate,
                         max_seconds=a.max_seconds)
    print(json.dumps({k: res["summary"][k] for k in ("frames", "end_to_end_fps", "device", "detector", "model_trained")}))
    print(f"events: {len(res['events'])}  alerts: {len(res['alerts'])}  -> {a.out}/(annotated_video.mp4, events.json, metrics.json)")


if __name__ == "__main__":
    main()
