"""python -m aegis.live [--source 0] [--checkpoint aegis_best.pt] -- local webcam window (press q to quit)."""
from __future__ import annotations

import argparse

from .camera import open_source
from .camera.latest_only import LatestOnlySource
from .config import load_config
from .pipeline.engine import AegisEngine
from .utils import setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--camera-id", default="CAM-001")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--headless", action="store_true", help="no window (servers/CI)")
    a = ap.parse_args(argv)
    setup_logging()
    eng = AegisEngine(load_config(a.config, a.policy), checkpoint=a.checkpoint)
    src = LatestOnlySource(open_source(a.source, a.camera_id))
    eng.run_source(src, annotate=False, display=not a.headless, max_seconds=a.max_seconds,
                   on_result=lambda r: [print("ALERT", x["kind"], x["event"], x["camera_id"]) for x in r["alerts"]])


if __name__ == "__main__":
    main()
