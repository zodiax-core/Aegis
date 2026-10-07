"""python -m aegis.benchmark [--policy FAST|BALANCED|ACCURATE|all] [--cameras N] [--seconds S]
Synthetic input by default: measures the pipeline, not accuracy. Numbers are specific to the machine they ran on."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psutil

from .camera.mock_source import MockSource
from .config import load_config
from .pipeline.engine import AegisEngine
from .semantic.laya_adapter import MockVerifier, NullVerifier
from .utils import setup_logging


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        pass
    return "CPU", 0.0


def run_one(policy: str, args) -> dict:
    cfg = load_config(args.config, policy)
    if args.no_pretrained:
        cfg.model.visual_pretrained = False
    if args.backend:
        cfg.perception.backend = args.backend
    cfg.runtime.device = args.device
    ver = MockVerifier(latency_ms=args.laya_ms) if args.mock_laya else None
    eng = AegisEngine(cfg, checkpoint=args.checkpoint, verifier=ver)
    proc = psutil.Process()
    src = MockSource("CAM-001", n_frames=int(args.seconds * 30), fps=30, size=(args.width, args.height))
    t0 = time.perf_counter()
    r = eng.run_source(src, annotate=False)
    wall = time.perf_counter() - t0
    lat = eng.prof.summary()
    tick = lat.get("analysis_tick", lat.get("total", {}))
    name, gmem = gpu_info()
    n_det = eng.prof.counters.get("detect_frames", 0)
    return {
        "policy": policy, "device": eng.device, "gpu": name, "detector": eng.detector.name,
        "input_resolution": f"{args.width}x{args.height}", "model_image_size": cfg.sampling.image_size,
        "detection_fps": round(n_det / (lat["yolo"]["total_ms"] / 1000), 1) if "yolo" in lat else None,
        "end_to_end_fps": round(r["summary"]["frames"] / wall, 1),
        "p50_latency_ms": round(tick.get("p50_ms", 0), 1), "p95_latency_ms": round(tick.get("p95_ms", 0), 1),
        "gpu_memory_mb": round(gmem, 1), "cpu_rss_mb": round(proc.memory_info().rss / 2**20, 1),
        "laya_backend": eng.verifier.name,
        "laya_calls_per_min": round(eng.prof.counters.get("laya_calls", 0) / (args.seconds / 60.0), 1),
        "stage_p50_ms": {k: round(v["p50_ms"], 1) for k, v in lat.items()},
        "note": "synthetic frames; random-weights model unless --checkpoint; detector may be a fallback (see 'detector')",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default="BALANCED")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default=None, help="yolo | motion | auto")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="skip downloading backbone weights")
    ap.add_argument("--mock-laya", action="store_true")
    ap.add_argument("--laya-ms", type=float, default=80.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    pols = ["FAST", "BALANCED", "ACCURATE"] if args.policy.lower() == "all" else [args.policy.upper()]
    rows = [run_one(p, args) for p in pols]
    for r in rows:
        print("=" * 56)
        for k in ["policy", "gpu", "detector", "input_resolution", "model_image_size", "detection_fps", "end_to_end_fps",
                  "p50_latency_ms", "p95_latency_ms", "gpu_memory_mb", "cpu_rss_mb", "laya_backend", "laya_calls_per_min"]:
            print(f"{k:>20}: {r[k]}")
        print(f"{'stage p50 (ms)':>20}: {r['stage_p50_ms']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
