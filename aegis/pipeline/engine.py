"""AegisEngine: one inference engine for files, webcams, RTSP and many cameras.
The stages are separate methods so the async pipeline can run them on different threads."""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..alerts.alert_manager import Alert, AlertManager
from ..camera import open_source
from ..camera.camera_source import CameraSource
from ..camera.frame import Frame
from ..config import AegisConfig, load_config
from ..events.unknown import UnknownEventDetector
from ..model.aegis_model import AegisModel
from ..model.checkpoint import load_model_checkpoint
from ..model.visual_encoder import VisualEncoder
from ..perception.yolo_detector import build_detector
from ..semantic.laya_adapter import SemanticResult, SemanticVerifier, build_verifier
from ..semantic.laya_policy import LayaDecision, LayaPolicy, PolicyContext
from ..utils import Profiler, log, resolve_device
from .state import CameraState


@dataclass
class Pending:
    """A window that has been scored by the temporal model and is waiting for (optional) Laya + fusion."""
    camera_id: str
    frame: Frame
    span: Tuple[float, float]
    vis: np.ndarray
    obj: np.ndarray
    probs: np.ndarray
    anomaly: float
    confidence: float
    decision: LayaDecision
    dets: list
    tracks: list


class AegisEngine:
    def __init__(self, cfg: Optional[AegisConfig] = None, checkpoint: Optional[str] = None,
                 detector=None, visual_encoder: Optional[VisualEncoder] = None, model: Optional[AegisModel] = None,
                 verifier: Optional[SemanticVerifier] = None, profiler: Optional[Profiler] = None,
                 alert_manager: Optional[AlertManager] = None):
        self.cfg = cfg or load_config()
        self.device = resolve_device(self.cfg.runtime.device)
        self.prof = profiler or Profiler()
        self.model_trained = False
        ckpt = None
        if checkpoint:
            model, ckpt = load_model_checkpoint(checkpoint, self.device)
            saved = ckpt["config"]
            self.cfg.model = type(self.cfg.model)(**saved["model"])
            self.cfg.sampling.clip_length = saved["sampling"]["clip_length"]
            self.cfg.sampling.image_size = saved["sampling"]["image_size"]
            self.model_trained = True
        self.detector = detector or build_detector(self.cfg.perception, self.device)
        self.visual = (visual_encoder or VisualEncoder(self.cfg.model.visual_backbone, self.cfg.model.visual_pretrained,
                                                       self.cfg.sampling.image_size)).to(self.device).eval()
        if model is None:
            model = AegisModel(self.cfg.model, self.visual.out_dim)
            log.warning("No checkpoint given: model is RANDOMLY INITIALISED. Outputs are meaningless until trained.")
        self.model = model.to(self.device).eval()
        self.class_names = self.model.class_names
        self.normal_index = self.class_names.index("NORMAL") if "NORMAL" in self.class_names else 0
        self.verifier = verifier if verifier is not None else build_verifier(self.cfg.laya, self.device)
        self.laya_policy = LayaPolicy(self.cfg.laya)
        self.unknown = UnknownEventDetector(self.cfg.events, self.normal_index)
        self.alerts = alert_manager or AlertManager(self.cfg.events)
        self._states: Dict[str, CameraState] = {}
        self._lock = threading.RLock()          # guards _states / per-camera lists
        self._model_lock = threading.Lock()     # serialises GPU model calls across worker threads
        self._amp = self.device == "cuda" and self.cfg.runtime.precision.lower() == "fp16"

    # ------------------------------------------------------------------ helpers
    def _state(self, camera_id: str, fps: float) -> CameraState:
        with self._lock:
            st = self._states.get(camera_id)
            if st is None:
                st = self._states[camera_id] = CameraState(camera_id, self.cfg, fps)
            return st

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def _infer_ctx(self):
        with torch.inference_mode():
            if self._amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    yield
            else:
                yield

    def _encode(self, image: np.ndarray) -> np.ndarray:
        with self.prof.time("preprocess"):
            x = self.visual.preprocess([image])
        with self.prof.time("visual_encoder"), self._model_lock, self._infer_ctx():
            e = self.visual(x)
            self._sync()
        return e[0].float().cpu().numpy()

    def _run_model(self, vis: np.ndarray, obj: np.ndarray, laya: Optional[np.ndarray] = None):
        dev = self.device
        v = torch.from_numpy(vis).unsqueeze(0).to(dev)
        o = torch.from_numpy(obj).unsqueeze(0).to(dev)
        lt = torch.from_numpy(laya).unsqueeze(0).to(dev) if laya is not None else None
        with self._model_lock, self._infer_ctx():
            out = self.model(v, o, lt, None if lt is None else torch.ones(1, dtype=torch.bool, device=dev))
            probs = torch.softmax(out["event_logits"].float(), -1)[0].cpu().numpy()
            anomaly = float(torch.sigmoid(out["anomaly_logit"].float())[0])
            conf = float(torch.sigmoid(out["confidence_logit"].float())[0])
            self._sync()
        return probs, anomaly, conf

    # ------------------------------------------------------------------ stages
    def stage_detect(self, items: List[Tuple[CameraState, Frame]]) -> None:
        """Batched detection across cameras (micro-batching), then per-camera tracking."""
        if not items:
            return
        with self.prof.time("yolo"):
            dets = self.detector.detect_batch([f.image for _, f in items])
        for (st, f), d in zip(items, dets):
            with self.prof.time("tracking"):
                st.apply_detections(f, d)
        self.prof.count("detect_frames", len(items))

    def stage_temporal(self, st: CameraState, frame: Frame, obj_vec: Optional[np.ndarray] = None,
                       summary: Optional[dict] = None, dets: Optional[list] = None) -> Optional[Pending]:
        emb = self._encode(frame.image)
        st.push_temporal(frame, emb, obj_vec)
        self.prof.count("temporal_frames")
        if not st.buffer.ready:
            return None
        vis, obj = st.buffer.window()
        with self.prof.time("temporal_model"):
            probs, anomaly, conf = self._run_model(vis, obj)
        top = int(np.argmax(probs))
        ctx = PolicyContext(st.camera_id, frame.timestamp, anomaly, probs, self.class_names[top], float(probs[top]),
                            list(st.recent_top[-6:]), summary if summary is not None else st.summary)
        decision = self.laya_policy.decide(ctx) if self.verifier.available else LayaDecision(False, ["verifier_unavailable"])
        dets = st.dets if dets is None else dets
        return Pending(st.camera_id, frame, st.buffer.span(), vis, obj, probs, anomaly, conf, decision,
                       [d.to_dict() for d in dets], [t.to_dict() for t in st.tracks.active(frame.timestamp)])

    def stage_semantic(self, pending: Pending) -> Optional[SemanticResult]:
        st = self._states[pending.camera_id]
        images = st.buffer.keyframes(self.cfg.laya.max_keyframes)
        self.laya_policy.record_call(pending.camera_id, pending.frame.timestamp)
        with self.prof.time("laya"):
            res = self.verifier.verify(images)
        self.prof.count("laya_calls")
        return res

    def finalize(self, pending: Pending, sem: Optional[SemanticResult]) -> Tuple[dict, List[Alert]]:
        probs, anomaly, conf = pending.probs, pending.anomaly, pending.confidence
        if sem is not None:
            with self.prof.time("fusion"):
                probs, anomaly, conf = self._run_model(pending.vis, pending.obj, sem.vector())
        top = int(np.argmax(probs))
        sev_laya = sem.values.get("severity") if sem is not None else None
        a = self.unknown.assess(probs, anomaly, sev_laya)
        if a.status == "known":
            name = self.class_names[top]
        elif a.status == "unknown":
            name = "UNKNOWN"
        else:
            name = "NORMAL"
        prob = float(probs[self.normal_index]) if a.status == "normal" else float(probs[top])
        event = {
            "camera_id": pending.camera_id, "timestamp": round(pending.frame.timestamp, 3),
            "window": [round(pending.span[0], 3), round(pending.span[1], 3)],
            "event": name, "status": a.status, "probability": round(prob, 4), "top_class": self.class_names[top],
            "anomaly_score": round(anomaly, 4), "confidence": round(conf, 4),
            "severity": round(float(sev_laya if sev_laya is not None else anomaly), 4),
            "severity_source": "laya" if sev_laya is not None else "anomaly_score",
            "class_probs": {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
            "assessment": a.reasons,
            "laya": {"called": sem is not None, "reasons": pending.decision.reasons,
                     "values": sem.values if sem else None, "latency_ms": sem.latency_ms if sem else None},
            "model_trained": self.model_trained,
        }
        st = self._states[pending.camera_id]
        with self._lock:
            st.recent_top = (st.recent_top + [self.class_names[top]])[-8:]
            st.last_event = event
        alerts = self.alerts.update(pending.camera_id, event, pending.frame.timestamp)
        return event, alerts

    # ------------------------------------------------------------------ public API
    def process_frame(self, frame: Union[Frame, np.ndarray], camera_id: str = "CAM-001",
                      source_fps: Optional[float] = None) -> dict:
        t0 = time.perf_counter()
        st = self._state(frame.camera_id if isinstance(frame, Frame) else camera_id,
                         source_fps or self.cfg.sampling.capture_fps)
        if not isinstance(frame, Frame):
            frame = Frame(frame, st.camera_id, st.n_frames, st.n_frames / st.sampler.source_fps, time.monotonic())
        dec = st.decide(frame)
        alerts: List[Alert] = []
        new_event = laya_called = False
        if dec.detect:
            self.stage_detect([(st, frame)])
        if dec.temporal:
            p = self.stage_temporal(st, frame)
            if p is not None:
                sem = self.stage_semantic(p) if p.decision.call else None
                _, alerts = self.finalize(p, sem)
                new_event, laya_called = True, sem is not None
        ms = (time.perf_counter() - t0) * 1000.0
        self.prof.add("total", ms)
        if dec.detect or dec.temporal:
            self.prof.add("analysis_tick", ms)      # frames where real work happened
        ev = st.last_event
        return {
            "camera_id": st.camera_id, "timestamp": round(frame.timestamp, 3),
            "events": [ev] if ev else [],
            "detections": [d.to_dict() for d in st.dets], "tracks": [t.to_dict() for t in st.tracks.active(frame.timestamp)],
            "anomaly_score": ev["anomaly_score"] if ev else 0.0, "latency_ms": round(ms, 2),
            "alerts": [a.to_dict() for a in alerts],
            "analysis": {"detected": dec.detect, "temporal": dec.temporal, "new_event": new_event, "laya_called": laya_called},
        }

    def run_source(self, source: CameraSource, output_dir: Optional[str] = None, annotate: bool = True,
                   display: bool = False, max_frames: Optional[int] = None, max_seconds: Optional[float] = None,
                   on_result: Optional[Callable[[dict], None]] = None) -> dict:
        """Shared loop behind process_video / process_camera / process_rtsp."""
        import cv2
        from ..viz import draw_overlay
        writer, timeline, all_alerts = None, [], []
        n, misses, t_start, fps_ema, last_t = 0, 0, time.perf_counter(), 0.0, time.perf_counter()
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        try:
            while True:
                frame = source.read()
                if frame is None:
                    misses += 1
                    if not source.is_live or misses > 100:
                        break
                    continue
                misses = 0
                res = self.process_frame(frame, source_fps=source.fps)
                n += 1
                now = time.perf_counter()
                fps_ema = 0.9 * fps_ema + 0.1 / max(now - last_t, 1e-6) if fps_ema else 1.0 / max(now - last_t, 1e-6)
                last_t = now
                if res["analysis"]["new_event"]:
                    timeline.append(res["events"][0])
                all_alerts += res["alerts"]
                if on_result:
                    on_result(res)
                if annotate or display:
                    vis = draw_overlay(frame.image, res, fps_ema)
                    if output_dir and annotate:
                        if writer is None:
                            h, w = vis.shape[:2]
                            writer = cv2.VideoWriter(os.path.join(output_dir, "annotated_video.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), min(source.fps, 60.0), (w, h))
                        writer.write(vis)
                    if display:
                        cv2.imshow(frame.camera_id, vis)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                if max_frames and n >= max_frames:
                    break
                if max_seconds and time.perf_counter() - t_start > max_seconds:
                    break
        finally:
            if writer is not None:
                writer.release()
            source.release()
            if display:
                cv2.destroyAllWindows()
        all_alerts += [a.to_dict() for a in self.alerts.flush(time.time() if source.is_live else last_media_ts(timeline))]
        wall = time.perf_counter() - t_start
        summary = {"camera_id": source.camera_id, "frames": n, "wall_seconds": round(wall, 2),
                   "end_to_end_fps": round(n / wall, 2) if wall > 0 else 0.0,
                   "device": self.device, "model_trained": self.model_trained, "detector": self.detector.name,
                   "laya_backend": self.verifier.name, "latency": self.prof.summary(), "counters": dict(self.prof.counters)}
        result = {"events": timeline, "alerts": all_alerts, "summary": summary}
        if output_dir:
            with open(os.path.join(output_dir, "events.json"), "w") as f:
                json.dump({"events": timeline, "alerts": all_alerts}, f, indent=1)
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(summary, f, indent=1)
        return result

    def process_video(self, path: str, camera_id: str = "CAM-001", **kw) -> dict:
        return self.run_source(open_source(path, camera_id), **kw)

    def process_camera(self, index: int = 0, camera_id: str = "CAM-001", **kw) -> dict:
        return self.run_source(open_source(str(index), camera_id), **kw)

    def process_rtsp(self, url: str, camera_id: str = "CAM-001", **kw) -> dict:
        return self.run_source(open_source(url, camera_id), **kw)


def last_media_ts(timeline: List[dict]) -> float:
    return timeline[-1]["timestamp"] if timeline else 0.0
