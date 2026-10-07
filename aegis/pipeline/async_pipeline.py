"""Producer/consumer pipeline: capture -> detection(+micro-batching) -> temporal -> semantic -> alerts.
Bounded queues; live sources drop the OLDEST frame when a stage falls behind (fresh beats complete)."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from ..camera.camera_source import CameraSource
from ..camera.frame import Frame
from ..utils import log
from .engine import AegisEngine, Pending


class BoundedQueue:
    def __init__(self, maxsize: int):
        self.max = maxsize
        self._q: Deque[Any] = deque()
        self._cv = threading.Condition()
        self.dropped = 0
        self.closed = False

    def put(self, item, drop_oldest: bool = True) -> None:
        with self._cv:
            while len(self._q) >= self.max and not self.closed:
                if drop_oldest:
                    self._q.popleft()
                    self.dropped += 1
                else:
                    self._cv.wait(0.05)          # backpressure (file sources: never lose frames)
            self._q.append(item)
            self._cv.notify_all()

    def try_put(self, item) -> bool:
        with self._cv:
            if len(self._q) >= self.max:
                return False
            self._q.append(item)
            self._cv.notify_all()
            return True

    def get(self, timeout: float = 0.1):
        with self._cv:
            if not self._q and not self.closed:
                self._cv.wait(timeout)
            if self._q:
                item = self._q.popleft()
                self._cv.notify_all()
                return item
            return None

    def get_batch(self, max_n: int, wait_ms: float) -> List[Any]:
        first = self.get(0.1)
        if first is None:
            return []
        batch, deadline = [first], time.monotonic() + wait_ms / 1000.0
        while len(batch) < max_n:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            with self._cv:
                if not self._q:
                    self._cv.wait(left)
                if self._q:
                    batch.append(self._q.popleft())
                    self._cv.notify_all()
        return batch

    def __len__(self):
        return len(self._q)

    def close(self):
        with self._cv:
            self.closed = True
            self._cv.notify_all()


@dataclass
class _Work:
    frame: Frame
    obj_vec: np.ndarray
    summary: dict
    dets: list


class AsyncPipeline:
    def __init__(self, engine: AegisEngine, sources: Dict[str, CameraSource],
                 on_result: Optional[Callable[[dict], None]] = None):
        self.e, self.sources, self.on_result = engine, sources, on_result
        r = engine.cfg.runtime
        self.frame_q = BoundedQueue(r.frame_queue_size * max(1, len(sources)))
        self.det_q = BoundedQueue(r.detection_queue_size * max(1, len(sources)))
        self.sem_q = BoundedQueue(r.semantic_queue_size)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._capturing = len(sources)
        self._cap_lock = threading.Lock()
        self.results: List[dict] = []
        self.stats = {"laya_dropped": 0}

    # --- stages ---------------------------------------------------------------------------
    def _capture(self, cam_id: str, src: CameraSource):
        misses = 0
        while not self._stop.is_set():
            f = src.read()
            if f is None:
                misses += 1
                if not src.is_live or misses > 200:
                    break
                continue
            misses = 0
            self.frame_q.put(f, drop_oldest=src.is_live)
        with self._cap_lock:
            self._capturing -= 1

    def _detect(self):
        rt = self.e.cfg.runtime
        while not self._stop.is_set():
            batch = self.frame_q.get_batch(rt.max_batch_size, rt.max_batch_wait_ms)
            if not batch:
                if self._capturing <= 0 and len(self.frame_q) == 0:
                    break
                continue
            todo, decisions = [], []
            for f in batch:
                st = self.e._state(f.camera_id, self.sources[f.camera_id].fps)
                d = st.decide(f)
                decisions.append((st, f, d))
                if d.detect:
                    todo.append((st, f))
            self.e.stage_detect(todo)        # ONE batched detector call for all cameras in the micro-batch
            for st, f, d in decisions:
                if d.temporal:
                    self.det_q.put(_Work(f, st.obj_vec.copy(), dict(st.summary), list(st.dets)),
                                   drop_oldest=self.sources[f.camera_id].is_live)
        self.det_q.close()

    def _temporal(self):
        while not self._stop.is_set():
            w = self.det_q.get(0.1)
            if w is None:
                if self.det_q.closed and len(self.det_q) == 0:
                    break
                continue
            st = self.e._state(w.frame.camera_id, 30.0)
            p = self.e.stage_temporal(st, w.frame, w.obj_vec, w.summary, w.dets)
            if p is None:
                continue
            if p.decision.call and self.sem_q.try_put(p):
                continue                       # semantic worker will finalise
            if p.decision.call:
                self.stats["laya_dropped"] += 1   # semantic worker is saturated: proceed without Laya
            self._publish(p, None)
        self.sem_q.close()

    def _semantic(self):
        while not self._stop.is_set():
            p: Optional[Pending] = self.sem_q.get(0.1)
            if p is None:
                if self.sem_q.closed and len(self.sem_q) == 0:
                    break
                continue
            self._publish(p, self.e.stage_semantic(p))

    def _publish(self, p: Pending, sem):
        event, alerts = self.e.finalize(p, sem)
        res = {"camera_id": p.camera_id, "timestamp": event["timestamp"], "events": [event],
               "detections": p.dets, "tracks": p.tracks, "anomaly_score": event["anomaly_score"],
               "latency_ms": round((time.monotonic() - p.frame.capture_time) * 1000.0, 2),
               "alerts": [a.to_dict() for a in alerts]}
        self.e.prof.add("end_to_end_latency", res["latency_ms"])
        self.results.append(res)
        if self.on_result:
            self.on_result(res)

    # --- lifecycle ------------------------------------------------------------------------
    def start(self) -> "AsyncPipeline":
        for cid, s in self.sources.items():
            self._threads.append(threading.Thread(target=self._capture, args=(cid, s), name=f"cap-{cid}", daemon=True))
        for fn, nm in ((self._detect, "detect"), (self._temporal, "temporal"), (self._semantic, "semantic")):
            self._threads.append(threading.Thread(target=fn, name=nm, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def join(self, timeout: Optional[float] = None) -> None:
        end = None if timeout is None else time.monotonic() + timeout
        for t in self._threads:
            t.join(None if end is None else max(0.0, end - time.monotonic()))

    def stop(self) -> None:
        self._stop.set()
        for q in (self.frame_q, self.det_q, self.sem_q):
            q.close()
        self.join(3.0)
        for s in self.sources.values():
            s.release()
        self.e.alerts.flush(time.time())

    def report(self) -> dict:
        return {"frames_dropped": self.frame_q.dropped, "detections_dropped": self.det_q.dropped,
                **self.stats, "results": len(self.results), "latency": self.e.prof.summary()}
