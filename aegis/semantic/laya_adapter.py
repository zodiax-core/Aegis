"""Isolates Laya Vision behind SemanticVerifier. Laya Vision is an experimental fork with its own
API; nothing outside this file may import `laya`."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import LayaConfig
from ..utils import log
from .laya_questions import LAYA_KEYS, LAYA_QUESTIONS, SEVERITY_LEVELS


@dataclass
class SemanticResult:
    values: Dict[str, float]           # question key -> P(true) or severity in [0,1]
    latency_ms: float = 0.0
    n_images: int = 0
    backend: str = ""

    def vector(self) -> np.ndarray:
        return np.array([self.values.get(k, 0.0) for k in LAYA_KEYS], np.float32)


class SemanticVerifier(ABC):
    available: bool = False
    name: str = "none"

    @abstractmethod
    def verify(self, images_rgb: List[np.ndarray], context: Optional[str] = None) -> Optional[SemanticResult]: ...


class NullVerifier(SemanticVerifier):
    name = "null"

    def verify(self, images_rgb, context=None):
        return None


class MockVerifier(SemanticVerifier):
    """Deterministic stand-in for tests/benchmarks: returns fixed values, optional fake latency."""
    available = True
    name = "mock"

    def __init__(self, values: Optional[Dict[str, float]] = None, latency_ms: float = 0.0):
        self.values = values or {k: 0.5 for k in LAYA_KEYS}
        self.latency_ms = latency_ms
        self.calls = 0

    def verify(self, images_rgb, context=None):
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        return SemanticResult(dict(self.values), self.latency_ms, len(images_rgb), self.name)


class LayaAdapter(SemanticVerifier):
    name = "laya-vision"

    def __init__(self, cfg: LayaConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = device
        self.agent = None
        self.error: Optional[str] = None
        self.resolved_revision: Optional[str] = cfg.revision
        self.available = self._load()

    def _load(self) -> bool:
        try:
            import laya  # Laya *Vision* fork; the PyPI text-only 'laya' has no load_vlm
            if not hasattr(laya, "load_vlm"):
                raise ImportError("installed 'laya' has no load_vlm; install the laya-vision fork "
                                  "(see requirements-laya.txt)")
            if self.resolved_revision is None:
                try:
                    from huggingface_hub import HfApi
                    self.resolved_revision = HfApi().model_info(self.cfg.checkpoint).sha
                except Exception as e:
                    log.warning("Could not resolve Laya revision (%s); weights are UNPINNED", e)
            kw = {"revision": self.resolved_revision} if self.resolved_revision else {}
            self.agent = laya.load_vlm(self.cfg.checkpoint, **kw)
            log.info("Laya Vision loaded: %s @ %s", self.cfg.checkpoint, self.resolved_revision)
            return True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            log.warning("Laya Vision unavailable (%s). Semantic verification disabled; the rest of AEGIS runs.",
                        self.error)
            return False

    def verify(self, images_rgb, context=None):
        if not self.available or not images_rgb:
            return None
        from PIL import Image
        t0 = time.perf_counter()
        noul: Dict[str, List[float]] = {k: [] for k, q in LAYA_QUESTIONS.items() if q["type"] == "noul"}
        sev: List[float] = []
        for img in images_rgb:
            state = {"image": Image.fromarray(img)}
            if context:
                state["note"] = context
            ans = self.agent.predict(state, LAYA_QUESTIONS)["answers"]
            for k in noul:
                noul[k].append(float(ans[k]["noul"]))
            sev.append(float(ans["severity"]["score"]) / max(1, SEVERITY_LEVELS - 1))
        values = {k: float(np.max(v)) for k, v in noul.items()}   # seen in ANY keyframe
        values["severity"] = float(np.mean(sev))
        return SemanticResult(values, (time.perf_counter() - t0) * 1000.0, len(images_rgb), self.name)


def build_verifier(cfg: LayaConfig, device: str = "auto") -> SemanticVerifier:
    if not cfg.enabled:
        return NullVerifier()
    v = LayaAdapter(cfg, device)
    return v if v.available else NullVerifier()
