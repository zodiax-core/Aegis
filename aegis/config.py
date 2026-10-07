"""Single source of truth for every tunable in AEGIS. No torch import here."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SamplingConfig:
    capture_fps: float = 30.0      # nominal; real sources report their own fps
    detection_fps: float = 10.0    # YOLO + tracking rate
    temporal_fps: float = 5.0      # visual encoder + temporal buffer rate
    frame_stride: int = 0          # >0: analyse every Nth captured frame instead of temporal_fps
    clip_length: int = 16          # frames per temporal window (research baseline)
    buffer_size: int = 0           # 0 -> clip_length
    image_size: int = 224          # visual encoder input
    thumb_max_side: int = 336      # keyframe thumbnails kept for Laya


@dataclass
class PerceptionConfig:
    backend: str = "auto"          # auto | yolo | motion  (motion = weight-free fallback for tests/demos)
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    half: bool = True              # only applied on CUDA
    tracker: str = "bytetrack"     # bytetrack | iou
    track_buffer: int = 30
    relevant_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "bus", "motorcycle", "bicycle", "knife", "gun", "pistol", "rifle"])


@dataclass
class ModelConfig:
    visual_backbone: str = "resnet18"
    visual_pretrained: bool = True
    d_model: int = 256
    n_heads: int = 4
    temporal_layers: int = 2
    fusion_layers: int = 1
    ff_dim: int = 512
    dropout: float = 0.2
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "FIGHT"])


@dataclass
class LayaConfig:
    enabled: bool = True
    checkpoint: str = "thaitea/laya-vision"
    revision: Optional[str] = None       # None -> resolve current sha at load time and log it
    max_keyframes: int = 2
    # policy (NOT calibrated; tune on validation data)
    skip_below_anomaly: float = 0.30
    confirm_above_anomaly: float = 0.85
    high_anomaly_action: str = "confirm"  # confirm | skip
    trigger_min_anomaly: float = 0.20
    low_confidence: float = 0.55
    entropy_trigger: float = 0.70
    flip_trigger: int = 2
    motion_trigger: float = 0.60          # normalised px/s of fastest person
    close_pair_distance: float = 0.15
    enable_object_triggers: bool = True
    min_interval_sec: float = 2.0         # per camera
    max_calls_per_minute: int = 20        # per camera


@dataclass
class EventConfig:
    # unknown-event detector
    anomaly_threshold: float = 0.50
    known_min_prob: float = 0.60
    unknown_entropy: float = 0.75
    # alerts
    alert_threshold: float = 0.70
    min_consecutive: int = 3
    cooldown_sec: float = 5.0
    close_after_sec: float = 4.0
    update_interval_sec: float = 2.0


@dataclass
class RuntimeConfig:
    device: str = "auto"           # auto | cuda | cpu
    precision: str = "fp16"        # fp32 | fp16 (fp16 only used on CUDA); int8 only via TensorRT export
    policy: str = "BALANCED"       # FAST | BALANCED | ACCURATE
    max_batch_size: int = 4
    max_batch_wait_ms: float = 15.0
    frame_queue_size: int = 8
    detection_queue_size: int = 8
    semantic_queue_size: int = 4


@dataclass
class DatasetSource:
    name: str = ""
    path: str = ""
    license: str = "unknown - check source terms"
    version: str = "1"
    source: str = ""


@dataclass
class DataConfig:
    datasets: List[DatasetSource] = field(default_factory=list)
    label_mapping: str = "configs/label_mapping.yaml"
    cache_dir: str = "cache"
    output_dir: str = "outputs"
    split_fractions: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    official_test: bool = True     # datasets shipping their own val/test dirs keep them as test
    seed: int = 1337
    min_video_sec: float = 1.0
    window_stride: int = 8         # training window hop (in temporal samples)


@dataclass
class TrainConfig:
    train_yolo: bool = False
    train_visual_encoder: bool = False
    train_temporal_encoder: bool = True
    train_fusion: bool = True
    train_event_head: bool = True
    epochs: int = 25
    batch_size: int = 64           # window mode: windows per step. MIL mode: BAGS (videos) per step,
                                   # so windows per step = batch_size * bag_size.
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 6
    laya_dropout: float = 0.5      # fraction of training windows with Laya features masked out
    balance: str = "sampler"       # sampler | class_weights | none
    amp: bool = True
    # ---- loss term weights. The anomaly target is ~81% positive on UCF-Crime so that term
    # collapses to a constant, and the confidence term is fitted to the event head's own error rate;
    # at 1.0/1.0/0.5 they dominated the reported total while the event head sat at chance.
    w_event: float = 1.0
    w_anomaly: float = 0.3
    w_confidence: float = 0.1
    monitor: str = "video_macro_f1"  # early-stopping metric; falls back to macro_f1 if absent
    # ---- multiple-instance learning over video bags (fixes weak window labels + length imbalance)
    mil: bool = False              # dataclass default off so --dry-run/smoke tests are untouched;
                                   # configs/default.yaml turns it on for real training.
    mil_k: int = 3                 # windows per bag supervised as anomalous / used for video voting
    bag_size: int = 12             # windows sampled per video per epoch == the per-video window cap
    mil_smooth: float = 0.0        # temporal smoothness on bag anomaly scores (off by default)
    mil_sparse: float = 0.0        # L1 sparsity on positive-bag anomaly scores (off by default)
    # ---- supervised contrastive loss (Orion and later) ----------------------------------
    # Penalises confusion classes (ASSAULT/ABUSE/ARREST) being too close in latent space.
    # 0.0 = disabled (backward-compatible with Vesper checkpoints).
    w_contrastive: float = 0.0
    contrastive_margin: float = 0.4   # minimum cosine distance between hard class pairs


@dataclass
class AegisConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    events: EventConfig = field(default_factory=EventConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def buffer_size(self) -> int:
        return max(self.sampling.buffer_size or self.sampling.clip_length, self.sampling.clip_length)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# Presets are starting points, not claims of optimality. Benchmark them (python -m aegis.benchmark).
POLICY_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "FAST": {
        "sampling": {"detection_fps": 5.0, "temporal_fps": 3.0, "image_size": 160},
        "perception": {"imgsz": 480},
        "laya": {"max_keyframes": 1, "min_interval_sec": 5.0, "max_calls_per_minute": 6,
                 "enable_object_triggers": False},
    },
    "BALANCED": {},
    "ACCURATE": {
        "sampling": {"detection_fps": 15.0, "temporal_fps": 8.0, "image_size": 224},
        "perception": {"imgsz": 960},
        "laya": {"max_keyframes": 3, "min_interval_sec": 1.0, "max_calls_per_minute": 40},
    },
}


def _build(cls, data: Dict[str, Any]):
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"Unknown config key '{k}' for {cls.__name__}")
        f = known[k]
        sample = f.default_factory() if callable(f.default_factory) else None
        if is_dataclass(sample) and isinstance(v, dict):
            kwargs[k] = _build(type(sample), v)
        elif k == "datasets" and cls is DataConfig:
            kwargs[k] = [DatasetSource(**d) if isinstance(d, dict) else d for d in v]
        else:
            if isinstance(f.default, float) and isinstance(v, (str, int)):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    pass
            elif isinstance(f.default, int) and isinstance(v, (str, float)) and not isinstance(f.default, bool):
                try:
                    v = int(v)
                except (ValueError, TypeError):
                    pass
            kwargs[k] = v
    return cls(**kwargs)


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_from_dict(d: Dict[str, Any], apply_policy: bool = True) -> AegisConfig:
    cfg = _build(AegisConfig, d)
    if apply_policy:
        cfg = with_policy(cfg, cfg.runtime.policy, explicit=d)
    return cfg


def with_policy(cfg: AegisConfig, policy: str, explicit: Optional[Dict[str, Any]] = None) -> AegisConfig:
    """Apply a preset. Values the user set explicitly in their YAML always win over the preset."""
    policy = policy.upper()
    if policy not in POLICY_PRESETS:
        raise ValueError(f"Unknown policy '{policy}'. Choose from {list(POLICY_PRESETS)}")
    merged = _deep_update(cfg.to_dict(), _deep_update(POLICY_PRESETS[policy], explicit or {}))
    merged["runtime"]["policy"] = policy
    return _build(AegisConfig, merged)


def load_config(path: Optional[str] = None, policy: Optional[str] = None) -> AegisConfig:
    d: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    if policy:
        d = _deep_update(d, {"runtime": {"policy": policy}})
    return config_from_dict(d)
