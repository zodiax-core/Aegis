"""Phase 5+6: Real dataloader test + model forward/backward on actual UFC_crime data."""
from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader

from aegis.config import load_config
from aegis.data.label_mapper import LabelMapper
from aegis.data.manifest import scan_dataset, merge_manifests
from aegis.data.splitter import split_videos
from aegis.data.cache import FeatureCache
from aegis.data.dataset import FeatureWindowDataset
from aegis.data.preprocessing import build_feature_cache
from aegis.perception.yolo_detector import build_detector
from aegis.model.visual_encoder import VisualEncoder
from aegis.model.aegis_model import AegisModel
from aegis.training.trainer import train

cfg = load_config("configs/default.yaml")
mapper = LabelMapper.from_yaml(cfg.data.label_mapping)
device = "cpu"

# Scan UFC_crime only (RWF_2000 is empty)
ds_cfg = next(d for d in cfg.data.datasets if d.name == "UFC_crime")
m = scan_dataset(ds_cfg.name, ds_cfg.path, mapper, ds_cfg.license, ds_cfg.version, ds_cfg.source, probe=True)
records = merge_manifests([m])
print(f"Records: {len(records)}, classes: {m.classes}")

split = split_videos(records, cfg.data.split_fractions, cfg.data.seed)
classes = mapper.class_list([r.label for r in records])
print(f"Classes: {classes}")
print(f"Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

# Build detector and visual encoder
det = build_detector(cfg.perception, device)
ve = VisualEncoder(cfg.model.visual_backbone, cfg.model.visual_pretrained, cfg.sampling.image_size).to(device)
print(f"Detector: {det.name}, VisualEncoder tag: {ve.tag}, out_dim: {ve.out_dim}")

# Extract features for only first 3 training videos
test_recs = [r for r in records if r.video_id in split["train"][:3]]
print(f"Using {len(test_recs)} videos for smoke test")
cache = FeatureCache("cache_test")
keys = build_feature_cache(test_recs, cfg, det, ve, cache, lambda i, n: print(f"  feature {i}/{n}"))
print(f"Feature cache keys: {len(keys)}")

# Build dataset
by_id = {r.video_id: r for r in test_recs}
ds = FeatureWindowDataset(test_recs, keys, cache, classes, cfg.sampling.clip_length, cfg.data.window_stride, mapper.normal_label)
print(f"Dataset windows: {len(ds)}")

assert len(ds) > 0, "No windows in dataset - videos may be too short or feature extraction failed"

item = ds[0]
print(f"Sample vis: {item['vis'].shape}, obj: {item['obj'].shape}, event: {item['event']}, anomaly: {item['anomaly']}")

dl = DataLoader(ds, batch_size=min(4, len(ds)))
batch = next(iter(dl))
print(f"Batch vis: {batch['vis'].shape}, obj: {batch['obj'].shape}")
print("PHASE 5 - REAL DATALOADER TEST: PASS")

# Phase 6: model forward + backward
print("\n--- PHASE 6: MODEL FORWARD + BACKWARD ---")
cfg.model.class_names = classes
model = AegisModel(cfg.model, ve.out_dim, class_names=classes).to(device)
model.train()

vis = batch["vis"].to(device)
obj = batch["obj"].to(device)
laya = batch["laya"].to(device)
mask = batch["laya_mask"].to(device)
y = batch["event"].to(device)
anom = batch["anomaly"].to(device)

print(f"Forward pass on vis={vis.shape}, obj={obj.shape}")
out = model(vis, obj, laya, mask)
print(f"Model outputs: {list(out.keys())}")
for k, v in out.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# Loss
from aegis.training.trainer import compute_loss
loss = compute_loss(out, y, anom, device)
print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()
print("Backward: PASS")

# Optimizer step
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
opt.step()
opt.zero_grad()
print("Optimizer step: PASS")

print("\nPHASE 6 - MODEL FORWARD + BACKWARD: PASS")
