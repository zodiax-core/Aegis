"""python -m aegis.training.train --config configs/default.yaml --out runs/exp1 [--dry-run]
Builds the canonical dataset (manifest -> split -> leakage check -> feature cache) then trains stage 1.
--dry-run uses synthetic features for 3 steps to verify plumbing (metrics from it are meaningless)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch

from ..config import load_config
from ..data.cache import FeatureCache
from ..data.dataset import FeatureWindowDataset, SyntheticFeatureDataset, VideoBagDataset
from ..data.label_mapper import LabelMapper
from ..data.manifest import merge_manifests, save_manifests, scan_dataset
from ..data.preprocessing import build_feature_cache
from ..data.splitter import save_split_manifest, split_videos
from ..data.validator import check_leakage, validate_manifest
from ..model.aegis_model import AegisModel
from ..model.visual_encoder import VisualEncoder
from ..perception.features import OBJ_FEAT_DIM
from ..perception.yolo_detector import build_detector
from ..utils import resolve_device, setup_logging, log
from .trainer import evaluate, loss_weights, train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config, a.policy)
    device = resolve_device(cfg.runtime.device)
    os.makedirs(a.out, exist_ok=True)
    if a.dry_run:
        names = ["NORMAL", "FIGHT"]
        cfg.model.class_names = names
        model = AegisModel(cfg.model, 512, class_names=names)
        cfg.train.epochs = 1
        tr = SyntheticFeatureDataset(64, cfg.sampling.clip_length, 512, OBJ_FEAT_DIM, 2)
        va = SyntheticFeatureDataset(32, cfg.sampling.clip_length, 512, OBJ_FEAT_DIM, 2, seed=1)
        res = train(model, tr, va, cfg, a.out, device, max_steps=3)
        print("dry-run OK (synthetic data; metrics meaningless):", res["best_val_macro_f1"])
        return
    mapper = LabelMapper.from_yaml(cfg.data.label_mapping)
    manifests = [scan_dataset(d.name, d.path, mapper, d.license, d.version, d.source) for d in cfg.data.datasets]
    records = merge_manifests(manifests)
    rep = validate_manifest(records, cfg.data.min_video_sec)
    for w in rep.warnings[:10]:
        log.warning(w)
    if not rep.ok:
        raise SystemExit("dataset validation failed:\n" + "\n".join(rep.errors[:20]))
    split = split_videos(records, cfg.data.split_fractions, cfg.data.seed, cfg.data.official_test)
    check_leakage(records, split, strict=True)
    classes = mapper.class_list([r.label for r in records])
    cfg.model.class_names = classes
    save_manifests(manifests, a.out)
    save_split_manifest(split, records, os.path.join(a.out, "split_manifest.json"), cfg.data.seed, cfg.data.split_fractions,
                        {"classes": classes, "clip_length": cfg.sampling.clip_length})
    shutil.copy(cfg.data.label_mapping, os.path.join(a.out, "label_mapping.yaml"))
    cfg.save_yaml(os.path.join(a.out, "config.yaml"))
    det = build_detector(cfg.perception, device)
    ve = VisualEncoder(cfg.model.visual_backbone, cfg.model.visual_pretrained, cfg.sampling.image_size).to(device)
    cache = FeatureCache(cfg.data.cache_dir)
    keys = build_feature_cache(records, cfg, det, ve, cache, lambda i, n: i % 50 == 0 and log.info("features %d/%d", i, n))
    by_id = {r.video_id: r for r in records}
    # Eval always uses windows (every window of every video) so trainer.evaluate can aggregate to
    # one prediction per video. Training uses video bags when MIL is on - see VideoBagDataset.
    mk = lambda ids, training: FeatureWindowDataset([by_id[i] for i in ids], keys, cache, classes, cfg.sampling.clip_length,
                                                    cfg.data.window_stride, mapper.normal_label,
                                                    laya_dropout=cfg.train.laya_dropout, training=training)
    va, te = mk(split["val"], False), mk(split["test"], False)
    if cfg.train.mil:
        tr = VideoBagDataset([by_id[i] for i in split["train"]], keys, cache, classes, cfg.sampling.clip_length,
                             cfg.data.window_stride, mapper.normal_label, bag_size=cfg.train.bag_size,
                             laya_dropout=cfg.train.laya_dropout, training=True, seed=cfg.data.seed)
        log.info("MIL: %d train bags x %d windows (%d bags/step); val/test windows: %d/%d",
                 len(tr), cfg.train.bag_size, cfg.train.batch_size, len(va), len(te))
    else:
        tr = mk(split["train"], True)
        log.info("windows train/val/test: %d/%d/%d", len(tr), len(va), len(te))
    model = AegisModel(cfg.model, ve.out_dim, class_names=classes)
    t0 = time.time()
    res = train(model, tr, va, cfg, a.out, device, visual_tag=ve.tag)
    from torch.utils.data import DataLoader
    best = torch.load(os.path.join(a.out, "aegis_best.pt"), map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, DataLoader(te, batch_size=cfg.train.batch_size), device,
                    w=loss_weights(cfg), mil_k=cfg.train.mil_k) if len(te) else {}
    metrics = {"best_val_macro_f1": res["best_val_macro_f1"], "best_val_video_macro_f1": res["best_val_video_macro_f1"],
               "monitor": res["best_monitor"], "best_score": res["best_score"],
               "objective": "MIL/bag" if cfg.train.mil else "window",
               "test": test, "train_seconds": time.time() - t0}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    # The headline comparison: a model that only learnt the class prior scores the baseline below.
    for lvl in ("", "video_"):
        b = test.get(f"{lvl}baseline")
        if b:
            log.info("test %smacro_f1 %.4f vs majority %.4f / random %.4f", lvl,
                     test.get(f"{lvl}macro_f1", float("nan")), b["majority_macro_f1"], b["random_macro_f1"])
    json.dump({"visual_encoder": ve.tag, "detector": det.name, "device": device, "classes": classes,
               "model": model.metadata(), "laya": {"checkpoint": cfg.laya.checkpoint, "revision": cfg.laya.revision},
               "note": "Laya features are NOT used in stage-1 training (masked); fusion learns to ignore them."},
              open(os.path.join(a.out, "model_metadata.json"), "w"), indent=1)
    print("done:", json.dumps(metrics)[:400])


if __name__ == "__main__":
    main()
