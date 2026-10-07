"""Audit all datasets in datasets/ using AEGIS data manifest and validation tools."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

from aegis.config import load_config
from aegis.data.label_mapper import LabelMapper
from aegis.data.manifest import scan_dataset, merge_manifests, save_manifests
from aegis.data.splitter import split_videos, save_split_manifest
from aegis.data.validator import validate_manifest, check_leakage

def audit():
    cfg = load_config("configs/default.yaml")
    mapper = LabelMapper.from_yaml(cfg.data.label_mapping)
    
    print("=" * 60)
    print("AEGIS DATASET AUDIT")
    print("=" * 60)
    
    manifests = []
    for d in cfg.data.datasets:
        print(f"\nScanning dataset: {d.name} at {d.path}")
        if not os.path.exists(d.path):
            print(f"  [ERROR] Path does not exist: {d.path}")
            continue
        try:
            m = scan_dataset(d.name, d.path, mapper, d.license, d.version, d.source, probe=True)
            manifests.append(m)
            print(f"  Videos found: {m.num_videos}")
            print(f"  Classes: {m.classes}")
            print(f"  Class distribution: {m.class_distribution}")
            print(f"  Resolutions: {m.resolutions}")
        except Exception as e:
            print(f"  [ERROR] scan failed: {e}")
            raise
            
    records = merge_manifests(manifests)
    print(f"\nTotal video records merged: {len(records)}")
    
    # Check corrupt, unreadable, or short files
    rep = validate_manifest(records, min_seconds=cfg.data.min_video_sec)
    print(f"Validation report: OK={rep.ok}, Errors={len(rep.errors)}, Warnings={len(rep.warnings)}")
    if rep.errors:
        print("Validation errors:")
        for err in rep.errors[:20]:
            print(f"  - {err}")
    if rep.warnings:
        print(f"Validation warnings (showing up to 10):")
        for w in rep.warnings[:10]:
            print(f"  - {w}")
            
    # Check for duplicates by fingerprint
    fps = Counter(r.fingerprint for r in records if r.fingerprint)
    dups = {k: v for k, v in fps.items() if v > 1}
    print(f"Duplicate files by fingerprint: {len(dups)}")
    
    # Split & leakage check
    print("\nRunning dataset splitter (fractions: 0.7, 0.15, 0.15)...")
    split = split_videos(records, cfg.data.split_fractions, cfg.data.seed, cfg.data.official_test)
    print(f"Split counts: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    
    print("\nRunning leakage check...")
    leakage_rep = check_leakage(records, split, strict=True)
    print(f"Leakage check: OK={leakage_rep.ok}, Errors={len(leakage_rep.errors)}, Warnings={len(leakage_rep.warnings)}")
    
    # Save manifests
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    save_manifests(manifests, out_dir)
    save_split_manifest(split, records, os.path.join(out_dir, "split_manifest.json"), cfg.data.seed, cfg.data.split_fractions)
    print(f"Saved dataset_manifest.json and split_manifest.json to {out_dir}/")
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    audit()
