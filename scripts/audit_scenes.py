"""Report videos that are probably the SAME camera/scene, and whether they straddle train/val/test.

Why this exists
---------------
`aegis/data/manifest.py` sets `group_id` to the video_id, and `path` is unique per video. That means
two of the three leakage checks in `check_leakage` (`aegis/data/validator.py`) can never fire -
only the `fingerprint` check (byte-identical files) does anything. UCF-Crime filenames
(`Abuse001_x264.mp4`) carry no camera identity, so if several clips come from one fixed camera they
can land in different splits and the model can score well by recognising the scene.

How it works
------------
The mean cached visual embedding of a video is a strong scene descriptor - fixed CCTV cameras
produce near-identical backgrounds - and those embeddings are ALREADY in the feature cache, so this
costs zero extra video decoding.

This script only REPORTS. It does not change `group_id` or the split. Scene leakage inflates
validation scores rather than depressing them, so it is not the cause of a low macro-F1; it is a
reason not to trust a good one. Measure first, then decide whether rewiring the split is worth it.

    python scripts/audit_scenes.py --run runs/baseline_tier0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.config import load_config
from aegis.data.cache import FeatureCache, cache_key
from aegis.data.label_mapper import LabelMapper
from aegis.data.manifest import merge_manifests, scan_dataset
from aegis.data.preprocessing import sampling_signature
from aegis.data.splitter import split_videos


def _encoder_and_detector(cfg, run: str | None):
    """Prefer the exact tags a run recorded; fall back to deriving them from config. Both feed
    `cache_key`, so getting them wrong just means 'nothing is cached'."""
    meta = os.path.join(run or "", "model_metadata.json")
    if run and os.path.isfile(meta):
        d = json.load(open(meta))
        return d["visual_encoder"], d["detector"], f"{meta} (recorded by the run)"
    tag = f"{cfg.model.visual_backbone}-{'in1k' if cfg.model.visual_pretrained else 'rand'}-{cfg.sampling.image_size}"
    det = f"yolo:{cfg.perception.weights}" if cfg.perception.backend != "motion" else "motion-blobs(fallback)"
    return tag, det, "derived from config (pass --run for the exact tags)"


def _clusters(sim: np.ndarray, thr: float) -> list[list[int]]:
    """Single-linkage via union-find: anything transitively similar is one scene."""
    parent = list(range(len(sim)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(np.triu(sim >= thr, k=1))):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[a] = b
    out: dict[int, list[int]] = {}
    for i in range(len(sim)):
        out.setdefault(find(i), []).append(i)
    return sorted(out.values(), key=len, reverse=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default=None, help="run dir with model_metadata.json + split_manifest.json")
    ap.add_argument("--threshold", type=float, default=0.92, help="cosine similarity for 'same scene'")
    ap.add_argument("--out", default="outputs/scene_audit.json")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    mapper = LabelMapper.from_yaml(cfg.data.label_mapping)
    records = merge_manifests([scan_dataset(d.name, d.path, mapper, d.license, d.version, d.source)
                               for d in cfg.data.datasets])
    sm = os.path.join(a.run or "", "split_manifest.json")
    if a.run and os.path.isfile(sm):
        split, split_src = json.load(open(sm))["splits"], sm
    else:
        split = split_videos(records, cfg.data.split_fractions, cfg.data.seed, cfg.data.official_test)
        split_src = f"recomputed (seed={cfg.data.seed})"
    where = {v: s for s, ids in split.items() for v in ids}

    tag, det, tag_src = _encoder_and_detector(cfg, a.run)
    sig = sampling_signature(cfg)
    cache = FeatureCache(cfg.data.cache_dir)
    print(f"encoder/detector: {tag} / {det}  [{tag_src}]")
    print(f"split: {split_src}")

    vids, embs, labels = [], [], []
    for r in records:
        z = cache.get(cache_key(r.fingerprint, sig, tag, det, cfg.perception.tracker))
        if z is None:
            continue
        v = z["vis"].astype(np.float32).mean(0)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        vids.append(r.video_id); embs.append(v / n); labels.append(r.label)
    if len(vids) < 2:
        print(f"only {len(vids)} of {len(records)} videos are in the cache at '{cfg.data.cache_dir}'.\n"
              f"Build the feature cache first (python -m aegis.training.train), or pass --run <dir>.")
        return 1

    E = np.stack(embs)
    sim = E @ E.T
    np.fill_diagonal(sim, -1.0)
    off = sim[np.triu_indices(len(vids), k=1)]
    pct = {f"p{p}": float(np.percentile(off, p)) for p in (50, 90, 99, 99.9)}
    print(f"\n{len(vids)}/{len(records)} videos cached. pairwise cosine: "
          + "  ".join(f"{k}={v:.3f}" for k, v in pct.items()) + f"  max={float(off.max()):.3f}")

    groups = [c for c in _clusters(sim, a.threshold) if len(c) > 1]
    straddling = [c for c in groups if len({where.get(vids[i]) for i in c} - {None}) > 1]
    print(f"\nthreshold {a.threshold}: {len(groups)} multi-video scene clusters "
          f"({sum(len(c) for c in groups)} videos), {len(straddling)} straddle splits")

    for c in straddling[: a.top]:
        print(f"  cluster of {len(c)}:")
        for i in c:
            print(f"    {where.get(vids[i], '-'):5s} {labels[i]:10s} {vids[i]}")

    pairs = [(float(sim[i, j]), vids[i], vids[j], where.get(vids[i]), where.get(vids[j]))
             for i, j in zip(*np.triu_indices(len(vids), k=1))
             if where.get(vids[i]) != where.get(vids[j]) and sim[i, j] >= a.threshold]
    pairs.sort(reverse=True)
    if pairs:
        print(f"\ntop cross-split pairs (these are the actionable ones):")
        for s, u, v, su, sv in pairs[: a.top]:
            print(f"  {s:.3f}  {su}:{u}  <->  {sv}:{v}")

    # A ready-to-use group_id assignment, if the numbers above say it is worth wiring in.
    proposed = {}
    for n, c in enumerate(_clusters(sim, a.threshold)):
        for i in c:
            proposed[vids[i]] = f"scene_{n:04d}" if len(c) > 1 else vids[i]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"threshold": a.threshold, "encoder": tag, "detector": det, "split_source": split_src,
                   "n_videos_cached": len(vids), "n_videos_total": len(records), "percentiles": pct,
                   "n_clusters_multi": len(groups), "n_clusters_straddling": len(straddling),
                   "clusters": [[vids[i] for i in c] for c in groups],
                   "straddling": [[{"video_id": vids[i], "split": where.get(vids[i]), "label": labels[i]}
                                   for i in c] for c in straddling],
                   "cross_split_pairs": [{"similarity": s, "a": u, "b": v, "split_a": su, "split_b": sv}
                                         for s, u, v, su, sv in pairs],
                   "proposed_group_ids": proposed}, f, indent=1)
    print(f"\nwrote {a.out}")
    if straddling:
        print("Next step: these clusters share a scene across splits, so val/test scores are "
              "optimistic. Wiring 'proposed_group_ids' into manifest.py:group_id makes "
              "split_videos() keep them together (it already moves whole groups).")
    else:
        print("No cross-split scene clusters at this threshold - group_id can stay as-is. "
              "Lower --threshold to probe harder before concluding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
