# ARCHITECTURE_AUDIT

> **Read this first.** I (the assistant that wrote this patch) only saw a *screenshot of your file tree*, not the
> contents of your repo. Everything under "CURRENT" below is reconstructed from (a) file names and (b) the notebook I
> generated earlier, which your modules were presumably split from. Treat it as **unverified**. Run
> `python scripts/audit_repo.py` in your repo to generate `ARCHITECTURE_AUDIT_AUTO.md` (AST/text facts: imports, unused
> definitions, hyperparameters, Colab/Modal signals) and correct this file where they disagree.

## CURRENT ARCHITECTURE (assumed from the earlier notebook)
ResNet-50 (ImageNet, frozen or last-block) per frame -> projection -> 3-layer Transformer over 16 frames + CLS ->
event head (10 classes) + anomaly head. Baseline mean-pool model also existed. YOLO produced 6 class-count features;
Laya Vision was a post-hoc keyframe verifier on flagged windows, fused by a hand-written weighted average (not learned).

## CURRENT DATA PIPELINE (assumed)
`datasets/<name>/<class>/*.mp4` scanned by folder name; hard-coded class list; per-video random clips decoded on the fly
with a per-clip .pt cache; video-level split. Your tree shows `RWF_2000/` and `UFC_crime/{Abuse,Normal}`.

## CURRENT INFERENCE PIPELINE (assumed)
Offline sliding windows over a whole video file, then smoothing/cooldown. Webcam mode was a print loop. No RTSP, no
multi-camera, no threading, no latency accounting.

## CURRENT DEPENDENCIES
torch, torchvision, opencv, ultralytics, scikit-learn, pandas, matplotlib, seaborn, tqdm, `laya`.

## UNUSED COMPONENTS (to verify with the audit script)
- Object features were computed but, in the earlier notebook, **never concatenated into the model** (only a toggle in the
  ablation harness re-scored the same model), so "+Objects" was effectively a no-op. If your `yolo_features.py`/`fusion.py`
  came from that, they are dead weight for training.
- `pandas`/`seaborn`/`matplotlib` are only needed for plots.

## COLAB-SPECIFIC COMPONENTS (to verify)
`/content/...` paths, `!pip install` lines, `google.colab` imports if any survived the split.

## PROBLEMS
1. **Laya install was probably wrong.** The earlier notebook ran `pip install laya`; the PyPI package is the text model and
   lacks `load_vlm`. Laya Vision must come from the fork (`requirements-laya.txt`). If your `laya.py` failed silently it
   would just have disabled verification.
2. Per-clip video decoding every epoch is slow/expensive; features from frozen encoders should be cached once.
3. Detection results were not tracked, so no motion/interaction features.
4. Laya fusion was an untrained weighted average; Laya was never seen in training.
5. Video-level labels applied to every window of a long video (weak/noisy labels, especially UCF-Crime).
6. Fixed offline design: no rate control, no queues, no drop-stale policy, no multi-camera.
7. Hard-coded class list; labels from different datasets not reconciled explicitly.
8. Compute-heavy defaults (ResNet-50, 224px, per-frame encoding of overlapping windows -> each frame re-encoded ~8x).

## PROPOSED ARCHITECTURE (implemented in this patch — see README)
Camera sources -> FrameSampler (capture/detect/temporal rates) -> YOLO (batched) -> per-camera tracker -> object/track
features; temporal frames -> visual encoder ONCE per frame -> TemporalBuffer (embeddings + object vectors) -> temporal
encoder -> fusion (temporal + visual + object + optional Laya tokens) -> event/anomaly/confidence heads ->
UnknownEventDetector -> LayaPolicy (selective) -> re-fusion with Laya -> AlertManager (incidents).
