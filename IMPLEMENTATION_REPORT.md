# Implementation report (requested format)

FILES CHANGED: none of yours were edited in place. `scripts/migrate_legacy.py` (run by you) moves old flat modules to
`aegis/legacy/` and rewrites imports in them and in `modal/train_modal.py` (.bak kept); `requirements.txt` replaced (old kept as `.old`).
FILES CREATED: everything else in this zip (see README layout) + tests/ + configs/.
FILES REMOVED: none.

OLD ARCHITECTURE (assumed, see ARCHITECTURE_AUDIT.md): ResNet-50 + Transformer + heads; YOLO/Laya post-hoc; offline only.
NEW ARCHITECTURE: see README.

DATASET CHANGES: manifests (+licence, hash, class distribution, resolutions), explicit label ontology (RWF Fight/UCF Fighting
-> FIGHT, both explicit), source-group splitting, leakage validator (video id, group, path, file-content fingerprint),
deterministic fp16 feature cache, optional temporal annotations, weak video-level labels otherwise.
  NOTE your tree shows `UFC_crime` (sic); mapping has both `UCF_crime` and `UFC_crime`. `configs/default.yaml` points at UFC_crime.

STATUS (verified = ran here on CPU):
- YOLO: adapter written; NOT run (ultralytics not installable here). Falls back to a motion-blob stand-in that is NOT a detector.
- Tracker: IoU tracker verified; BYTETracker adapter NOT run (needs ultralytics); auto-falls back to IoU.
- Laya: adapter + policy written; policy verified; real Laya Vision NOT run (fork + HF weights unavailable) - MockVerifier only.
- Temporal model / fusion / heads: verified forward, Laya-masked == no-Laya, checkpoint round-trip, ONNX parity (<1e-3), dry-run train step.
- Async pipeline: verified with 3 mock cameras (micro-batched detection, no drops at file speed).
LIVE CAMERA: code + headless mock run verified; real webcam NOT tested. RTSP: implemented, NOT tested against a camera.
CUDA: code paths written, NOT tested (test auto-skips). TensorRT: export code written, NOT tested; INT8 deliberately refused.

TESTS: 21 passed, 1 skipped (CUDA) on CPU. Failures found and fixed during development: ONNX export left the model in train
mode (real bug, fixed); leakage validator correctly rejected my own duplicate-content test videos.

CURRENT FPS / LATENCY / GPU MEMORY: CPU sandbox, 1 core, fallback detector, random weights, synthetic 1280x720 - e2e FPS
FAST 174 / BALANCED 87 / ACCURATE 55, but p50/p95 are noisy at 6 s runs. RSS ~870 MB. **These say nothing about your GPU.** GPU memory: not measured.

REMAINING WORK (you, on the target machine): install ultralytics + torch(CUDA); run pytest; run benchmark with real YOLO on GPU;
test webcam + one RTSP camera; install Laya Vision fork, choose+pin revision, test; validate dataset on your real folders
(the label mapper will error on any folder not in the YAML - intended); decide temporal annotations for UCF-Crime.
Known limitations: UnknownEventDetector thresholds and LayaPolicy thresholds are uncalibrated defaults; stage-1 training
masks Laya (no Laya features cached yet), so fusion of Laya is untrained until you add a Laya-feature cache; INT8/TensorRT unvalidated;
MotionBlob fallback must not be used for real results.

TRAINING IS READY: NO - not until pytest + benchmark pass on your GPU box with real YOLO, and one real video/webcam run looks sane.
