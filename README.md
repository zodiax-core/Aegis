# AEGIS

### Real-time video anomaly detection with spatial context, temporal reasoning, and actionable alerts.

AEGIS is a modular research and inference stack for turning live or recorded video into
structured event decisions. It combines visual embeddings, object/track features, temporal
attention, optional semantic verification, and an alert state machine in one configurable
pipeline.

> **Status:** active research code. The core test suite passes, but production decisions still
> require validation on the target cameras, hardware, and deployment data.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.2+">
  <img src="https://img.shields.io/badge/license-not%20specified-lightgrey" alt="License not specified">
</p>

## Why AEGIS?

Most video anomaly prototypes stop at frame classification. AEGIS is designed around the
operational loop:

- **See:** visual encoders and YOLO-based perception extract appearance, objects, tracks, and motion.
- **Remember:** a bounded temporal buffer and attention model reason over short video windows.
- **Decide:** event, anomaly, confidence, and unknown-event heads produce explicit signals.
- **Verify:** Laya Vision can be called selectively through a rate-limited semantic policy.
- **Respond:** alert incidents move through `OPEN → UPDATE → CLOSE` with cooldown merging.
- **Scale:** the same engine supports files, webcams, RTSP streams, mocks, and asynchronous
  multi-camera pipelines.

## Architecture

```mermaid
flowchart LR
    S[Video source<br/>file · webcam · RTSP · mock] --> P[Perception<br/>YOLO · tracking · features]
    P --> B[Temporal buffer<br/>sampling · bounded queues]
    B --> V[Visual encoder]
    B --> T[Temporal encoder]
    P --> F[Feature fusion]
    V --> F
    T --> F
    F --> H[Event heads<br/>class · anomaly · confidence]
    H --> Q{Semantic policy}
    Q -->|selective| L[Laya Vision]
    H --> A[Alert manager<br/>open · update · close]
    L --> A
```

### Repository map

| Area | Responsibility |
| --- | --- |
| [`aegis/camera/`](aegis/camera/) | File, webcam, RTSP, mock, and latest-frame sources |
| [`aegis/perception/`](aegis/perception/) | Detection, tracking, object schemas, and motion features |
| [`aegis/pipeline/`](aegis/pipeline/) | Sampling, camera state, engine orchestration, and async multi-camera execution |
| [`aegis/model/`](aegis/model/) | Visual/temporal encoders, fusion, heads, and checkpoint loading |
| [`aegis/semantic/`](aegis/semantic/) | Optional Laya adapter, questions, and call policy |
| [`aegis/alerts/`](aegis/alerts/) | Incident lifecycle, cooldowns, and event policy |
| [`aegis/data/`](aegis/data/) | Manifests, label mapping, splitting, validation, caching, and preprocessing |
| [`aegis/training/`](aegis/training/) | Dry-run and configurable training entry points |
| [`aegis/export/`](aegis/export/) | ONNX export and runtime parity checks |
| [`web/`](web/) | Flask dashboard for uploading and analyzing videos |
| [`configs/`](configs/) | Default, baseline, MIL, Orion, and label-mapping configurations |
| [`tests/`](tests/) | Unit and integration coverage for data, model, pipeline, and training |

## Quick start

### 1. Install

Use a Python 3.10+ environment. Install a PyTorch build compatible with your machine first
(CUDA users should follow the [official PyTorch selector](https://pytorch.org/get-started/locally/)):

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Laya Vision is optional and intentionally isolated:

```bash
pip install -r requirements-laya.txt
```

### 2. Verify the installation

```bash
python -m pytest tests -q
```

The suite is designed to exercise the weight-free paths where possible. GPU, camera, RTSP,
TensorRT, and external model downloads should be validated separately on the deployment
machine.

### 3. Run inference

```bash
# Recorded video
python -m aegis.infer \
  --source path/to/video.mp4 \
  --checkpoint path/to/aegis_best.pt \
  --out outputs/demo

# Synthetic source for a hardware-free smoke test
python -m aegis.infer --source mock --checkpoint path/to/aegis_best.pt --out outputs/mock

# Live webcam
python -m aegis.live --source 0

# RTSP
python -m aegis.infer --source rtsp://user:password@camera/stream \
  --checkpoint path/to/aegis_best.pt \
  --out outputs/rtsp
```

Inference writes structured artifacts such as `events.json`, `metrics.json`, and an annotated
video to the selected output directory. Checkpoints and datasets are deliberately not committed
to this repository; provide their paths locally or through your deployment artifact store.

## Web dashboard

The included Flask application provides a browser workflow for uploading a video, selecting an
available checkpoint, and inspecting the predicted class, anomaly score, confidence, and timeline.

```bash
python web/app.py
```

Then open <http://127.0.0.1:5000>. Put local `.pt`/`.pth` checkpoints in `web/models/`; that
directory is ignored by Git so large weights stay out of source control.

## Training and configuration

AEGIS keeps runtime behavior in YAML rather than scattering constants through the code:

```bash
# Validate the training/data plumbing without a real dataset
python -m aegis.training.train --dry-run --out outputs/dry-run

# Start a configured experiment
python -m aegis.training.train \
  --config configs/default.yaml \
  --out runs/experiment-01
```

Useful starting points:

- `configs/default.yaml` — general-purpose baseline
- `configs/baseline.yaml` — baseline experiment
- `configs/mil.yaml` — multiple-instance learning configuration
- `configs/orion.yaml` — extended pose/contrastive experiment
- `configs/label_mapping.yaml` — explicit dataset label normalization

The `FAST`, `BALANCED`, and `ACCURATE` runtime policies trade sampling cost for coverage.
Benchmark before choosing a policy:

```bash
python -m aegis.benchmark --policy all --seconds 20
```

## Data and model artifacts

This repository contains source code and lightweight configuration, not a redistribution of
third-party datasets or trained weights. The following local paths are ignored by design:

```text
datasets/     cache/        cache_test/
checkpoints/  runs/         outputs/
*.pt          *.onnx        *.engine
```

Review each dataset's license and terms before use. For reproducible experiments, keep the
dataset manifest, label mapping, split manifest, checkpoint metadata, and training configuration
together with the experiment record.

## Design notes

- Frames are encoded once and reused from the temporal buffer instead of re-encoding every
  overlapping window.
- Object and track features are model inputs, not merely display metadata.
- Semantic verification is policy-controlled and rate-limited per camera.
- Unknown-event scoring is a heuristic and should be calibrated against validation data.
- Live pipelines use bounded queues so latency remains bounded under load.
- Explicit label mapping turns unknown labels into errors instead of silent class merges.

More detailed engineering and research context is available in
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md),
[`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md),
[`PROJECT_DEEP_DIVE_ANALYSIS.md`](PROJECT_DEEP_DIVE_ANALYSIS.md), and
[`LITERATURE_REVIEW_20_PAPERS.md`](LITERATURE_REVIEW_20_PAPERS.md).

## Roadmap

- [ ] Add a reproducible CI workflow for CPU-only tests
- [ ] Calibrate anomaly/unknown thresholds on held-out camera data
- [ ] Add a documented model/artifact release workflow
- [ ] Expand deployment examples for GPU, ONNX Runtime, and TensorRT
- [ ] Add authenticated multi-camera dashboard deployment guidance

## Contributing

Small, focused pull requests are welcome. Before opening one:

1. Keep generated artifacts, datasets, credentials, and model weights out of commits.
2. Add or update tests for behavior changes.
3. Run `python -m pytest tests -q`.
4. Explain hardware, dataset, and checkpoint assumptions in the pull request.

## License

No license has been selected for this repository yet. Until one is added, all rights are
reserved by the copyright holder.
