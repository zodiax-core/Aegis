"""AEGIS Modal Training - Complete file for GPU training on Modal cloud.

Usage:
  # 1. GPU smoke test (verify GPU + AEGIS imports + tiny forward pass):
  modal run modal/train_modal.py::gpu_smoke_test

  # 2. Tiny training smoke test (3 steps, no full training):
  modal run modal/train_modal.py::training_smoke_test

  # 3. FULL TRAINING (expensive - starts a real training run):
  modal run modal/train_modal.py::run_training

Dataset is read from Modal Volume "aegis-data" at /vol/datasets/.
Cache and checkpoints are also persisted to the volume.

Volume layout:
  /vol/
    datasets/
      UFC_crime/
        Abuse/    Arrest/    Assault/    Burglary/    Normal/
      RWF_2000/   (if populated)
    cache/
    checkpoints/
    outputs/
"""
from __future__ import annotations

import modal

# On Windows clients, direct worker input-plane gRPC channels frequently time out.
# Forcing _input_plane_url = None ensures invocations route through Modal's control plane.
try:
    import modal._functions
    _orig_call_fn = modal._functions._Function._call_function
    async def _safe_call_fn(self, args, kwargs):
        self._input_plane_url = None
        return await _orig_call_fn(self, args, kwargs)
    modal._functions._Function._call_function = _safe_call_fn
except Exception:
    pass

# ------------------------------------------------------------------
# Persistent volume (datasets, cache, checkpoints survive between runs)
# ------------------------------------------------------------------
vol = modal.Volume.from_name("aegis-data", create_if_missing=False)
VOL_MOUNT = "/vol"

# ------------------------------------------------------------------
# Image: Python 3.11, CUDA 12.4, all AEGIS dependencies
# ------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        # Core: install CUDA build of torch explicitly
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy",
        "opencv-python-headless",  # headless for servers (no display needed)
        "pyyaml",
        "psutil",
        "scikit-learn",
        "ultralytics",           # YOLO11 + BYTETracker
        "lap>=0.5.12",           # Linear assignment for ByteTrack
        "Pillow",
        "onnx",
        "modal",
    )
    .pip_install(
        # Laya Vision fork (installs transformers, huggingface_hub, safetensors, tokenizers)
        "git+https://github.com/r33drichards/laya-vision",
        "huggingface_hub",
    )
    # Copy the AEGIS project source into the image
    .add_local_dir("aegis", remote_path="/app/aegis")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_file("yolo11n.pt", remote_path="/app/yolo11n.pt")
    # yolo11n-pose.pt is optional; download_pose_weights() fetches it into the volume.
    # We add a sentinel so the image build never fails if the file is absent locally.
)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = modal.App("aegis-training")


# ------------------------------------------------------------------
# DATA SETUP HELPERS (run once; CPU-only, very cheap)
# ------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOL_MOUNT: vol},
    timeout=10800,  # 3 hour ceiling — extraction of 2000+ files takes time
)
def download_rwf2000():
    """Download and extract RWF-2000 from Hugging Face into /vol/datasets/RWF_2000/.

    The HuggingFace repo (DanJoshua/RWF-2000) ships the dataset as a single
    RWF-2000.tar.gz archive.  This function:
      1. Downloads the archive (~12 GB compressed) via hf_hub_download
      2. Extracts it in-place into /vol/datasets/RWF_2000/
      3. Deletes the archive to reclaim volume space
      4. Commits the volume so the files survive container shutdown

    Run once before orion training:
        modal run modal/train_modal.py::download_rwf2000

    Cost: ~$0.05 CPU time (I/O bound, ~15-30 min at datacenter speeds).
    The volume persists across all future training runs.
    """
    import sys, os, tarfile, shutil
    sys.path.insert(0, "/app")
    from huggingface_hub import hf_hub_download

    dest = os.path.join(VOL_MOUNT, "datasets", "RWF_2000")
    os.makedirs(dest, exist_ok=True)

    # Count existing video files
    existing = sum(
        1 for _, _, files in os.walk(dest) for f in files
        if f.lower().endswith((".avi", ".mp4"))
    )
    if existing >= 1800:   # allow small margin; full set has 2000
        print(f"RWF-2000 already extracted ({existing} video files). Skipping.")
        return
    if existing > 0:
        print(f"Partial extraction detected ({existing}/2000 files). Re-extracting...")

    # Step 1: Download the tar.gz archive from HuggingFace
    archive_dest = os.path.join(dest, "RWF-2000.tar.gz")
    if not os.path.exists(archive_dest):
        print("Downloading RWF-2000.tar.gz from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id="DanJoshua/RWF-2000",
            filename="RWF-2000.tar.gz",
            repo_type="dataset",
            local_dir=dest,
        )
        print(f"Archive saved to: {downloaded}")
        archive_dest = downloaded
    else:
        print(f"Archive already on volume at {archive_dest}, skipping download.")

    # Step 2: Extract — with long-filename sanitisation
    print(f"Extracting {archive_dest} -> {dest} ...")
    import hashlib

    def _safe_name(name: str, max_bytes: int = 200) -> str:
        """Truncate a filename so its UTF-8 encoding fits within max_bytes,
        keeping the extension and appending a short hash for uniqueness."""
        base, ext = os.path.splitext(name)
        ext_bytes = ext.encode("utf-8")
        # If even the extension is too long, hash the whole thing
        if len(ext_bytes) >= max_bytes:
            return hashlib.md5(name.encode("utf-8")).hexdigest() + ".avi"
        avail = max_bytes - len(ext_bytes)
        base_bytes = base.encode("utf-8")
        if len(base_bytes) <= avail:
            return name   # already short enough
        # Truncate base to avail-8 bytes (leave 8 bytes for hash suffix)
        trunc = base_bytes[:avail - 8].decode("utf-8", errors="ignore")
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return trunc + "_" + h + ext

    with tarfile.open(archive_dest, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)
        print(f"  {total} entries in archive.")
        skipped, extracted, errors = 0, 0, 0
        for i, m in enumerate(members):
            # Fix path: rebuild member name with safe per-component filenames
            parts = m.name.replace("\\", "/").split("/")
            safe_parts = [_safe_name(p) if p else p for p in parts]
            m.name = "/".join(safe_parts)

            target = os.path.join(dest, m.name)
            # Skip already-extracted files (resume support)
            if m.isfile() and os.path.exists(target):
                skipped += 1
                continue
            try:
                tar.extract(m, dest)
                extracted += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARNING skipping {m.name}: {e}")
            if i % 200 == 0:
                print(f"  Processed {i}/{total} (extracted={extracted} skipped={skipped} errors={errors})")
    print(f"Extraction complete: extracted={extracted} skipped={skipped} errors={errors}")

    # Step 3: Clean up archive to reclaim volume space
    try:
        os.remove(archive_dest)
        print(f"Deleted archive {archive_dest} to save space.")
    except Exception as e:
        print(f"WARNING: Could not delete archive: {e}")

    # Step 4: Report
    final = sum(
        1 for _, _, files in os.walk(dest) for f in files
        if f.lower().endswith((".avi", ".mp4"))
    )
    print(f"Done. {final} video files in {dest}")

    print("Volume committing...")
    vol.commit()
    print("Volume committed.")


@app.function(
    image=image,
    volumes={VOL_MOUNT: vol},
    timeout=300,
)
def download_pose_weights():
    """Pre-download yolo11n-pose.pt into /vol/weights/ so training containers
    never hit the network during feature extraction.

    Run once before orion training:
        modal run modal/train_modal.py::download_pose_weights

    Cost: ~$0.01 CPU time.  Weight is ~6 MB.
    """
    import sys, os, shutil, glob
    sys.path.insert(0, "/app")

    dest = os.path.join(VOL_MOUNT, "weights", "yolo11n-pose.pt")
    if os.path.exists(dest):
        print(f"yolo11n-pose.pt already on volume at {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("Downloading yolo11n-pose.pt via ultralytics...")
    from ultralytics import YOLO
    YOLO("yolo11n-pose.pt")   # triggers ultralytics auto-download to ~/.config/Ultralytics/

    # Find where ultralytics cached it
    candidates = (
        glob.glob(os.path.expanduser("~/.config/Ultralytics/yolo11n-pose.pt"))
        + glob.glob("/root/.config/Ultralytics/yolo11n-pose.pt")
        + glob.glob("/root/yolo11n-pose.pt")
        + glob.glob("/tmp/yolo11n-pose.pt")
    )
    if not candidates:
        # Search common ultralytics cache locations
        for root, dirs, files in os.walk("/root"):
            for f in files:
                if f == "yolo11n-pose.pt":
                    candidates.append(os.path.join(root, f))
            if candidates:
                break
    if not candidates:
        print("WARNING: Could not find cached yolo11n-pose.pt. Training will download it at runtime.")
        return

    shutil.copy(candidates[0], dest)
    print(f"Saved yolo11n-pose.pt to {dest}  ({os.path.getsize(dest)//1024} KB)")
    vol.commit()
    print("Volume committed.")


# GPU SMOKE TEST (Phase 12)
# Verifies: CUDA, GPU name/VRAM, AEGIS imports, YOLO, tiny forward pass
# ------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L40S",
    volumes={VOL_MOUNT: vol},
    timeout=300,
)
def gpu_smoke_test():
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    import torch

    print("=" * 60)
    print("AEGIS GPU SMOKE TEST")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA is NOT available on this Modal GPU!"
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.1f} GB")

    # AEGIS imports
    from aegis.config import load_config
    from aegis.model.aegis_model import AegisModel
    from aegis.model.visual_encoder import VisualEncoder
    from aegis.perception.yolo_detector import build_detector
    from aegis.perception.tracker import make_tracker
    from aegis.data.label_mapper import LabelMapper
    print("AEGIS imports: OK")

    # YOLO
    cfg = load_config("configs/default.yaml")
    det = build_detector(cfg.perception, device="cuda")
    print(f"YOLO detector: {det.name}")

    # Tracker
    tracker = make_tracker(cfg.perception, fps=30.0)
    print(f"Tracker: {tracker.__class__.__name__}")

    # Laya import check (no inference - model weights may not be cached yet)
    try:
        import laya
        print(f"Laya package: {laya.__version__ if hasattr(laya, '__version__') else 'installed'}, load_vlm: {hasattr(laya, 'load_vlm')}")
    except ImportError as e:
        print(f"WARNING: Laya not importable: {e}")

    # Volume check - verify dataset is accessible
    dataset_root = f"{VOL_MOUNT}/datasets/UFC_crime"
    if os.path.isdir(dataset_root):
        classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        total_files = sum(
            len([f for f in os.listdir(os.path.join(dataset_root, c)) if f.endswith(".mp4")])
            for c in classes if os.path.isdir(os.path.join(dataset_root, c))
        )
        print(f"Dataset on volume: UFC_crime classes={classes}, total videos={total_files}")
    else:
        print(f"WARNING: Dataset not found at {dataset_root} - run dataset upload first!")

    # Tiny AEGIS forward pass on GPU
    print("\n--- Tiny forward pass on CUDA ---")
    names = ["NORMAL", "FIGHT", "ABUSE", "ARREST", "ASSAULT"]
    cfg.model.class_names = names
    ve = VisualEncoder(cfg.model.visual_backbone, cfg.model.visual_pretrained, cfg.sampling.image_size)
    model = AegisModel(cfg.model, ve.out_dim, class_names=names).to("cuda")
    model.eval()

    T = cfg.sampling.clip_length
    from aegis.perception.features import OBJ_FEAT_DIM
    from aegis.semantic.laya_questions import LAYA_FEAT_DIM
    vis = torch.randn(2, T, ve.out_dim, device="cuda")
    obj = torch.randn(2, T, OBJ_FEAT_DIM, device="cuda")
    laya_feat = torch.zeros(2, LAYA_FEAT_DIM, device="cuda")
    laya_mask = torch.tensor([False, False], device="cuda")

    with torch.no_grad():
        out = model(vis, obj, laya_feat, laya_mask)
    print(f"Forward pass outputs: {list(out.keys())}")
    print(f"event_logits shape: {out['event_logits'].shape}")
    torch.cuda.synchronize()
    print("GPU forward pass: PASS")

    print("=" * 60)
    print("GPU SMOKE TEST: ALL PASS")
    print("=" * 60)


# ------------------------------------------------------------------
# TRAINING SMOKE TEST
# Runs 3 real training steps on 3 real videos from the volume.
# Does NOT start full training.
# ------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L40S",
    volumes={VOL_MOUNT: vol},
    timeout=600,
)
def training_smoke_test():
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    import torch
    from aegis.config import load_config
    from aegis.data.label_mapper import LabelMapper
    from aegis.data.manifest import scan_dataset, merge_manifests
    from aegis.data.splitter import split_videos
    from aegis.data.cache import FeatureCache
    from aegis.data.dataset import FeatureWindowDataset
    from aegis.data.preprocessing import build_feature_cache
    from aegis.model.aegis_model import AegisModel
    from aegis.model.visual_encoder import VisualEncoder
    from aegis.perception.yolo_detector import build_detector
    from aegis.training.trainer import compute_loss
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Point datasets at the Modal volume
    cfg = load_config("configs/default.yaml")
    for d in cfg.data.datasets:
        d.path = os.path.join(VOL_MOUNT, d.path)  # e.g. /vol/datasets/UFC_crime
    cfg.data.cache_dir = os.path.join(VOL_MOUNT, "cache")

    mapper = LabelMapper.from_yaml(cfg.data.label_mapping)

    # Scan only datasets that exist on the volume
    manifests = []
    for d in cfg.data.datasets:
        if os.path.isdir(d.path):
            print(f"Scanning {d.name} at {d.path}")
            m = scan_dataset(d.name, d.path, mapper, d.license, d.version, d.source, probe=True)
            manifests.append(m)
            print(f"  {m.num_videos} videos, classes: {m.classes}")
        else:
            print(f"WARNING: {d.name} not on volume ({d.path}) - skipping")

    assert manifests, "No datasets found on the Modal volume!"
    records = merge_manifests(manifests)
    split = split_videos(records, cfg.data.split_fractions, cfg.data.seed)
    classes = mapper.class_list([r.label for r in records])
    print(f"Classes: {classes}")
    print(f"Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

    # Use only first 3 training videos for smoke test
    test_recs = [r for r in records if r.video_id in split["train"][:3]]
    det = build_detector(cfg.perception, device)
    ve = VisualEncoder(cfg.model.visual_backbone, cfg.model.visual_pretrained, cfg.sampling.image_size).to(device)
    cache = FeatureCache(cfg.data.cache_dir)
    keys = build_feature_cache(test_recs, cfg, det, ve, cache, lambda i, n: print(f"  features {i}/{n}"))

    ds = FeatureWindowDataset(test_recs, keys, cache, classes, cfg.sampling.clip_length, cfg.data.window_stride, mapper.normal_label)
    print(f"Dataset windows: {len(ds)}")
    assert len(ds) > 0, "No windows - feature extraction failed!"

    dl = DataLoader(ds, batch_size=min(4, len(ds)))
    batch = next(iter(dl))

    cfg.model.class_names = classes
    model = AegisModel(cfg.model, ve.out_dim, class_names=classes).to(device)
    model.train()
    model.set_trainable(True, True, True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    for step in range(3):
        vis = batch["vis"].to(device)
        obj = batch["obj"].to(device)
        laya = batch["laya"].to(device)
        mask = batch["laya_mask"].to(device)
        y = batch["event"].to(device)
        anom = batch["anomaly"].to(device)
        out = model(vis, obj, laya, mask)
        loss = compute_loss(out, y, anom, device)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Step {step+1}: loss={loss.item():.4f}")

    print("\nTRAINING SMOKE TEST: PASS (3 steps on real data)")


# ------------------------------------------------------------------
# FULL TRAINING
# Usage:
#   # Baseline (window-mode, the control run):
#   modal run modal/train_modal.py::run_training --run-name baseline_tier0 --config configs/baseline.yaml
#
#   # MIL (bag-mode, should beat baseline):
#   modal run modal/train_modal.py::run_training --run-name mil_tier1 --config configs/mil.yaml
#
# Results saved to /vol/outputs/<run_name>/
# ------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L40S",
    volumes={VOL_MOUNT: vol},
    timeout=86400,  # 24 hours max
)
def run_training(run_name: str = "baseline_tier0", config: str = "configs/baseline.yaml"):
    import sys, os, yaml, subprocess
    sys.path.insert(0, "/app")
    os.chdir("/app")

    out_dir = os.path.join(VOL_MOUNT, "outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load whichever config was requested (baseline.yaml or mil.yaml)
    cfg_path = os.path.join("/app", config)
    print(f"Loading config: {cfg_path}")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    # Patch dataset paths and cache dir to use the Modal Volume.
    # Drop any dataset whose folder is missing on the volume so scan_dataset never
    # crashes with FileNotFoundError (e.g. RWF_2000 not uploaded yet).
    patched_datasets = []
    for ds in raw.get("data", {}).get("datasets", []):
        local_rel = ds["path"]                              # e.g. "datasets/UFC_crime"
        vol_path = os.path.join(VOL_MOUNT, local_rel)      # /vol/datasets/UFC_crime
        ds["path"] = vol_path
        exists = os.path.isdir(vol_path)
        n = sum(1 for _, _, files in os.walk(vol_path) for f in files
                if f.lower().endswith((".mp4", ".avi"))) if exists else 0
        status = f"OK ({n} videos)" if exists else "MISSING - skipped"
        print(f"  {ds['name']}: {vol_path} -> {status}")
        if exists and n > 0:
            patched_datasets.append(ds)
    if not patched_datasets:
        raise RuntimeError("No datasets found on the Modal Volume. Run upload_datasets_to_modal.py first.")
    raw["data"]["datasets"] = patched_datasets
    raw.setdefault("data", {})["cache_dir"] = os.path.join(VOL_MOUNT, "cache")
    raw.setdefault("data", {})["output_dir"] = os.path.join(VOL_MOUNT, "outputs")

    # If config uses pose weights, symlink them from the volume into /app/ so YOLO finds them
    perception_weights = raw.get("perception", {}).get("weights", "")
    if perception_weights and "-pose" in perception_weights:
        vol_weight = os.path.join(VOL_MOUNT, "weights", perception_weights)
        app_weight = os.path.join("/app", perception_weights)
        if os.path.exists(vol_weight) and not os.path.exists(app_weight):
            os.symlink(vol_weight, app_weight)
            print(f"Symlinked pose weights: {vol_weight} -> {app_weight}")
        elif not os.path.exists(vol_weight):
            print(f"WARNING: {vol_weight} not on volume. Training will attempt auto-download.")

    patched_cfg = os.path.join(out_dir, "config_modal.yaml")
    with open(patched_cfg, "w") as f:
        yaml.safe_dump(raw, f)
    print(f"Patched config -> {patched_cfg}")
    print(f"Output dir     -> {out_dir}")

    # Stream stdout/stderr live so errors appear in Modal logs, not as a silent exit-1
    result = subprocess.run(
        [sys.executable, "-m", "aegis.training.train",
         "--config", patched_cfg,
         "--out", out_dir],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training subprocess exited with code {result.returncode}. "
            "Scroll up in the Modal logs for the full traceback."
        )

    # Commit all changes to the volume so they persist after the container exits
    vol.commit()
    print(f"\nTraining complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    # Allow direct local testing (will use CPU)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dry-run":
        print("Modal dry-run: importing OK")