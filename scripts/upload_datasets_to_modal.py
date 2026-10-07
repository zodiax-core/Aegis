"""Upload local datasets/ directory to the Modal Volume 'aegis-data'.

Usage:
    python scripts/upload_datasets_to_modal.py

This script uploads any local dataset folders that are not yet on the volume.
It does NOT re-upload folders that already exist (safe to re-run).

Volume layout after upload:
    /vol/datasets/UFC_crime/Abuse/    Arrest/    Assault/    Burglary/    Normal/
    /vol/datasets/RWF_2000/Fight/     NonFight/   (if populated locally)
"""
from __future__ import annotations

import os
import subprocess
import sys


VOLUME_NAME = "aegis-data"
LOCAL_DATASETS = "datasets"
REMOTE_DATASETS = "datasets"

# Classes to upload for UFC_crime
UFC_CRIME_CLASSES = ["Abuse", "Arrest", "Assault", "Burglary", "Normal"]
RWF_2000_CLASSES = ["Fight", "NonFight"]


def modal_ls(remote_path: str) -> list[str]:
    """List entries in a Modal volume path."""
    result = subprocess.run(
        ["modal", "volume", "ls", VOLUME_NAME, remote_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and not l.startswith("C:\\") and "DeprecationWarning" not in l]
    return lines


def modal_put(local_path: str, remote_path: str) -> bool:
    """Upload a local directory to Modal volume."""
    print(f"  Uploading {local_path} -> {VOLUME_NAME}:{remote_path}")
    result = subprocess.run(
        ["modal", "volume", "put", VOLUME_NAME, local_path, remote_path],
        capture_output=False,
    )
    return result.returncode == 0


def count_files(path: str) -> int:
    """Count video files in directory."""
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.lower().endswith(".mp4"))


def main():
    print("=" * 60)
    print(f"AEGIS Dataset Upload to Modal Volume: {VOLUME_NAME}")
    print("=" * 60)

    # Check existing volume contents
    existing = modal_ls(REMOTE_DATASETS)
    print(f"Already on volume under '{REMOTE_DATASETS}': {existing}")

    uploaded = 0
    skipped = 0
    errors = 0

    datasets = {
        "UFC_crime": UFC_CRIME_CLASSES,
        "RWF_2000": RWF_2000_CLASSES,
    }

    for ds_name, classes in datasets.items():
        local_ds = os.path.join(LOCAL_DATASETS, ds_name)
        if not os.path.isdir(local_ds):
            print(f"\nSkipping {ds_name}: not found locally at {local_ds}")
            continue

        print(f"\n{ds_name}:")
        # Check which classes already exist on volume
        existing_classes = modal_ls(f"{REMOTE_DATASETS}/{ds_name}")
        existing_class_names = set(
            os.path.basename(p.rstrip("/")) for p in existing_classes
        )

        for cls in classes:
            local_cls = os.path.join(local_ds, cls)
            remote_cls = f"{REMOTE_DATASETS}/{ds_name}/{cls}"

            if not os.path.isdir(local_cls):
                n = 0
                print(f"  {cls}: not found locally - skipping")
                skipped += 1
                continue

            n = count_files(local_cls)
            if cls in existing_class_names:
                print(f"  {cls}: already on volume ({n} local videos) - skipping")
                skipped += 1
                continue

            print(f"  {cls}: uploading {n} videos...")
            ok = modal_put(local_cls, remote_cls)
            if ok:
                uploaded += 1
                print(f"  {cls}: upload DONE")
            else:
                errors += 1
                print(f"  {cls}: upload FAILED")

    print("\n" + "=" * 60)
    print(f"Upload complete: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    print(f"\nVerify with: modal volume ls {VOLUME_NAME} datasets")
    print("=" * 60)

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
