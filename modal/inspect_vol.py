"""One-shot Modal script to inspect what's inside /vol/datasets/RWF_2000/"""
import modal

vol = modal.Volume.from_name("aegis-data", create_if_missing=False)
VOL_MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)

app = modal.App("aegis-inspect")

@app.function(image=image, volumes={VOL_MOUNT: vol}, timeout=120)
def inspect_rwf():
    import os

    dest = "/vol/datasets/RWF_2000"
    print(f"=== Contents of {dest} ===")
    if not os.path.exists(dest):
        print("Directory does not exist!")
        return

    for root, dirs, files in os.walk(dest):
        level = root.replace(dest, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files[:10]:  # show first 10 per dir
            fsize = os.path.getsize(os.path.join(root, f))
            print(f"{subindent}{f}  ({fsize // 1024} KB)")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")
        if level >= 3:  # don't go too deep
            break

    # Count by extension
    ext_counts = {}
    total_size = 0
    for root, dirs, files in os.walk(dest):
        for f in files:
            ext = os.path.splitext(f)[1].lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass

    print("\n=== File types ===")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext}: {count} files")
    print(f"\nTotal size: {total_size // (1024*1024)} MB")
