"""Moves your OLD flat modules into aegis/legacy/ so they survive as a baseline and don't collide with the new
packages (aegis/model.py would be shadowed by aegis/model/). Dry-run by default.

  python scripts/migrate_legacy.py            # show what would happen
  python scripts/migrate_legacy.py --apply    # do it (backups: *.bak, requirements.txt.old)

Run this from your repo root BEFORE copying the new files over."""
import argparse, os, re, shutil, sys

OLD = ["config", "dataset", "evaluation", "fusion", "inference", "laya", "model", "training", "yolo_features"]


def rewrite(src: str) -> str:
    for n in OLD:
        src = re.sub(rf"from\s+aegis\.{n}\s+import", f"from aegis.legacy.{n} import", src)
        src = re.sub(rf"from\s+\.{n}\s+import", f"from aegis.legacy.{n} import", src)
        src = re.sub(rf"import\s+aegis\.{n}\b", f"import aegis.legacy.{n}", src)
    src = re.sub(r"from\s+aegis\s+import\s+([\w, ]+)", lambda m: _split_import(m.group(1)), src)
    return src


def _split_import(names: str) -> str:
    parts = [p.strip() for p in names.split(",")]
    old = [p for p in parts if p in OLD]
    keep = [p for p in parts if p not in OLD]
    lines = ([f"from aegis import {', '.join(keep)}"] if keep else []) + [f"from aegis.legacy import {p}" for p in old]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    pkg = os.path.join(root, "aegis")
    if not os.path.isdir(pkg):
        sys.exit(f"no aegis/ package in {root}")
    leg = os.path.join(pkg, "legacy")
    moves = [(os.path.join(pkg, f"{n}.py"), os.path.join(leg, f"{n}.py")) for n in OLD if os.path.isfile(os.path.join(pkg, f"{n}.py"))]
    if not moves:
        print("nothing to migrate (already done?)")
    for s, d in moves:
        print(("MOVE " if a.apply else "would move ") + os.path.relpath(s, root) + " -> " + os.path.relpath(d, root))
    targets = [d for _, d in moves] + [os.path.join(root, "modal", "train_modal.py")]
    for t in targets:
        print(("REWRITE imports in " if a.apply else "would rewrite imports in ") + os.path.relpath(t, root))
    print(("BACKUP " if a.apply else "would back up ") + "requirements.txt -> requirements.txt.old")
    if not a.apply:
        print("\n(dry run) re-run with --apply")
        return
    os.makedirs(leg, exist_ok=True)
    open(os.path.join(leg, "__init__.py"), "a").close()
    for s, d in moves:
        shutil.move(s, d)
    for t in targets:
        if os.path.isfile(t):
            src = open(t).read()
            new = rewrite(src)
            if new != src:
                shutil.copy(t, t + ".bak")
                open(t, "w").write(new)
    req = os.path.join(root, "requirements.txt")
    if os.path.isfile(req):
        shutil.copy(req, req + ".old")
    print("done. Now copy the new files over the repo and run: python -m pytest tests -q")


main()
