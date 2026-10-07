"""Static audit of a repo: imports, what is defined, what is actually referenced, Colab/Modal specifics.
Writes ARCHITECTURE_AUDIT_AUTO.md. Facts only (AST + text search); it does not run your code.
  python scripts/audit_repo.py --root . --out ARCHITECTURE_AUDIT_AUTO.md"""
import argparse, ast, os, re, sys

HINTS = {
    "colab": [r"google\.colab", r"/content/", r"^\s*!pip", r"drive\.mount"],
    "modal": [r"^\s*import modal", r"modal\.App|modal\.Stub|modal\.Image|@app\.function"],
    "yolo": [r"ultralytics", r"\bYOLO\("],
    "laya": [r"\blaya\b"],
    "training": [r"optimizer|\.backward\(\)|loss\.backward"],
}
CFG = re.compile(r"\b(batch_size|clip_length|frame_stride|image_size|epochs|learning_rate|lr|backbone_name|weight_decay)\b\s*[:=]\s*([^\n#,)]+)")


def files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in {"__pycache__", ".git", "datasets", "checkpoints", "outputs", "venv", ".venv", "node_modules"}]
        for f in fn:
            if f.endswith((".py", ".ipynb", ".txt", ".md")):
                yield os.path.join(dp, f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", default="ARCHITECTURE_AUDIT_AUTO.md")
    a = ap.parse_args()
    py = [f for f in files(a.root) if f.endswith(".py")]
    text = {f: open(f, errors="ignore").read() for f in py}
    defs, imports, out = {}, {}, []
    for f, s in text.items():
        try:
            t = ast.parse(s)
        except SyntaxError as e:
            out.append(f"- SYNTAX ERROR {f}: {e}"); continue
        imports[f] = sorted({(n.module or "") .split(".")[0] if isinstance(n, ast.ImportFrom) else al.name.split(".")[0]
                             for n in ast.walk(t) if isinstance(n, (ast.Import, ast.ImportFrom)) for al in (n.names if isinstance(n, ast.Import) else [n])})
        defs[f] = [n.name for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    alltext = "\n".join(text.values())
    R = ["# ARCHITECTURE_AUDIT_AUTO (generated, facts only)\n"]
    R.append("## Files\n")
    for f in sorted(text):
        R.append(f"### {os.path.relpath(f, a.root)}  ({len(text[f].splitlines())} lines)")
        R.append(f"- imports: {', '.join(i for i in imports.get(f, []) if i) or '-'}")
        tags = [k for k, pats in HINTS.items() if any(re.search(p, text[f], re.M) for p in pats)]
        R.append(f"- signals: {', '.join(tags) or '-'}")
        unused = [d for d in defs.get(f, []) if len(re.findall(rf"\b{re.escape(d)}\b", alltext)) <= 1]
        R.append(f"- defined but never referenced anywhere in repo: {', '.join(unused) or '-'}")
        cfg = {m.group(1): m.group(2).strip() for m in CFG.finditer(text[f])}
        if cfg: R.append(f"- hyperparameter-looking assignments: {cfg}")
        R.append("")
    R.append("## Imported but possibly unused (import present, no other mention of the name)\n")
    for f, s in text.items():
        try: t = ast.parse(s)
        except SyntaxError: continue
        for n in ast.walk(t):
            names = [(al.asname or al.name).split(".")[0] for al in n.names] if isinstance(n, (ast.Import, ast.ImportFrom)) else []
            for nm in names:
                if nm != "*" and len(re.findall(rf"\b{re.escape(nm)}\b", s)) <= 1:
                    R.append(f"- {os.path.relpath(f, a.root)}: `{nm}`")
    open(a.out, "w").write("\n".join(R) + "\n" + "\n".join(out))
    print("wrote", a.out)

main()
