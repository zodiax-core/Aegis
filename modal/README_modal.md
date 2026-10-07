# Modal

`modal/train_modal.py` is intentionally NOT modified in this patch. When you're ready, the training entrypoint to
call inside your Modal function (L40S) is:

    python -m aegis.training.train --config configs/default.yaml --out /vol/runs/exp1

(preprocessing + feature caching + split + leakage check + training + test eval in one command; mount your dataset
and cache dir on a Modal Volume and point `data.datasets[*].path` / `data.cache_dir` at it.)
Local development never imports modal. Dry-run plumbing check (no data, CPU ok):
`python -m aegis.training.train --dry-run --out /tmp/x`
