"""Stage-1 training on cached features (frozen YOLO + visual encoder). Not run in development.

Two training objectives live here:

* **window mode** - one sample = one temporal window, labelled with its video's (weak) label.
  This is the historical behaviour and is what evaluation always uses.
* **bag mode (MIL)** - one sample = one video, carrying `bag_size` windows. Selected automatically
  when the training dataset sets `is_bag = True` (see `aegis.data.dataset.VideoBagDataset`).
  Required because a video-level label stamped onto every window of a 15-minute video is wrong for
  ~95% of those windows.

Every loss is reported **per term**. A sum of several terms can fall steadily while the term you
actually care about stays at chance, which looks like "training works" and is the single most
common way to ship a model with low loss and near-zero macro-F1.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import AegisConfig
from ..model.aegis_model import AegisModel
from ..model.checkpoint import save_checkpoint
from ..utils import log

# Historical weights, kept so the back-compat `compute_loss` wrapper is bit-identical to the old one.
LEGACY_W = {"event": 1.0, "anomaly": 1.0, "confidence": 0.5}


def validate_train_config(cfg: AegisConfig) -> None:
    t = cfg.train
    if t.train_visual_encoder or t.train_yolo:
        raise ValueError("train_visual_encoder / train_yolo require raw-frame training, which this cached-feature "
                         "trainer does not do. Keep them False for stage 1.")


def loss_weights(cfg: AegisConfig) -> Dict[str, float]:
    t = cfg.train
    return {"event": t.w_event, "anomaly": t.w_anomaly, "confidence": t.w_confidence}


def normal_index(class_names: Sequence[str], normal_label: str = "NORMAL") -> int:
    return class_names.index(normal_label) if normal_label in class_names else 0


# ---- losses -------------------------------------------------------------------------------------

def compute_losses(out: dict, y: torch.Tensor, ya: torch.Tensor, weight: Optional[torch.Tensor] = None,
                   w: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Window-mode objective. Returns (backward-able total, per-term floats for logging).

    The per-term dict is the whole point: `anomaly` is ~81% positive on this dataset and collapses
    to a constant almost immediately, and `confidence` is trained against the event head's own
    `correct` mask, so it is *rewarded* when the classifier is reliably wrong. Both drag the total
    down without the event head learning anything.
    """
    w = w or LEGACY_W
    l_evt = F.cross_entropy(out["event_logits"], y, weight=weight)
    l_an = F.binary_cross_entropy_with_logits(out["anomaly_logit"], ya)
    correct = (out["event_logits"].argmax(-1) == y).float().detach()
    l_conf = F.binary_cross_entropy_with_logits(out["confidence_logit"], correct)
    total = w["event"] * l_evt + w["anomaly"] * l_an + w["confidence"] * l_conf
    # .item() instead of float() avoids "Converting a tensor with requires_grad=True to a scalar"
    return total, {"event": l_evt.item(), "anomaly": l_an.item(), "confidence": l_conf.item(), "total": total.item()}


def compute_loss(out: dict, y: torch.Tensor, ya: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Back-compat scalar loss. Callers: scripts/smoke_test_real.py, modal/train_modal.py."""
    return compute_losses(out, y, ya, w=LEGACY_W)[0]


def compute_mil_losses(out: dict, y: torch.Tensor, ya: torch.Tensor, n_bags: int, k: int, normal: int,
                       weight: Optional[torch.Tensor] = None, w: Optional[Dict[str, float]] = None,
                       mil_k: int = 3, smooth: float = 0.0, sparse: float = 0.0
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Multiple-instance objective over video bags. `out` holds (n_bags*k) flattened windows;
    `y`/`ya` are per-BAG labels (length n_bags).

    The asymmetry is what makes this work on weak labels: **negative bags are clean, positive bags
    are weak.** Every window of a NORMAL video really is normal, so all of them are supervised
    toward normal. Only *some* windows of an ABUSE video are abuse, so only the top-`mil_k` windows
    (ranked by the model's own anomaly score) are supervised toward ABUSE.

    Positive and negative contributions are averaged **per bag**, not per window, so the balance
    does not silently shift when `bag_size` or `mil_k` changes.
    """
    w = w or LEGACY_W
    kk = max(1, min(mil_k, k))
    ev = out["event_logits"].view(n_bags, k, -1)
    an = out["anomaly_logit"].view(n_bags, k)
    cf = out["confidence_logit"].view(n_bags, k)
    pos = ya > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    zero = an.new_zeros(())

    # --- anomaly head: top-k up for positive bags, every window down for negative bags
    l_pos, l_neg = zero, zero
    if n_pos:
        top = an[pos].topk(kk, dim=1).values.mean(1)                     # (P,)
        l_pos = F.binary_cross_entropy_with_logits(top, torch.ones_like(top))
    if n_neg:
        neg = an[~pos]
        l_neg = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    l_an = (n_pos * l_pos + n_neg * l_neg) / max(1, n_pos + n_neg)

    # --- event + confidence heads on the same selected windows
    def _ce(logits, targets, conf):
        l_e = F.cross_entropy(logits, targets, weight=weight)
        ok = (logits.argmax(-1) == targets).float().detach()
        return l_e, F.binary_cross_entropy_with_logits(conf, ok)

    e_pos = e_neg = c_pos = c_neg = zero
    if n_pos:
        ep, yp = ev[pos], y[pos]
        idx = an[pos].topk(kk, dim=1).indices                            # (P,kk)
        g = ep.gather(1, idx.unsqueeze(-1).expand(-1, -1, ep.shape[-1])).reshape(-1, ep.shape[-1])
        e_pos, c_pos = _ce(g, yp.unsqueeze(1).expand(-1, kk).reshape(-1),
                           cf[pos].gather(1, idx).reshape(-1))
    if n_neg:
        en = ev[~pos].reshape(-1, ev.shape[-1])
        e_neg, c_neg = _ce(en, torch.full((en.shape[0],), normal, device=ev.device, dtype=torch.long),
                           cf[~pos].reshape(-1))
    d = max(1, n_pos + n_neg)
    l_evt = (n_pos * e_pos + n_neg * e_neg) / d
    l_conf = (n_pos * c_pos + n_neg * c_neg) / d

    # --- optional Sultani-style regularisers (default off; bag windows are time-ordered)
    s = torch.sigmoid(an)
    l_smooth = ((s[:, 1:] - s[:, :-1]) ** 2).mean() if k > 1 else zero
    l_sparse = s[pos].mean() if n_pos else zero

    total = (w["event"] * l_evt + w["anomaly"] * l_an + w["confidence"] * l_conf
             + smooth * l_smooth + sparse * l_sparse)
    # .item() to avoid requires_grad UserWarning when converting to scalar
    return total, {"event": l_evt.item(), "anomaly": l_an.item(), "confidence": l_conf.item(),
                   "mil_pos": l_pos.item(), "mil_neg": l_neg.item(),
                   "smooth": l_smooth.item(), "sparse": l_sparse.item(), "total": total.item()}


def compute_contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    class_names: Sequence[str],
    margin: float = 0.4,
    hard_pairs: Optional[List[Tuple[str, str]]] = None,
) -> torch.Tensor:
    """Cosine margin loss that penalises confusion classes being too close in latent space.

    For every cross-class pair ``(classA, classB)`` in ``hard_pairs``, any (anchor, negative)
    combination where cosine-similarity > ``(1 - margin)`` incurs a hinge penalty.  The loss is
    zero when the two classes are already well-separated.

    This requires NO extra labelled data — it piggybacks on the same MIL bags already in the
    batch.  Set ``w_contrastive = 0.0`` in the config to disable it entirely (the default,
    backward-compatible with Vesper checkpoints).

    Args:
        z           : (B, D) CLS-token embeddings from TemporalEncoder.
        y           : (B,)  class labels (integer indices into class_names).
        class_names : ordered list of class name strings.
        margin      : minimum required cosine distance between hard class pairs.
        hard_pairs  : list of (classA, classB) tuples to enforce separation on.
                      Defaults to the three confusion pairs: ASSAULT/ARREST, ASSAULT/ABUSE,
                      ABUSE/ARREST.
    """
    if hard_pairs is None:
        hard_pairs = [("ASSAULT", "ARREST"), ("ASSAULT", "ABUSE"), ("ABUSE", "ARREST")]

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    z_norm = F.normalize(z.float(), dim=-1)      # unit vectors, stable under fp16
    loss = z_norm.new_zeros(())
    n_active = 0

    for a_name, b_name in hard_pairs:
        if a_name not in name_to_idx or b_name not in name_to_idx:
            continue
        ma = (y == name_to_idx[a_name])
        mb = (y == name_to_idx[b_name])
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        za = z_norm[ma]           # (nA, D)
        zb = z_norm[mb]           # (nB, D)
        sim = za @ zb.T           # (nA, nB)  cosine similarities in [-1, 1]
        # Hinge: penalise similarities greater than (1 - margin)
        loss = loss + F.relu(sim - (1.0 - margin)).mean()
        n_active += 1

    return loss / max(1, n_active)


# ---- metrics ------------------------------------------------------------------------------------

def _metrics(y, p, prob_anom, anom, names) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
    # Always pass labels=range(len(names)) so sklearn never silently drops a class from the macro
    # average just because it is absent from both y_true and y_pred in this split.  Without this,
    # a model that predicts only ABUSE on a val set where only ABUSE videos appear gets macro_f1=1.0
    # instead of the correct 0.2 (1/5 classes predicted correctly).
    labs = list(range(len(names)))
    m = {"accuracy": float((y == p).mean()),
         "macro_f1": float(f1_score(y, p, average="macro", labels=labs, zero_division=0)),
         "macro_precision": float(precision_score(y, p, average="macro", labels=labs, zero_division=0)),
         "macro_recall": float(recall_score(y, p, average="macro", labels=labs, zero_division=0)),
         "per_class_f1": {n: float(v) for n, v in zip(names, f1_score(y, p, average=None, labels=labs, zero_division=0))}}
    if len(set(anom.tolist())) == 2:
        m["anomaly_roc_auc"] = float(roc_auc_score(anom, prob_anom))
        m["anomaly_pr_auc"] = float(average_precision_score(anom, prob_anom))
    return m


def baselines(y: np.ndarray, names: Sequence[str], seed: int = 0) -> dict:
    """Trivial reference points. Without these a macro-F1 of 0.30 is uninterpretable - on a 5-class
    imbalanced set, always predicting the majority class already scores ~0.11, and a model that has
    learnt nothing but the prior can look like it is 'almost working'."""
    from sklearn.metrics import f1_score
    labels = range(len(names))
    maj = int(np.bincount(y, minlength=len(names)).argmax())
    rand = np.random.RandomState(seed).randint(0, len(names), size=len(y))
    return {"majority_class": names[maj],
            "majority_macro_f1": float(f1_score(y, np.full_like(y, maj), average="macro", labels=labels, zero_division=0)),
            "majority_accuracy": float((y == maj).mean()),
            "random_macro_f1": float(f1_score(y, rand, average="macro", labels=labels, zero_division=0))}


def aggregate_videos(video_ids: Sequence[str], y: np.ndarray, prob_evt: np.ndarray, prob_anom: np.ndarray,
                     anom: np.ndarray, normal: int, mil_k: int = 3):
    """Window predictions -> ONE prediction per video.

    Window-level metrics are dominated by long videos (Abuse042 contributes ~590 windows, Abuse029
    ~5), so they measure "how well do we do on the longest clips". Video-level gives every video
    one vote, and is the number comparable to published UCF-Crime results.

    Video anomaly score = max over its windows. Video event = argmax of the mean class posterior
    over the `mil_k` most anomalous windows (a single event in a long normal clip must not be
    averaged away).
    """
    order: List[str] = []
    buckets: Dict[str, List[int]] = {}
    for i, v in enumerate(video_ids):
        if v not in buckets:
            buckets[v] = []
            order.append(v)
        buckets[v].append(i)
    yv, pv, pav, anv = [], [], [], []
    for v in order:
        ix = np.asarray(buckets[v])
        pa = prob_anom[ix]
        top = ix[np.argsort(-pa)[:max(1, min(mil_k, len(ix)))]]
        pv.append(int(prob_evt[top].mean(0).argmax()))
        pav.append(float(pa.max()))
        anv.append(int(anom[ix].max()))                       # a video is anomalous if any window is
        lab = y[ix]
        nz = lab[lab != normal]
        yv.append(int(np.bincount(nz).argmax()) if len(nz) else normal)
    return np.asarray(yv), np.asarray(pv), np.asarray(pav), np.asarray(anv)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested metric dicts into scalar history columns. The old history line dropped every
    dict value, which is why `per_class_f1` was computed every epoch and never seen."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}_"))
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[f"{prefix}{k}"] = v
    return out


@torch.no_grad()
def evaluate(model, loader, device, w: Optional[Dict[str, float]] = None, mil_k: int = 3) -> dict:
    """Window-level AND video-level metrics, plus the loss broken down by term using the SAME
    formula as training (the old version reported event-CE only, so train/val loss were different
    quantities and could not be compared)."""
    from sklearn.metrics import confusion_matrix
    model.eval()
    names = model.class_names
    nrm = normal_index(names)
    ys, ps, pe, pa, ya, vids = [], [], [], [], [], []
    acc = {"event": 0.0, "anomaly": 0.0, "confidence": 0.0, "total": 0.0}
    for b in loader:
        out = model(b["vis"].to(device), b["obj"].to(device), b["laya"].to(device), b["laya_mask"].to(device))
        y, yab = b["event"].to(device), b["anomaly"].to(device)
        _, parts = compute_losses(out, y, yab, w=w)
        for k in acc:
            acc[k] += parts[k] * len(y)
        ys.append(y.cpu()); ps.append(out["event_logits"].argmax(-1).cpu())
        pe.append(torch.softmax(out["event_logits"].float(), -1).cpu())
        pa.append(torch.sigmoid(out["anomaly_logit"].float()).cpu()); ya.append(b["anomaly"])
        vids.extend(list(b["video_id"]))
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    pe_, pa_, ya_ = torch.cat(pe).numpy(), torch.cat(pa).numpy(), torch.cat(ya).numpy().astype(int)
    n = max(1, len(y))

    m = _metrics(y, p, pa_, ya_, names)
    m.update({f"{k}_loss": v / n for k, v in acc.items()})
    m["baseline"] = baselines(y, names)
    m["confusion_matrix"] = confusion_matrix(y, p, labels=range(len(names))).tolist()
    m["n_windows"] = int(len(y))
    if vids:
        yv, pv, pav, anv = aggregate_videos(vids, y, pe_, pa_, ya_, nrm, mil_k)
        vm = _metrics(yv, pv, pav, anv, names)
        vm["baseline"] = baselines(yv, names)
        vm["confusion_matrix"] = confusion_matrix(yv, pv, labels=range(len(names))).tolist()
        m.update({f"video_{k}": v for k, v in vm.items()})
        m["n_videos"] = int(len(yv))
    return m


# ---- training -----------------------------------------------------------------------------------

def _forward_bag(model, b, device):
    """(B,k,T,D) bags -> one flat (B*k,T,D) forward pass. AegisModel.forward keys off vis.shape[0],
    so it is shape-agnostic on the batch dim and needs no change."""
    vis, obj = b["vis"].to(device), b["obj"].to(device)
    B, k = vis.shape[:2]
    laya = b["laya"].to(device).unsqueeze(1).expand(-1, k, -1).reshape(B * k, -1)
    mask = b["laya_mask"].to(device).unsqueeze(1).expand(-1, k).reshape(B * k)
    vis_flat = vis.reshape(B * k, *vis.shape[2:])
    obj_flat = obj.reshape(B * k, *obj.shape[2:])
    out = model(vis_flat, obj_flat, laya, mask)
    # Also expose the per-bag CLS token (mean over k windows) for contrastive loss.
    # TemporalEncoder already returns it; AegisModel currently discards it.
    # We approximate per-bag z as the mean event logit embedding — using the CLS slot
    # requires AegisModel changes we defer; the logit embedding is sufficient.
    out["_cls_z"] = out["event_logits"].view(B, k, -1).mean(1)   # (B, C) — proxy for z
    return out, B, k


def train(model: AegisModel, train_ds, val_ds, cfg: AegisConfig, out_dir: str, device: str = "cpu",
          max_steps: Optional[int] = None, visual_tag: str = "") -> dict:
    validate_train_config(cfg)
    t = cfg.train
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    model.set_trainable(t.train_temporal_encoder, t.train_fusion, t.train_event_head)
    w = loss_weights(cfg)
    nrm = normal_index(model.class_names)
    # Bag mode is decided by the DATASET, not the config, so --dry-run (SyntheticFeatureDataset)
    # and the smoke-test scripts keep working regardless of cfg.train.mil.
    is_bag = bool(getattr(train_ds, "is_bag", False))

    if len(train_ds) == 0:
        raise RuntimeError("train_ds has 0 samples — feature cache may be empty or failed to extract")
    raw_labels = train_ds.labels() if hasattr(train_ds, "labels") else train_ds.y.numpy()
    labels = np.asarray(raw_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(model.class_names)).astype(np.float64)
    weight = None
    sampler = None
    if t.balance == "sampler":
        sw = (1.0 / np.clip(counts, 1, None))[labels]
        sampler = WeightedRandomSampler(sw, len(labels), replacement=True)
    elif t.balance == "class_weights":
        weight = torch.tensor(counts.sum() / (len(counts) * np.clip(counts, 1, None)), dtype=torch.float32, device=device)
    tl = DataLoader(train_ds, batch_size=t.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0)
    vl = DataLoader(val_ds, batch_size=t.batch_size if not is_bag else max(1, t.batch_size * t.bag_size),
                    shuffle=False, num_workers=0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t.lr), weight_decay=float(t.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t.epochs))
    use_amp = bool(t.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info("objective=%s balance=%s weights=%s monitor=%s", "MIL/bag" if is_bag else "window", t.balance, w, t.monitor)

    best, best_vm, bad, hist, steps = -1.0, {}, 0, [], 0
    for ep in range(t.epochs):
        model.train()
        run: Dict[str, float] = {}
        n = 0
        for b in tl:
            y, ya = b["event"].to(device), b["anomaly"].to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                if is_bag:
                    out, B, k_bag = _forward_bag(model, b, device)
                    loss, parts = compute_mil_losses(out, y, ya, B, k_bag, nrm, weight, w, t.mil_k,
                                                     t.mil_smooth, t.mil_sparse)
                    # Supervised contrastive loss on per-bag CLS embeddings
                    w_con = getattr(t, "w_contrastive", 0.0)
                    if w_con > 0.0 and "_cls_z" in out:
                        margin = getattr(t, "contrastive_margin", 0.4)
                        l_con = compute_contrastive_loss(
                            out["_cls_z"], y, model.class_names, margin=margin
                        )
                        loss = loss + w_con * l_con
                        parts["contrastive"] = l_con.item()
                        parts["total"] += w_con * l_con.item()
                    else:
                        parts["contrastive"] = 0.0
                else:
                    out = model(b["vis"].to(device), b["obj"].to(device),
                                b["laya"].to(device), b["laya_mask"].to(device))
                    loss, parts = compute_losses(out, y, ya, weight, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(opt); scaler.update()
            for kk, vv in parts.items():
                run[kk] = run.get(kk, 0.0) + vv * len(y)
            n += len(y); steps += 1
            if max_steps and steps >= max_steps:
                break
        sched.step()
        vm = evaluate(model, vl, device, w=w, mil_k=t.mil_k)
        row = {"epoch": ep + 1, "train_loss": run.get("total", 0.0) / max(1, n),
               **{f"train_{k}_loss": v / max(1, n) for k, v in run.items()},
               **_flatten({f"val_{k}": v for k, v in vm.items()})}
        hist.append(row)

        mon = t.monitor if t.monitor in vm else "macro_f1"
        score = float(vm.get(mon, -1.0))
        log.info("epoch %d train_loss %.4f (evt %.4f an %.4f cf %.4f) val_%s %.4f [majority %.4f]",
                 ep + 1, row["train_loss"], row.get("train_event_loss", float("nan")),
                 row.get("train_anomaly_loss", float("nan")), row.get("train_confidence_loss", float("nan")),
                 mon, score, vm.get("video_baseline", vm.get("baseline", {})).get("majority_macro_f1", float("nan")))
        save_checkpoint(os.path.join(out_dir, "aegis_last.pt"), model, cfg, vm, ep, opt, visual_tag)
        if score > best:
            best, best_vm, bad = score, vm, 0
            save_checkpoint(os.path.join(out_dir, "aegis_best.pt"), model, cfg, vm, ep, opt, visual_tag)
            with open(os.path.join(out_dir, "confusion_matrix.json"), "w") as f:
                json.dump({"epoch": ep + 1, "monitor": mon, "score": score, "class_names": model.class_names,
                           "window": vm.get("confusion_matrix"), "video": vm.get("video_confusion_matrix"),
                           "window_baseline": vm.get("baseline"), "video_baseline": vm.get("video_baseline")},
                          f, indent=1)
        else:
            bad += 1
            if bad >= t.patience:
                log.info("early stopping at epoch %d", ep + 1)
                break
        if max_steps and steps >= max_steps:
            break
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return {"best_val_macro_f1": float(best_vm.get("macro_f1", -1.0)), "best_monitor": t.monitor,
            "best_score": best, "best_val_video_macro_f1": float(best_vm.get("video_macro_f1", -1.0)),
            "best_metrics": best_vm, "history": hist}
