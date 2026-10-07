from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(split_dict: dict, class_names: list):
    counts = np.array([len(split_dict.get(c, [])) for c in class_names], dtype=np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(class_names) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloader(dataset: VideoClipDataset, cfg: AegisConfig, train_split: dict = None, shuffle=True):
    if cfg.class_balance_strategy == "weighted_sampler" and train_split is not None:
        class_counts = np.array([len(train_split.get(c, [])) for c in cfg.class_names], dtype=np.float32)
        class_counts = np.clip(class_counts, 1, None)
        class_weight_lookup = 1.0 / class_counts
        sample_weights = [class_weight_lookup[lbl] for _, _, lbl in dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_aegis_model(model, train_loader, val_loader, cfg: AegisConfig, class_weights=None, run_name="aegis"):
    device = cfg.device
    model = model.to(device)

    if cfg.class_balance_strategy == "focal_loss":
        event_criterion = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights.to(device) if class_weights is not None else None)
    else:
        w = class_weights.to(device) if (cfg.class_balance_strategy == "class_weights" and class_weights is not None) else None
        event_criterion = nn.CrossEntropyLoss(weight=w)
    anomaly_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    scaler = GradScaler(enabled=cfg.use_amp)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    run_dir = os.path.join(cfg.runs_dir, f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            clip = batch["clip"].to(device, non_blocking=True)
            event_labels = batch["class_label"].to(device)
            anomaly_labels = batch["anomaly_label"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                event_logits, anomaly_logit, _ = model(clip)
                loss = event_criterion(event_logits, event_labels) + anomaly_criterion(anomaly_logit, anomaly_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * clip.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_f1, _, _ = evaluate_model(model, val_loader, cfg, event_criterion, anomaly_criterion)

        if cfg.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_macro_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, cfg, val_f1, history,
                             path=os.path.join(cfg.output_dir, "aegis_best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (no val_F1 improvement for "
                      f"{cfg.early_stopping_patience} epochs).")
                break

    save_checkpoint(model, optimizer, epoch, cfg, best_val_f1, history,
                     path=os.path.join(cfg.output_dir, "aegis_final.pt"))

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    return history, best_val_f1

def save_checkpoint(model, optimizer, epoch, cfg: AegisConfig, val_f1, history, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "class_names": cfg.class_names,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "metrics": {"val_macro_f1": val_f1, "history": history},
        "model_architecture": model.__class__.__name__,
    }, path)
    print(f"Saved checkpoint: {path} (val_macro_F1={val_f1:.4f})")


def load_checkpoint(path, cfg: AegisConfig, model_cls=None):
    """Loads a checkpoint from scratch and reconstructs the model — proves the .pt
    file is actually sufficient to reproduce inference, not just a state_dict fragment."""
    ckpt = torch.load(path, map_location=cfg.device)
    model_cls = model_cls or AegisTemporalModel
    model = model_cls(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.device).eval()
    print(f"Loaded {ckpt['model_architecture']} from {path} "
          f"(epoch {ckpt['epoch']}, val_macro_F1={ckpt['metrics']['val_macro_f1']:.4f})")
    return model, ckpt
