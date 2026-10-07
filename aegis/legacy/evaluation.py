def evaluate_model(model, loader, cfg: AegisConfig, event_criterion=None, anomaly_criterion=None):
    model.eval()
    device = cfg.device
    all_preds, all_labels, all_anomaly_probs, all_anomaly_labels = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        clip = batch["clip"].to(device)
        event_labels = batch["class_label"].to(device)
        anomaly_labels = batch["anomaly_label"].float().to(device)

        event_logits, anomaly_logit, _ = model(clip)

        if event_criterion is not None and anomaly_criterion is not None:
            loss = event_criterion(event_logits, event_labels) + anomaly_criterion(anomaly_logit, anomaly_labels)
            total_loss += loss.item() * clip.size(0)

        preds = event_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(event_labels.cpu().numpy())
        all_anomaly_probs.extend(torch.sigmoid(anomaly_logit).cpu().numpy())
        all_anomaly_labels.extend(anomaly_labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset) if (event_criterion is not None) else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, (all_labels, all_preds), (all_anomaly_labels, all_anomaly_probs)


def full_evaluation_report(model, loader, cfg: AegisConfig, split_name="test"):
    _, macro_f1, (labels, preds), (anomaly_labels, anomaly_probs) = evaluate_model(model, loader, cfg)

    report = classification_report(labels, preds, target_names=cfg.class_names,
                                    zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(cfg.output_dir, f"classification_report_{split_name}.csv"))
    print(report_df)

    cm = confusion_matrix(labels, preds, labels=list(range(cfg.num_classes)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=cfg.class_names, yticklabels=cfg.class_names, cmap="Blues")
    plt.title(f"Confusion matrix — {split_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "confusion_matrix.png"))
    plt.show()

    metrics = {"macro_f1": macro_f1}
    try:
        metrics["anomaly_roc_auc"] = roc_auc_score(anomaly_labels, anomaly_probs)
        prec, rec, _ = precision_recall_curve(anomaly_labels, anomaly_probs)
        metrics["anomaly_pr_auc"] = sk_auc(rec, prec)
    except ValueError:
        metrics["anomaly_roc_auc"] = None
        metrics["anomaly_pr_auc"] = None

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSummary metrics:", metrics)
    return metrics


def benchmark_inference_speed(model, cfg: AegisConfig, n_iters=30):
    model.eval()
    dummy = torch.randn(1, cfg.clip_length, 3, cfg.image_size, cfg.image_size, device=cfg.device)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model(dummy)
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    elapsed = time.time() - start
    latency_ms = (elapsed / n_iters) * 1000
    fps = cfg.clip_length / (latency_ms / 1000)
    print(f"Latency: {latency_ms:.1f} ms/window | Effective FPS: {fps:.1f} "
          f"(clip_length={cfg.clip_length} frames/window)")
    return latency_ms, fps
