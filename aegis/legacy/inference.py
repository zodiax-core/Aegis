class AlertStateMachine:
    """Implements smoothing_consecutive_windows + alert_cooldown_sec from cfg."""

    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.consecutive_count = 0
        self.last_alert_time = -1e9
        self.active_event = None

    def update(self, timestamp: float, event_class: str, confidence: float):
        cfg = self.cfg
        above_threshold = confidence >= cfg.alert_confidence_threshold

        if above_threshold and event_class == self.active_event:
            self.consecutive_count += 1
        elif above_threshold:
            self.active_event = event_class
            self.consecutive_count = 1
        else:
            self.active_event = None
            self.consecutive_count = 0

        should_fire = (
            above_threshold
            and self.consecutive_count >= cfg.smoothing_consecutive_windows
            and (timestamp - self.last_alert_time) >= cfg.alert_cooldown_sec
        )
        if should_fire:
            self.last_alert_time = timestamp
        return should_fire


@torch.no_grad()
def predict_video(video_path: str, model, cfg: AegisConfig, yolo_model=None, save_annotated=True):
    model.eval()
    fps, n_frames = video_metadata(video_path)
    window_frames = int(cfg.window_length_sec * fps)
    hop_frames = int(cfg.window_stride_sec * fps)

    events, alerts = [], []
    alert_machine = AlertStateMachine(cfg)
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    writer = None
    if save_annotated:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(cfg.output_dir, "sample_annotated_video.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_start = 0
    overall_anomaly_scores = []

    while frame_start + window_frames <= n_frames:
        clip = sample_clip(video_path, frame_start, clip_length=cfg.clip_length,
                            stride=max(1, window_frames // cfg.clip_length))
        if clip is None:
            frame_start += hop_frames
            continue

        clip_batch = clip.unsqueeze(0).to(cfg.device)
        event_logits, anomaly_logit, _ = model(clip_batch)
        event_probs = torch.softmax(event_logits, dim=1)[0].cpu().numpy()
        anomaly_prob = torch.sigmoid(anomaly_logit)[0].item()
        overall_anomaly_scores.append(anomaly_prob)

        top_idx = int(np.argmax(event_probs))
        top_class = cfg.class_names[top_idx]
        top_prob = float(event_probs[top_idx])

        laya_result = {}
        if top_class != "Normal" and should_verify_with_laya(top_prob, cfg):
            raw_frames = [((f.numpy().transpose(1, 2, 0) *
                            np.array([0.229, 0.224, 0.225]) +
                            np.array([0.485, 0.456, 0.406])) * 255).clip(0, 255).astype(np.uint8)
                          for f in clip]
            candidate_classes = [c for c, p in zip(cfg.class_names, event_probs) if p > 0.1 and c != "Normal"]
            laya_result = laya_verify_clip(raw_frames, candidate_classes, cfg)

        final_prob, contributors = fuse_predictions(top_prob, laya_result, top_class, cfg)

        window_start_sec = frame_start / fps
        window_end_sec = (frame_start + window_frames) / fps

        fired = alert_machine.update(window_end_sec, top_class, final_prob)
        event_record = {
            "event": top_class,
            "start_time": round(window_start_sec, 2),
            "end_time": round(window_end_sec, 2),
            "confidence": round(final_prob, 3),
            "anomaly_score": round(anomaly_prob, 3),
            "verification": {"laya": laya_result.get(top_class, {}).get("laya_prob")} if laya_result else None,
        }
        events.append(event_record)
        if fired:
            alerts.append(event_record)

        frame_start += hop_frames

    cap.release()

    if writer is not None:
        _annotate_and_write_video(video_path, cfg, events, writer, fps)
        writer.release()

    elapsed = time.time() - start_time
    result = {
        "events": events,
        "overall_anomaly_score": float(np.mean(overall_anomaly_scores)) if overall_anomaly_scores else 0.0,
        "processing_time": round(elapsed, 2),
        "fps": round(n_frames / elapsed, 1) if elapsed > 0 else None,
        "alerts": alerts,
    }

    with open(os.path.join(cfg.output_dir, "inference_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def _annotate_and_write_video(video_path, cfg, events, writer, fps):
    """Draws a lightweight AEGIS overlay onto each frame using the nearest window's result."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        current = next((e for e in events if e["start_time"] <= t <= e["end_time"]), None)

        if current and current["event"] != "Normal" and current["confidence"] >= cfg.alert_confidence_threshold:
            status, color = "ALERT", (0, 0, 255)
        elif current:
            status, color = "NORMAL", (0, 200, 0)
        else:
            status, color, current = "NORMAL", (0, 200, 0), {"event": "Normal", "confidence": 0.0, "anomaly_score": 0.0}

        cv2.rectangle(frame, (0, 0), (330, 100), (20, 20, 20), -1)
        cv2.putText(frame, "AEGIS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"EVENT: {current['event']}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"CONF: {current['confidence']*100:.0f}%", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"ANOMALY: {current['anomaly_score']*100:.0f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"STATUS: {status}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        writer.write(frame)
        frame_idx += 1
    cap.release()

print("predict_video() ready.")
