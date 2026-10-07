def load_yolo(cfg: AegisConfig):
    if not cfg.use_yolo:
        return None
    try:
        from ultralytics import YOLO
        return YOLO(cfg.yolo_checkpoint)
    except Exception as e:
        print(f"YOLO unavailable ({e}) — continuing without object features. "
              f"The temporal model and Laya layer still work; only the object-context "
              f"channel in the fusion model is disabled.")
        return None


yolo_model = load_yolo(cfg)

# Object classes we actually care about for surveillance context
RELEVANT_YOLO_CLASSES = {"person", "car", "truck", "motorcycle", "bicycle", "bus"}

def extract_object_features(clip_frames_rgb: list, yolo_model, every_n: int = None):
    """clip_frames_rgb: list of HxWx3 uint8 RGB numpy arrays (raw, pre-normalization).
    Returns a fixed-size feature vector: counts of each relevant class, averaged over
    the sampled frames."""
    every_n = every_n or cfg.yolo_sample_every_n_frames
    counts = {c: [] for c in RELEVANT_YOLO_CLASSES}

    if yolo_model is None:
        return np.zeros(len(RELEVANT_YOLO_CLASSES), dtype=np.float32)

    for i, frame in enumerate(clip_frames_rgb):
        if i % every_n != 0:
            continue
        result = yolo_model.predict(frame, verbose=False)[0]
        frame_counts = {c: 0 for c in RELEVANT_YOLO_CLASSES}
        for cls_id in result.boxes.cls.tolist():
            name = yolo_model.names[int(cls_id)]
            if name in RELEVANT_YOLO_CLASSES:
                frame_counts[name] += 1
        for c in RELEVANT_YOLO_CLASSES:
            counts[c].append(frame_counts[c])

    vec = np.array([np.mean(v) if v else 0.0 for v in counts.values()], dtype=np.float32)
    return vec

OBJECT_FEATURE_DIM = len(RELEVANT_YOLO_CLASSES)
print(f"Object feature dim: {OBJECT_FEATURE_DIM} ({sorted(RELEVANT_YOLO_CLASSES)})")
