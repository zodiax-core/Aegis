_laya_agent = None

def get_laya_agent(cfg: AegisConfig):
    global _laya_agent
    if not cfg.laya_enabled:
        return None
    if _laya_agent is not None:
        return _laya_agent
    try:
        import laya
        _laya_agent = laya.load_vlm(cfg.laya_checkpoint)
        print(f"Laya Vision loaded: {cfg.laya_checkpoint}")
    except Exception as e:
        print(f"Laya Vision unavailable ({e}) — the temporal model still runs standalone; "
              f"only the verification/fusion step is skipped.")
        _laya_agent = None
    return _laya_agent


# Typed questions per candidate event class. Kept short and specific — Laya scores each
# in a single forward pass per image, so precision in the question matters more than quantity.
LAYA_QUESTIONS = {
    "Fighting": {"type": "noul", "instructions": "Does this frame show people in physical confrontation, such as hitting, grappling, or striking each other?"},
    "Assault":  {"type": "noul", "instructions": "Does this frame show one person physically attacking another?"},
    "RoadAccident": {"type": "noul", "instructions": "Does this frame show a vehicle collision or a vehicle that has crashed?"},
    "Shooting": {"type": "noul", "instructions": "Does this frame show a person holding or firing a firearm?"},
    "Explosion": {"type": "noul", "instructions": "Does this frame show an explosion, fireball, or blast?"},
    "Arson": {"type": "noul", "instructions": "Does this frame show fire deliberately burning a structure or object?"},
    "Robbery": {"type": "noul", "instructions": "Does this frame show one person forcibly taking property from another, e.g. at gunpoint or by force?"},
    "severity": {"type": "score", "instructions": "How severe is the apparent danger or violence in this frame?",
                 "criteria": ["none: no danger or violence visible", "mild: tense but no physical harm",
                              "moderate: physical altercation or clear hazard", "severe: serious violence, injury, or major hazard"]},
}


def select_keyframes(clip_frames_rgb: list, k: int) -> list:
    """Evenly-spaced keyframes from a clip — cheap, deterministic, avoids biasing toward
    the first or last frame of the window."""
    n = len(clip_frames_rgb)
    if n <= k:
        return clip_frames_rgb
    idxs = np.linspace(0, n - 1, k).astype(int)
    return [clip_frames_rgb[i] for i in idxs]


def laya_verify_clip(clip_frames_rgb: list, candidate_classes: list, cfg: AegisConfig):
    """Runs Laya Vision on a few keyframes of a suspicious window, only for the classes
    the temporal model actually flagged as plausible. Returns a dict:
    {class_name: {"laya_prob": float, "severity_score": float}}"""
    agent = get_laya_agent(cfg)
    if agent is None:
        return {}

    from PIL import Image
    keyframes = select_keyframes(clip_frames_rgb, cfg.laya_max_keyframes_per_window)

    per_class_scores = {c: [] for c in candidate_classes if c in LAYA_QUESTIONS}
    severity_scores = []

    for frame in keyframes:
        image = Image.fromarray(frame)
        questions = {c: LAYA_QUESTIONS[c] for c in per_class_scores}
        questions["severity"] = LAYA_QUESTIONS["severity"]

        result = agent.predict({"image": image}, questions)

        for c in per_class_scores:
            per_class_scores[c].append(result["answers"][c]["noul"])
        severity_scores.append(result["answers"]["severity"]["score"])

    out = {}
    for c, scores in per_class_scores.items():
        out[c] = {
            "laya_prob": float(np.mean(scores)) if scores else 0.0,
            "severity_score": float(np.mean(severity_scores)) if severity_scores else 0.0,
        }
    return out


def should_verify_with_laya(top_prob: float, cfg: AegisConfig) -> bool:
    """Cascade gate: only call Laya when the temporal model is confident-but-not-certain,
    or genuinely unsure — never on every window (would destroy throughput)."""
    if not cfg.laya_enabled:
        return False
    lo, hi = cfg.laya_uncertainty_band
    is_suspicious = top_prob >= cfg.laya_prob_threshold
    is_uncertain = lo <= top_prob <= hi
    return is_suspicious or is_uncertain


print("Laya Vision verification layer defined. Call get_laya_agent(cfg) once to warm-load it.")
