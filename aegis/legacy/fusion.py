def fuse_predictions(temporal_prob: float, laya_result: dict, event_class: str, cfg: AegisConfig):
    """temporal_prob: the temporal model's probability for `event_class`.
    laya_result: output of laya_verify_clip for this window (may be empty if not consulted).
    Returns (final_prob, contributors_dict) for transparency/logging."""
    if not laya_result or event_class not in laya_result:
        return temporal_prob, {"temporal": temporal_prob, "laya": None, "strategy": "temporal_only"}

    laya_prob = laya_result[event_class]["laya_prob"]

    if cfg.fusion_strategy == "average":
        final = 0.5 * temporal_prob + 0.5 * laya_prob

    elif cfg.fusion_strategy == "threshold_gate":
        # Laya can veto: if it strongly disagrees (says "no" with high confidence),
        # damp the temporal model's score rather than trusting it blindly.
        if laya_prob < 0.15:
            final = temporal_prob * 0.3
        else:
            final = max(temporal_prob, laya_prob * 0.9)

    else:  # confidence_weighted (default)
        laya_confidence = abs(laya_prob - 0.5) * 2       # 0 (unsure) .. 1 (certain)
        w = cfg.laya_fusion_weight * laya_confidence
        final = (1 - w) * temporal_prob + w * laya_prob

    final = float(np.clip(final, 0.0, 1.0))
    return final, {"temporal": temporal_prob, "laya": laya_prob, "strategy": cfg.fusion_strategy}


print(f"Fusion strategy: {cfg.fusion_strategy}")
