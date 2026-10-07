"""Tests for the stage-1 training objective: loss decomposition, MIL over video bags, and
video-level evaluation. These guard the three mechanisms that previously produced a falling loss
with a near-chance macro-F1."""
import json, numpy as np, pytest, torch

from aegis.config import load_config
from aegis.data.cache import FeatureCache
from aegis.data.dataset import FeatureWindowDataset, VideoBagDataset
from aegis.data.manifest import VideoRecord
from aegis.model.aegis_model import AegisModel
from aegis.perception.features import OBJ_FEAT_DIM
from aegis.training.trainer import (aggregate_videos, baselines, compute_loss, compute_losses,
                                    compute_mil_losses, evaluate, normal_index, train, _flatten)

T, VD = 4, 16                      # clip_length, visual dim - tiny so these run in milliseconds
CLASSES = ["NORMAL", "ABUSE"]


def _bag_cache(tmp_path, lengths):
    """A FeatureCache populated directly (no video decoding). lengths = {video_id: n_samples}."""
    cache = FeatureCache(str(tmp_path / "cache"))
    recs, keys = [], {}
    for i, (vid, n) in enumerate(lengths.items()):
        lab = "NORMAL" if vid.startswith("n") else "ABUSE"
        rng = np.random.RandomState(i)
        k = f"key{i}"
        cache.put(k, rng.randn(n, VD), rng.randn(n, OBJ_FEAT_DIM), np.arange(n, dtype=np.float32))
        recs.append(VideoRecord(vid, "DS", f"/{vid}.mp4", lab, lab, vid, None, f"fp-{vid}"))
        keys[vid] = k
    return recs, keys, cache


def _mk_bags(tmp_path, lengths, bag_size=4, training=False, stride=1, **kw):
    recs, keys, cache = _bag_cache(tmp_path, lengths)
    return VideoBagDataset(recs, keys, cache, CLASSES, T, stride, "NORMAL", bag_size=bag_size,
                           training=training, **kw)


def _out(anom, evt, conf=None):
    """Fake model outputs. anom: (N,) logits, evt: (N,C) logits."""
    a = torch.tensor(anom, dtype=torch.float32)
    return {"anomaly_logit": a, "event_logits": torch.tensor(evt, dtype=torch.float32),
            "confidence_logit": torch.tensor(conf if conf is not None else [0.0] * len(anom), dtype=torch.float32)}


# ---- Stage A: loss decomposition and baselines ---------------------------------------------------

def test_compute_losses_reports_every_term_separately():
    out = _out([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])
    y, ya = torch.tensor([0, 1]), torch.tensor([0.0, 1.0])
    total, parts = compute_losses(out, y, ya, w={"event": 1.0, "anomaly": 0.3, "confidence": 0.1})
    assert set(parts) == {"event", "anomaly", "confidence", "total"}
    # The whole point: the reported total must be reconstructible from the terms, so a falling
    # total can be attributed to the term that is actually moving.
    assert parts["total"] == pytest.approx(parts["event"] + 0.3 * parts["anomaly"] + 0.1 * parts["confidence"], rel=1e-5)
    assert total.requires_grad is False and float(total) == pytest.approx(parts["total"], rel=1e-6)


def test_compute_loss_backcompat_signature():
    """scripts/smoke_test_real.py and modal/train_modal.py call this positionally."""
    out = _out([0.0], [[0.0, 0.0]])
    loss = compute_loss(out, torch.tensor([1]), torch.tensor([1.0]), "cpu")
    assert loss.ndim == 0 and float(loss) > 0


def test_baselines_make_a_low_macro_f1_interpretable():
    y = np.array([0] * 90 + [1] * 10)
    b = baselines(y, ["NORMAL", "ABUSE"])
    assert b["majority_class"] == "NORMAL"
    assert b["majority_accuracy"] == pytest.approx(0.9)
    # Always predicting the majority class: F1 of 0.947 on NORMAL, 0.0 on ABUSE -> macro 0.474.
    assert b["majority_macro_f1"] == pytest.approx(0.4736, abs=1e-3)
    assert 0.0 <= b["random_macro_f1"] <= 1.0


def test_flatten_surfaces_per_class_f1():
    flat = _flatten({"val_macro_f1": 0.3, "val_per_class_f1": {"NORMAL": 0.6, "ABUSE": 0.0},
                     "val_confusion_matrix": [[1, 0], [0, 1]]})
    assert flat["val_per_class_f1_ABUSE"] == 0.0 and flat["val_macro_f1"] == 0.3
    assert "val_confusion_matrix" not in flat          # lists go to their own file, not history


def test_normal_index():
    assert normal_index(["NORMAL", "ABUSE"]) == 0 and normal_index(["ABUSE", "NORMAL"]) == 1
    assert normal_index(["A", "B"]) == 0               # no NORMAL class -> index 0


# ---- Stage A: video-level evaluation -------------------------------------------------------------

def test_video_aggregation_gives_each_video_one_vote():
    """A 10-window normal video must not outvote a 1-window anomalous one, which is exactly what
    window-level metrics let it do."""
    vids = ["long"] * 10 + ["short"]
    y = np.array([0] * 10 + [1])
    prob_anom = np.array([0.1] * 10 + [0.9])
    prob_evt = np.zeros((11, 2)); prob_evt[:10, 0] = 1.0; prob_evt[10, 1] = 1.0
    anom = np.array([0] * 10 + [1])
    yv, pv, pav, anv = aggregate_videos(vids, y, prob_evt, prob_anom, anom, normal=0, mil_k=3)
    assert len(yv) == 2                                 # two videos, not eleven windows
    assert list(yv) == [0, 1] and list(pv) == [0, 1] and list(anv) == [0, 1]
    assert pav[0] == pytest.approx(0.1) and pav[1] == pytest.approx(0.9)


def test_video_event_uses_most_anomalous_windows_not_the_mean():
    """One abusive window inside a long quiet clip must survive aggregation."""
    vids = ["v"] * 11
    y = np.array([1] * 11)
    prob_anom = np.array([0.01] * 10 + [0.99])
    prob_evt = np.zeros((11, 2)); prob_evt[:10, 0] = 1.0; prob_evt[10, 1] = 1.0
    yv, pv, pav, _ = aggregate_videos(vids, y, prob_evt, prob_anom, np.ones(11, int), normal=0, mil_k=1)
    assert list(yv) == [1] and list(pv) == [1] and pav[0] == pytest.approx(0.99)


# ---- Stage B: MIL loss ---------------------------------------------------------------------------

def test_mil_anomaly_loss_is_asymmetric_between_positive_and_negative_bags():
    """The property that makes MIL work on weak labels. Identical features, opposite bag labels:
    a positive bag only needs its top window to fire, a negative bag needs ALL windows quiet."""
    anom = [5.0, -5.0, -5.0, -5.0]                      # one window fires
    evt = [[0.0, 0.0]] * 4
    pos = compute_mil_losses(_out(anom, evt), torch.tensor([1]), torch.tensor([1.0]),
                             n_bags=1, k=4, normal=0, mil_k=1)[1]
    neg = compute_mil_losses(_out(anom, evt), torch.tensor([0]), torch.tensor([0.0]),
                             n_bags=1, k=4, normal=0, mil_k=1)[1]
    assert pos["anomaly"] < 0.05                        # top-1 satisfied the positive bag
    assert neg["anomaly"] > 1.0                         # the firing window violates the clean bag
    quiet = compute_mil_losses(_out([-5.0] * 4, evt), torch.tensor([0]), torch.tensor([0.0]),
                               n_bags=1, k=4, normal=0, mil_k=1)[1]
    assert quiet["anomaly"] < 0.05                      # all-quiet negative bag is satisfied


def test_mil_event_loss_follows_the_anomaly_ranking():
    """The event head is supervised on the windows the model itself called anomalous. Putting the
    correct class on a window the model thinks is quiet must NOT earn a low loss."""
    right = _out([5.0, -5.0], [[-5.0, 5.0], [5.0, -5.0]])   # firing window predicts ABUSE: good
    wrong = _out([5.0, -5.0], [[5.0, -5.0], [-5.0, 5.0]])   # ABUSE only on the quiet window: bad
    args = dict(n_bags=1, k=2, normal=0, mil_k=1)
    lo = compute_mil_losses(right, torch.tensor([1]), torch.tensor([1.0]), **args)[1]["event"]
    hi = compute_mil_losses(wrong, torch.tensor([1]), torch.tensor([1.0]), **args)[1]["event"]
    assert lo < 0.05 < 1.0 < hi


def test_mil_negative_bag_supervises_every_window_as_normal():
    anom, evt = [0.0, 0.0], [[5.0, -5.0], [5.0, -5.0]]      # both windows predict NORMAL
    good = compute_mil_losses(_out(anom, evt), torch.tensor([0]), torch.tensor([0.0]),
                              n_bags=1, k=2, normal=0, mil_k=1)[1]["event"]
    bad = compute_mil_losses(_out(anom, [[-5.0, 5.0], [5.0, -5.0]]), torch.tensor([0]), torch.tensor([0.0]),
                             n_bags=1, k=2, normal=0, mil_k=1)[1]["event"]
    assert good < 0.05 and bad > good                       # one non-normal window is still penalised


def test_mil_per_bag_averaging_is_independent_of_bag_size():
    """Positive/negative terms are averaged per BAG, so changing bag_size or mil_k must not
    silently re-weight the objective."""
    def run(k):
        anom = [5.0] + [-5.0] * (k - 1)
        return compute_mil_losses(_out(anom, [[0.0, 0.0]] * k), torch.tensor([1]), torch.tensor([1.0]),
                                  n_bags=1, k=k, normal=0, mil_k=1)[1]["anomaly"]
    assert run(4) == pytest.approx(run(12), rel=1e-5)


def test_mil_regularisers_are_off_by_default_and_wire_up():
    out = _out([5.0, -5.0, 5.0, -5.0], [[0.0, 0.0]] * 4)
    y, ya = torch.tensor([1]), torch.tensor([1.0])
    base = compute_mil_losses(out, y, ya, 1, 4, 0, mil_k=1)[1]
    reg = compute_mil_losses(out, y, ya, 1, 4, 0, mil_k=1, smooth=1.0, sparse=1.0)[1]
    assert base["smooth"] > 0 and base["sparse"] > 0        # measured...
    assert base["total"] == pytest.approx(reg["total"] - base["smooth"] - base["sparse"], rel=1e-5)  # ...but unweighted


# ---- Stage B: bag dataset ------------------------------------------------------------------------

def test_bag_dataset_shapes_and_one_label_per_video(tmp_path):
    ds = _mk_bags(tmp_path, {"a1": 20, "n1": 20}, bag_size=4)
    assert len(ds) == 2 and ds.is_bag is True
    item = ds[0]
    assert item["vis"].shape == (4, T, VD) and item["obj"].shape == (4, T, OBJ_FEAT_DIM)
    assert item["event"].ndim == 0 and item["anomaly"].ndim == 0
    assert list(ds.labels()) == [CLASSES.index("ABUSE"), CLASSES.index("NORMAL")]


def test_bag_dataset_caps_long_videos_removing_the_length_imbalance(tmp_path):
    """The 120:1 window ratio between a 949-second and a 10-second video is what let a handful of
    long clips own the gradient. Every video now contributes exactly bag_size windows."""
    ds = _mk_bags(tmp_path, {"a_long": 400, "a_short": 8}, bag_size=5)
    assert ds[0]["vis"].shape[0] == ds[1]["vis"].shape[0] == 5
    flat = FeatureWindowDataset(*_bag_cache(tmp_path, {"a_long": 400, "a_short": 8})[:3],
                                CLASSES, T, 1, "NORMAL")
    counts = {v: sum(1 for it in flat.items if it[0] == v) for v in ("a_long", "a_short")}
    assert counts["a_long"] > 50 * counts["a_short"]        # the imbalance the bag view removes


def test_bag_dataset_eval_is_deterministic_and_time_ordered(tmp_path):
    ds = _mk_bags(tmp_path, {"a1": 40}, bag_size=5, training=False)
    s1, s2 = ds[0]["starts"].tolist(), ds[0]["starts"].tolist()
    assert s1 == s2                                          # deterministic for eval
    assert s1 == sorted(s1)                                  # time-ordered for the smoothness term
    assert s1[0] == 0 and s1[-1] == 40 - T                   # spans the whole video


def test_bag_dataset_training_sampling_varies_but_stays_ordered(tmp_path):
    ds = _mk_bags(tmp_path, {"a1": 60}, bag_size=5, training=True)
    draws = [tuple(ds[0]["starts"].tolist()) for _ in range(8)]
    assert len(set(draws)) > 1                               # stochastic -> acts as augmentation
    assert all(list(d) == sorted(d) for d in draws)


def test_bag_dataset_handles_videos_shorter_than_the_bag(tmp_path):
    ds = _mk_bags(tmp_path, {"a1": T + 1}, bag_size=6)        # only 2 possible windows
    assert ds[0]["vis"].shape[0] == 6                         # padded by repetition, never ragged


def test_bag_dataset_skips_videos_shorter_than_clip_length(tmp_path):
    assert len(_mk_bags(tmp_path, {"a1": T - 1, "a2": 20})) == 1


def test_bag_dataset_annotations_restrict_positive_windows(tmp_path):
    """The drop-in point for the official UCF-Crime temporal annotations."""
    recs, keys, cache = _bag_cache(tmp_path, {"a1": 40})
    ann = {"a1": [[0.0, 6.0]]}                                # event covers samples 0..6
    ds = VideoBagDataset(recs, keys, cache, CLASSES, T, 1, "NORMAL", bag_size=3, annotations=ann)
    starts = ds.videos[0][2]
    # Candidates are cut down to the annotated event, with the usual one-window boundary bleed
    # (a 4-sample window starting at 4 still overlaps [0,6] by half), so we assert the property
    # rather than an exact endpoint: every candidate must genuinely overlap the event.
    assert 0 in starts and max(starts) < 40 and len(starts) < 15
    assert all(s + T <= 8 for s in starts)                    # nothing far past the event survives
    empty = VideoBagDataset(recs, keys, cache, CLASSES, T, 1, "NORMAL", bag_size=3,
                            annotations={"a1": [[900.0, 901.0]]})
    assert len(empty.videos[0][2]) > 1                        # bad annotation must not empty the bag


# ---- end-to-end ----------------------------------------------------------------------------------

def test_mil_training_loop_runs_and_logs_decomposed_losses(tmp_path):
    cfg = load_config()
    cfg.model.visual_pretrained = False
    cfg.model.class_names = CLASSES
    cfg.train.epochs, cfg.train.batch_size, cfg.train.bag_size, cfg.train.mil_k = 2, 2, 4, 2
    cfg.train.amp = False
    lengths = {f"a{i}": 24 for i in range(3)} | {f"n{i}": 24 for i in range(3)}
    tr = _mk_bags(tmp_path, lengths, bag_size=4, training=True)
    recs, keys, cache = _bag_cache(tmp_path / "v", lengths)
    va = FeatureWindowDataset(recs, keys, cache, CLASSES, T, 8, "NORMAL")
    out = str(tmp_path / "run")
    res = train(AegisModel(cfg.model, VD, class_names=CLASSES), tr, va, cfg, out, "cpu")

    row = res["history"][0]
    for k in ("train_event_loss", "train_anomaly_loss", "train_confidence_loss", "train_mil_pos_loss"):
        assert k in row, f"missing decomposed loss {k}"
    # train and val totals are now the same quantity, so comparing them is meaningful
    assert "val_total_loss" in row and "val_event_loss" in row
    assert "val_per_class_f1_ABUSE" in row                    # no longer silently discarded
    assert "val_video_macro_f1" in row and "val_baseline_majority_macro_f1" in row
    cm = json.load(open(f"{out}/confusion_matrix.json"))
    assert cm["class_names"] == CLASSES and len(cm["window"]) == 2
    assert res["best_monitor"] == "video_macro_f1"


def test_window_mode_still_works_when_dataset_is_not_a_bag(tmp_path):
    """The MIL switch is driven by the dataset, not the config, so --dry-run and the smoke tests
    are unaffected by cfg.train.mil."""
    cfg = load_config()
    cfg.model.visual_pretrained = False
    cfg.model.class_names = CLASSES
    cfg.train.epochs, cfg.train.batch_size, cfg.train.amp, cfg.train.mil = 1, 4, False, True
    lengths = {f"a{i}": 24 for i in range(2)} | {f"n{i}": 24 for i in range(2)}
    recs, keys, cache = _bag_cache(tmp_path, lengths)
    ds = FeatureWindowDataset(recs, keys, cache, CLASSES, T, 8, "NORMAL")
    res = train(AegisModel(cfg.model, VD, class_names=CLASSES), ds, ds, cfg, str(tmp_path / "r"), "cpu")
    assert "train_event_loss" in res["history"][0] and "train_mil_pos_loss" not in res["history"][0]


def test_evaluate_returns_both_window_and_video_metrics(tmp_path):
    cfg = load_config()
    cfg.model.visual_pretrained = False
    cfg.model.class_names = CLASSES
    recs, keys, cache = _bag_cache(tmp_path, {"a1": 30, "n1": 30})
    ds = FeatureWindowDataset(recs, keys, cache, CLASSES, T, 4, "NORMAL")
    m = evaluate(AegisModel(cfg.model, VD, class_names=CLASSES).eval(),
                 torch.utils.data.DataLoader(ds, batch_size=8), "cpu")
    assert m["n_videos"] == 2 and m["n_windows"] > 2
    assert {"event_loss", "anomaly_loss", "confidence_loss", "total_loss"} <= set(m)
    assert "video_macro_f1" in m and "video_baseline" in m
    assert len(m["confusion_matrix"]) == len(CLASSES)
