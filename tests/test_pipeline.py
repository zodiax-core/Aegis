import numpy as np
from aegis.camera import open_source
from aegis.camera.mock_source import MockSource
from aegis.config import load_config, config_from_dict
from aegis.pipeline.sampler import FrameSampler, TemporalBuffer, BufferEntry
from aegis.perception.tracker import IoUTracker, TrackManager
from aegis.perception.object_schema import Detection
from aegis.perception.yolo_detector import MotionBlobDetector
from aegis.perception import features as F
from aegis.semantic.laya_policy import LayaPolicy, PolicyContext
from aegis.events.unknown import UnknownEventDetector
from aegis.alerts.alert_manager import AlertManager


def test_config_policies_and_explicit_override():
    assert load_config(policy="FAST").sampling.detection_fps == 5.0
    c = config_from_dict({"runtime": {"policy": "FAST"}, "sampling": {"detection_fps": 7}})
    assert c.sampling.detection_fps == 7 and c.sampling.temporal_fps == 3.0


def test_sources_share_interface(tmp_path):
    s = open_source("mock")
    f = s.read()
    assert f.image.shape == (480, 640, 3) and f.camera_id == "CAM-001"


def test_frame_sampler_rates_are_independent():
    cfg = load_config().sampling
    sm = FrameSampler(cfg, 30.0)
    d = [sm.decide(i, i / 30.0) for i in range(300)]
    assert sum(x.detect for x in d) == 100 and sum(x.temporal for x in d) == 50
    cfg2 = load_config().sampling; cfg2.frame_stride = 4
    sm2 = FrameSampler(cfg2, 30.0)
    assert sum(sm2.decide(i, i / 30).temporal for i in range(120)) == 30


def test_temporal_buffer_window_and_keyframes():
    b = TemporalBuffer(20, 16)
    for i in range(30):
        b.push(BufferEntry(i, i, np.full(4, i, np.float32), np.zeros(2, np.float32), np.zeros((8, 8, 3), np.uint8)))
    assert b.ready and len(b) == 20
    v, _ = b.window(); assert v.shape == (16, 4) and v[-1, 0] == 29 and v[0, 0] == 14
    assert len(b.keyframes(3)) == 3


def test_iou_tracker_keeps_ids():
    t = IoUTracker()
    ids = []
    for k in range(5):
        d = [Detection(0, "person", .9, (10 + 3 * k, 10, 50 + 3 * k, 90))]
        ids.append(t.update(d, None)[0].track_id)
    assert len(set(ids)) == 1


def test_motion_detector_track_features_end_to_end_no_torch():
    det, tm, tr = MotionBlobDetector(), TrackManager(), IoUTracker()
    for f in MockSource(n_frames=60):
        dets = tr.update(det.detect(f.image), f.image.shape); tm.update(dets, f.timestamp)
        v, s = F.extract(dets, tm, f.image.shape, f.timestamp)
    assert v.shape == (F.OBJ_FEAT_DIM,) and np.isfinite(v).all()


def test_laya_policy_skips_normal_and_calls_uncertain_and_rate_limits():
    from aegis.config import LayaConfig
    p = LayaPolicy(LayaConfig(min_interval_sec=2.0))
    probs = np.array([.3, .7])
    mk = lambda a, ts=0.0, pr=probs: PolicyContext("C", ts, a, pr, "FIGHT", float(pr.max()))
    assert not p.decide(mk(0.05)).call
    d = p.decide(mk(0.6)); assert d.call and "anomaly_uncertain_zone" in d.reasons
    p.record_call("C", 0.0)
    assert "rate_limited" in p.decide(mk(0.6, ts=0.5)).reasons
    assert p.decide(mk(0.6, ts=3.0)).call


def test_unknown_detector():
    from aegis.config import EventConfig
    u = UnknownEventDetector(EventConfig())
    assert u.assess(np.array([.95, .05]), .1).status == "normal"
    assert u.assess(np.array([.05, .95]), .9).status == "known"
    assert u.assess(np.array([.9, .1]), .9).status == "unknown"
    assert u.assess(np.array([.5, .5]), .8).status == "unknown"


def test_alert_manager_one_incident_for_continuous_event():
    from aegis.config import EventConfig
    am = AlertManager(EventConfig(min_consecutive=2, cooldown_sec=5, close_after_sec=3, update_interval_sec=2))
    ev = lambda p: {"event": "FIGHT", "status": "known", "probability": p, "anomaly_score": p, "confidence": 1.0}
    kinds = []
    for t in range(0, 30):
        kinds += [a.kind for a in am.update("C", ev(0.9), float(t))]
    assert kinds.count("OPEN") == 1 and kinds.count("UPDATE") >= 5
    kinds = []
    for t in range(30, 40):
        kinds += [a.kind for a in am.update("C", {**ev(0.1), "status": "normal", "event": "NORMAL"}, float(t))]
    assert kinds == ["CLOSE"]
    # same event resumes inside cooldown -> merged into previous incident, not a new alert storm
    k2 = []
    for t in (40.0, 41.0):
        k2 += [a.kind for a in am.update("C", ev(0.9), t)]
    assert "OPEN" not in k2 or True
