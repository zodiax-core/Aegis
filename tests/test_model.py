import numpy as np, pytest, torch
from aegis.config import load_config
from aegis.model.aegis_model import AegisModel
from aegis.model.visual_encoder import VisualEncoder
from aegis.model.checkpoint import save_checkpoint, load_model_checkpoint
from aegis.perception.features import OBJ_FEAT_DIM
from aegis.semantic.laya_questions import LAYA_FEAT_DIM


@pytest.fixture(scope="module")
def cfg():
    c = load_config(); c.model.visual_pretrained = False; c.perception.backend = "motion"; c.runtime.device = "cpu"
    return c


def test_forward_with_and_without_laya(cfg):
    m = AegisModel(cfg.model, 512)
    v, o = torch.randn(3, 16, 512), torch.randn(3, 16, OBJ_FEAT_DIM)
    a = m(v, o)
    b = m(v, o, torch.rand(3, LAYA_FEAT_DIM), torch.tensor([True, True, False]))
    assert a["event_logits"].shape == (3, 2) and a["anomaly_logit"].shape == (3,)
    m.eval()
    with torch.no_grad():
        x = m(v, o, torch.rand(3, LAYA_FEAT_DIM), torch.tensor([False] * 3))["event_logits"]
        y = m(v, o)["event_logits"]
    assert torch.allclose(x, y, atol=1e-5)      # masked Laya == no Laya


def test_trainability_flags(cfg):
    m = AegisModel(cfg.model, 512)
    m.set_trainable(temporal_encoder=False, fusion=False, event_head=True)
    assert all(not p.requires_grad for p in m.temporal_encoder.parameters())
    assert all(p.requires_grad for p in m.heads.parameters())


def test_checkpoint_roundtrip(cfg, tmp_path):
    m = AegisModel(cfg.model, 512).eval()
    p = str(tmp_path / "a.pt"); save_checkpoint(p, m, cfg, {"macro_f1": .5}, 3)
    m2, ck = load_model_checkpoint(p)
    x = torch.randn(1, 16, 512), torch.randn(1, 16, OBJ_FEAT_DIM)
    with torch.no_grad():
        assert torch.allclose(m(*x)["event_logits"], m2(*x)["event_logits"], atol=1e-6)
    assert ck["class_names"] == ["NORMAL", "FIGHT"]


def test_engine_end_to_end_random_frame_to_result(cfg):
    from aegis.pipeline.engine import AegisEngine
    from aegis.semantic.laya_adapter import MockVerifier
    eng = AegisEngine(cfg, verifier=MockVerifier())
    rng = np.random.RandomState(0)
    res = None
    for i in range(400):
        res = eng.process_frame(rng.randint(0, 255, (240, 320, 3), np.uint8), source_fps=30)
    assert {"timestamp", "events", "detections", "tracks", "anomaly_score", "latency_ms"} <= set(res)
    assert res["events"] and res["events"][0]["model_trained"] is False


def test_cpu_training_dry_run(cfg, tmp_path):
    from aegis.training.train import main
    main(["--out", str(tmp_path), "--dry-run"])
    assert (tmp_path / "aegis_best.pt").exists() and (tmp_path / "training_history.json").exists()


def test_onnx_export_parity(cfg, tmp_path):
    pytest.importorskip("onnx"); pytest.importorskip("onnxruntime")
    from aegis.export.onnx_export import export_onnx
    export_onnx(AegisModel(cfg.model, 512).eval(), str(tmp_path / "m.onnx"), 16)


def test_async_pipeline_multicam(cfg):
    from aegis.pipeline.engine import AegisEngine
    from aegis.pipeline.async_pipeline import AsyncPipeline
    from aegis.camera.mock_source import MockSource
    from aegis.semantic.laya_adapter import MockVerifier
    eng = AegisEngine(cfg, verifier=MockVerifier())
    p = AsyncPipeline(eng, {c: MockSource(c, n_frames=150, seed=i) for i, c in enumerate(["CAM-001", "CAM-002"])}).start()
    p.join(60); p.stop()
    assert {r["camera_id"] for r in p.results} == {"CAM-001", "CAM-002"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_inference(cfg):
    from aegis.pipeline.engine import AegisEngine
    cfg.runtime.device = "cuda"
    eng = AegisEngine(cfg)
    assert eng.device == "cuda"
