import os, cv2, numpy as np, pytest
from aegis.data.label_mapper import LabelMapper, UnmappedLabelError
from aegis.data.manifest import scan_dataset, merge_manifests, VideoRecord
from aegis.data.splitter import split_videos
from aegis.data.validator import check_leakage, LeakageError
from aegis.data.cache import FeatureCache, cache_key

MAP = os.path.join(os.path.dirname(__file__), "..", "configs", "label_mapping.yaml")


def _vid(path, n=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    import zlib
    rng = np.random.RandomState(zlib.crc32(path.encode()))
    for i in range(n):
        w.write(rng.randint(0, 255, (48, 64, 3), np.uint8))
    w.release()


@pytest.fixture
def ds(tmp_path):
    for i in range(6):
        _vid(str(tmp_path / "RWF_2000" / "Fight" / f"f{i}.mp4"))
        _vid(str(tmp_path / "RWF_2000" / "NonFight" / f"n{i}.mp4"))
    return str(tmp_path / "RWF_2000")


def test_label_mapping_explicit():
    m = LabelMapper.from_yaml(MAP)
    assert m.map("RWF_2000", "Fight") == "FIGHT" and m.map("RWF_2000", "nonfight") == "NORMAL"
    with pytest.raises(UnmappedLabelError):
        m.map("RWF_2000", "Violence")
    with pytest.raises(UnmappedLabelError):
        m.map("SomeNewDataset", "Fight")


def test_scan_split_and_no_leakage(ds):
    m = LabelMapper.from_yaml(MAP)
    man = scan_dataset("RWF_2000", ds, m)
    assert man.num_videos == 12 and man.class_distribution == {"FIGHT": 6, "NORMAL": 6}
    recs = merge_manifests([man])
    split = split_videos(recs, seed=1)
    assert sum(len(v) for v in split.values()) == 12
    assert check_leakage(recs, split).ok
    assert m.class_list(["FIGHT", "NORMAL"]) == ["NORMAL", "FIGHT"]


def test_unmapped_folder_errors(tmp_path):
    _vid(str(tmp_path / "RWF_2000" / "Weird" / "a.mp4"))
    with pytest.raises(UnmappedLabelError):
        scan_dataset("RWF_2000", str(tmp_path / "RWF_2000"), LabelMapper.from_yaml(MAP))


def test_leakage_detected_for_shared_group_and_content(ds):
    m = LabelMapper.from_yaml(MAP)
    recs = scan_dataset("RWF_2000", ds, m).videos
    recs[1].group_id = recs[0].group_id            # same source video, different clips
    bad = {"train": [recs[0].video_id], "val": [recs[1].video_id], "test": []}
    with pytest.raises(LeakageError):
        check_leakage(recs, bad)
    recs2 = scan_dataset("RWF_2000", ds, m).videos
    recs2[2].fingerprint = recs2[3].fingerprint     # duplicate file content
    assert not check_leakage(recs2, {"train": [recs2[2].video_id], "val": [recs2[3].video_id], "test": []}, strict=False).ok
    dup = {"train": [recs2[0].video_id], "val": [recs2[0].video_id], "test": []}
    assert not check_leakage(recs2, dup, strict=False).ok


def test_cache_roundtrip_fp16_and_deterministic_key(tmp_path):
    c = FeatureCache(str(tmp_path))
    k = cache_key("fp", {"a": 1}, "enc", "det", "trk")
    assert k == cache_key("fp", {"a": 1}, "enc", "det", "trk") != cache_key("fp", {"a": 2}, "enc", "det", "trk")
    c.put(k, np.random.randn(5, 8), np.random.randn(5, 4), np.arange(5))
    z = c.get(k)
    assert z["vis"].dtype == np.float16 and z["vis"].shape == (5, 8)


def test_merge_manifests_deduplicates_fingerprint(ds):
    m = LabelMapper.from_yaml(MAP)
    man = scan_dataset("RWF_2000", ds, m)
    # Artificially duplicate the first video's fingerprint in the second
    man.videos[1].fingerprint = man.videos[0].fingerprint
    recs = merge_manifests([man], dedup_fingerprint=True)
    assert len(recs) == 11
    assert recs[0].video_id == man.videos[0].video_id
    assert all(r.video_id != man.videos[1].video_id for r in recs)

