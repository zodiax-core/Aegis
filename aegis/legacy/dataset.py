def scan_dataset(root: str, class_names):
    """Return {class_name: [video_paths]} for whatever actually exists on disk."""
    found = {}
    missing_classes = []
    if not os.path.isdir(root):
        return found, class_names  # everything missing
    for c in class_names:
        class_dir = os.path.join(root, c)
        if not os.path.isdir(class_dir):
            missing_classes.append(c)
            continue
        vids = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
        if vids:
            found[c] = vids
        else:
            missing_classes.append(c)
    return found, missing_classes


ucf_found, ucf_missing = scan_dataset(cfg.ucf_crime_dir, cfg.class_names)
total_videos = sum(len(v) for v in ucf_found.values())

if total_videos == 0:
    print("No dataset found yet at", cfg.ucf_crime_dir)
    print("Download UCF-Crime / RWF-2000 per the instructions above, place them at the paths "
          "in `cfg`, then re-run this cell. Every downstream cell depends on this data existing — "
          "there is nothing meaningful to preprocess or train on without it.")
else:
    print(f"Found {total_videos} videos across {len(ucf_found)} classes.")
    if ucf_missing:
        print("Missing/empty classes (will be skipped):", ucf_missing)
    for c, vids in ucf_found.items():
        print(f"  {c:15s} {len(vids):4d} videos")

print(f"Expected UCF-Crime path: {cfg.ucf_crime_dir}")
print(f"Expected RWF-2000 path: {cfg.rwf2000_dir}")

print("\nListing contents of dataset root:")
!ls -F {cfg.dataset_root}

print("\nListing contents of UCF_Crime directory (if it exists):")
!ls -F {cfg.ucf_crime_dir}

print("\nListing contents of RWF-2000 directory (if it exists):")
!ls -F {cfg.rwf2000_dir}




#CELL GROUP 5

import cv2
import pandas as pd
import matplotlib.pyplot as plt

def probe_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return dict(path=path, fps=fps, n_frames=n_frames, duration_sec=duration, width=w, height=h)


def explore_dataset(found: dict, sample_per_class: int = 15):
    rows = []
    for cls, vids in found.items():
        for p in vids[:sample_per_class]:
            try:
                info = probe_video(p)
                info["class"] = cls
                rows.append(info)
            except Exception as e:
                print(f"  could not probe {p}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("Nothing to explore yet — dataset is empty.")
        return df

    print(df.groupby("class")["duration_sec"].describe()[["count", "mean", "min", "max"]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["class"].value_counts().plot(kind="bar", ax=axes[0], title="Sampled videos per class")
    df["duration_sec"].hist(bins=30, ax=axes[1])
    axes[1].set_title("Duration distribution (sec)")
    plt.tight_layout()
    plt.show()
    return df


if total_videos > 0:
    explore_df = explore_dataset(ucf_found)





#CELL GROUP 6

import hashlib
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((cfg.image_size, cfg.image_size)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _cache_key(video_path: str, start_frame: int, clip_length: int, stride: int) -> str:
    raw = f"{video_path}|{start_frame}|{clip_length}|{stride}|{cfg.image_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def sample_clip(video_path: str, start_frame: int, clip_length: int = None, stride: int = None):
    """Read `clip_length` frames starting at `start_frame`, every `stride`-th frame.
    Returns a (clip_length, 3, H, W) float tensor, or None if the video is too short/corrupt.
    Cached to disk under cfg.cache_dir."""
    clip_length = clip_length or cfg.clip_length
    stride = stride or cfg.frame_stride

    key = _cache_key(video_path, start_frame, clip_length, stride)
    cache_path = os.path.join(cfg.cache_dir, key + ".pt")
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path)
        except Exception:
            pass  # fall through and re-decode if cache is corrupt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(clip_length):
        for _ in range(stride):
            ok, frame = cap.read()
            if not ok:
                break
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess_tf(frame))
    cap.release()

    if len(frames) < clip_length:
        return None  # too short — caller should skip this window

    clip = torch.stack(frames, dim=0)  # (T, C, H, W)
    try:
        torch.save(clip, cache_path)
    except Exception:
        pass
    return clip


def video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n_frames


print("Preprocessing pipeline ready. Cache dir:", cfg.cache_dir)





#CELL GROUP 7

from torch.utils.data import Dataset

def video_level_split(found: dict, cfg: AegisConfig, seed: int = SEED):
    """Split at the video level. Returns three dicts: {class: [paths]} for train/val/test."""
    rng = random.Random(seed)
    train, val, test = {}, {}, {}
    for cls, vids in found.items():
        vids = vids[:]
        rng.shuffle(vids)
        n = len(vids)
        n_train = max(1, int(n * cfg.train_frac))
        n_val = max(1, int(n * cfg.val_frac)) if n > 2 else 0
        train[cls] = vids[:n_train]
        val[cls] = vids[n_train:n_train + n_val]
        test[cls] = vids[n_train + n_val:]
        if not test[cls]:  # guarantee non-empty test where possible
            if len(train[cls]) > 1:
                test[cls] = [train[cls].pop()]
    return train, val, test


class VideoClipDataset(Dataset):
    """Each item is one temporal clip (clip_length frames), its class label,
    the anomaly label (binary), the source video id, and the clip's start timestamp."""

    def __init__(self, split_dict: dict, cfg: AegisConfig, clips_per_video: int = 4, training: bool = True):
        self.cfg = cfg
        self.training = training
        self.class_to_idx = {c: i for i, c in enumerate(cfg.class_names)}
        self.samples = []  # (video_path, start_frame, class_idx)

        for cls, vids in split_dict.items():
            if cls not in self.class_to_idx:
                continue
            cidx = self.class_to_idx[cls]
            for vp in vids:
                fps, n_frames = video_metadata(vp)
                needed = cfg.clip_length * cfg.frame_stride
                if n_frames <= needed:
                    self.samples.append((vp, 0, cidx))
                    continue
                max_start = n_frames - needed
                starts = sorted(random.Random(hash(vp) % (2**31)).sample(
                    range(0, max_start), min(clips_per_video, max_start)))
                for s in starts:
                    self.samples.append((vp, s, cidx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame, class_idx = self.samples[idx]
        clip = sample_clip(video_path, start_frame)
        if clip is None:
            # fall back to a different sample rather than crashing the loader
            return self.__getitem__((idx + 1) % len(self.samples))
        anomaly_label = 0 if self.cfg.class_names[class_idx] == "Normal" else 1
        return {
            "clip": clip,                       # (T, C, H, W)
            "class_label": class_idx,
            "anomaly_label": anomaly_label,
            "video_path": video_path,
            "start_frame": start_frame,
        }


if total_videos > 0:
    train_split, val_split, test_split = video_level_split(ucf_found, cfg)
    print("Train videos:", sum(len(v) for v in train_split.values()))
    print("Val videos:  ", sum(len(v) for v in val_split.values()))
    print("Test videos: ", sum(len(v) for v in test_split.values()))

    # sanity check: no video path should appear in more than one split
    train_paths = {p for v in train_split.values() for p in v}
    val_paths = {p for v in val_split.values() for p in v}
    test_paths = {p for v in test_split.values() for p in v}
    assert not (train_paths & val_paths), "Data leakage: video in both train and val"
    assert not (train_paths & test_paths), "Data leakage: video in both train and test"
    assert not (val_paths & test_paths), "Data leakage: video in both val and test"
    print("No video-level leakage between splits — confirmed.")
