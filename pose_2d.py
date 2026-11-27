"""2D keypoint extraction with MediaPipe PoseLandmarker + .npz caching.

Output per video: ``keypoints`` array of shape (num_frames, 33, 3) holding
``[x_px, y_px, visibility]``. Frames without a detection are ``[nan, nan, 0]``.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from landmarks import NUM_LANDMARKS

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_heavy.task")
CACHE_VERSION = 1


@dataclass
class PoseSequence:
    keypoints: np.ndarray  # (N, 33, 3) float32 [x_px, y_px, visibility]
    fps: float
    width: int
    height: int
    timestamps_ms: np.ndarray  # (N,) int64
    video_path: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])


# --------------------------------------------------------------------------- #
# Model download
# --------------------------------------------------------------------------- #
def ensure_model(model_path: str | os.PathLike = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Return the model path, downloading it atomically if it is missing."""
    model_path = Path(model_path)
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading pose model to %s", model_path)
    fd, tmp = tempfile.mkstemp(dir=model_path.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, model_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return model_path


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def make_timestamps_ms(num_frames: int, fps: float) -> np.ndarray:
    """Strictly increasing integer ms timestamps (required by VIDEO mode)."""
    ts = np.round(np.arange(num_frames) * 1000.0 / fps).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1
    return ts


def extract_pose_2d(
    video_path: str | os.PathLike,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    log_every: int = 500,
) -> PoseSequence:
    """Run MediaPipe PoseLandmarker (VIDEO mode, 1 pose) over every frame."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    frames: list[np.ndarray] = []
    timestamps: list[int] = []
    last_ts = -1
    i = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            ts = max(int(round(i * 1000.0 / fps)), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, ts)

            kp = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
            kp[:, 2] = 0.0
            if result.pose_landmarks:
                for j, lm in enumerate(result.pose_landmarks[0][:NUM_LANDMARKS]):
                    kp[j, 0] = lm.x * width
                    kp[j, 1] = lm.y * height
                    kp[j, 2] = lm.visibility if lm.visibility is not None else 1.0
            frames.append(kp)
            timestamps.append(ts)
            i += 1
            if log_every and i % log_every == 0:
                log.info("%s: %d frames processed", Path(video_path).name, i)
    cap.release()

    keypoints = (
        np.stack(frames).astype(np.float32)
        if frames
        else np.zeros((0, NUM_LANDMARKS, 3), np.float32)
    )
    return PoseSequence(
        keypoints=keypoints,
        fps=fps,
        width=width,
        height=height,
        timestamps_ms=np.asarray(timestamps, dtype=np.int64),
        video_path=str(video_path),
    )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def video_fingerprint(video_path: str | os.PathLike, model_path: str | os.PathLike) -> str:
    st = os.stat(video_path)
    return f"v{CACHE_VERSION}|{st.st_size}|{int(st.st_mtime)}|{Path(model_path).name}"


def save_npz(seq: PoseSequence, cache_path: str | os.PathLike, fingerprint: str = "") -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        keypoints=seq.keypoints.astype(np.float32),
        fps=np.float64(seq.fps),
        width=np.int64(seq.width),
        height=np.int64(seq.height),
        timestamps_ms=seq.timestamps_ms.astype(np.int64),
        video_path=np.str_(seq.video_path),
        fingerprint=np.str_(fingerprint),
    )
    return cache_path


def load_npz(cache_path: str | os.PathLike) -> tuple[PoseSequence, str]:
    with np.load(cache_path, allow_pickle=False) as d:
        seq = PoseSequence(
            keypoints=d["keypoints"].astype(np.float32),
            fps=float(d["fps"]),
            width=int(d["width"]),
            height=int(d["height"]),
            timestamps_ms=d["timestamps_ms"].astype(np.int64),
            video_path=str(d["video_path"]),
        )
        fingerprint = str(d["fingerprint"]) if "fingerprint" in d.files else ""
    return seq, fingerprint


def load_or_extract(
    video_path: str | os.PathLike,
    cache_path: str | os.PathLike,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    force: bool = False,
) -> PoseSequence:
    """Load cached 2D keypoints if still valid for this video, else extract and cache."""
    cache_path = Path(cache_path)
    fp = video_fingerprint(video_path, model_path)
    if cache_path.is_file() and not force:
        try:
            seq, cached_fp = load_npz(cache_path)
            if cached_fp == fp:
                log.info("Using cached 2D pose: %s", cache_path)
                return seq
            log.info("Cache stale for %s, re-extracting", video_path)
        except Exception as exc:  # corrupt cache -> re-extract
            log.warning("Failed to read cache %s (%s), re-extracting", cache_path, exc)
    seq = extract_pose_2d(video_path, model_path)
    save_npz(seq, cache_path, fp)
    return seq


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract & cache MediaPipe 2D pose from a video.")
    ap.add_argument("video")
    ap.add_argument("--cache", help="output .npz (default: <video>_pose2d.npz)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache = args.cache or str(Path(args.video).with_suffix("")) + "_pose2d.npz"
    seq = load_or_extract(args.video, cache, args.model, force=args.force)
    detected = np.mean(seq.keypoints[:, 0, 2] > 0) if seq.num_frames else 0.0
    print(f"{seq.num_frames} frames @ {seq.fps:.3f} fps, {seq.width}x{seq.height}, "
          f"detection rate {detected:.1%} -> {cache}")


if __name__ == "__main__":
    main()
