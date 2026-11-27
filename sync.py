"""Clap-based synchronisation of cam1/cam2 2D pose sequences.

Offset convention used across the pipeline::

    cam2_frame = cam1_frame + frame_offset      (cam1 is the master timeline)

Precedence: manual override, then visual clap (cross-checked with >2.5 kHz
audio transients), then cross-correlation of vertical hip/ankle velocity.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

import landmarks as L

log = logging.getLogger(__name__)

DEFAULT_SYNC_PARAMS: dict[str, Any] = {
    "clap_threshold_sw": 0.35,
    "min_closing_speed_sw_per_s": 3.0,
    "closing_lookback_s": 0.3,
    "clap_refractory_s": 0.4,
    "visibility_min": 0.5,
    "audio_highpass_hz": 2500,
    "audio_window_s": 0.5,
    "audio_confirm_tolerance_frames": 1,
    "audio_sr": 22050,
    "motion_xcorr_max_lag_s": 10.0,
}


@dataclass
class ClapEvent:
    frame: int
    depth: float  # min hand distance, in shoulder widths
    closing_speed: float  # peak closing speed before the minimum, shoulder widths / s


@dataclass
class SyncResult:
    frame_offset: int
    offset_sec: float
    method: str  # "manual" | "clap" | "clap+audio" | "motion_xcorr"
    confidence: float
    fps_cam1: float
    fps_cam2: float
    clap_frames_cam1: list[int] = field(default_factory=list)
    clap_frames_cam2: list[int] = field(default_factory=list)
    audio_check: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs no longer than ``max_gap`` frames."""
    x = np.asarray(x, dtype=np.float64).copy()
    nan = np.isnan(x)
    if not nan.any() or nan.all():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~nan], x[~nan])
    # only accept fills inside short interior gaps
    run_start = None
    for i in range(len(x) + 1):
        is_nan = i < len(x) and nan[i]
        if is_nan and run_start is None:
            run_start = i
        elif not is_nan and run_start is not None:
            run_len = i - run_start
            interior = run_start > 0 and i < len(x)
            if interior and run_len <= max_gap:
                x[run_start:i] = filled[run_start:i]
            run_start = None
    return x


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x * 0.0


def _midpoint(kp: np.ndarray, a: int, b: int, vis_min: float) -> np.ndarray:
    """(N, 2) midpoint of landmarks a and b. NaN where either one is low-visibility."""
    pts = kp[:, [a, b], :2].astype(np.float64)
    ok = (kp[:, a, L.VIS] >= vis_min) & (kp[:, b, L.VIS] >= vis_min)
    mid = pts.mean(axis=1)
    mid[~ok] = np.nan
    return mid


# --------------------------------------------------------------------------- #
# Visual clap detector
# --------------------------------------------------------------------------- #
def hand_distance_curve(kp: np.ndarray, vis_min: float = 0.5, max_gap: int = 5) -> np.ndarray:
    """Per-frame left/right hand distance in shoulder widths.

    Each hand is the midpoint of its wrist (15/16) and index finger (19/20).
    The distance is divided by the shoulder width (11-12).
    """
    hand_l = _midpoint(kp, L.LEFT_WRIST, L.LEFT_INDEX, vis_min)
    hand_r = _midpoint(kp, L.RIGHT_WRIST, L.RIGHT_INDEX, vis_min)
    sh_ok = (kp[:, L.LEFT_SHOULDER, L.VIS] >= vis_min) & (kp[:, L.RIGHT_SHOULDER, L.VIS] >= vis_min)
    sw = np.linalg.norm(kp[:, L.LEFT_SHOULDER, :2] - kp[:, L.RIGHT_SHOULDER, :2], axis=1).astype(np.float64)
    sw[~sh_ok | (sw < 1e-6)] = np.nan
    # shoulder width changes slowly, so bridge any interior gap
    sw = _interp_gaps(sw, max_gap=10**9)
    d = np.linalg.norm(hand_l - hand_r, axis=1) / sw
    d = _interp_gaps(d, max_gap)
    valid = ~np.isnan(d)
    if valid.sum() >= 3:
        sm = d.copy()
        sm[valid] = median_filter(d[valid], size=3, mode="nearest")
        d = sm
    return d


def detect_claps(
    curve: np.ndarray,
    fps: float,
    threshold: float = 0.35,
    min_closing_speed: float = 3.0,
    lookback_s: float = 0.3,
    refractory_s: float = 0.4,
) -> list[ClapEvent]:
    """Sharp local minima of the hand-distance curve preceded by rapid closing."""
    d = np.asarray(curve, dtype=np.float64)
    if len(d) < 3 or np.all(np.isnan(d)):
        return []
    fill = np.nanmax(d) if np.isfinite(np.nanmax(d)) else 10.0
    d_f = np.where(np.isnan(d), fill, d)
    closing = -np.gradient(d_f) * fps  # positive when hands approach
    closing[np.isnan(d)] = 0.0

    peaks, _ = signal.find_peaks(-d_f, height=-threshold, distance=max(1, int(round(refractory_s * fps))))
    lb = max(1, int(round(lookback_s * fps)))
    events: list[ClapEvent] = []
    for p in peaks:
        lo = max(0, p - lb)
        speed = float(closing[lo : p + 1].max()) if p >= lo else 0.0
        if speed >= min_closing_speed:
            events.append(ClapEvent(frame=int(p), depth=float(d_f[p]), closing_speed=speed))
    return events


def match_clap_offset(
    frames1: Sequence[int], frames2: Sequence[int], tolerance: int = 1
) -> tuple[int, float, list[tuple[int, int]]]:
    """Consensus offset between clap frames of two cameras.

    Returns ``(frame_offset, confidence, matched_pairs)``. The confidence is the
    fraction of the smaller clap set that agrees with the offset.
    """
    f1, f2 = list(map(int, frames1)), list(map(int, frames2))
    if not f1 or not f2:
        raise ValueError("Need at least one clap in each camera")
    candidates = sorted({b - a for a in f1 for b in f2})
    best = None
    for c in candidates:
        pairs, resid = [], 0
        used: set[int] = set()
        for a in f1:
            options = [(abs(b - a - c), b) for b in f2 if b not in used and abs(b - a - c) <= tolerance]
            if options:
                r, b = min(options)
                used.add(b)
                pairs.append((a, b))
                resid += r
        key = (len(pairs), -resid, -abs(c))
        if best is None or key > best[0]:
            best = (key, c, pairs)
    _, c, pairs = best
    offset = int(round(float(np.median([b - a for a, b in pairs]))))
    confidence = len(pairs) / min(len(f1), len(f2))
    return offset, confidence, pairs


# --------------------------------------------------------------------------- #
# Audio cross-check (> 2.5 kHz transients)
# --------------------------------------------------------------------------- #
def load_video_audio(video_path: str, sr: int = 22050) -> np.ndarray:
    """Decode the audio track of a video to mono float32 using the bundled ffmpeg."""
    import imageio_ffmpeg

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg audio decode failed for {video_path}: {proc.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def highpass_transients(audio: np.ndarray, sr: int, cutoff_hz: float = 2500.0, smooth_ms: float = 5.0) -> np.ndarray:
    """Envelope of high-passed audio (Butterworth order 4, zero phase)."""
    cutoff = min(cutoff_hz, 0.45 * sr)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    hp = signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float64))
    env = np.abs(hp)
    win = max(1, int(round(smooth_ms * 1e-3 * sr)))
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(env, size=win)


def audio_clap_time(env: np.ndarray, sr: int, around_s: float, window_s: float = 0.5) -> float | None:
    """Time (s) of the strongest transient within ``±window_s`` of ``around_s``.

    The onset is taken as the first sample reaching 50% of the local peak, which
    is more stable than the argmax of a ringing clap.
    """
    lo = max(0, int((around_s - window_s) * sr))
    hi = min(len(env), int((around_s + window_s) * sr))
    if hi - lo < 2:
        return None
    seg = env[lo:hi]
    peak = int(np.argmax(seg))
    if seg[peak] <= 0:
        return None
    floor = np.median(seg)
    level = floor + 0.5 * (seg[peak] - floor)
    onset = int(np.argmax(seg[: peak + 1] >= level))
    return (lo + onset) / sr


def audio_offset_check(
    video1: str,
    video2: str,
    pairs: list[tuple[int, int]],
    fps1: float,
    fps2: float,
    visual_offset: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    sr = int(params["audio_sr"])
    try:
        env1 = highpass_transients(load_video_audio(video1, sr), sr, params["audio_highpass_hz"])
        env2 = highpass_transients(load_video_audio(video2, sr), sr, params["audio_highpass_hz"])
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    offsets_s = []
    for a, b in pairs:
        t1 = audio_clap_time(env1, sr, a / fps1, params["audio_window_s"])
        t2 = audio_clap_time(env2, sr, b / fps2, params["audio_window_s"])
        if t1 is not None and t2 is not None:
            # cam2_time = cam1_time + offset_s  ->  in cam1 frame units
            offsets_s.append(t2 - t1)
    if not offsets_s:
        return {"status": "no_transient"}
    off_s = float(np.median(offsets_s))
    off_frames = int(round(off_s * fps1))
    tol = int(params["audio_confirm_tolerance_frames"])
    return {
        "status": "confirmed" if abs(off_frames - visual_offset) <= tol else "disagrees",
        "audio_offset_sec": off_s,
        "audio_offset_frames": off_frames,
        "per_clap_offset_sec": offsets_s,
    }


# --------------------------------------------------------------------------- #
# Fallback: vertical hip/ankle velocity cross-correlation
# --------------------------------------------------------------------------- #
def vertical_motion_signal(kp: np.ndarray, vis_min: float = 0.5) -> np.ndarray:
    idx = [L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_ANKLE, L.RIGHT_ANKLE]
    y = kp[:, idx, L.Y].astype(np.float64)
    y[kp[:, idx, L.VIS] < vis_min] = np.nan
    with np.errstate(all="ignore"):
        y_mean = np.nanmean(y, axis=1)
    y_mean = _interp_gaps(y_mean, max_gap=10)
    vel = np.gradient(np.nan_to_num(y_mean, nan=np.nanmean(y_mean) if np.isfinite(y_mean).any() else 0.0))
    vel[np.isnan(y_mean)] = 0.0
    return _zscore(vel)


def motion_xcorr_offset(kp1: np.ndarray, kp2: np.ndarray, fps: float, max_lag_s: float = 10.0,
                        vis_min: float = 0.5) -> tuple[int, float]:
    """Lag k maximising corr(s2[n + k], s1[n]), so cam2_frame = cam1_frame + k."""
    s1 = vertical_motion_signal(kp1, vis_min)
    s2 = vertical_motion_signal(kp2, vis_min)
    c = signal.correlate(s2, s1, mode="full", method="fft")
    lags = signal.correlation_lags(len(s2), len(s1), mode="full")
    max_lag = int(round(max_lag_s * fps))
    m = np.abs(lags) <= max_lag
    c, lags = c[m], lags[m]
    k = int(np.argmax(c))
    denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    conf = float(c[k] / denom) if denom > 0 else 0.0
    return int(lags[k]), conf


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def synchronize(
    pose1,
    pose2,
    video1: str | None = None,
    video2: str | None = None,
    manual_offset: int | None = None,
    params: dict[str, Any] | None = None,
) -> SyncResult:
    """Synchronise two ``pose_2d.PoseSequence`` objects (or anything with .keypoints/.fps)."""
    p = {**DEFAULT_SYNC_PARAMS, **(params or {})}
    fps1, fps2 = float(pose1.fps), float(pose2.fps)
    warnings: list[str] = []
    if abs(fps1 - fps2) / max(fps1, fps2) > 0.005:
        msg = f"Camera fps differ ({fps1:.3f} vs {fps2:.3f}); frame offsets are in cam1 frames"
        log.warning(msg)
        warnings.append(msg)

    def result(off: int, method: str, conf: float, **kw) -> SyncResult:
        return SyncResult(frame_offset=int(off), offset_sec=off / fps1, method=method,
                          confidence=float(conf), fps_cam1=fps1, fps_cam2=fps2, warnings=warnings, **kw)

    if manual_offset is not None:
        return result(int(manual_offset), "manual", 1.0)

    vis = p["visibility_min"]
    claps = []
    for pose in (pose1, pose2):
        curve = hand_distance_curve(pose.keypoints, vis_min=vis)
        claps.append(detect_claps(curve, float(pose.fps), p["clap_threshold_sw"],
                                  p["min_closing_speed_sw_per_s"], p["closing_lookback_s"],
                                  p["clap_refractory_s"]))
    c1 = [e.frame for e in claps[0]]
    c2 = [e.frame for e in claps[1]]

    if c1 and c2:
        # convert cam2 clap frames onto cam1 frame units if fps differ
        c2_in_cam1 = [int(round(f * fps1 / fps2)) for f in c2]
        offset, conf, pairs_cam1 = match_clap_offset(c1, c2_in_cam1)
        back = dict(zip(c2_in_cam1, c2))
        pairs = [(a, back[b]) for a, b in pairs_cam1]
        method, audio = "clap", {"status": "not_run"}
        if video1 and video2:
            audio = audio_offset_check(video1, video2, pairs, fps1, fps2, offset, p)
            if audio.get("status") == "confirmed":
                method = "clap+audio"
            elif audio.get("status") == "disagrees":
                msg = (f"Audio transient offset {audio['audio_offset_frames']} disagrees with "
                       f"visual clap offset {offset}")
                log.warning(msg)
                warnings.append(msg)
        return result(offset, method, conf, clap_frames_cam1=c1, clap_frames_cam2=c2, audio_check=audio)

    log.info("No clap found in both cameras (cam1=%d, cam2=%d); using motion xcorr", len(c1), len(c2))
    offset, conf = motion_xcorr_offset(pose1.keypoints, pose2.keypoints, fps1, p["motion_xcorr_max_lag_s"], vis)
    return result(offset, "motion_xcorr", conf, clap_frames_cam1=c1, clap_frames_cam2=c2)


def align_sequences(kp1: np.ndarray, kp2: np.ndarray, frame_offset: int):
    """Trim both sequences to their overlap.

    Returns ``(kp1_aligned, kp2_aligned, cam1_frame_indices)``.
    """
    k = int(frame_offset)
    n0 = max(0, -k)
    n1 = min(len(kp1), len(kp2) - k)
    if n1 <= n0:
        raise ValueError(f"No overlap between cameras for frame_offset={k}")
    return kp1[n0:n1], kp2[n0 + k : n1 + k], np.arange(n0, n1)


def main(argv: list[str] | None = None) -> None:
    from pose_2d import load_npz

    ap = argparse.ArgumentParser(description="Synchronise cam1/cam2 2D pose caches via clap detection.")
    ap.add_argument("cam1_npz")
    ap.add_argument("cam2_npz")
    ap.add_argument("--video1")
    ap.add_argument("--video2")
    ap.add_argument("--offset", type=int, help="manual frame_offset override")
    ap.add_argument("--out", help="write sync.json here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p1, _ = load_npz(args.cam1_npz)
    p2, _ = load_npz(args.cam2_npz)
    res = synchronize(p1, p2, args.video1, args.video2, args.offset)
    text = json.dumps(res.to_dict(), indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
