"""Synthetic tests: no video, audio or model files required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import landmarks as L  # noqa: E402
import sync  # noqa: E402
from pose_2d import PoseSequence, load_npz, make_timestamps_ms, save_npz  # noqa: E402

FPS = 30.0
SW = 100.0  # shoulder width in px


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #
def standing_skeleton(n: int, rng: np.random.Generator, cx: float = 640.0) -> np.ndarray:
    kp = np.zeros((n, 33, 3), np.float32)
    kp[:, :, L.VIS] = 0.99
    kp[:, L.LEFT_SHOULDER, :2] = [cx + SW / 2, 300]
    kp[:, L.RIGHT_SHOULDER, :2] = [cx - SW / 2, 300]
    kp[:, L.LEFT_HIP, :2] = [cx + 40, 500]
    kp[:, L.RIGHT_HIP, :2] = [cx - 40, 500]
    kp[:, L.LEFT_ANKLE, :2] = [cx + 40, 800]
    kp[:, L.RIGHT_ANKLE, :2] = [cx - 40, 800]
    # hands hanging apart: ~1.6 shoulder widths
    kp[:, L.LEFT_WRIST, :2] = [cx + 80, 520]
    kp[:, L.RIGHT_WRIST, :2] = [cx - 80, 520]
    kp[:, L.LEFT_INDEX, :2] = [cx + 80, 540]
    kp[:, L.RIGHT_INDEX, :2] = [cx - 80, 540]
    kp[:, :, :2] += rng.normal(0, 0.5, size=(n, 33, 2))
    return kp


def set_hand_separation(kp: np.ndarray, sep_sw: np.ndarray, cx: float = 640.0) -> None:
    """Place both hands symmetrically at ``sep_sw`` shoulder widths apart."""
    half = sep_sw * SW / 2
    for wrist, index, sign in ((L.LEFT_WRIST, L.LEFT_INDEX, 1), (L.RIGHT_WRIST, L.RIGHT_INDEX, -1)):
        kp[:, wrist, 0] = cx + sign * half
        kp[:, index, 0] = cx + sign * half
        kp[:, wrist, 1] = 400
        kp[:, index, 1] = 410


def clap_profile(n: int, clap_frames: list[int], open_sw: float = 1.6, closed_sw: float = 0.08,
                 close_frames: int = 4, hold_frames: int = 2, reopen_frames: int = 8) -> np.ndarray:
    sep = np.full(n, open_sw)
    for c in clap_frames:
        for i in range(close_frames):  # fast close -> ~ (1.5/4)*30 ≈ 11 sw/s
            f = c - close_frames + i
            if 0 <= f < n:
                sep[f] = open_sw + (closed_sw - open_sw) * (i + 1) / close_frames
        for i in range(hold_frames):
            if 0 <= c + i < n:
                sep[c + i] = closed_sw
        for i in range(reopen_frames):
            f = c + hold_frames + i
            if 0 <= f < n:
                sep[f] = closed_sw + (open_sw - closed_sw) * (i + 1) / reopen_frames
    return sep


def make_pose(kp: np.ndarray, fps: float = FPS) -> PoseSequence:
    return PoseSequence(kp, fps, 1280, 720, make_timestamps_ms(len(kp), fps))


def clap_session(offset: int, claps_cam1: list[int], n: int = 900, seed: int = 0):
    rng = np.random.default_rng(seed)
    kp1 = standing_skeleton(n, rng)
    set_hand_separation(kp1, clap_profile(n, claps_cam1))
    kp2 = standing_skeleton(n, rng, cx=500.0)
    set_hand_separation(kp2, clap_profile(n, [c + offset for c in claps_cam1]), cx=500.0)
    return make_pose(kp1), make_pose(kp2)


# --------------------------------------------------------------------------- #
# Clap sync tests
# --------------------------------------------------------------------------- #
def test_hand_distance_is_normalised_by_shoulder_width():
    rng = np.random.default_rng(1)
    kp = standing_skeleton(10, rng)
    set_hand_separation(kp, np.full(10, 1.0))
    d = sync.hand_distance_curve(kp)
    assert np.allclose(d, 1.0, atol=0.03)


def test_detect_planted_claps():
    rng = np.random.default_rng(2)
    n, planted = 600, [120, 400]
    kp = standing_skeleton(n, rng)
    set_hand_separation(kp, clap_profile(n, planted))
    events = sync.detect_claps(sync.hand_distance_curve(kp), FPS)
    frames = [e.frame for e in events]
    assert len(frames) == 2
    for got, want in zip(frames, planted):
        assert abs(got - want) <= 1
    assert all(e.depth < 0.35 and e.closing_speed > 3.0 for e in events)


def test_slow_hands_together_is_not_a_clap():
    rng = np.random.default_rng(3)
    n = 300
    kp = standing_skeleton(n, rng)
    # hands drift together over 2 s (~0.75 sw/s) and stay together
    sep = np.concatenate([np.linspace(1.6, 0.1, 60), np.full(n - 60, 0.1)])
    set_hand_separation(kp, sep)
    assert sync.detect_claps(sync.hand_distance_curve(kp), FPS) == []


@pytest.mark.parametrize("offset", [37, -12, 0])
def test_synchronize_recovers_clap_offset(offset):
    p1, p2 = clap_session(offset, [150, 600])
    res = sync.synchronize(p1, p2)
    assert res.method == "clap"
    assert res.frame_offset == offset
    assert res.confidence == pytest.approx(1.0)
    assert res.offset_sec == pytest.approx(offset / FPS)


def test_match_clap_offset_ignores_spurious_detection():
    off, conf, pairs = sync.match_clap_offset([100, 400, 700], [125, 300, 425, 726])
    assert off == 25
    assert len(pairs) == 3 and conf == pytest.approx(1.0)


def test_manual_override_wins():
    p1, p2 = clap_session(20, [150])
    res = sync.synchronize(p1, p2, manual_offset=-5)
    assert res.method == "manual" and res.frame_offset == -5


def test_motion_xcorr_fallback_without_claps():
    rng = np.random.default_rng(4)
    n, offset = 1200, 23
    # non-periodic "stepping" bob: smoothed noise on hip/ankle Y
    base = np.convolve(rng.normal(0, 1, n + 200), np.ones(9) / 9, mode="same") * 15
    kp1 = standing_skeleton(n, rng)
    kp2 = standing_skeleton(n, rng, cx=500.0)
    for j in (L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_ANKLE, L.RIGHT_ANKLE):
        kp1[:, j, L.Y] += base[100 : 100 + n]
        # cam2 frame (n + offset) shows the same moment as cam1 frame n; scaled differently
        kp2[:, j, L.Y] += 0.7 * base[100 - offset : 100 - offset + n]
    res = sync.synchronize(make_pose(kp1), make_pose(kp2))
    assert res.method == "motion_xcorr"
    assert res.frame_offset == offset
    assert res.confidence > 0.5


def test_audio_highpass_locates_click_in_low_frequency_noise():
    sr, dur, click_t = 22050, 4.0, 2.317
    rng = np.random.default_rng(5)
    t = np.arange(int(sr * dur)) / sr
    audio = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)  # bass / room hum
    audio += 0.01 * rng.normal(size=t.size)
    k = int(click_t * sr)
    audio[k : k + 200] += rng.normal(0, 0.6, 200) * np.exp(-np.arange(200) / 40)  # broadband clap
    env = sync.highpass_transients(audio, sr, 2500)
    got = sync.audio_clap_time(env, sr, around_s=2.3, window_s=0.5)
    assert got is not None
    assert abs(got - click_t) * FPS <= 1.0


def test_align_sequences_overlap():
    kp1 = np.arange(100)[:, None, None] * np.ones((1, 33, 3))
    kp2 = np.arange(90)[:, None, None] * np.ones((1, 33, 3))
    a1, a2, idx = sync.align_sequences(kp1, kp2, 10)
    assert len(a1) == len(a2) == len(idx) == 80
    assert np.all(a2[:, 0, 0] - a1[:, 0, 0] == 10)
    a1, a2, idx = sync.align_sequences(kp1, kp2, -5)
    assert idx[0] == 5 and np.all(a1[:, 0, 0] - a2[:, 0, 0] == 5)


def test_sync_result_is_json_serialisable():
    import json

    p1, p2 = clap_session(7, [200])
    json.dumps(sync.synchronize(p1, p2).to_dict())


# --------------------------------------------------------------------------- #
# pose_2d cache (no mediapipe needed)
# --------------------------------------------------------------------------- #
def test_timestamps_strictly_increasing():
    ts = make_timestamps_ms(1000, 29.97)
    assert np.all(np.diff(ts) > 0)


def test_pose_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    kp = standing_skeleton(50, rng)
    kp[10] = np.nan
    kp[10, :, L.VIS] = 0
    seq = make_pose(kp)
    path = save_npz(seq, tmp_path / "c.npz", fingerprint="abc")
    loaded, fp = load_npz(path)
    assert fp == "abc"
    assert loaded.fps == FPS and loaded.width == 1280 and loaded.height == 720
    np.testing.assert_array_equal(np.isnan(loaded.keypoints), np.isnan(kp))
    np.testing.assert_allclose(np.nan_to_num(loaded.keypoints), np.nan_to_num(kp))
