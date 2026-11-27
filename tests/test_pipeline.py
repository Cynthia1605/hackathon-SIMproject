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


# =========================================================================== #
# Step 2: music_analyzer
# =========================================================================== #
import music_analyzer as MA  # noqa: E402

SR = 22050


def synth_percussion(duration_s: float, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Kick on beats (random accents), conga hits on random off-beats, hi-hats."""
    n = int(duration_s * SR)
    y = np.zeros(n)
    beat = 60.0 / bpm

    def add(t0, burst):
        k = int(t0 * SR)
        if k < n:
            m = min(len(burst), n - k)
            y[k : k + m] += burst[:m]

    tk = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (60 + 60 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 18)
    tc = np.arange(int(0.15 * SR)) / SR
    conga = np.sin(2 * np.pi * 190 * tc) * np.exp(-tc * 25)
    hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.005 * SR))
    t = 0.0
    while t < duration_s:
        add(t, kick * rng.uniform(0.5, 1.0))
        if rng.random() < 0.6:
            add(t + beat * rng.choice([0.5, 0.75]), conga * rng.uniform(0.4, 1.0))
        add(t + beat * 0.5, hat * 0.15)
        t += beat
    return (y / np.abs(y).max()).astype(np.float32)


def noisy_room_recording(song: np.ndarray, offset_s: float, duration_s: float, rng) -> np.ndarray:
    """Song played at cam1 time ``offset_s`` (may be < 0), buried in room noise."""
    n = int(duration_s * SR)
    out = np.zeros(n)
    k = int(round(offset_s * SR))
    src0, dst0 = max(0, -k), max(0, k)
    m = min(len(song) - src0, n - dst0)
    out[dst0 : dst0 + m] = 0.35 * song[src0 : src0 + m]
    t = np.arange(n) / SR
    rumble = np.cumsum(rng.normal(0, 1, n))  # brown noise
    rumble = (rumble - np.convolve(rumble, np.ones(2001) / 2001, mode="same")) / 60
    hum = 0.4 * np.sin(2 * np.pi * 30 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)
    hiss = rng.normal(0, 0.25, n)
    from scipy.signal import butter, sosfilt

    hiss = sosfilt(butter(4, 3000, "highpass", fs=SR, output="sos"), hiss)
    chatter = np.zeros(n)  # random broadband speech-ish bursts
    for _ in range(int(duration_s)):
        s0 = rng.integers(0, n - SR // 2)
        chatter[s0 : s0 + SR // 4] += rng.normal(0, 0.3, SR // 4)
    return (out + rumble + hum + hiss + chatter).astype(np.float32)


@pytest.fixture(scope="module")
def song_analysis():
    rng = np.random.default_rng(10)
    y = synth_percussion(30.0, 120.0, rng)
    return y, MA.analyze_song_audio(y, SR)


def test_song_bpm_and_beats(song_analysis):
    _, song = song_analysis
    assert song.bpm == pytest.approx(120.0, rel=0.03)
    assert len(song.beat_times_sec) > 40
    assert np.median(np.diff(song.beat_times_sec)) == pytest.approx(0.5, rel=0.03)


@pytest.mark.parametrize("true_offset", [4.37, -3.21])
def test_mp3_offset_recovered_through_room_noise(song_analysis, true_offset):
    y, song = song_analysis
    rec = noisy_room_recording(y, true_offset, 34.0, np.random.default_rng(11))
    off, conf = MA.estimate_mp3_offset(song, rec, SR, (60, 250))
    assert off == pytest.approx(true_offset, abs=0.03)
    assert conf > 0.2


def test_align_music_end_to_end_and_manual_override(song_analysis, monkeypatch, tmp_path):
    y, song = song_analysis
    rec = noisy_room_recording(y, 2.5, 34.0, np.random.default_rng(12))
    import soundfile as sf

    song_path = tmp_path / "song.wav"
    sf.write(song_path, y, SR)
    monkeypatch.setattr(sync, "load_video_audio", lambda path, sr=SR: rec)

    res = MA.align_music(str(song_path), "cam1.mp4")
    assert res.method == "xcorr"
    assert res.mp3_offset_sec == pytest.approx(2.5, abs=0.03)
    assert res.aligned_beat_times_sec[0] == pytest.approx(res.beat_times_mp3_sec[0] + res.mp3_offset_sec)
    assert res.beat_period_sec == pytest.approx(0.5, rel=0.03)

    man = MA.align_music(str(song_path), "cam1.mp4", manual_offset_sec=1.25, song=song)
    assert man.method == "manual" and man.mp3_offset_sec == 1.25
    assert man.aligned_beat_times_sec[0] == pytest.approx(song.beat_times_sec[0] + 1.25)
    import json

    json.dumps(man.to_dict())


def test_align_music_handles_missing_video_audio(song_analysis, monkeypatch):
    _, song = song_analysis

    def boom(path, sr=SR):
        raise RuntimeError("no audio stream")

    monkeypatch.setattr(sync, "load_video_audio", boom)
    res = MA.align_music("unused.mp3", "cam1.mp4", song=song)
    assert res.method == "unaligned" and res.mp3_offset_sec == 0.0 and res.warnings


# =========================================================================== #
# Step 2: calibrate_3d
# =========================================================================== #
import calibrate_3d as C3  # noqa: E402

HEIGHT = 1.70
EYE_Y = HEIGHT * 0.935


def body_template() -> np.ndarray:
    """33 landmarks in metres, body frame: Y up, facing +Z, subject's left = +X."""
    T = np.zeros((33, 3))
    T[0] = [0, EYE_Y - 0.03, 0.10]
    for i, x in zip((1, 2, 3), (0.015, 0.032, 0.048)):
        T[i] = [x, EYE_Y, 0.08]
        T[i + 3] = [-x, EYE_Y, 0.08]
    T[7], T[8] = [0.075, EYE_Y - 0.03, 0.0], [-0.075, EYE_Y - 0.03, 0.0]
    T[9], T[10] = [0.025, EYE_Y - 0.08, 0.08], [-0.025, EYE_Y - 0.08, 0.08]
    T[11], T[12] = [0.19, 1.42, 0.0], [-0.19, 1.42, 0.0]
    T[13], T[14] = [0.25, 1.15, -0.02], [-0.25, 1.15, -0.02]
    T[15], T[16] = [0.27, 0.92, 0.06], [-0.27, 0.92, 0.06]
    for i, d in zip((17, 19, 21), ([0.01, -0.07, 0.05], [0.0, -0.09, 0.07], [-0.03, -0.05, 0.08])):
        T[i] = T[15] + d
        T[i + 1] = T[16] + np.array([-d[0], d[1], d[2]])
    T[23], T[24] = [0.10, 0.95, 0.0], [-0.10, 0.95, 0.0]
    T[25], T[26] = [0.10, 0.50, 0.03], [-0.10, 0.50, 0.03]
    T[27], T[28] = [0.10, 0.08, 0.0], [-0.10, 0.08, 0.0]
    T[29], T[30] = [0.10, 0.0, -0.05], [-0.10, 0.0, -0.05]
    T[31], T[32] = [0.11, 0.0, 0.15], [-0.11, 0.0, 0.15]
    return T


def synth_dancer(n: int = 300, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """World (Y-up, floor Y=0) trajectory and a per-frame 'both feet grounded' mask."""
    T = body_template()
    t = np.arange(n) / fps
    pts = np.zeros((n, 33, 3))
    grounded = np.ones(n, bool)
    left_leg = [25, 27, 29, 31]
    right_leg = [26, 28, 30, 32]
    for i in range(n):
        B = T.copy()
        if t[i] > 3.0:  # stepping: alternate foot lifts
            phase = np.sin(2 * np.pi * 1.0 * t[i])
            lift_l, lift_r = 0.10 * max(0.0, phase), 0.10 * max(0.0, -phase)
            B[left_leg, 1] += lift_l * np.array([0.4, 1, 1, 1])
            B[right_leg, 1] += lift_r * np.array([0.4, 1, 1, 1])
            B[[15, 17, 19, 21], 2] += 0.15 * np.sin(2 * np.pi * 0.7 * t[i])
            grounded[i] = max(lift_l, lift_r) < 0.005
        yaw = np.deg2rad(35) * np.sin(2 * np.pi * 0.15 * t[i])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        trans = np.array([0.6 * np.sin(2 * np.pi * 0.08 * t[i]), 0.0, 0.4 * np.cos(2 * np.pi * 0.11 * t[i])])
        pts[i] = B @ Ry.T + trans
    return pts, grounded


def look_at(cam_pos, target):
    f = target - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


def project(world_pts, R, C, K, rng, noise_px=0.5):
    cam = (world_pts - C) @ R.T
    uv = cam[..., :2] / cam[..., 2:3]
    px = uv * K[0, 0] + K[:2, 2]
    px += rng.normal(0, noise_px, px.shape)
    out = np.concatenate([px, np.full(px.shape[:-1] + (1,), 0.95)], axis=-1)
    return out.astype(np.float32)


def two_camera_views(world, angle_deg=(-40, 35), radius=4.0, cam_h=1.3, seed=20):
    rng = np.random.default_rng(seed)
    K = C3.approx_intrinsics(1280, 720, 0.85)
    target = np.array([0.0, 1.0, 0.0])
    views = []
    for a in angle_deg:
        a = np.deg2rad(a)
        C = np.array([radius * np.sin(a), cam_h, radius * np.cos(a)])
        views.append(project(world, look_at(C, target), C, K, rng))
    return views


@pytest.fixture(scope="module")
def calibrated_dancer():
    world, grounded = synth_dancer()
    kp1, kp2 = two_camera_views(world)
    pts, calib = C3.calibrate_and_triangulate(kp1, kp2, (1280, 720), (1280, 720), HEIGHT)
    return world, grounded, pts, calib


def test_calibration_geometry(calibrated_dancer):
    _, _, pts, calib = calibrated_dancer
    assert calib.n_inliers > 0.9 * calib.n_correspondences
    assert calib.reprojection_error_px < 2.0
    # true relative rotation between the cameras is 75 degrees
    ang = np.rad2deg(np.arccos((np.trace(calib.R) - 1) / 2))
    assert ang == pytest.approx(75, abs=2)
    assert pts.shape == (300, 33, 3)


def test_metric_height_recovered_within_2pct(calibrated_dancer):
    _, _, pts, _ = calibrated_dancer
    eye_y = 0.5 * (pts[:, L.LEFT_EYE, 1] + pts[:, L.RIGHT_EYE, 1])
    est_height = np.median(eye_y) / 0.935
    assert est_height == pytest.approx(HEIGHT, rel=0.02)
    sw = np.median(np.linalg.norm(pts[:, 11] - pts[:, 12], axis=1))
    assert sw == pytest.approx(0.38, rel=0.02)


def test_floor_plane_at_y0_and_y_up(calibrated_dancer):
    _, grounded, pts, _ = calibrated_dancer
    feet = pts[grounded][:, C3.FOOT_IDX, 1]
    assert abs(np.median(feet)) < 0.015
    assert np.percentile(np.abs(feet), 95) < 0.04
    assert np.median(pts[:, L.NOSE, 1]) > 1.4  # +Y is up
    # stepping foot lifts are positive heights
    assert np.nanmax(pts[:, L.LEFT_HEEL, 1]) > 0.07


def test_triangulation_propagates_nan():
    world, _ = synth_dancer(120)
    kp1, kp2 = two_camera_views(world)
    kp1[50] = np.nan
    kp1[50, :, L.VIS] = 0
    kp2[60, L.LEFT_WRIST, L.VIS] = 0.1
    pts, _ = C3.calibrate_and_triangulate(kp1, kp2, (1280, 720), (1280, 720), HEIGHT)
    assert np.all(np.isnan(pts[50]))
    assert np.all(np.isnan(pts[60, L.LEFT_WRIST]))
    assert np.all(np.isfinite(pts[61]))


def test_calibrate_session_uses_sync_offset():
    world, _ = synth_dancer(200)
    kp1, kp2 = two_camera_views(world)
    off = 12  # cam2 started recording earlier, so its frame index is ahead
    pad = np.full((off, 33, 3), np.nan, np.float32)
    pad[..., 2] = 0
    kp2_shifted = np.concatenate([pad, kp2])
    pts, calib, idx = C3.calibrate_session(make_pose(kp1), make_pose(kp2_shifted), off, HEIGHT)
    assert len(idx) == 200 and calib.reprojection_error_px < 2.0


# =========================================================================== #
# Step 2: filters
# =========================================================================== #
import filters as FL  # noqa: E402


def jittery_trajectory(n=600, fps=30.0, noise=0.01, seed=30):
    """Slow sway (idle/standing-like) plus 1 cm triangulation jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    truth = np.zeros((n, 33, 3))
    truth[..., 0] = 0.05 * np.sin(2 * np.pi * 0.3 * t)[:, None]
    truth[..., 1] = 1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t)[:, None] + np.arange(33) * 0.01
    truth[..., 2] = 0.04 * np.cos(2 * np.pi * 0.2 * t)[:, None]
    return truth, truth + rng.normal(0, noise, truth.shape)


def test_one_euro_reduces_jitter():
    truth, noisy = jittery_trajectory()
    out = FL.filter_points3d(noisy, 30.0)
    raw_err = np.sqrt(np.mean((noisy - truth) ** 2))
    f_err = np.sqrt(np.mean((out[30:] - truth[30:]) ** 2))
    assert f_err < 0.7 * raw_err
    raw_jit = np.mean(np.abs(np.diff(noisy, 2, axis=0)))
    f_jit = np.mean(np.abs(np.diff(out, 2, axis=0)))
    assert f_jit < 0.4 * raw_jit


def test_one_euro_preserves_sharp_footfall_deceleration():
    fps = 30.0
    t = np.arange(120) / fps
    x = np.zeros((120, 1, 3))
    x[:, 0, 0] = np.where(t < 2.0, 1.5 * t, 3.0)  # foot slides at 1.5 m/s, then plants
    noisy = x + np.random.default_rng(31).normal(0, 0.01, x.shape)
    out = FL.filter_points3d(noisy, fps)
    settle = int(np.argmax(np.abs(out[60:, 0, 0] - 3.0) < 0.02))
    assert settle <= 3  # within ~100 ms of the plant
    assert np.std(out[90:, 0, 0]) < 0.006


@pytest.mark.parametrize("gap", [3, 20])
def test_one_euro_recovers_after_nan_dropout(gap):
    truth, noisy = jittery_trajectory()
    noisy[200 : 200 + gap] = np.nan  # full-frame dropout
    noisy[300:310, 5] = np.nan  # single-joint dropout
    out = FL.filter_points3d(noisy, 30.0)
    assert np.all(np.isnan(out[200 : 200 + gap]))
    assert np.all(np.isnan(out[300:310, 5]))
    after = np.concatenate([np.arange(200 + gap, 300), np.arange(310, 600)])
    assert np.all(np.isfinite(out[after]))
    assert np.all(np.isfinite(out[300:310, 6]))
    # first valid frame after the gap is close to the truth (no stale state), then settles
    assert np.max(np.abs(out[200 + gap] - truth[200 + gap])) < 0.05
    settle = slice(200 + gap + 10, 300)
    assert np.sqrt(np.mean((out[settle] - truth[settle]) ** 2)) < 0.007


def test_one_euro_passes_through_constant_and_init():
    x = np.ones((50, 2, 3))
    x[0, 1] = np.nan
    out = FL.one_euro_filter(x, 30.0)
    assert np.isnan(out[0, 1]).all()
    np.testing.assert_allclose(out[1:], 1.0)


# =========================================================================== #
# Step 3: synthetic end-to-end salsa timeline
# =========================================================================== #
import epoch_segmenter as ES  # noqa: E402
import main as M  # noqa: E402
import salsa_rhythm as SR_  # noqa: E402

BPM = 180.0
BEAT = 60.0 / BPM
ACTIVE_WINDOWS = [(10.0, 26.0), (34.0, 50.0)]
STEP_PLAN = {  # beat -> (foot, target z of that foot in body frame, lift height)
    1: ("L", 0.25, 0.06), 2: ("R", -0.05, 0.035), 3: ("L", 0.0, 0.06),
    5: ("R", -0.25, 0.06), 6: ("L", 0.05, 0.035), 7: ("R", 0.0, 0.06),
}
LEG = {"L": {"knee": 25, "foot": [27, 29, 31]}, "R": {"knee": 26, "foot": [28, 30, 32]}}


def _smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def salsa_timeline(height=1.70, fps=30.0, duration=50.0, mp3_offset=1.7, late_s=(0.0, 0.06),
                   jitter_s=0.010, seed=40):
    """World-frame 3D trajectory of a dancer: idle + clap, then 1-2-3 5-6-7 epochs.

    Returns (points (N,33,3), truth_steps, beat_grid_cam1, clap_time).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    t = np.arange(n) / fps
    s = height / 1.70
    T = body_template() * s
    grid = mp3_offset + np.arange(0, int((duration + 2) / BEAT)) * BEAT

    # ---- per-foot step events (touchdown time, target z, lift) ----
    truth, events = [], {"L": [], "R": []}
    for w, (w0, w1) in enumerate(ACTIVE_WINDOWS):
        beats = grid[(grid >= w0 + 0.3) & (grid <= w1 - 0.1)]
        for c, g in enumerate(beats):
            b = c % 8 + 1
            if b not in STEP_PLAN:
                continue
            foot, z, lift = STEP_PLAN[b]
            td = g + late_s[w] + rng.normal(0, jitter_s)
            events[foot].append((td, z * s, lift))
            truth.append({"t": td, "foot": foot, "beat": b, "epoch": w,
                          "timing_type": "SLOW" if b in (3, 7) else "QUICK", "beat_time": g})
    def foot_track(foot):
        z, y = np.zeros(n), np.zeros(n)
        cur = 0.0
        for td, target, lift in events[foot]:
            dur = 0.22
            u = (t - (td - dur)) / dur
            m = (u >= 0) & (u <= 1)
            z[t > td] = target
            z[m] = cur + (target - cur) * _smoothstep(u[m])
            y[m] = lift * np.sin(np.pi * u[m])
            cur = target
        # before the first event the foot stays at 0; z after last event persists
        first = events[foot][0][0] - 0.22 if events[foot] else np.inf
        z[t < first] = 0.0
        return z, y

    tracks = {f: foot_track(f) for f in ("L", "R")}
    pts = np.zeros((n, 33, 3))
    clap_t = 5.0
    for i in range(n):
        B = T.copy()
        zl, yl = tracks["L"][0][i], tracks["L"][1][i]
        zr, yr = tracks["R"][0][i], tracks["R"][1][i]
        for f, zz, yy in (("L", zl, yl), ("R", zr, yr)):
            B[LEG[f]["foot"], 2] += zz
            B[LEG[f]["foot"], 1] += yy
            B[LEG[f]["knee"], 2] += 0.5 * zz
            B[LEG[f]["knee"], 1] += 0.4 * yy
        upper = [j for j in range(33) if j not in (25, 26, 27, 28, 29, 30, 31, 32)]
        B[upper, 2] += 0.5 * (zl + zr)
        B[upper, 1] += 0.01 * np.sin(2 * np.pi * 0.2 * t[i])  # breathing / sway
        # idle clap: raise hands to chest, clap at clap_t, lower
        raise_w = _smoothstep((t[i] - 4.0) / 0.5) - _smoothstep((t[i] - 6.0) / 0.5)
        if raise_w > 0:
            dtc = t[i] - clap_t
            close = 1.0 - (_smoothstep((dtc + 0.12) / 0.12) - _smoothstep((dtc - 0.06) / 0.3))
            half = 0.015 + (0.22 - 0.015) * close
            for hand, sign in (([15, 17, 19, 21], 1), ([16, 18, 20, 22], -1)):
                tgt = B[hand].copy()
                tgt[:, 0] = sign * half
                tgt[:, 1] = 1.25 * s + (B[hand, 1] - B[hand[0], 1])
                tgt[:, 2] = 0.30
                B[hand] = (1 - raise_w) * B[hand] + raise_w * tgt
        yaw = np.deg2rad(12) * np.sin(2 * np.pi * t[i] / 60)
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        pts[i] = B @ Ry.T + np.array([0.1, 0.0, 0.15])
    return pts, truth, grid, clap_t


def synthetic_subject(height, offset_frames, seed, late_s=(0.0, 0.06)):
    world, truth, grid, clap_t = salsa_timeline(height=height, seed=seed, late_s=late_s)
    kp1, kp2 = two_camera_views(world, seed=seed + 1)
    if offset_frames >= 0:  # cam2 started earlier: cam2_frame = cam1_frame + offset
        kp2 = np.concatenate([np.repeat(kp2[:1], offset_frames, axis=0), kp2])
    else:  # cam1 started earlier
        kp2 = kp2[-offset_frames:]
    song = MA.SongAnalysis(bpm=BPM, beat_times_sec=grid - 1.7, onset_env=np.zeros(1), sr=SR,
                           hop_length=512, duration_sec=60.0)
    music = MA.build_alignment(song, 1.7, "xcorr", 0.9)
    return make_pose(kp1), make_pose(kp2), music, truth, clap_t


def match_truth(steps, truth, tol=0.1):
    """Greedy 1:1 match of detected steps to truth by time."""
    tt = np.array([x["t"] for x in truth])
    pairs, used = [], set()
    for s in steps:
        j = int(np.argmin(np.abs(tt - s["timestamp_sec"])))
        if abs(tt[j] - s["timestamp_sec"]) <= tol and j not in used:
            used.add(j)
            pairs.append((s, truth[j]))
    return pairs


SUBJECTS = {  # id -> (height, cam2 frame offset, seed)
    "subject_A": (1.70, 9, 40),
    "subject_B": (1.62, -6, 50),
}


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    """Run main.main() on two synthetic subjects with pose + music I/O stubbed."""
    tmp = tmp_path_factory.mktemp("e2e")
    data = {sid: synthetic_subject(h, off, seed) for sid, (h, off, seed) in SUBJECTS.items()}
    cfg = json.loads((Path(__file__).resolve().parents[1] / "sessions.example.json").read_text())
    cfg["sessions"] = [
        {"id": sid, "cam1": f"data/{sid}/cam1.mp4", "cam2": f"data/{sid}/cam2.mp4",
         "song_mp3": f"data/{sid}/song.mp3", "height_meters": h,
         "overrides": {"frame_offset": None, "mp3_offset_sec": None}, "params": {}}
        for sid, (h, _, _) in SUBJECTS.items()
    ]
    cfg_path = tmp / "sessions.json"
    cfg_path.write_text(json.dumps(cfg))

    def fake_pose(video_path, cache_path, model_path=None, force=False):
        p = Path(video_path)
        sid, cam = p.parent.name, p.stem
        return data[sid][0] if cam == "cam1" else data[sid][1]

    def fake_music(song_mp3, cam1_video, manual_offset_sec=None, params=None, song=None):
        return data[Path(song_mp3).parent.name][2]

    mp = pytest.MonkeyPatch()
    mp.setattr(M.pose_2d, "load_or_extract", fake_pose)
    mp.setattr(M.music_analyzer, "align_music", fake_music)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = M.main(["--config", str(cfg_path)])
    mp.undo()
    reports = {sid: json.loads((tmp / "outputs" / sid / "epochs_and_steps.json").read_text()) for sid in SUBJECTS}
    return {"rc": rc, "stdout": buf.getvalue(), "tmp": tmp, "data": data, "reports": reports}


import json  # noqa: E402


def test_e2e_runs_and_prints_summary(e2e_run):
    assert e2e_run["rc"] == 0
    out = e2e_run["stdout"]
    for sid in SUBJECTS:
        assert sid in out
    assert "BPM" in out and "QQS acc" in out and "100%" in out


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_outputs_written(e2e_run, sid):
    d = e2e_run["tmp"] / "outputs" / sid
    for f in ("sync.json", "epochs_and_steps.json", "steps.csv", "rhythm_analysis.png"):
        assert (d / f).is_file() and (d / f).stat().st_size > 0
    assert (d / "rhythm_analysis.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    import csv

    with open(d / "steps.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == M.CSV_COLUMNS
    assert len(rows) == len(e2e_run["reports"][sid]["steps"])
    assert {r["subject_id"] for r in rows} == {sid}


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_sync_and_epoch_boundaries(e2e_run, sid):
    rep = e2e_run["reports"][sid]
    assert rep["sync"]["method"] == "clap"
    assert rep["sync"]["frame_offset"] == SUBJECTS[sid][1]
    assert len(rep["beat_grid_sec"]) > 100 and np.median(np.diff(rep["beat_grid_sec"])) == pytest.approx(BEAT, rel=0.01)
    eps = rep["epochs"]
    assert len(eps) == 2
    for ep, (w0, w1) in zip(eps, ACTIVE_WINDOWS):
        assert ep["label"] == "ACTIVE_MOVE"
        assert abs(ep["start_sec"] - w0) <= 0.75
        assert abs(ep["end_sec"] - w1) <= 0.75
    clap_t = e2e_run["data"][sid][4]
    assert not any(ep["start_sec"] <= clap_t <= ep["end_sec"] for ep in eps)


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_no_false_steps_in_idle_or_clap(e2e_run, sid):
    steps = e2e_run["reports"][sid]["steps"]
    idle = [(0.0, ACTIVE_WINDOWS[0][0]), (ACTIVE_WINDOWS[0][1], ACTIVE_WINDOWS[1][0])]
    for s in steps:
        assert not any(a <= s["timestamp_sec"] < b for a, b in idle), s
    assert all(s["timing_type"] != "TAP" for s in steps)


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_quick_slow_classification_is_perfect(e2e_run, sid):
    rep, truth = e2e_run["reports"][sid], e2e_run["data"][sid][3]
    steps = rep["steps"]
    pairs = match_truth(steps, truth)
    assert len(steps) == len(truth) == len(pairs) == 72
    for s, tr in pairs:
        assert s["foot"] == tr["foot"]
        assert s["timing_type"] == tr["timing_type"], (s, tr)
        assert s["beat_count"] == tr["beat"], (s, tr)
        if tr["beat"] in (3, 7):
            assert s["hold_after"] == ("4-hold" if tr["beat"] == 3 else "8-hold")
    assert rep["summary"]["qqs_pattern_accuracy"] == pytest.approx(1.0)
    labels = [s["label"] for s in steps[:6]]
    assert labels == ["L1", "R2", "L3-SLOW", "R5", "L6", "R7-SLOW"]


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_beat_offsets_and_drift(e2e_run, sid):
    rep, truth = e2e_run["reports"][sid], e2e_run["data"][sid][3]
    for s, tr in match_truth(rep["steps"], truth):
        true_off = (tr["t"] - tr["beat_time"]) * 1000
        assert s["beat_offset_ms"] == pytest.approx(true_off, abs=35)
        assert s["nearest_music_beat_sec"] == pytest.approx(tr["beat_time"], abs=1e-6)
    m0, m1 = (e["summary"]["mean_offset_ms"] for e in rep["epochs"])
    assert abs(m0) < 30  # epoch 1 danced on the beat
    assert m1 - m0 == pytest.approx(60, abs=20)  # epoch 2 deliberately 60 ms late
    lengths = [s["step_length_cm"] for s in rep["steps"]]
    assert 15 < np.median(lengths) < 35


@pytest.mark.parametrize("sid", list(SUBJECTS))
def test_e2e_footwork_exported(e2e_run, sid):
    rep = e2e_run["reports"][sid]
    fw = rep["footwork"]
    assert fw["available"] and fw["method"] == ["foot_facing"]
    m = fw["metrics"]
    s = SUBJECTS[sid][0] / 1.70
    # synthetic dancer: L forward 25 cm (x s) on 1 from a +5 cm replace position; R back symmetric
    assert m["fwd_reach_cm"] == pytest.approx(20 * s, abs=4) and m["back_reach_cm"] == pytest.approx(20 * s, abs=4)
    assert m["fwd_back_symmetry_pct"] > 85 and m["placement_err_cm"] < 8
    assert all(d["dist_cm"] < 10 for d in m["drift"])
    l1 = next(st for st in rep["steps"] if st["beat_count"] == 1)
    assert l1["floor_x_cm"] < 0 and l1["floor_y_cm"] > 15  # left foot, on the left, forward


# --------------------------------------------------------------------------- #
# Step 3 unit tests
# --------------------------------------------------------------------------- #
def _feet_only_trajectory(n, fps, active_mask, seed=60):
    """Simple feet-only 3D track: alternating steps (L/R) where active_mask is True."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    pts = np.zeros((n, 33, 3))
    pts[:, :, 1] = 1.0
    for f, (idxs, ph) in {"L": ([27, 29, 31], 0.0), "R": ([28, 30, 32], np.pi)}.items():
        osc = np.where(active_mask, 0.12 * np.sin(2 * np.pi * 1.5 * t + ph), 0.0)
        lift = np.where(active_mask, 0.05 * np.maximum(0, np.sin(2 * np.pi * 1.5 * t + ph)), 0.0)
        pts[:, idxs, 2] = osc[:, None]
        pts[:, idxs, 1] = lift[:, None]
        pts[:, idxs, 0] = 0.1 if f == "L" else -0.1
    pts[:, [23, 24], 1] = 0.95
    return pts + rng.normal(0, 0.002, pts.shape)


def test_segmenter_idle_and_clapping_only_gives_no_epochs():
    fps, n = 30.0, 600
    pts = _feet_only_trajectory(n, fps, np.zeros(n, bool))
    t = np.arange(n) / fps
    pts[:, [15, 17, 19], 0] = (0.2 * np.abs(np.sin(2 * np.pi * 2 * t)))[:, None]  # vigorous clapping
    epochs, diag = ES.segment_epochs(pts, fps)
    assert epochs == []


def test_segmenter_merges_short_holds_and_drops_short_bursts():
    fps, n = 30.0, 900
    t = np.arange(n) / fps
    active = ((t >= 3) & (t < 9)) | ((t >= 9.6) & (t < 15)) | ((t >= 22) & (t < 23.5))
    epochs, _ = ES.segment_epochs(_feet_only_trajectory(n, fps, active), fps)
    assert len(epochs) == 1  # 0.6 s hold merged, 1.5 s burst dropped
    assert abs(epochs[0].start_sec - 3) <= 0.75 and abs(epochs[0].end_sec - 15) <= 0.75
    d = epochs[0].to_dict()
    assert d["label"] == "ACTIVE_MOVE" and d["cam1_end_frame"] >= d["cam1_start_frame"]


def test_segmenter_ignores_walking_in():
    fps, n = 30.0, 450
    t = np.arange(n) / fps
    active = np.ones(n, bool)
    pts = _feet_only_trajectory(n, fps, active)
    pts[:, :, 2] += (1.2 * t)[:, None]  # whole body travels 1.2 m/s
    assert ES.segment_epochs(pts, fps)[0] == []


def test_same_foot_double_trigger_is_tap():
    p = SR_.DEFAULT_RHYTHM_PARAMS
    fps = 30.0
    ev = [(0, "L"), (10, "R"), (20, "L"), (25, "L"), (40, "R")]
    out = SR_.classify_epoch_steps(ev, fps, BEAT, p)
    assert [o["timing_type"] for o in out] == ["QUICK", "QUICK", "SLOW", "TAP", "QUICK"]


def test_beat_period_fallback_and_octave_correction():
    un = MA.MusicAlignment(0.0, float("nan"), 0.0, [], "unaligned", 0.0)
    per, grid, w = SR_.resolve_beat_period(un, 0.333)
    assert per == pytest.approx(0.333) and len(grid) == 0 and w
    half = MA.MusicAlignment(90.0, 0.667, 0.0, list(np.arange(0, 10, 0.6667)), "xcorr", 0.8)
    per, grid, w = SR_.resolve_beat_period(half, 0.333)
    assert per == pytest.approx(0.3333, abs=0.002) and len(grid) > 25 and w
    assert SR_.step_interval_period(np.array([0, 0.33, 0.66, 1.33, 1.66, 2.0, 2.66])) == pytest.approx(0.33, abs=0.01)


def test_load_config_merges_and_resolves(tmp_path):
    cfg = {
        "output_root": "out", "cache_dir": "cache", "model_path": "m.task",
        "defaults": {"sync": {"clap_threshold_sw": 0.4}, "segmenter": {"min_epoch_s": 3.0}},
        "sessions": [{"id": "s1", "cam1": "a.mp4", "cam2": "/abs/b.mp4", "song_mp3": "s.mp3",
                      "height_meters": 1.6, "params": {"segmenter": {"min_epoch_s": 4.0}}}],
    }
    (tmp_path / "sessions.json").write_text(json.dumps(cfg))
    c = M.load_config(tmp_path / "sessions.json")
    s = c["sessions"][0]
    assert s["cam1"] == str(tmp_path / "a.mp4") and s["cam2"] == "/abs/b.mp4"
    assert c["output_root"] == str(tmp_path / "out")
    assert s["params"]["sync"]["clap_threshold_sw"] == 0.4
    assert s["params"]["sync"]["min_closing_speed_sw_per_s"] == 3.0  # module default kept
    assert s["params"]["segmenter"]["min_epoch_s"] == 4.0  # session beats defaults
    assert s["overrides"] == {"frame_offset": None, "mp3_offset_sec": None}
    bad = dict(cfg, sessions=[{"id": "x", "cam1": "a"}])
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        M.load_config(tmp_path / "bad.json")
