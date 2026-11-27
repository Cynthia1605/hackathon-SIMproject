"""Overlay (v2) and dashboard (v2) tests on synthetic data (no real video or model needed)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard as DB  # noqa: E402
import landmarks as L  # noqa: E402
import overlay_video as OV  # noqa: E402
from pose_2d import PoseSequence, make_timestamps_ms, save_npz  # noqa: E402

FPS = 30.0
PERIOD = 60.0 / 180.0  # 180 BPM
PATTERN = [(1, "L", "QUICK"), (2, "R", "QUICK"), (3, "L", "SLOW"), (5, "R", "QUICK"), (6, "L", "QUICK"), (7, "R", "SLOW")]


# --------------------------------------------------------------------------- #
# Synthetic report (same schema as epochs_and_steps.json)
# --------------------------------------------------------------------------- #
def make_report(offsets_fn=lambda i, beat, foot: 0.0, epochs=((2.0, 10.0),), mp3_offset=0.0, phase_beats=0,
                qqs=1.0):
    """Steps on a 180 BPM grid inside each epoch; offsets_fn(i, beat, foot) gives ms."""
    grid = mp3_offset + np.arange(0, 80) * PERIOD
    steps = []
    for eid, (a, b) in enumerate(epochs):
        idx = np.flatnonzero((grid >= a + 0.2) & (grid <= b - 0.2))
        for c, k in enumerate(idx[phase_beats:]):
            beat = c % 8 + 1
            plan = [p for p in PATTERN if p[0] == beat]
            if not plan:
                continue
            _, foot, kind = plan[0]
            off = offsets_fn(len(steps), beat, foot)
            t = float(grid[k] + off / 1000.0)
            steps.append({
                "epoch_id": eid, "idx": int(t * FPS), "cam1_frame": int(round(t * FPS)), "timestamp_sec": t,
                "foot": foot, "timing_type": kind, "beat_count": beat,
                "hold_after": "4-hold" if beat == 3 else "8-hold" if beat == 7 else None,
                "interval_sec": 2 * PERIOD if kind == "SLOW" else PERIOD,
                "nearest_music_beat_sec": float(grid[k]), "beat_offset_ms": off,
                "step_length_cm": 25.0 if foot == "L" else 22.0,
                "label": f"{foot}{beat}" + ("-SLOW" if kind == "SLOW" else ""),
            })
    return {
        "subject_id": "synthetic",
        "fps": FPS,
        "music": {"bpm": 180.0, "beat_period_sec": PERIOD, "mp3_offset_sec": mp3_offset,
                  "aligned_beat_times_sec": grid.tolist(), "method": "xcorr", "confidence": 0.8},
        "epochs": [{"epoch_id": i, "start_sec": a, "end_sec": b, "label": "ACTIVE_MOVE"} for i, (a, b) in enumerate(epochs)],
        "steps": steps,
        "summary": {"beat_period_sec": PERIOD, "qqs_pattern_accuracy": qqs},
        "warnings": [],
    }


def skeleton_kp(w=640, h=360) -> np.ndarray:
    """A plausible standing 2D skeleton (33, 3) in pixels (subject's left = image right)."""
    cx, top, bh = w / 2, 0.12 * h, 0.8 * h
    rel = {
        0: (0, 0.05), 2: (0.02, 0.03), 5: (-0.02, 0.03), 7: (0.04, 0.05), 8: (-0.04, 0.05),
        11: (0.1, 0.18), 12: (-0.1, 0.18), 13: (0.14, 0.33), 14: (-0.14, 0.33), 15: (0.15, 0.47), 16: (-0.15, 0.47),
        17: (0.16, 0.5), 18: (-0.16, 0.5), 19: (0.15, 0.51), 20: (-0.15, 0.51), 21: (0.14, 0.5), 22: (-0.14, 0.5),
        23: (0.06, 0.5), 24: (-0.06, 0.5), 25: (0.07, 0.72), 26: (-0.07, 0.72), 27: (0.07, 0.93), 28: (-0.07, 0.93),
        29: (0.06, 0.96), 30: (-0.08, 0.96), 31: (0.11, 0.98), 32: (-0.03, 0.98),
    }
    kp = np.zeros((33, 3), np.float32)
    for j in range(33):
        dx, dy = rel.get(j, (0.0, 0.04))
        kp[j] = [cx + dx * bh, top + dy * bh, 0.95]
    return kp


def close(px, bgr, tol=12):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(px, bgr))


# --------------------------------------------------------------------------- #
# Overlay: English labels & rhythm state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("off,cat,label", [
    (0, "on_time", "ON TIME"), (45, "on_time", "ON TIME"), (-45, "on_time", "ON TIME"),
    (-46, "early", "EARLY"), (60, "late", "LATE"), (90, "late", "LATE"),
    (91, "off_late", "LATE / OFF-BEAT"), (-150, "off_early", "EARLY / OFF-BEAT"), (None, "none", "NO BEAT REF"),
])
def test_classify_offset_thresholds_english(off, cat, label):
    assert OV.classify_offset(off) == (cat, label)


def test_offset_badges():
    assert OV.offset_badge(12) == "ON TIME (+12ms)"
    assert OV.offset_badge(-68) == "EARLY (-68ms)"
    assert OV.offset_badge(115) == "LATE / OFF-BEAT (+115ms)"
    assert OV.CATEGORY_BGR["on_time"] == OV.GREEN and OV.CATEGORY_BGR["early"] == OV.AMBER
    assert OV.CATEGORY_BGR["off_late"] == OV.RED
    assert OV.LEFT_COL == OV.hex_bgr("#00E5FF") and OV.RIGHT_COL == OV.hex_bgr("#FF4081")


def test_timeline_count_phase_and_state():
    rep = make_report(epochs=((2.0, 10.0), (14.0, 20.0)))
    tl = OV.RhythmTimeline.from_report(rep)
    for s in rep["steps"]:
        assert tl.beat_count_at(s["nearest_music_beat_sec"] + 0.01) == s["beat_count"]
    first = rep["steps"][0]
    assert tl.beat_count_at(first["nearest_music_beat_sec"] + 3 * PERIOD + 0.01) == 4  # the hold beat
    assert tl.beat_progress_at(first["nearest_music_beat_sec"] + PERIOD / 2) == pytest.approx(0.5, abs=0.01)
    idle = tl.state_at(12.0)
    assert idle["active"] is False and idle["status"] == "IDLE"
    act = tl.state_at(first["timestamp_sec"] + 0.05)
    assert act["active"] and act["status"] == "DANCING"
    co = act["callouts"]
    assert len(co) == 1 and co[0]["foot_label"] == "LEFT FOOT" and co[0]["beat"] == 1
    assert co[0]["badge"] == "ON TIME (+0ms)" and not co[0]["is_error"]
    assert tl.state_at(first["timestamp_sec"] + 0.7)["callouts"][0]["foot"] == "R"  # L1 expired, R2 shown
    assert tl.state_at(0.5)["callouts"] == []


def test_state_error_callouts_and_flash():
    rep = make_report(lambda i, beat, foot: 115.0 if beat == 3 else (-68.0 if beat == 2 else 10.0))
    tl = OV.RhythmTimeline.from_report(rep)
    s3 = next(s for s in rep["steps"] if s["beat_count"] == 3)
    st = tl.state_at(s3["timestamp_sec"] + 0.1)
    c3 = next(c for c in st["callouts"] if c["beat"] == 3)
    assert c3["is_error"] and c3["category"] == "off_late" and c3["badge"] == "LATE / OFF-BEAT (+115ms)"
    assert c3["foot_label"] == "LEFT FOOT"
    c2 = next((c for c in st["callouts"] if c["beat"] == 2), None)
    if c2 is not None:
        assert c2["badge"] == "EARLY (-68ms)" and c2["category"] == "early"
    assert 3 in st["error_flash"] and st["error_flash"][3] == pytest.approx(0.1, abs=1e-6)
    assert tl.state_at(s3["timestamp_sec"] + 0.7)["error_flash"].get(3) is None  # flash over after 0.6 s
    ok, n = st["on_time"]
    assert n >= 3 and ok < n


def _half_time_report():
    """Report whose song grid is half-time (librosa octave error) but steps were scored at full tempo."""
    rep = make_report(lambda i, b, f: 20.0)
    grid = np.asarray(rep["music"]["aligned_beat_times_sec"])
    rep["music"]["aligned_beat_times_sec"] = grid[::2].tolist()  # every other beat only
    rep["music"]["beat_period_sec"] = 2 * PERIOD
    rep["music"]["bpm"] = 90.0
    return rep, grid


def test_octave_corrected_grid_used_for_counts_and_rescoring():
    rep, full = _half_time_report()
    np.testing.assert_allclose(OV.analysis_beat_grid(rep)[:20], full[:20])
    tl = OV.RhythmTimeline.from_report(rep)
    for s in rep["steps"]:
        assert tl.beat_count_at(s["nearest_music_beat_sec"] + 0.01) == s["beat_count"]
    d = DB.build_dashboard_data(rep, mp3_offset=0.01)  # music 10 ms later than the scored 0.0 s
    offs = [s["offset_ms"] for s in d["steps"]]
    assert max(abs(o) for o in offs) <= PERIOD * 500 + 1  # never more than half a (real) beat
    assert all(abs(o - 10.0) < 1 for o in offs)
    # an exported grid takes precedence
    rep["beat_grid_sec"] = full.tolist()
    np.testing.assert_allclose(OV.analysis_beat_grid(rep), full)


def test_timeline_per_epoch_phase():
    rep = make_report(epochs=((2.0, 10.0),))
    rep2 = make_report(epochs=((14.0, 22.0),), phase_beats=2)
    for s in rep2["steps"]:
        s["epoch_id"] = 1
    rep["steps"] += rep2["steps"]
    rep["epochs"].append({"epoch_id": 1, "start_sec": 14.0, "end_sec": 22.0})
    tl = OV.RhythmTimeline.from_report(rep)
    for s in rep["steps"]:
        assert tl.beat_count_at(s["nearest_music_beat_sec"] + 0.01) == s["beat_count"]


# --------------------------------------------------------------------------- #
# Overlay: drawing
# --------------------------------------------------------------------------- #
def test_skeleton_visual_hierarchy():
    frame = np.zeros((360, 640, 3), np.uint8)
    kp = skeleton_kp()
    OV.draw_skeleton(frame, kp, scale=0.5)
    lx, ly = kp[L.LEFT_ANKLE, :2].astype(int)
    rx, ry = kp[L.RIGHT_ANKLE, :2].astype(int)
    assert close(frame[ly, lx], OV.LEFT_COL, 2)  # crisp electric cyan joint
    assert close(frame[ry, rx], OV.RIGHT_COL, 2)  # crisp magenta/coral joint
    mx, my = ((kp[25, :2] + kp[27, :2]) / 2).astype(int)
    assert close(frame[my, mx], OV.LEFT_COL, 2)  # leg line
    # glow halo just beside the leg line: tinted, but not full intensity
    halo = frame[my, mx + 7].astype(int)
    assert 0 < halo.max() < 230
    # upper body is faint: ~35% of the slate colour on black
    sx, sy = ((kp[11, :2] + kp[13, :2]) / 2).astype(int)
    px = frame[sy, sx].astype(int)
    assert 10 < px.max() <= int(0.4 * max(OV.UPPER_COL)) + 2
    f2 = np.zeros_like(frame)
    kp2 = kp.copy()
    kp2[:, 2] = 0.1
    OV.draw_skeleton(f2, kp2)
    assert f2.max() == 0


def test_callout_beside_offending_foot():
    rep = make_report(lambda i, beat, foot: 115.0 if beat == 3 else 0.0)
    tl = OV.RhythmTimeline.from_report(rep)
    s3 = next(s for s in rep["steps"] if s["beat_count"] == 3)
    frame = np.zeros((360, 640, 3), np.uint8)
    kp = skeleton_kp()
    st = tl.state_at(s3["timestamp_sec"] + 0.2)
    drawn = OV.draw_callouts(frame, st, kp)
    d3 = next(d for d in drawn if d["foot"] == "L")
    cx, cy = d3["center"]
    foot = OV.foot_center(kp, "L")
    assert abs(cx - foot[0]) <= 1 and abs(cy - foot[1]) <= 1
    assert d3["category"] == "off_late"
    # red ring around the left foot
    ring = frame[cy, cx - d3["ring_r"] - 1 : cx - d3["ring_r"] + 3]
    assert any(close(p, OV.RED, 40) for p in ring)
    # the tag sits beside the foot, on the outside of the body (left foot = image right here)
    tx, ty, tw, th = d3["tag"]
    hips_x = kp[[23, 24], 0].mean()
    assert tx > hips_x and abs(tx - cx) < 120 and ty < cy
    # no callouts once the 0.6 s window has passed
    assert OV.draw_callouts(np.zeros_like(frame), tl.state_at(s3["timestamp_sec"] + 0.65), kp) == [] or \
        all(d["foot"] != "L" for d in OV.draw_callouts(np.zeros_like(frame), tl.state_at(s3["timestamp_sec"] + 0.65), kp))


def test_beat_strip_active_and_error_flash():
    rep = make_report(lambda i, beat, foot: 115.0 if beat == 3 else 0.0)
    tl = OV.RhythmTimeline.from_report(rep)
    s1 = rep["steps"][0]
    s3 = next(s for s in rep["steps"] if s["beat_count"] == 3)

    def cell_px(frame, box, n):
        x0, y0, sw, sh = box
        s = frame.shape[0] / 720.0
        cells_x0 = x0 + int(84 * s)
        gap = int(6 * s)
        cw = (x0 + sw - int(10 * s) - cells_x0 - 7 * gap) // 8
        return frame[y0 + int(12 * s), cells_x0 + (n - 1) * (cw + gap) + 3]

    f1 = np.full((360, 640, 3), 80, np.uint8)
    box = OV.draw_beat_strip(f1, tl.state_at(s1["nearest_music_beat_sec"] + 0.05))
    assert close(cell_px(f1, box, 1), OV.BEAT_ON, 25)  # beat 1 lit amber
    assert not close(cell_px(f1, box, 2), OV.BEAT_ON, 25)
    f3 = np.full((360, 640, 3), 80, np.uint8)
    box = OV.draw_beat_strip(f3, tl.state_at(s3["timestamp_sec"] + 0.05))
    assert close(cell_px(f3, box, 3), OV.RED, 40)  # error on beat 3 flashes red (wins over active)
    assert box[1] + box[3] <= 360 and box[1] > 280  # strip at the bottom


def test_draw_hud_status_english():
    rep = make_report()
    tl = OV.RhythmTimeline.from_report(rep)
    s0 = rep["steps"][0]
    active = np.full((360, 640, 3), 255, np.uint8)
    OV.draw_hud(active, tl.state_at(s0["timestamp_sec"] + 0.05), skeleton_kp())
    idle = np.full((360, 640, 3), 255, np.uint8)
    OV.draw_hud(idle, tl.state_at(14.0))
    assert close(active[17, 10], OV.GREEN, 22) and close(idle[17, 10], OV.GREY, 22)  # chip padding, 94% opaque
    assert (active[150:250, 520:] == 255).all()  # video untouched away from HUD/feet


def _write_synthetic_video(path: Path, n: int = 48, w: int = 640, h: int = 360):
    import cv2

    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), 60 + i, np.uint8))
    wr.release()


@pytest.fixture
def synthetic_assets(tmp_path):
    vid = tmp_path / "cam1.mp4"
    _write_synthetic_video(vid)
    kp = np.repeat(skeleton_kp()[None], 48, axis=0)
    npz = save_npz(PoseSequence(kp, FPS, 640, 360, make_timestamps_ms(48, FPS)), tmp_path / "cam1_pose2d.npz")
    rep = make_report(epochs=((0.3, 1.5),), mp3_offset=0.0)
    (tmp_path / "report.json").write_text(json.dumps(rep))
    (tmp_path / "sync.json").write_text(json.dumps({"method": "clap", "frame_offset": 9}))
    return vid, npz, tmp_path


def _frames(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_render_overlay_opencv_only(synthetic_assets):
    vid, npz, tmp = synthetic_assets
    out = tmp / "overlay_cam1.mp4"
    info = OV.render_overlay(vid, npz, tmp / "report.json", out, tmp / "sync.json", use_ffmpeg=False)
    assert out.is_file() and not (tmp / "overlay_cam1.raw.mp4").exists()
    assert info["frames"] == 48 and info["codec"] in ("avc1", "mp4v")
    frames = _frames(out)
    assert len(frames) == 48
    kp = skeleton_kp()
    x, y = kp[L.LEFT_ANKLE, :2].astype(int)
    b, g, r = frames[20][y, x].astype(int)  # electric cyan (lossy codec: approximate)
    assert b > 170 and g > 160 and r < 100


def test_render_overlay_ffmpeg_h264(synthetic_assets):
    ffmpeg = OV.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    vid, npz, tmp = synthetic_assets
    out = tmp / "overlay_cam1.mp4"
    info = OV.render_overlay(vid, npz, tmp / "report.json", out, tmp / "sync.json", max_seconds=1.0)
    assert info["codec"] == "h264" and info["frames"] == 30
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(out)], stderr=subprocess.PIPE).stderr.decode()
    assert "h264" in probe and "yuv420p" in probe
    assert len(_frames(out)) == 30


# --------------------------------------------------------------------------- #
# Dashboard: scoring, verdicts, coaching (English)
# --------------------------------------------------------------------------- #
def test_timing_credit_and_score():
    half = PERIOD * 500
    assert DB.timing_credit(0, half) == 1.0 and DB.timing_credit(45, half) == 1.0
    assert DB.timing_credit(90, half) == pytest.approx(0.5)
    assert DB.timing_credit(half, half) == 0.0
    assert DB.tempo_score([0, 10, -20], half, 1.0) == pytest.approx(100)
    assert DB.tempo_score([], half, 1.0) is None


def test_dashboard_excellent_dancer():
    rng = np.random.default_rng(1)
    d = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.normal(0, 12))))
    assert d["verdict"]["text"] == "Excellent rhythm — on tempo" and d["verdict"]["level"] == "excellent"
    assert d["score"] >= 90
    k = d["kpis"]
    assert k["on_tempo_pct"] > 95 and k["bpm"] == 180.0 and k["qqs_accuracy_pct"] == 100.0
    assert any("Beats 1 and 5 land on time" in n["text"] for n in d["coaching"])
    s0 = d["steps"][0]
    assert s0["status"] == "On Time" and s0["foot"] == "L" and s0["beat"] == 1 and not s0["is_error"]
    json.dumps(d, ensure_ascii=False)


def test_dashboard_rushing_dancer():
    d = DB.build_dashboard_data(make_report(lambda i, b, f: -60.0 + (i % 3) * 5))
    assert "rush" in d["verdict"]["text"]
    assert d["kpis"]["mean_offset_ms"] < -50 and d["kpis"]["n_errors"] == d["kpis"]["n_steps"]
    assert any("come early" in n["text"] for n in d["coaching"])
    assert {s["status"] for s in d["steps"]} == {"Early"} and all(s["is_error"] for s in d["steps"])


def test_dashboard_off_tempo_dancer():
    rng = np.random.default_rng(2)
    d = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.choice([-1, 1]) * rng.uniform(110, 160)),
                                            qqs=0.4))
    assert d["verdict"]["text"] == "Off tempo" and d["score"] < 50
    assert {s["status"] for s in d["steps"]} <= {"Early (off-beat)", "Late (off-beat)"}


def test_coaching_slow_rush_asymmetry_drift():
    rush = DB.build_dashboard_data(make_report(lambda i, beat, foot: -70.0 if beat in (3, 7) else 0.0))
    texts = " ".join(n["text"] for n in rush["coaching"])
    assert "Slow steps 3 and 7 rush" in texts and "Beats 1 and 5 land on time" in texts
    asym = DB.build_dashboard_data(make_report(lambda i, beat, foot: 60.0 if foot == "R" else 5.0))
    assert "Right foot is less precise" in " ".join(n["text"] for n in asym["coaching"])
    sym = DB.build_dashboard_data(make_report(lambda i, beat, foot: 10.0))
    assert "Good left/right symmetry" in " ".join(n["text"] for n in sym["coaching"])
    drift = make_report(epochs=((2.0, 10.0), (14.0, 22.0)))
    for s in drift["steps"]:
        s["beat_offset_ms"] = 0.0 if s["epoch_id"] == 0 else 70.0
    assert "drifts later" in " ".join(n["text"] for n in DB.build_dashboard_data(drift)["coaching"])


def test_offbeat_grid_warning_and_wrapped_offsets():
    rng = np.random.default_rng(3)
    wrapped = DB.build_dashboard_data(make_report(lambda i, b, f: (150.0 if i % 2 else -150.0) + float(rng.normal(0, 8))))
    assert any("off-beat" in w for w in wrapped["warnings"])
    texts = " ".join(n["text"] for n in wrapped["coaching"])
    assert "land on time" not in texts and "Good left/right symmetry" not in texts
    assert "irregular" in texts and "Both feet" in texts
    ok = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.normal(0, 8))))
    assert not any("off-beat" in w for w in ok["warnings"])


def test_shift_report_rescoring():
    rep = make_report(lambda i, b, f: 60.0)  # dancer consistently 60 ms late
    d0 = DB.build_dashboard_data(rep)
    assert d0["mp3_offset_sec"] == 0.0 and d0["mp3_offset_default_sec"] == 0.0
    assert d0["kpis"]["on_tempo_pct"] == 0
    d1 = DB.build_dashboard_data(rep, mp3_offset=0.06)  # move the music 60 ms later
    assert d1["mp3_offset_sec"] == pytest.approx(0.06) and d1["mp3_offset_default_sec"] == 0.0
    assert d1["kpis"]["on_tempo_pct"] == 100 and abs(d1["kpis"]["mean_offset_ms"]) < 1
    assert d1["beats"][0]["t"] == pytest.approx(0.06)
    assert rep["music"]["mp3_offset_sec"] == 0.0  # original untouched
    # a shift of more than half a beat wraps to the nearest beat
    d2 = DB.build_dashboard_data(rep, mp3_offset=PERIOD)
    assert abs(d2["kpis"]["mean_offset_ms"] - 60) < 1


def test_default_offset_is_zero_not_the_stale_report_value():
    """An old report with a bogus auto-aligned offset (-130.757 s) must play from 0.0 s."""
    good = make_report(lambda i, b, f: 10.0)  # scored with the song starting at 0.0 s
    stale = DB.shift_report(good, -130.757)  # what an old xcorr run left in epochs_and_steps.json
    assert stale["music"]["mp3_offset_sec"] == pytest.approx(-130.757)
    d = DB.build_dashboard_data(stale)
    assert d["mp3_offset_sec"] == 0.0 and d["mp3_offset_default_sec"] == 0.0
    assert d["beats"][0]["t"] == pytest.approx(0.0)
    assert d["kpis"]["on_tempo_pct"] == 100 and d["kpis"]["mean_offset_ms"] == pytest.approx(10, abs=0.5)
    # an explicit sessions.json override is used as the default instead
    d2 = DB.build_dashboard_data(good, default_offset=0.01)
    assert d2["mp3_offset_sec"] == pytest.approx(0.01) and d2["mp3_offset_default_sec"] == pytest.approx(0.01)
    assert d2["kpis"]["mean_offset_ms"] == pytest.approx(0, abs=0.5)


def test_page_has_no_alignment_status_label():
    html = DB.INDEX_HTML
    assert "music alignment" not in html and "analysed offset" not in html and "ANALYZED" not in html
    assert "primeAudio" in html and "audio blocked" in html  # audio starts inside the user's click


def test_dashboard_without_steps():
    rep = make_report()
    rep["steps"] = []
    d = DB.build_dashboard_data(rep)
    assert d["score"] is None and d["verdict"]["text"] == "No steps detected"
    assert d["coaching"] and d["kpis"]["n_steps"] == 0


def test_page_is_english_single_screen():
    html = DB.INDEX_HTML
    for s in ("Rhythm Dashboard", "Coaching diagnosis", "Step error timeline", "Rhythm error &amp; step log",
              "Errors only", "All steps", "Tap Beat 1 Now", "Save Offset to sessions.json", "Music offset",
              "Time (s)", "Foot", "Beat", "Error (ms)", "Status", "0.5x", "0.75x", "1.0x"):
        assert s in html, s
    assert "height:100vh" in html and "overflow:hidden" in html and "grid-template-columns:minmax(0,55fr) minmax(0,45fr)" in html
    for spanish in ("Veredicto", "Pasos", "cámara lenta", "Recomendaciones", "Desfase"):
        assert spanish not in html
    assert "<script src=" not in html  # self-contained


# --------------------------------------------------------------------------- #
# Dashboard: HTTP server, save-offset endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(tmp_path):
    video = tmp_path / "overlay_cam1.mp4"
    video.write_bytes(bytes(range(256)) * 40)
    audio = tmp_path / "song_sync.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 1000)
    (tmp_path / "secret.txt").write_text("no")
    cfg = {"sessions": [{"id": "s1", "cam1": "a.mp4", "cam2": "b.mp4", "song_mp3": "s.mp3", "height_meters": 1.7,
                         "overrides": {"frame_offset": 3, "mp3_offset_sec": None}, "params": {"x": 1}},
                        {"id": "s2", "cam1": "c.mp4", "cam2": "d.mp4", "song_mp3": "s.mp3", "height_meters": 1.6}]}
    cfg_path = tmp_path / "sessions.json"
    cfg_path.write_text(json.dumps(cfg))
    media = {"video": True, "video_is_overlay": True, "audio": True, "audio_format": "wav"}
    srv = DB.make_server(DB.shift_report(make_report(), -130.757), media, video, audio, "127.0.0.1", 0,
                         save_cb=lambda v: DB.save_offset_to_config(cfg_path, "s1", v))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", video, cfg_path
    srv.shutdown()
    srv.server_close()


def _req(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_server_serves_page_and_data(server):
    base, _, _ = server
    st, hd, body = _req(base + "/")
    assert st == 200 and "text/html" in hd["Content-Type"] and b"Rhythm Dashboard" in body
    st, _, body = _req(base + "/api/data")
    d = json.loads(body)
    assert d["subject_id"] == "synthetic" and d["steps"] and d["media"]["can_save"] is True
    assert d["mp3_offset_sec"] == 0.0  # stale -130.757 s report offset ignored: song plays from 0.0 s
    st, _, body = _req(base + "/api/data?mp3_offset=0.6")
    assert json.loads(body)["mp3_offset_sec"] == pytest.approx(0.6)
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/api/data?mp3_offset=nan")
    assert e.value.code == 400


def test_server_range_requests(server):
    base, video, _ = server
    raw = video.read_bytes()
    st, hd, body = _req(base + "/media/video", headers={"Range": "bytes=100-199"})
    assert st == 206 and body == raw[100:200] and hd["Content-Range"] == f"bytes 100-199/{len(raw)}"
    st, hd, body = _req(base + "/media/video", headers={"Range": "bytes=-50"})
    assert st == 206 and body == raw[-50:]
    st, hd, _ = _req(base + "/media/audio", headers={"Range": "bytes=0-9"})
    assert st == 206 and hd["Content-Type"] == "audio/wav"
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/media/video", headers={"Range": f"bytes={len(raw) + 10}-"})
    assert e.value.code == 416


@pytest.mark.parametrize("path", ["/secret.txt", "/media/../secret.txt", "/media/other", "/sessions.json"])
def test_server_only_serves_whitelisted_routes(server, path):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + path)
    assert e.value.code == 404


def test_save_offset_endpoint_persists(server):
    base, _, cfg_path = server
    st, _, body = _req(base + "/api/save-offset", json.dumps({"mp3_offset_sec": 1.23456}).encode(),
                       {"Content-Type": "application/json", "Origin": base})
    res = json.loads(body)
    assert st == 200 and res["ok"] and res["mp3_offset_sec"] == pytest.approx(1.2346)
    cfg = json.loads(cfg_path.read_text())
    s1, s2 = cfg["sessions"]
    assert s1["overrides"] == {"frame_offset": 3, "mp3_offset_sec": 1.2346}  # other overrides kept
    assert s1["params"] == {"x": 1} and "overrides" not in s2  # rest of the file untouched
    _, _, body = _req(base + "/api/data")
    d = json.loads(body)
    assert d["media"]["saved_offset"] == pytest.approx(1.2346)
    assert d["mp3_offset_sec"] == pytest.approx(1.2346) and d["mp3_offset_default_sec"] == pytest.approx(1.2346)


@pytest.mark.parametrize("data,headers,code", [
    (b'{"mp3_offset_sec": 1.0}', {"Content-Type": "text/plain"}, 415),  # simple (CSRF-able) request refused
    (b'{"mp3_offset_sec": 1.0}', {"Content-Type": "application/json", "Origin": "http://evil.example"}, 403),
    (b'{"mp3_offset_sec": "abc"}', {"Content-Type": "application/json"}, 400),
    (b'{"mp3_offset_sec": 1e9}', {"Content-Type": "application/json"}, 400),
    (b'{"wrong": 1}', {"Content-Type": "application/json"}, 400),
    (b'not json', {"Content-Type": "application/json"}, 400),
])
def test_save_offset_rejects_bad_requests(server, data, headers, code):
    base, _, cfg_path = server
    before = cfg_path.read_text()
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/api/save-offset", data, headers)
    assert e.value.code == code
    assert cfg_path.read_text() == before


def test_save_offset_read_only(tmp_path):
    srv = DB.make_server(make_report(), {"video": False, "audio": False}, None, None, "127.0.0.1", 0, save_cb=None)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base + "/api/save-offset", b'{"mp3_offset_sec": 1.0}', {"Content-Type": "application/json"})
        assert e.value.code == 403
        assert json.loads(_req(base + "/api/data")[2])["media"]["can_save"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_save_offset_to_config_validation(tmp_path):
    p = tmp_path / "sessions.json"
    p.write_text(json.dumps({"sessions": [{"id": "a"}]}))
    assert DB.save_offset_to_config(p, "a", -0.25)["mp3_offset_sec"] == -0.25
    assert json.loads(p.read_text())["sessions"][0]["overrides"]["mp3_offset_sec"] == -0.25
    with pytest.raises(KeyError):
        DB.save_offset_to_config(p, "nope", 1.0)
    with pytest.raises(ValueError):
        DB.save_offset_to_config(p, "a", float("inf"))


def test_load_subject_prepares_wav_and_falls_back_to_original_video(tmp_path):
    ffmpeg = OV.find_ffmpeg()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cam1.mp4").write_bytes(b"x")
    out = tmp_path / "outputs" / "s1"
    out.mkdir(parents=True)
    (out / "epochs_and_steps.json").write_text(json.dumps(make_report()))
    song = tmp_path / "data" / "song.mp3"
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        str(song)], check=True)
    cfg = {"sessions": [{"id": "s1", "cam1": "data/cam1.mp4", "cam2": "data/cam2.mp4", "song_mp3": "data/song.mp3",
                         "height_meters": 1.7}]}
    (tmp_path / "sessions.json").write_text(json.dumps(cfg))
    report, media, video, audio, notes, sid, floor = DB.load_subject(str(tmp_path / "sessions.json"), "s1")
    assert sid == "s1" and video == tmp_path / "data" / "cam1.mp4" and not media["video_is_overlay"]
    assert any("overlay_video.py" in n for n in notes)
    assert floor["available"] is False and any("Footwork map unavailable" in n for n in notes)  # no points3d.npz
    assert media["default_offset"] == 0.0 and media["offset_source"] == "default"
    cfg["sessions"][0]["overrides"] = {"mp3_offset_sec": 0.42}
    (tmp_path / "sessions.json").write_text(json.dumps(cfg))
    _, media2, *_ = DB.load_subject(str(tmp_path / "sessions.json"), "s1", prepare_audio=False)
    assert media2["default_offset"] == 0.42 and media2["offset_source"] == "sessions.json"
    if ffmpeg:
        assert audio == out / "song_sync.wav" and audio.read_bytes()[:4] == b"RIFF" and media["audio_format"] == "wav"
    with pytest.raises(SystemExit):
        DB.load_subject(str(tmp_path / "sessions.json"), "nope")



# =========================================================================== #
# Top-down footwork reconstruction (footwork.py + dashboard floor map)
# =========================================================================== #
import footwork as FW  # noqa: E402


def make_floor_dancer(yaw_deg=35.0, origin=(1.2, -0.4), height=1.70, back_scale=1.0, drift_cm=(0.0, 0.0),
                      cycles=4, fps=30.0, toes=True, offsets_fn=lambda i, b, f: 0.0):
    """Synthetic metric 3D feet doing the ideal On-1 basic, facing ``yaw`` in an arbitrary room position.

    Returns (points3d, cam1_frames, fps, report). The world frame has floor Y = 0. A drifting
    dancer closes each cycle (beats 6 and 7) at a centre shifted by ``drift_cm`` (right, forward),
    which becomes the next cycle's stance.
    """
    geo = FW.ideal_geometry(height)
    a = np.radians(yaw_deg)
    f = np.array([np.sin(a), np.cos(a)])  # facing (X, Z)
    r = np.array([-f[1], f[0]])  # dancer's right
    o = np.asarray(origin, dtype=float)
    to_world = lambda x_cm, y_cm: o + (x_cm * r + y_cm * f) / 100.0  # noqa: E731
    beat_frames = 10
    n = int(fps) + cycles * 8 * beat_frames + 30
    hs, R = geo["half_stance_cm"], geo["reach_cm"]
    pos = {"L": np.zeros((n, 2)), "R": np.zeros((n, 2))}
    cur = {"L": np.array([-hs, 0.0]), "R": np.array([hs, 0.0])}
    events, steps = [], []
    anchor = np.zeros(2)
    frame = int(fps)
    plan = {1: ("L", (-hs, R)), 2: ("R", (hs, 0)), 3: ("L", (-hs, 0)), 5: ("R", (hs, -R * back_scale)),
            6: ("L", (-hs, 0)), 7: ("R", (hs, 0))}
    for c in range(cycles):
        for b in range(1, 9):
            if b == 6:
                anchor = anchor + np.asarray(drift_cm, dtype=float)
            if b in plan:
                foot, (x, y) = plan[b]
                events.append((frame, foot, anchor + np.array([x, y])))
                i = len(steps)
                off = offsets_fn(i, b, foot)
                steps.append({"epoch_id": 0, "idx": frame, "cam1_frame": frame, "timestamp_sec": frame / fps,
                              "foot": foot, "beat_count": b, "timing_type": "SLOW" if b in (3, 7) else "QUICK",
                              "beat_offset_ms": off, "label": f"{foot}{b}", "nearest_music_beat_sec": frame / fps})
            frame += beat_frames
    ev = sorted(events, key=lambda e: e[0])
    k = 0
    for t in range(n):
        while k < len(ev) and ev[k][0] <= t:
            cur[ev[k][1]] = ev[k][2]
            k += 1
        pos["L"][t], pos["R"][t] = cur["L"], cur["R"]
    pts = np.full((n, 33, 3), np.nan)
    for foot, (ank, heel, toe) in {"L": (27, 29, 31), "R": (28, 30, 32)}.items():
        w = np.array([to_world(*p) for p in pos[foot]])  # (n, 2) X, Z
        for j, d, y in ((heel, -0.10, 0.0), (toe, 0.14, 0.0), (ank, -0.04, 0.08)):
            pts[:, j, 0] = w[:, 0] + d * f[0]
            pts[:, j, 2] = w[:, 1] + d * f[1]
            pts[:, j, 1] = y
        if not toes:
            pts[:, toe] = np.nan
            pts[:, heel] = np.nan
    report = {"steps": steps,
              "epochs": [{"epoch_id": 0, "start_idx": int(fps) - 5, "end_idx": n, "start_sec": (fps - 5) / fps,
                          "end_sec": n / fps}]}
    return pts, np.arange(n), fps, report, geo


def test_ideal_geometry_scales_with_height():
    g = FW.ideal_geometry(1.70)
    assert 25 <= g["reach_cm"] <= 30 and g["half_stance_cm"] == pytest.approx(9.35, abs=0.1)
    assert g["targets"][1] == ("L", -g["half_stance_cm"], g["reach_cm"])
    assert g["targets"][5][0] == "R" and g["targets"][5][2] == -g["reach_cm"]
    assert FW.ideal_geometry(1.55)["reach_cm"] < FW.ideal_geometry(1.85)["reach_cm"]


@pytest.mark.parametrize("yaw", [0.0, 35.0, -120.0, 180.0])
def test_floor_normalisation_faces_front(yaw):
    pts, frames, fps, rep, geo = make_floor_dancer(yaw_deg=yaw)
    fw = FW.compute_footwork(pts, frames, fps, rep, 1.70)
    assert fw["available"] and fw["method"] == ["foot_facing"]
    st = {(rep["steps"][f["i"]]["beat_count"]): f for f in fw["steps"][:6]}
    hs, R = geo["half_stance_cm"], geo["reach_cm"]
    assert st[1]["x"] == pytest.approx(-hs, abs=0.5) and st[1]["y"] == pytest.approx(R, abs=0.5)  # left fwd, on the left
    assert st[5]["x"] == pytest.approx(hs, abs=0.5) and st[5]["y"] == pytest.approx(-R, abs=0.5)  # right back
    assert st[3]["y"] == pytest.approx(0, abs=0.5) and abs(st[1]["heading_deg"]) < 1
    # live track is in the same frame
    i = int(np.searchsorted(fw["track"]["t"], 0.5))
    assert fw["track"]["L"][i] == pytest.approx([-hs, 0.0], abs=0.6)


def test_floor_normalisation_pca_fallback_without_toes():
    pts, frames, fps, rep, geo = make_floor_dancer(yaw_deg=70.0, toes=False)
    fw = FW.compute_footwork(pts, frames, fps, rep, 1.70)
    assert fw["method"] == ["pca"]
    b1 = next(f for f in fw["steps"] if rep["steps"][f["i"]]["beat_count"] == 1)
    b5 = next(f for f in fw["steps"] if rep["steps"][f["i"]]["beat_count"] == 5)
    assert b1["y"] > 20 and b5["y"] < -20  # sign fixed by the beat-1 step


def test_ideal_dancer_spatial_metrics():
    pts, frames, fps, rep, geo = make_floor_dancer()
    m = FW.compute_footwork(pts, frames, fps, rep, 1.70)["metrics"]
    R = geo["reach_cm"]
    assert m["placement_err_cm"] < 0.5 and all(v < 0.5 for v in m["placement_err_by_beat_cm"].values())
    assert m["fwd_reach_cm"] == pytest.approx(R, abs=0.5) and m["back_reach_cm"] == pytest.approx(R, abs=0.5)
    assert m["fwd_back_symmetry_pct"] == pytest.approx(100, abs=1)
    assert m["big_step_len_L_cm"] == pytest.approx(R, abs=0.5) and m["big_step_len_R_cm"] == pytest.approx(R, abs=0.5)
    assert m["step_len_L_cm"] < R and m["step_len_R_cm"] < R  # replace steps (beats 2/6) move ~0 cm
    assert all(d["dist_cm"] < 0.5 for d in m["drift"])
    notes = FW.spatial_notes(m)
    assert len(notes) == 1 and notes[0]["kind"] == "ok" and "balanced" in notes[0]["text"]


def test_short_backward_step_metrics_and_note():
    pts, frames, fps, rep, geo = make_floor_dancer(back_scale=0.55)
    fw = FW.compute_footwork(pts, frames, fps, rep, 1.70)
    m, R = fw["metrics"], geo["reach_cm"]
    assert m["back_reach_cm"] == pytest.approx(0.55 * R, abs=0.5)
    assert m["fwd_back_symmetry_pct"] == pytest.approx(55, abs=1.5)
    assert m["placement_err_by_beat_cm"][5] == pytest.approx(0.45 * R, abs=0.5)
    assert m["placement_err_by_beat_cm"][1] < 0.5
    b5 = next(f for f in fw["steps"] if rep["steps"][f["i"]]["beat_count"] == 5)
    assert b5["err_cm"] == pytest.approx(0.45 * R, abs=0.5)
    assert (b5["target_x"], b5["target_y"]) == pytest.approx((geo["half_stance_cm"], -R), abs=0.5)
    texts = " ".join(n["text"] for n in FW.spatial_notes(m))
    assert f"Right backward step (beat 5) is {0.45 * R:.0f} cm shorter than left forward step (beat 1)" in texts


def test_drift_detected_and_not_counted_as_placement_error():
    pts, frames, fps, rep, geo = make_floor_dancer(drift_cm=(-7.0, 0.0), cycles=5)
    m = FW.compute_footwork(pts, frames, fps, rep, 1.70)["metrics"]
    d = m["drift"][0]
    assert d["bars"] == 4 and d["dx_cm"] == pytest.approx(-28, abs=1) and abs(d["dy_cm"]) < 1
    # beats 1-5 sit on their cycle's ideal pattern; only the drifting closing steps (6, 7) are off by ~7 cm
    assert all(m["placement_err_by_beat_cm"][b] < 0.5 for b in (1, 2, 3, 5))
    assert m["placement_err_by_beat_cm"][6] == pytest.approx(7, abs=0.5) and m["placement_err_cm"] < 3
    texts = " ".join(n["text"] for n in FW.spatial_notes(m))
    assert "drifts 28 cm left over 4 bars" in texts


def test_assign_cycles():
    steps = [{"beat_count": b, "epoch_id": e} for b, e in
             [(1, 0), (2, 0), (3, 0), (5, 0), (6, 0), (7, 0), (1, 0), (2, 0), (None, 0), (3, 0), (1, 1), (5, 1)]]
    assert FW.assign_cycles(steps) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_dashboard_floor_data_and_spatial_coaching():
    pts, frames, fps, rep, geo = make_floor_dancer(back_scale=0.55)
    full = make_report()  # timing data for the dashboard
    rep2 = dict(full, steps=rep["steps"], epochs=[dict(full["epochs"][0], **rep["epochs"][0])])
    floor = FW.compute_footwork(pts, frames, fps, rep2, 1.70)
    d = DB.build_dashboard_data(rep2, floor=floor)
    assert d["floor"]["available"] and len(d["floor"]["ideal"]["targets"]) == 6
    s5 = next(s for s in d["steps"] if s["beat"] == 5)
    assert s5["floor"]["err_cm"] == pytest.approx(0.45 * geo["reach_cm"], abs=0.5)
    texts = [n["text"] for n in d["coaching"]]
    assert any("backward step (beat 5)" in t for t in texts[:3])  # spatial bullet near the top
    json.dumps(d)
    assert DB.build_dashboard_data(rep2)["floor"]["available"] is False


def test_page_has_footwork_map_controls():
    html = DB.INDEX_HTML
    for sname in ("Camera + Skeleton", "Top-Down Footwork Map", "Split View", "Current 8-Count Cycle",
                  "All Steps Overlay", "FRONT / FACING", "Position offset", "Fwd / back symmetry"):
        assert sname in html, sname


def test_load_floor_from_points3d(tmp_path):
    pts, frames, fps, rep, geo = make_floor_dancer()
    np.savez_compressed(tmp_path / "points3d.npz", points3d=pts.astype(np.float32), cam1_frames=frames, fps=fps)
    fl = DB.load_floor(tmp_path, rep, 1.70)
    assert fl["available"] and fl["metrics"]["placement_err_cm"] < 0.5
    (tmp_path / "points3d.npz").unlink()
    assert DB.load_floor(tmp_path, dict(rep, footwork=fl), 1.70) is not None  # falls back to the report
    assert DB.load_floor(tmp_path, rep, 1.70)["available"] is False
