"""Overlay rendering and dashboard tests on synthetic data (no real video or model needed)."""

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
def make_report(offsets_fn=lambda i, beat, foot: 0.0, epochs=((2.0, 10.0),), mp3_offset=0.5, phase_beats=0,
                qqs=1.0):
    """Steps on a 180 BPM grid inside each epoch; offsets_fn(i, beat, foot) gives ms."""
    grid = mp3_offset + np.arange(0, 60) * PERIOD
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
    """A plausible standing 2D skeleton (33, 3) in pixels."""
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


# --------------------------------------------------------------------------- #
# Overlay: pure logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("off,cat,label", [
    (0, "on_time", "A TIEMPO"), (45, "on_time", "A TIEMPO"), (-45, "on_time", "A TIEMPO"),
    (-46, "slight", "LEVEMENTE ADELANTADO"), (60, "slight", "LEVEMENTE ATRASADO"), (90, "slight", "LEVEMENTE ATRASADO"),
    (91, "off", "FUERA DE TEMPO"), (-150, "off", "FUERA DE TEMPO"), (None, "none", "SIN REFERENCIA"),
])
def test_classify_offset_thresholds(off, cat, label):
    assert OV.classify_offset(off) == (cat, label)


def test_timeline_count_phase_and_state():
    rep = make_report(epochs=((2.0, 10.0), (14.0, 20.0)), phase_beats=0)
    tl = OV.RhythmTimeline.from_report(rep)
    for s in rep["steps"]:  # the beat count at each step's own beat matches its label
        assert tl.beat_count_at(s["nearest_music_beat_sec"] + 0.01) == s["beat_count"]
    first = rep["steps"][0]
    assert tl.beat_count_at(first["nearest_music_beat_sec"] + 3 * PERIOD + 0.01) == 4  # the hold beat
    idle = tl.state_at(12.0)
    assert idle["active"] is False and idle["status"] == "EN PAUSA / IDLE"
    act = tl.state_at(first["timestamp_sec"] + 0.05)
    assert act["active"] and act["status"] == "BAILANDO / ACTIVO"
    assert act["last_step"]["label"] == "L1" and act["feedback_label"] == "A TIEMPO" and act["step_flash"]
    assert tl.state_at(0.5)["last_step"] is None
    counts = [b["count"] for b in tl.beats_with_counts()]
    assert set(c for c in counts if c is not None) == set(range(1, 9))


def test_timeline_per_epoch_phase():
    # Epoch 2 starts its count 2 beats later in the grid: phases must differ per epoch
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
def test_draw_skeleton_highlights_lower_body():
    frame = np.zeros((360, 640, 3), np.uint8)
    kp = skeleton_kp()
    OV.draw_skeleton(frame, kp, scale=0.5)
    lx, ly = kp[L.LEFT_ANKLE, :2].astype(int)
    rx, ry = kp[L.RIGHT_ANKLE, :2].astype(int)
    assert tuple(frame[ly, lx]) == OV.NEON_LEFT
    assert tuple(frame[ry, rx]) == OV.NEON_RIGHT
    # knee-ankle segment midpoint is drawn neon (thick lower-body line)
    mx, my = ((kp[L.LEFT_KNEE, :2] + kp[L.LEFT_ANKLE, :2]) / 2).astype(int) if hasattr(L, "LEFT_KNEE") else \
        ((kp[25, :2] + kp[27, :2]) / 2).astype(int)
    assert tuple(frame[my, mx]) == OV.NEON_LEFT
    # upper body is drawn thinner/white
    sx, sy = kp[L.LEFT_SHOULDER, :2].astype(int)
    assert frame[sy, sx].min() > 200
    # invisible joints are not drawn
    f2 = np.zeros_like(frame)
    kp2 = kp.copy()
    kp2[:, 2] = 0.1
    OV.draw_skeleton(f2, kp2)
    assert f2.max() == 0


def test_draw_hud_badges_and_count():
    rep = make_report()
    tl = OV.RhythmTimeline.from_report(rep)
    s0 = rep["steps"][0]
    active = np.full((360, 640, 3), 255, np.uint8)
    OV.draw_hud(active, tl.state_at(s0["timestamp_sec"] + 0.05), skeleton_kp())
    idle = np.full((360, 640, 3), 255, np.uint8)
    OV.draw_hud(idle, tl.state_at(14.0))  # > 2.5 s after the last step: no step feedback
    # translucent dark panel in the top-left corner, video untouched elsewhere
    assert active[100:110, 200:260].mean() < 200
    assert (active[150:, 480:] == 255).all()  # (the footfall ring sits around the ankle)
    # status badge colour: green when active, grey when idle
    pa = active[16, 16].astype(int)  # badge padding, away from anti-aliased text
    pi = idle[16, 16].astype(int)
    assert tuple(pa) == OV.GREEN and tuple(pi) == OV.GREY
    # feedback badge (green, "A TIEMPO") appears only once a step has happened
    green_rows = np.all(active == np.array(OV.GREEN, np.uint8), axis=2).sum(axis=1)
    assert (green_rows[60:] > 20).any()
    assert not np.all(idle == np.array(OV.GREEN, np.uint8), axis=2).any()


def _write_synthetic_video(path: Path, n: int = 48, w: int = 640, h: int = 360):
    import cv2

    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 60 + i, np.uint8)
        wr.write(f)
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
    b, g, r = frames[20][y, x].astype(int)  # neon cyan (lossy codec: approximate)
    assert b > 180 and g > 180 and r < 90


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
# Dashboard: scoring, verdicts, coaching
# --------------------------------------------------------------------------- #
def test_timing_credit_and_score():
    half = PERIOD * 500
    assert DB.timing_credit(0, half) == 1.0 and DB.timing_credit(45, half) == 1.0
    assert DB.timing_credit(90, half) == pytest.approx(0.5)
    assert DB.timing_credit(half, half) == 0.0
    assert DB.tempo_score([0, 10, -20], half, 1.0) == pytest.approx(100)
    assert DB.tempo_score([], half, 1.0) is None


def test_dashboard_perfect_dancer_is_excellent():
    rng = np.random.default_rng(1)
    d = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.normal(0, 12))))
    assert d["verdict"]["texto"] == "¡Excelente ritmo! En tempo" and d["verdict"]["nivel"] == "excelente"
    assert d["score"] >= 90
    assert d["kpis"]["pct_a_tiempo"] > 95 and d["kpis"]["bpm"] == 180.0
    assert d["kpis"]["precision_qqs"] == 100.0
    assert any("1 y 5 caen a tiempo" in n["texto"] for n in d["coaching"])
    assert d["steps"][0]["category_es"] == "A tiempo"
    json.dumps(d, ensure_ascii=False)


def test_dashboard_rushing_dancer():
    d = DB.build_dashboard_data(make_report(lambda i, b, f: -60.0 + (i % 3) * 5))
    assert "adelantarse" in d["verdict"]["texto"]
    assert d["kpis"]["desfase_promedio_ms"] < -50
    assert any("adelantados" in n["texto"] for n in d["coaching"])
    assert {s["category"] for s in d["steps"]} == {"slight_early"}


def test_dashboard_off_tempo_dancer():
    rng = np.random.default_rng(2)
    d = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.choice([-1, 1]) * rng.uniform(110, 160)),
                                            qqs=0.4))
    assert d["verdict"]["texto"] == "Fuera de tiempo" and d["score"] < 50
    assert any("regularidad" in n["texto"] or "Fuera" in n["texto"] or "1 y 5" in n["texto"] for n in d["coaching"])


def test_coaching_slow_steps_rush_and_asymmetry_and_drift():
    rush = DB.build_dashboard_data(make_report(lambda i, beat, foot: -70.0 if beat in (3, 7) else 0.0))
    texts = " ".join(n["texto"] for n in rush["coaching"])
    assert "3 y 7 se apresuran" in texts and "1 y 5 caen a tiempo" in texts

    asym = DB.build_dashboard_data(make_report(lambda i, beat, foot: 60.0 if foot == "R" else 5.0))
    texts = " ".join(n["texto"] for n in asym["coaching"])
    assert "pie derecho es menos preciso" in texts

    sym = DB.build_dashboard_data(make_report(lambda i, beat, foot: 10.0))
    assert "Buena simetría" in " ".join(n["texto"] for n in sym["coaching"])

    drift = make_report(lambda i, b, f: 0.0 if i < 20 else 70.0, epochs=((2.0, 10.0), (14.0, 22.0)))
    for s in drift["steps"]:
        s["beat_offset_ms"] = 0.0 if s["epoch_id"] == 0 else 70.0
    texts = " ".join(n["texto"] for n in DB.build_dashboard_data(drift)["coaching"])
    assert "tiendes a atrasarte" in texts


def test_dashboard_flags_suspected_offbeat_grid():
    rng = np.random.default_rng(3)
    d = DB.build_dashboard_data(make_report(lambda i, b, f: 150.0 + float(rng.normal(0, 8))))
    assert any("contratiempo" in w for w in d["warnings"])
    wrapped = DB.build_dashboard_data(make_report(lambda i, b, f: (150.0 if i % 2 else -150.0) + float(rng.normal(0, 8))))
    assert any("contratiempo" in w for w in wrapped["warnings"])
    texts = " ".join(n["texto"] for n in wrapped["coaching"])
    assert "caen a tiempo" not in texts and "Buena simetría" not in texts  # +/- offsets must not cancel out
    assert "irregulares" in texts and "Ambos pies" in texts
    ok = DB.build_dashboard_data(make_report(lambda i, b, f: float(rng.normal(0, 8))))
    assert not any("contratiempo" in w for w in ok["warnings"])


def test_dashboard_without_steps():
    rep = make_report()
    rep["steps"] = []
    d = DB.build_dashboard_data(rep)
    assert d["score"] is None and d["verdict"]["texto"] == "Sin pasos detectados"
    assert d["coaching"] and d["kpis"]["n_pasos"] == 0


# --------------------------------------------------------------------------- #
# Dashboard: HTTP server
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(tmp_path):
    video = tmp_path / "overlay_cam1.mp4"
    video.write_bytes(bytes(range(256)) * 40)
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 1000)
    (tmp_path / "secret.txt").write_text("no")
    data = DB.build_dashboard_data(make_report(), {"video": True, "video_is_overlay": True, "audio": True})
    srv = DB.make_server(data, video, audio, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", video
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_server_serves_page_and_data(server):
    base, _ = server
    st, hd, body = _get(base + "/")
    html = body.decode("utf-8")
    assert st == 200 and "text/html" in hd["Content-Type"]
    for s in ("Veredicto general de tempo", "Pasos detectados", "cámara lenta", "Metrónomo", "Ajuste de audio"):
        assert s in html
    assert "<script src=" not in html  # self-contained, no CDN
    st, hd, body = _get(base + "/api/data")
    d = json.loads(body)
    assert st == 200 and d["subject_id"] == "synthetic" and d["steps"]


def test_server_range_requests_for_seeking(server):
    base, video = server
    raw = video.read_bytes()
    st, hd, body = _get(base + "/media/video", {"Range": "bytes=100-199"})
    assert st == 206 and body == raw[100:200]
    assert hd["Content-Range"] == f"bytes 100-199/{len(raw)}" and hd["Accept-Ranges"] == "bytes"
    st, hd, body = _get(base + "/media/video", {"Range": "bytes=-50"})
    assert st == 206 and body == raw[-50:]
    st, hd, body = _get(base + "/media/video")
    assert st == 200 and body == raw and hd["Content-Type"] == "video/mp4"
    st, hd, _ = _get(base + "/media/audio", {"Range": "bytes=0-9"})
    assert st == 206 and hd["Content-Type"] == "audio/mpeg"
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/media/video", {"Range": f"bytes={len(raw) + 10}-"})
    assert e.value.code == 416


@pytest.mark.parametrize("path", ["/secret.txt", "/media/../secret.txt", "/media/other", "/api/../secret.txt"])
def test_server_only_serves_whitelisted_routes(server, path):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + path)
    assert e.value.code == 404


def test_load_subject_falls_back_to_original_video(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cam1.mp4").write_bytes(b"x")
    out = tmp_path / "outputs" / "s1"
    out.mkdir(parents=True)
    (out / "epochs_and_steps.json").write_text(json.dumps(make_report()))
    cfg = {"sessions": [{"id": "s1", "cam1": "data/cam1.mp4", "cam2": "data/cam2.mp4", "song_mp3": "data/song.mp3",
                         "height_meters": 1.7}]}
    (tmp_path / "sessions.json").write_text(json.dumps(cfg))
    data, video, audio, notes = DB.load_subject(str(tmp_path / "sessions.json"), "s1")
    assert video == tmp_path / "data" / "cam1.mp4" and audio is None
    assert data["media"]["video"] and not data["media"]["video_is_overlay"]
    assert any("overlay_video.py" in n for n in notes) and any(".mp3" in n for n in notes)
    with pytest.raises(SystemExit):
        DB.load_subject(str(tmp_path / "sessions.json"), "nope")
