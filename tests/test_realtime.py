"""Phase 5 live coach tests: synthetic landmark streams, no webcam / MediaPipe needed."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import realtime as RT  # noqa: E402

MONO0 = 100.0  # synthetic monotonic time at song t = 0
PLAN = {1: ("L", (-1, 1)), 2: ("R", (1, 0)), 3: ("L", (-1, 0)), 5: ("R", (1, -1)), 6: ("L", (-1, 0)), 7: ("R", (1, 0))}


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def synth_live_stream(bpm=180.0, fps=30.0, bars=4, start_beat=8, offsets_fn=lambda i, c, f: 0.0, noise_m=0.003,
                      yaw_deg=0.0, height=1.70, back_scale=1.0, move_s=0.18, seed=0, tail_s=1.0, still_only=False,
                      clap=False):
    """MediaPipe-like world landmarks (hip-centred, y down, z away from camera) of a dancer doing the
    On-1 basic in front of a webcam, synchronised to a metronome whose song time 0 = mono MONO0.

    Returns (frames [(mono, kp2d, world)], truth [(touchdown mono, foot, count, offset_ms)]).
    """
    import footwork

    rng = np.random.default_rng(seed)
    geo = footwork.ideal_geometry(height)
    hs, R = geo["half_stance_cm"], geo["reach_cm"]
    period = 60.0 / bpm
    events, truth = [], []
    pos = {"L": np.array([-hs, 0.0]), "R": np.array([hs, 0.0])}
    i = 0
    if not still_only:
        for k in range(start_beat, start_beat + 8 * bars):
            c = k % 8 + 1
            if c not in PLAN:
                continue
            foot, (sx, sy) = PLAN[c]
            tgt = np.array([sx * hs, sy * R * (back_scale if c == 5 else 1.0)])
            off = float(offsets_fn(i, c, foot))
            td = MONO0 + k * period + off / 1000.0
            lift = 0.035 if c in (2, 6) else 0.05
            events.append((td, foot, tgt, lift))
            truth.append((td, foot, c, off))
            i += 1
    t_end = MONO0 + (start_beat + 8 * bars) * period + tail_s
    a = np.radians(yaw_deg)
    fwd = np.array([np.sin(a), -np.cos(a)])
    right = np.array([fwd[1], -fwd[0]])
    frames = []
    t = MONO0 + 0.3
    cur = {f: pos[f].copy() for f in pos}
    timeline = []  # (touchdown, foot, target, lift, start position)
    for td, foot, tgt, lift in events:
        timeline.append((td, foot, tgt, lift, cur[foot].copy()))
        cur[foot] = tgt
    while t < t_end:
        p, h = {f: pos[f].copy() for f in pos}, {"L": 0.0, "R": 0.0}
        for td, foot, tgt, lift, start in timeline:
            u = (t - (td - move_s)) / move_s
            if u >= 1:
                p[foot] = tgt.copy()
            elif u > 0:
                p[foot] = start + (tgt - start) * _smooth(u)
                h[foot] = lift * np.sin(np.pi * u)
        hip = 0.5 * (p["L"] + p["R"])
        world = np.full((33, 3), np.nan)
        for foot, (ank, heel, toe) in {"L": (27, 29, 31), "R": (28, 30, 32)}.items():
            v = (p[foot] - hip) / 100.0
            c = v[0] * right + v[1] * fwd
            y = 0.90 - h[foot]
            for j, d, dy in ((heel, -0.10, 0.0), (toe, 0.14, 0.0), (ank, -0.04, -0.08)):
                world[j] = [c[0] + d * fwd[0], y + dy, c[1] + d * fwd[1]]
        world[23], world[24] = [0.1, 0, 0], [-0.1, 0, 0]
        world[11], world[12] = [0.18, -0.5, 0], [-0.18, -0.5, 0]
        if clap:  # vigorous hand motion only
            for j in (15, 17, 19, 21):
                world[j] = [0.2 * abs(np.sin(8 * t)), -0.3, -0.2]
        world += rng.normal(0, noise_m, world.shape)
        kp = np.zeros((33, 3), np.float32)
        ok = np.isfinite(world[:, 0])
        kp[ok, 0] = 640 + 380 * (world[ok, 0] + hip[0] / 100.0)
        kp[ok, 1] = 300 + 380 * world[ok, 1]
        kp[ok, 2] = 0.95
        kp[~ok] = np.nan
        frames.append((t, kp, world))
        t += 1.0 / fps
    return frames, truth


def run_session(frames, bpm=180.0, heartbeat_s=0.25, height=1.70, grid=None, params=None):
    clock = RT.LiveClock()
    sess = RT.LiveSession(height, params, clock, synthetic_time=True)
    sess.set_grid(grid or RT.BeatGrid.metronome(bpm))
    last_hb = -1e9
    for t, kp, world in frames:
        if t - last_hb >= heartbeat_s:
            clock.update(t - MONO0, True, mono=t, source="metronome", bpm=bpm)
            last_hb = t
        sess.on_frame(t, kp, world)
    return sess


def match(steps, truth, tol=0.1):
    out = []
    for td, foot, c, off in truth:
        best = min(steps, key=lambda s: abs(s.t - td), default=None)
        if best is not None and abs(best.t - td) <= tol:
            out.append((best, (td, foot, c, off)))
    return out


# =========================================================================== #
# Clock & beat grid
# =========================================================================== #
def test_live_clock_maps_browser_audio_time():
    c = RT.LiveClock()
    for i in range(10):
        m = 50.0 + i * 0.2
        c.update(m - 48.0 + (0.002 if i % 2 else -0.002), True, mono=m, source="metronome", bpm=180)
    assert c.song_time(52.0, now=52.0) == pytest.approx(4.0, abs=0.003)
    assert c.to_mono(4.0) == pytest.approx(52.0, abs=0.003)
    assert c.song_time(52.0, now=60.0) is None  # stale heartbeats
    c.update(10.0, True, mono=52.2)  # seek / restart: estimate resets to the new position
    assert c.song_time(52.2, now=52.2) == pytest.approx(10.0)
    c.update(10.2, False, mono=52.4)  # paused
    assert c.song_time(52.4, now=52.4) is None
    assert c.status()["bpm"] == 180


def test_beat_grid_metronome_and_song():
    g = RT.BeatGrid.metronome(180)
    assert g.period == pytest.approx(1 / 3)
    k, bt = g.nearest(1.36)
    assert k == 4 and bt == pytest.approx(4 / 3) and g.count_of(k) == 5
    assert g.current(1.5)[0] == 5 and g.current(1.5)[1] == pytest.approx(0.5, abs=0.01)
    g.count_shift = 1
    assert g.count_of(0) == 2
    s = RT.BeatGrid.from_song([0.5, 1.0, 1.5, 2.0, 2.5], 120)
    assert s.bpm == pytest.approx(120) and s.nearest(1.62) == (2, 1.5) and s.nearest(9.0) is None
    assert s.current(2.25)[0] == 4 and s.current(0.1)[0] is None


# =========================================================================== #
# Streaming step detection & live beat-offset classification
# =========================================================================== #
OFFSETS = {3: -70.0, 5: 60.0, 7: 120.0}  # rushing slow 3, late 5, off-beat 7


def _check_detection(sess, truth, max_err_ms=30.0):
    m = match(sess.steps, truth)
    assert len(sess.steps) == len(truth) == len(m)
    for st, (td, foot, c, off) in m:
        assert st.foot == foot and st.count == c
        assert abs(st.t - td) * 1000 <= max_err_ms
        assert st.offset_ms == pytest.approx(off, abs=max_err_ms)
        assert st.category == RT.classify_offset(st.offset_ms)[0]
    lat = [s.latency_ms for s in sess.steps]
    assert np.mean(lat) <= 150 and max(lat) <= 200  # plant confirmed within the ~100-150 ms latency budget
    return m


def test_streaming_detection_and_classification_30fps():
    frames, truth = synth_live_stream(offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0))
    sess = run_session(frames)
    m = _check_detection(sess, truth)
    by = {(tr[2]): st for st, tr in m}
    assert by[1].status == "On Time" and by[1].timing_type == "QUICK"
    assert by[3].status == "Early" and by[3].timing_type == "SLOW" and by[3].label == "L3-SLOW"
    assert by[5].status == "Late" and by[7].status == "Late (off-beat)"
    st = sess.stats()
    assert st["qqs_accuracy_pct"] == 100 and st["one_foot"] == "L" and st["n_steps"] == len(truth)
    assert st["on_tempo_pct"] == pytest.approx(50, abs=1)  # 1, 2, 6 on time; 3, 5, 7 not
    assert 0 < st["tempo_score"] < 100


@pytest.mark.parametrize("kw", [dict(bpm=200.0), dict(yaw_deg=25.0, noise_m=0.005), dict(fps=24.0)])
def test_streaming_detection_robust(kw):
    frames, truth = synth_live_stream(offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0), **kw)
    sess = run_session(frames, bpm=kw.get("bpm", 180.0))
    m = match(sess.steps, truth)
    assert len(m) >= len(truth) - 1
    good = [st for st, tr in m if st.foot == tr[1] and st.count == tr[2] and abs(st.offset_ms - tr[3]) <= 40]
    assert len(good) >= 0.9 * len(truth)


@pytest.mark.parametrize("kw", [dict(still_only=True), dict(still_only=True, clap=True, noise_m=0.004)])
def test_no_false_steps_when_idle_or_clapping(kw):
    frames, _ = synth_live_stream(**kw, bars=3)
    sess = run_session(frames)
    assert sess.steps == []
    assert sess.overlay_state(frames[-1][0])["status"] == "IDLE"


def test_hold_and_tap_classification():
    # step on beat 4 (hold) instead of waiting, and a same-foot double trigger
    frames, truth = synth_live_stream(bars=1, start_beat=8)
    sess = run_session(frames)
    s = sess.steps[0]
    ev = RT.StepEvent(s.t + 4 * (1 / 3), "R", 0.1, 0.05, s.t + 4 * (1 / 3) + 0.1)  # lands on beat 4 of the next phrase? -> HOLD
    hold = sess._add_step(RT.StepEvent(sess.steps[-1].t + 1 / 3, "L" if sess.steps[-1].foot == "R" else "R", 0.1, 0.05,
                                       sess.steps[-1].t + 1 / 3 + 0.1))
    assert hold.count in (4, 8) and hold.timing_type == "HOLD" and hold.status == "Stepped on hold"
    tap = sess._add_step(RT.StepEvent(hold.t + 0.3, hold.foot, 0.05, 0.03, hold.t + 0.4))
    assert tap.timing_type == "TAP" and tap.status == "Tap" and tap.category == "none"
    assert sess.stats()["n_taps"] == 1
    del ev


def test_coaching_every_two_bars():
    frames, _ = synth_live_stream(offsets_fn=lambda i, c, f: -80.0 if c == 3 else 0.0, bars=5)
    sess = run_session(frames)
    assert sess.coach["kind"] == "warn" and sess.coach["bar"] is not None
    assert sess.coach["text"].startswith("Rushing beat 3 on left foot") and "hold your slow step" in sess.coach["text"]
    good = run_session(synth_live_stream(bars=5)[0])
    assert good.coach["kind"] == "ok" and "on tempo" in good.coach["text"]


def test_coaching_messages_unit():
    mk = lambda c, f, off, tt="QUICK": RT.LiveStep(0, 0, 0, f, 1.0, 1.0, off, c, RT.classify_offset(off)[0], "", tt, "", 100)  # noqa: E731
    assert "Late on beat 1, right foot" in RT.coaching_message([mk(1, "R", 95.0)])["text"]
    assert "Early on beat 5" in RT.coaching_message([mk(5, "L", -70.0)])["text"]
    assert "Stepping on the hold (beat 4)" in RT.coaching_message([mk(4, "L", 0.0, "HOLD")])["text"]
    assert RT.coaching_message([])["kind"] == "info"


# =========================================================================== #
# Live top-down floor map
# =========================================================================== #
def test_live_floor_positions_and_targets():
    import footwork

    geo = footwork.ideal_geometry(1.70)
    frames, truth = synth_live_stream(bars=2, yaw_deg=20.0)
    sess = run_session(frames)
    by = {}
    for st in sess.steps:
        by.setdefault(st.count, []).append(st)
    hs, R = geo["half_stance_cm"], geo["reach_cm"]
    for s in by[1]:
        assert s.foot == "L" and s.x == pytest.approx(-hs, abs=1.5) and s.y == pytest.approx(R, abs=2.0)
    for s in by[5]:
        assert s.x == pytest.approx(hs, abs=1.5) and s.y == pytest.approx(-R, abs=2.0)
    errs = [s.err_cm for s in sess.steps if s.err_cm is not None]
    assert len(errs) >= 10 and np.mean(errs) < 2.5
    fs = sess.floor_state()
    assert fs["ideal"]["one_foot"] == "L" and fs["live"] is not None and fs["steps"]
    json.dumps(sess.state_dict())


def test_right_foot_lead_gets_mirrored_targets():
    sess = run_session(synth_live_stream(bars=2)[0])
    # swap feet labels as if the dancer stepped right on 1 (follower)
    for s in sess.steps:
        s.foot = "R" if s.foot == "L" else "L"
    assert sess.one_foot() == "R"
    tx, ty = sess._target(1, "R")
    assert ty < 0 and tx > 0  # right foot goes back on 1
    assert sess._target(1, "L") is None


# =========================================================================== #
# Overlay rendering, model choice, export
# =========================================================================== #
def test_render_live_frame_draws_v2_overlay():
    import overlay_video as OV

    frames, truth = synth_live_stream(bars=1)
    sess = run_session(frames)
    s = sess.steps[2]
    t, kp, _ = min(frames, key=lambda f: abs(f[0] - (s.t + 0.1)))
    state = sess.overlay_state(t)
    assert state["callouts"] and state["callouts"][0]["badge"].endswith("ms)") and state["status"] == "DANCING"
    img = np.zeros((720, 1280, 3), np.uint8)
    out = RT.render_live_frame(img.copy(), kp, state, mirror=True)
    x, y = int(1279 - kp[27, 0]), int(kp[27, 1])  # mirrored left ankle
    assert any(np.abs(out[y + dy, x + dx].astype(int) - np.array(OV.LEFT_COL)).max() < 10
               for dy in range(-2, 3) for dx in range(-2, 3))
    assert out[690, 640].max() > 0  # bottom 8-count strip drawn
    assert RT.placeholder_jpeg()[:2] == b"\xff\xd8"


def test_choose_model_prefers_fast_models(tmp_path, monkeypatch):
    import pose_2d

    called = {}
    monkeypatch.setattr(pose_2d, "ensure_model", lambda p, url=None: called.setdefault("p", Path(p)))
    assert RT.choose_model(tmp_path).name == "pose_landmarker_heavy.task"  # fallback (download)
    (tmp_path / "pose_landmarker_full.task").write_bytes(b"x")
    assert RT.choose_model(tmp_path).name == "pose_landmarker_full.task"
    (tmp_path / "pose_landmarker_lite.task").write_bytes(b"x")
    assert RT.choose_model(tmp_path).name == "pose_landmarker_lite.task"


def test_to_report_is_recorded_session_compatible():
    import dashboard as DB

    frames, truth = synth_live_stream(offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0), bars=2)
    sess = run_session(frames)
    for s in sess.steps:
        s.t_rel = s.t - sess.t0
    rep = sess.to_report("live_test", 30.0, frames[-1][0] - frames[0][0], {"source": "metronome", "mp3_offset_sec": 0.0})
    json.dumps(rep)
    assert rep["live"] and rep["steps"] and rep["epochs"] and rep["beat_grid_sec"] and rep["footwork"]["available"]
    d = DB.build_dashboard_data(rep, floor=rep["footwork"])
    assert d["kpis"]["n_steps"] == len(truth) and d["floor"]["available"]
    assert d["kpis"]["qqs_accuracy_pct"] == 100
    s3 = next(s for s in d["steps"] if s["beat"] == 3)
    assert s3["offset_ms"] == pytest.approx(-70, abs=30) and s3["status"] == "Early"


def test_click_track(tmp_path):
    import soundfile as sf

    p = RT.write_click_track(tmp_path / "m.wav", [(0.5, 1), (0.8333, 2)], 1.2)
    y, sr = sf.read(p)
    assert len(y) == pytest.approx(1.2 * sr, abs=2)
    on = np.flatnonzero(np.abs(y) > 0.1)
    assert on[0] / sr == pytest.approx(0.5, abs=0.002) and np.abs(y).max() > 0.45


def test_main_skips_live_sessions(tmp_path, monkeypatch):
    import main as M

    cfg = {"sessions": [{"id": "live_1", "live": True, "cam1": "a.mp4", "cam2": "a.mp4", "song_mp3": "m.wav",
                         "height_meters": 1.7}]}
    (tmp_path / "sessions.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(M, "run_session", lambda *a, **k: pytest.fail("live session must not be batch-processed"))
    assert M.main(["--config", str(tmp_path / "sessions.json")]) == 0


# =========================================================================== #
# Threaded engine + /api/live/* without a webcam
# =========================================================================== #
class FakeSource:
    """Blank 640x480 frames at ~30 fps."""

    def __init__(self, idx):
        self.idx = idx

    def read(self):
        time.sleep(1 / 30)
        return True, np.full((480, 640, 3), 40, np.uint8)

    def release(self):
        pass


class FakeEstimator:
    """Returns the synthetic dancer's landmarks for the frame's capture time (song t = mono - T0)."""

    model = "fake-landmarker"

    def __init__(self, frames, t0_ref):
        self.frames, self.t0_ref = frames, t0_ref
        self.ts = np.array([f[0] for f in frames])

    def detect(self, frame, t_ms):
        s = t_ms / 1000.0 - self.t0_ref["T0"] + MONO0
        i = int(np.clip(np.searchsorted(self.ts, s), 0, len(self.ts) - 1))
        return self.frames[i][1], self.frames[i][2]

    def close(self):
        pass


@pytest.fixture
def live_server(tmp_path):
    import dashboard as DB

    cfg_path = tmp_path / "sessions.json"
    cfg_path.write_text(json.dumps({"output_root": "outputs", "cache_dir": "cache", "sessions": []}))
    (tmp_path / "data").mkdir()
    frames, truth = synth_live_stream(fps=120.0, bars=1, start_beat=4, offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0),
                                      tail_s=0.6)
    ref = {"T0": None}
    eng = RT.LiveEngine(config_path=str(cfg_path), source_factory=FakeSource,
                        estimator_factory=lambda: FakeEstimator(frames, ref))
    eng.camera_lister = lambda: [{"index": 0, "name": "Camera 0", "width": 640, "height": 480},
                                 {"index": 1, "name": "Camera 1", "width": 1280, "height": 720}]
    srv = DB.make_server(store=DB.config_store(str(cfg_path)), live=eng, host="127.0.0.1", port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", eng, ref, frames, truth, cfg_path
    srv._closing = True
    eng.stop()
    srv.shutdown()
    srv.server_close()


def _req(url, data=None, headers=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers=dict({"Content-Type": "application/json"} if data is not None else {}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


def test_live_engine_end_to_end_without_webcam(live_server):
    base, eng, ref, frames, truth, cfg_path = live_server
    st, hd, body = _req(base + "/")
    assert st == 200 and b"Live Webcam Coach" in body  # no recorded sessions yet -> redirected to /live
    html = _req(base + "/live")[2].decode()
    for s in ("Live Webcam Coach", "Recorded Sessions", "Start Live Session", "Live Camera + Skeleton",
              "Live Top-Down Footwork Map", "Split View", "Save Session", "Coaching diagnosis", "Tempo", "QQS accuracy",
              "Step &amp; error log", "/api/live/clock", "salsa clave"):
        assert s in html, s
    cams = json.loads(_req(base + "/api/live/cameras")[2])["cameras"]
    assert [c["index"] for c in cams] == [0, 1]
    assert _req(base + "/api/live/stream?once=1")[2][:2] == b"\xff\xd8"  # placeholder JPEG before start

    res = json.loads(_req(base + "/api/live/start", {"camera": 1, "source": "metronome", "bpm": 180,
                                                     "camera_latency_ms": 0, "mirror": True})[2])
    assert res["ok"] and res["model"] == "fake-landmarker" and res["opts"]["camera"] == 1
    ref["T0"] = time.monotonic() - 0.3  # synthetic stream starts at song t = 0.3 s
    t_stop = ref["T0"] + (frames[-1][0] - MONO0)
    while time.monotonic() < t_stop:  # browser heartbeats: song time = mono - T0
        _req(base + "/api/live/clock", {"source": "metronome", "bpm": 180, "playing": True,
                                         "audio_time": time.monotonic() - ref["T0"]})
        time.sleep(0.2)
    state = json.loads(_req(base + "/api/live/state")[2])
    assert state["running"] and state["fps"] > 10 and state["clock"]["playing"] and state["grid"]["bpm"] == 180
    assert state["model"] == "fake-landmarker"
    steps = state["steps"][::-1]
    assert len(steps) >= len(truth) - 1
    by = {s["beat"]: s for s in steps}
    assert by[3]["status"] == "Early" and by[7]["status"] == "Late (off-beat)" and by[1]["status"] == "On Time"
    assert by[5]["offset_ms"] == pytest.approx(60, abs=45)
    assert state["stats"]["qqs_accuracy_pct"] >= 80 and state["floor"]["steps"]

    # MJPEG stream delivers frames while running
    with urllib.request.urlopen(base + "/api/live/stream", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        chunk = r.read(4096)
    assert b"--frame" in chunk and b"\xff\xd8" in chunk

    # security: cross-origin and non-JSON posts are refused
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/api/live/stop", {}, {"Origin": "http://evil.example"})
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(base + "/api/live/stop", data=b"{}", headers={"Content-Type": "text/plain"}))
    assert e.value.code == 415

    saved = json.loads(_req(base + "/api/live/save", {})[2])
    sid = saved["subject_id"]
    out = Path(saved["dir"])
    assert sid.startswith("live_") and saved["url"] == f"/?subject={sid}" and saved["n_steps"] >= len(truth) - 1
    for f in ("cam1.mp4", "overlay_cam1.mp4", "epochs_and_steps.json", "steps.csv", "metronome.wav", "sync.json"):
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    cfg = json.loads(cfg_path.read_text())
    entry = cfg["sessions"][-1]
    assert entry["id"] == sid and entry["live"] is True and entry["overrides"]["mp3_offset_sec"] == 0.0

    # replay in Recorded Sessions mode
    assert sid in json.loads(_req(base + "/api/subjects")[2])["subjects"]
    d = json.loads(_req(base + f"/api/data?subject={sid}")[2])
    assert d["subject_id"] == sid and d["kpis"]["n_steps"] >= len(truth) - 1 and d["floor"]["available"]
    assert d["media"]["video_is_overlay"] and d["media"]["audio"]
    assert _req(base + f"/media/video?subject={sid}", headers={"Range": "bytes=0-99"})[0] == 206
    assert _req(base + f"/?subject={sid}")[0] == 200
