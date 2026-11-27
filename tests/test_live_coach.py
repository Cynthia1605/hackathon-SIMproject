"""Tests for the standalone lightweight live coach (no webcam / MediaPipe needed)."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live_coach as LC  # noqa: E402
from test_realtime import MONO0, match, synth_live_stream  # noqa: E402  (synthetic MediaPipe-like streams)

OFFSETS = {3: -70.0, 5: 60.0, 7: 120.0}


def run_coach(frames, bpm=180.0):
    clock = LC.BeatClock()
    coach = LC.Coach(clock, synthetic_time=True)
    last = -1e9
    for t, kp, world in frames:
        if t - last >= 0.2:  # browser /sync heartbeats: beat_time = seconds since beat 1
            clock.sync(t - MONO0, True, bpm, mono=t)
            last = t
        coach.on_pose(t, world)
    return coach


class Step:  # adapter for test_realtime.match()
    def __init__(self, d):
        self.__dict__.update(d)


# --------------------------------------------------------------------------- #
# Streaming step detector + beat classification
# --------------------------------------------------------------------------- #
def test_streaming_steps_on_synthetic_frames():
    frames, truth = synth_live_stream(offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0))
    coach = run_coach(frames)
    steps = [Step(s) for s in coach.steps]
    m = match(steps, truth)
    assert len(coach.steps) == len(truth) == len(m)
    for s, (td, foot, beat, off) in m:
        assert s.foot == foot and s.beat == beat
        assert abs(s.t - td) * 1000 <= 30 and s.offset_ms == pytest.approx(off, abs=30)
        want = "ON TIME" if abs(off) <= 45 else ("EARLY" if off < 0 else "LATE")
        assert s.status == want
    lat = [s["latency_ms"] for s in coach.steps]
    assert np.mean(lat) <= 150 and max(lat) <= 200
    st = coach.stats()
    assert st["n_steps"] == len(truth) and st["on_tempo_pct"] == pytest.approx(50, abs=1)
    assert 0 < st["tempo_score"] < 100 and st["mean_offset_ms"] == pytest.approx(np.mean([o for *_, o in truth]), abs=15)


@pytest.mark.parametrize("kw", [dict(bpm=200.0), dict(fps=24.0), dict(yaw_deg=25.0, noise_m=0.005)])
def test_streaming_steps_robust(kw):
    frames, truth = synth_live_stream(offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0), **kw)
    coach = run_coach(frames, kw.get("bpm", 180.0))
    m = match([Step(s) for s in coach.steps], truth)
    good = [s for s, tr in m if s.foot == tr[1] and s.beat == tr[2] and abs(s.offset_ms - tr[3]) <= 40]
    assert len(good) >= 0.9 * len(truth)


@pytest.mark.parametrize("kw", [dict(still_only=True), dict(still_only=True, clap=True, noise_m=0.004)])
def test_no_steps_when_idle(kw):
    assert run_coach(synth_live_stream(bars=3, **kw)[0]).steps == []


def test_footwork_pad_and_last_steps():
    coach = run_coach(synth_live_stream(bars=2)[0])
    pad = coach.pad()
    assert len(pad["targets"]) == 4 and len(pad["steps"]) == 8
    b1 = [s for s in coach.steps if s["beat"] == 1]
    b5 = [s for s in coach.steps if s["beat"] == 5]
    assert all(s["pos"][1] > 20 and s["pos"][0] < 0 for s in b1)  # left foot forward, on the left
    assert all(s["pos"][1] < -20 and s["pos"][0] > 0 for s in b5)  # right foot back
    last = coach.last_steps(8)
    assert len(last) == 8 and last[0]["t_rel"] >= last[-1]["t_rel"]
    json.dumps(pad)
    json.dumps(last)


def test_beat_clock():
    c = LC.BeatClock()
    for i in range(6):
        c.sync(1.0 + 0.2 * i, True, 180, mono=10.0 + 0.2 * i)
    assert c.beat_time(11.0, now=11.0) == pytest.approx(2.0)
    # mono 11.3833 -> beat time 2.3833 s = beat index 7 (count 8) + 50 ms
    off, beat = c.nearest(10.0 + 4 / 3 + 0.05, now=11.0)
    assert beat == 8 and off == pytest.approx(50, abs=0.5)
    count, prog = c.current(10.0 + 1 / 3 * 0.5 - 1.0, now=11.0)
    assert count == 1 and prog == pytest.approx(0.5, abs=0.01)
    assert c.nearest(11.0, now=20.0) is None  # stale
    c.sync(None, False)
    assert c.beat_time(11.0, now=11.0) is None
    assert LC.status_of(-46) == "EARLY" and LC.status_of(46) == "LATE" and LC.status_of(45) == "ON TIME"


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def test_ensure_model_lite_download_and_heavy_fallback(tmp_path):
    got = []
    assert LC.ensure_model(tmp_path, download=lambda url, dst: (got.append(url), dst.write_bytes(b"lite"))).name == \
        "pose_landmarker_lite.task"
    assert got == [LC.LITE_URL]
    assert LC.ensure_model(tmp_path, download=lambda *a: pytest.fail("must not download again")).name == \
        "pose_landmarker_lite.task"
    off = tmp_path / "offline"
    off.mkdir()

    def boom(url, dst):
        raise OSError("offline")

    with pytest.raises(RuntimeError):
        LC.ensure_model(off, download=boom)
    (off / "pose_landmarker_heavy.task").write_bytes(b"heavy")
    assert LC.ensure_model(off, download=boom).name == "pose_landmarker_heavy.task"


# --------------------------------------------------------------------------- #
# Overlay drawing + speed
# --------------------------------------------------------------------------- #
def _kp640(kp):
    k = kp.copy()
    k[:, 0] = (k[:, 0] - 640) * 0.5 + 320  # synthetic 1280-wide coordinates -> 640x360
    k[:, 1] = k[:, 1] * 0.5 - 20
    return k


def test_draw_overlay_legs_callout_and_beat_bar():
    frames, truth = synth_live_stream(bars=1)
    kp = _kp640(frames[60][1])
    img = np.zeros((360, 640, 3), np.uint8)
    call = [{"foot": "L", "beat": 3, "status": "LATE", "age": 0.1}]
    out = LC.draw_overlay(img, kp, 3, 0.4, call, {3: 0.1}, mirror=True)
    x, y = int(639 - kp[LC.L_HEEL, 0]), int(kp[LC.L_HEEL, 1])
    patch = out[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).astype(int)
    assert (np.abs(patch - np.array(LC.CYAN)).max(1) < 12).any()  # cyan left foot joint (mirrored)
    ax, ay = int(639 - kp[LC.L_ANK, 0]), int(kp[LC.L_ANK, 1])
    region = out[max(0, ay - 40):ay, max(0, ax - 260):min(640, ax + 260)].reshape(-1, 3).astype(int)
    assert (np.abs(region - np.array(LC.RED)).max(1) < 10).sum() > 200  # red LATE callout tag beside the ankle
    bar = out[330:352, :, :].reshape(-1, 3).astype(int)
    assert (np.abs(bar - np.array(LC.RED)).max(1) < 60).sum() > 100  # beat 3 flashes red
    assert LC.placeholder("x")[:2] == b"\xff\xd8"


def test_draw_and_encode_fast_enough_for_30fps():
    import cv2

    frames, _ = synth_live_stream(bars=1)
    kp = _kp640(frames[60][1])
    img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    call = [{"foot": "R", "beat": 2, "status": "ON TIME", "age": 0.2}]
    t0 = time.perf_counter()
    n = 60
    for _ in range(n):
        f = cv2.flip(img, 1)
        LC.draw_overlay(f, kp, 2, 0.5, call, {}, mirror=True)
        cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    per = (time.perf_counter() - t0) / n * 1000
    assert per < 15.0, f"{per:.1f} ms per frame"  # budget is 33 ms at 30 fps


# --------------------------------------------------------------------------- #
# Threads + HTTP endpoints, no webcam
# --------------------------------------------------------------------------- #
class FakeCam:
    """60 fps camera; each frame encodes its sequence number in the top-left pixels."""

    def __init__(self, idx):
        self.seq = 0

    def read(self):
        time.sleep(1 / 60)
        self.seq += 1
        f = np.full((360, 640, 3), 30, np.uint8)
        f[:32, :32] = (self.seq * 4) % 256  # flat grey marker survives JPEG: frame seq mod 64
        return True, f

    def release(self):
        pass


class SlowBackend:
    """Mimics LIVE_STREAM: submit() never blocks; frames arriving while busy (40 ms) are dropped."""

    name = "fake-lite"

    def __init__(self, on_result, frames, ref):
        self.on_result, self.frames, self.ref = on_result, frames, ref
        self.ts = np.array([f[0] for f in frames])
        self.busy = threading.Event()
        self.dropped = 0

    def submit(self, frame, ts):
        if self.busy.is_set():
            self.dropped += 1
            return
        self.busy.set()

        def work():
            time.sleep(0.04)
            s = ts - self.ref["T0"] + MONO0
            i = int(np.clip(np.searchsorted(self.ts, s), 0, len(self.ts) - 1))
            self.on_result(ts, _kp640(self.frames[i][1]), self.frames[i][2])
            self.busy.clear()

        threading.Thread(target=work, daemon=True).start()

    def close(self):
        pass


@pytest.fixture
def server(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "song.mp3").write_bytes(b"ID3" + b"\0" * 2000)
    frames, truth = synth_live_stream(fps=120.0, bars=1, start_beat=4, tail_s=0.6,
                                      offsets_fn=lambda i, c, f: OFFSETS.get(c, 0.0))
    ref = {"T0": time.monotonic()}
    backends = []

    def bf(cb):
        b = SlowBackend(cb, frames, ref)
        backends.append(b)
        return b

    app = LC.LiveCoach(tmp_path / "models", tmp_path / "data", camera_factory=FakeCam, backend_factory=bf)
    app.cam_latency = 0.0
    srv = LC.serve(app, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app, ref, frames, truth, backends
    srv.closing = True
    app.stop()
    srv.shutdown()
    srv.server_close()


def _req(url, body=None, headers=None):
    h = dict({"Content-Type": "application/json"} if body is not None else {}, **(headers or {}))
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


def test_state_endpoint_and_live_pipeline(server):
    import cv2

    base, app, ref, frames, truth, backends = server
    html = _req(base + "/")[2].decode()
    for s in ("Tempo score", "On-tempo", "Mean offset", "Footwork pad", "Last 8 steps", "/sync", "Start", "clave",
              "height:100vh", "65fr 35fr"):
        assert s in html, s
    assert "backdrop-filter" not in html and "box-shadow" not in html
    assert json.loads(_req(base + "/songs")[2])["songs"][0]["name"] == "song.mp3"
    assert _req(base + "/song?id=0", headers={"Range": "bytes=0-9"})[0] == 206
    st = json.loads(_req(base + "/state")[2])
    assert st["running"] is False and st["stats"]["n_steps"] == 0 and st["last_steps"] == []

    assert json.loads(_req(base + "/start", {"camera": 0, "mirror": True})[2])["model"] == "fake-lite"
    ref["T0"] = time.monotonic() - 0.3  # synthetic stream starts at beat time 0.3 s
    t_end = ref["T0"] + (frames[-1][0] - MONO0)
    while time.monotonic() < t_end:  # browser beat clock: beat_time = seconds since beat 1
        _req(base + "/sync", {"playing": True, "bpm": 180, "beat_time": time.monotonic() - ref["T0"]})
        time.sleep(0.1)
    st = json.loads(_req(base + "/state")[2])
    json.dumps(st)
    c = st["counters"]
    # latest-frame-only: a 40 ms model under a 60 fps camera skips frames instead of queueing them
    assert c["captured"] > 100 and c["results"] < 0.75 * c["captured"] and backends[0].dropped > 0
    assert st["fps"]["camera"] > 40 and st["fps"]["inference"] > 12 and st["fps"]["stream"] > 24
    assert st["pose_age_ms"] is not None and st["pose_age_ms"] < 150 and st["clock"]["synced"]
    assert st["count"] in range(1, 9)
    last = st["last_steps"]
    assert len(last) >= len(truth) - 1
    by = {s["beat"]: s for s in last}
    assert by[1]["status"] == "ON TIME" and by[3]["status"] == "EARLY" and by[7]["status"] == "LATE"
    assert st["stats"]["tempo_score"] is not None and st["pad"]["steps"]

    # zero visual lag: the streamed JPEG shows (almost) the newest captured frame
    jpg = _req(base + "/stream?once=1")[2]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    shown = int(round(float(img[8:24, -24:-8].mean()) / 4)) % 64  # mirrored: marker is top-right
    assert (app.counters["captured"] - shown) % 64 <= 4  # within ~4 frames (~65 ms) of the newest capture

    with urllib.request.urlopen(base + "/stream", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        assert b"\xff\xd8" in r.read(8192)

    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/stop", {}, {"Origin": "http://evil.example"})
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(base + "/stop", data=b"{}", headers={"Content-Type": "text/plain"}))
    assert e.value.code == 415
    assert json.loads(_req(base + "/stop", {})[2])["ok"]
    assert json.loads(_req(base + "/state")[2])["running"] is False


def test_live_coach_is_decoupled_from_offline_pipeline():
    src = (Path(LC.__file__)).read_text()
    for mod in ("import main", "import dashboard", "import overlay_video", "import realtime", "from realtime",
                "from overlay_video", "import librosa", "import scipy"):
        assert mod not in src, mod
