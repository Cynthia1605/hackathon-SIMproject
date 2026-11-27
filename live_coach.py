#!/usr/bin/env python3
"""Ultra-lightweight real-time webcam salsa coach (standalone, zero-lag).

    python live_coach.py                 # opens http://127.0.0.1:8770/
    python live_coach.py --camera 1 --port 8770 --no-browser

Completely separate from the two-camera offline pipeline (main.py / dashboard.py /
overlay_video.py are not imported). Dependencies: OpenCV, MediaPipe, NumPy, stdlib.

Anti-lag architecture (latest-frame-only, nothing ever queues):
* Capture thread: reads the webcam (CAP_DSHOW on Windows, 640x360, buffer 1) and
  overwrites a single ``latest_frame`` slot.
* Inference: MediaPipe Pose Landmarker *lite* in LIVE_STREAM mode. ``detect_async`` gets
  the newest frame only, and frames arriving while the model is busy are skipped. Results
  arrive in a callback that updates the latest pose and the step detector.
* Stream thread: at ~30 fps, draws the newest camera frame with the latest skeleton and
  HUD, then JPEG (q=70). The video never waits for inference.
* HTTP (stdlib): ``/`` UI, ``GET /stream`` MJPEG, ``GET /state`` JSON (polled every
  100 ms), ``POST /sync`` browser beat clock, ``POST /start`` and ``POST /stop``.

The browser plays the music (an .mp3 from data/ or a Web Audio clave/cowbell metronome,
120-200 BPM) and is the master beat clock. It posts ``beat_time``: seconds since beat 1
of its grid, after subtracting output latency. A step's timing error is measured against
the nearest beat of that grid. For songs, press "Tap 1" on a downbeat to anchor the grid
(BPM from the slider): no audio analysis library needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

log = logging.getLogger("live_coach")

LITE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
            "float16/latest/pose_landmarker_lite.task")
HERE = Path(__file__).resolve().parent

# MediaPipe landmark ids (only the salsa-relevant ones are tracked / drawn)
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 23, 24, 25, 26
L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = 27, 28, 29, 30, 31, 32
KEY = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE]
FOOT = {"L": [L_ANK, L_HEEL, L_TOE], "R": [R_ANK, R_HEEL, R_TOE]}
UPPER_LINES = [(L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP)]
LEG_LINES = {"L": [(L_HIP, L_KNEE), (L_KNEE, L_ANK), (L_ANK, L_HEEL), (L_HEEL, L_TOE), (L_ANK, L_TOE)],
             "R": [(R_HIP, R_KNEE), (R_KNEE, R_ANK), (R_ANK, R_HEEL), (R_HEEL, R_TOE), (R_ANK, R_TOE)]}

# BGR colours
CYAN = (255, 229, 0)  # #00E5FF left
MAGENTA = (129, 64, 255)  # #FF4081 right
GREEN, AMBER, RED = (94, 197, 34), (11, 158, 245), (68, 68, 239)
GREY, WHITE, DARK = (120, 120, 120), (255, 255, 255), (32, 18, 11)
FOOT_COL = {"L": CYAN, "R": MAGENTA}
STATUS_COL = {"ON TIME": GREEN, "EARLY": AMBER, "LATE": RED}

ON_TIME_MS = 45.0
CALLOUT_S = 0.5

PARAMS: dict[str, Any] = {
    "move_speed_mps": 0.25, "still_speed_mps": 0.10, "lift_m": 0.025, "still_s": 0.03, "still_window_s": 0.06,
    "min_travel_m": 0.03, "min_step_interval_s": 0.12, "buffer_s": 2.0, "plant_progress": 0.97,
    "lift_drop_frac": 0.2, "lift_floor_m": 0.006, "filter_min_cutoff": 3.0, "filter_beta": 3.0,
    "tap_window_s": 1.5, "camera_latency_ms": 40.0,
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def ensure_model(models_dir: str | Path, timeout: float = 30.0,
                 download: Callable[[str, Path], None] | None = None) -> Path:
    """models/pose_landmarker_lite.task, downloaded if missing; heavy only as an offline fallback."""
    d = Path(models_dir)
    lite, heavy = d / "pose_landmarker_lite.task", d / "pose_landmarker_heavy.task"
    if lite.is_file() and lite.stat().st_size > 0:
        return lite
    d.mkdir(parents=True, exist_ok=True)

    def _dl(url: str, dst: Path):
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        os.replace(tmp, dst)

    try:
        log.info("Downloading %s", LITE_URL)
        (download or _dl)(LITE_URL, lite)
        return lite
    except Exception as exc:
        if heavy.is_file() and heavy.stat().st_size > 0:
            log.warning("Lite model download failed (%s); falling back to the heavy model (slower)", exc)
            return heavy
        raise RuntimeError(f"No pose model: could not download {LITE_URL} ({exc})") from exc


# --------------------------------------------------------------------------- #
# Beat clock (browser = master)
# --------------------------------------------------------------------------- #
class BeatClock:
    """Maps the browser's beat clock (seconds since beat 1) onto time.monotonic()."""

    def __init__(self, window: int = 9, jump_s: float = 0.12, stale_s: float = 2.0):
        self.offsets: deque[float] = deque(maxlen=window)
        self.jump_s, self.stale_s = jump_s, stale_s
        self.playing, self.bpm, self.last = False, 180.0, None
        self.lock = threading.Lock()

    def sync(self, beat_time: float | None, playing: bool, bpm: float | None = None, mono: float | None = None):
        mono = time.monotonic() if mono is None else mono
        with self.lock:
            self.last = mono
            if bpm:
                b = float(np.clip(bpm, 40, 260))
                if abs(b - self.bpm) > 1e-6:
                    self.offsets.clear()
                self.bpm = b
            if not playing or beat_time is None or not math.isfinite(beat_time):
                self.playing = False
                self.offsets.clear()
                return
            off = float(beat_time) - mono
            if self.offsets and abs(off - float(np.median(self.offsets))) > self.jump_s:
                self.offsets.clear()
            self.offsets.append(off)
            self.playing = True

    @property
    def period(self) -> float:
        return 60.0 / self.bpm

    def beat_time(self, mono: float, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.playing or not self.offsets or self.last is None or now - self.last > self.stale_s:
                return None
            return mono + float(np.median(self.offsets))

    def nearest(self, mono: float, now: float | None = None):
        """(offset_ms, count 1-8) of the nearest beat, or None."""
        bt = self.beat_time(mono, now)
        if bt is None:
            return None
        k = round(bt / self.period)
        return (bt - k * self.period) * 1000.0, int(k % 8) + 1

    def current(self, mono: float, now: float | None = None):
        """(count 1-8, progress 0-1) at ``mono``."""
        bt = self.beat_time(mono, now)
        if bt is None:
            return None, 0.0
        k = math.floor(bt / self.period)
        return int(k % 8) + 1, (bt - k * self.period) / self.period

    def status(self) -> dict[str, Any]:
        with self.lock:
            age = None if self.last is None else time.monotonic() - self.last
            return {"synced": bool(self.playing and age is not None and age <= self.stale_s), "bpm": self.bpm,
                    "jitter_ms": float(np.std(self.offsets) * 1000) if len(self.offsets) > 1 else None}


# --------------------------------------------------------------------------- #
# Streaming step detector (pose_world_landmarks, metres, hip-centred, y down)
# --------------------------------------------------------------------------- #
class OneEuro:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(fc, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * fc * dt))

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if self.t is None or t - self.t > 0.5 or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        self.dx = self._a(self.dc, dt) * (x - self.x) / dt + (1 - self._a(self.dc, dt)) * self.dx
        a = self._a(self.mc + self.beta * float(np.linalg.norm(self.dx)), dt)
        self.x, self.t = a * x + (1 - a) * self.x, t
        return self.x.copy()


def foot_signals(world: np.ndarray):
    """Foot-centre (x, z) per foot and heights (+up) from world landmarks; None if feet missing."""
    lf, rf = world[FOOT["L"]], world[FOOT["R"]]
    if not (np.isfinite(lf).all() and np.isfinite(rf).all()):
        return None
    lm, rm = lf.mean(0), rf.mean(0)
    return lm[[0, 2]], rm[[0, 2]], -float(lm[1]), -float(rm[1])


class StepDetector:
    """Plant detection from the left-to-right foot vector (moving the hips does not change it)
    plus foot lift.

    The touchdown is refined on a rolling raw buffer: the moment the foot covered 97% of its
    travel, or its lift dropped. Latency is about 60-150 ms.
    """

    def __init__(self, p: dict[str, Any] | None = None):
        self.p = {**PARAMS, **(p or {})}
        self.f_rel = OneEuro(self.p["filter_min_cutoff"], self.p["filter_beta"])
        self.f_lift = OneEuro(3 * self.p["filter_min_cutoff"], 0.0)
        self.buf: deque = deque()
        self.state, self.prev, self.prev_feet = "STILL", None, None
        self.still_count, self.still_time, self.last_event = 0, 0.0, -1e9
        self._reset_motion(None, None)

    def _reset_motion(self, t0, rel0):
        self.move_t0, self.move_rel0 = t0, rel0
        self.max_lift, self.lift_side = 0.0, None
        self.lift_acc, self.hip_acc = {"L": 0.0, "R": 0.0}, {"L": 0.0, "R": 0.0}

    def update(self, t: float, world: np.ndarray | None) -> list[tuple[float, str, float]]:
        """Feed one frame; returns new steps as (touchdown_time, foot, detected_at)."""
        sig = None if world is None else foot_signals(world)
        if sig is None:
            return []
        lxz, rxz, hl, hr = sig
        rel = lxz - rxz
        self.buf.append((t, rel[0], rel[1], hl, hr))
        while self.buf and t - self.buf[0][0] > self.p["buffer_s"]:
            self.buf.popleft()
        rel_f = self.f_rel(t, rel)
        lift = float(self.f_lift(t, np.array([hl - hr]))[0])
        out = []
        if self.prev is None or t <= self.prev[0]:
            self.prev, self.prev_feet = (t, rel_f), (lxz, rxz)
            return out
        dt = t - self.prev[0]
        speed = float(np.linalg.norm(rel_f - self.prev[1]) / dt)
        lifted = abs(lift) > self.p["lift_m"]
        if self.state == "STILL" and (speed > self.p["move_speed_mps"] or lifted):
            self.state, self.still_count, self.still_time = "MOVING", 0, 0.0
            self._reset_motion(self.prev[0], self.prev[1].copy())
        if self.state == "MOVING":
            side = ("L" if lift > 0 else "R") if lifted else None
            if side and self.lift_side and side != self.lift_side:  # other foot lifts: previous step landed
                ev = self._emit(t, rel_f, t - dt)
                if ev:
                    out.append(ev)
                self._reset_motion(self.prev[0], self.prev[1].copy())
                self.still_count, self.still_time = 0, 0.0
            if side:
                self.lift_side = side
            w = max(speed, 0.05) * dt
            self.lift_acc["L"] += max(lift, 0.0) * w
            self.lift_acc["R"] += max(-lift, 0.0) * w
            self.max_lift = max(self.max_lift, abs(lift))
            self.hip_acc["L"] += float(np.linalg.norm(lxz - self.prev_feet[0]))
            self.hip_acc["R"] += float(np.linalg.norm(rxz - self.prev_feet[1]))
            j = len(self.buf) - 2
            while j > 0 and t - self.buf[j][0] < self.p["still_window_s"]:
                j -= 1
            raw_speed = math.hypot(rel[0] - self.buf[j][1], rel[1] - self.buf[j][2]) / max(t - self.buf[j][0], 1e-6)
            if raw_speed < self.p["still_speed_mps"] and abs(hl - hr) < 0.5 * self.p["lift_m"]:
                self.still_count += 1
                self.still_time += dt
            else:
                self.still_count, self.still_time = 0, 0.0
            if self.still_count >= 1 and self.still_time >= self.p["still_s"] - 1e-9:
                self.state = "STILL"
                ev = self._emit(t, rel_f, t)
                if ev:
                    out.append(ev)
        self.prev, self.prev_feet = (t, rel_f), (lxz, rxz)
        return out

    def _emit(self, t, rel_f, end_t):
        travel = float(np.linalg.norm(rel_f - self.move_rel0))
        if travel < self.p["min_travel_m"] and self.max_lift < self.p["lift_m"]:
            return None
        acc = self.lift_acc if self.lift_acc["L"] + self.lift_acc["R"] > 1e-6 else self.hip_acc
        foot = "L" if acc["L"] >= acc["R"] else "R"
        td = self._touchdown(self.move_t0, end_t)
        if td - self.last_event < self.p["min_step_interval_s"]:
            return None
        self.last_event = td
        return td, foot, t

    def _touchdown(self, t0, t1) -> float:
        b = np.asarray(self.buf)
        t, rel, lift = b[:, 0], b[:, 1:3], np.abs(b[:, 3] - b[:, 4])
        win = np.flatnonzero((t >= (t0 if t0 is not None else t[0]) - 0.1) & (t <= t1 + 1e-9))
        if len(win) < 3:
            return float(min(t1, t[-1]))
        cands = []
        start, tail = rel[win[0]], rel[win[-2:]].mean(0)
        d = tail - start
        travel = float(np.linalg.norm(d))
        if travel >= 2 * self.p["min_travel_m"]:
            prog = (rel[win] - start) @ d / (travel * travel)
            above = np.flatnonzero(prog >= self.p["plant_progress"])
            if len(above):
                j = above[0]
                if j > 0:
                    a = (self.p["plant_progress"] - prog[j - 1]) / max(prog[j] - prog[j - 1], 1e-9)
                    cands.append(t[win[j - 1]] + a * (t[win[j]] - t[win[j - 1]]))
                else:
                    cands.append(t[win[0]])
        lw = lift[win]
        if lw.max() >= self.p["lift_m"]:
            pk = int(np.argmax(lw))
            thr = max(self.p["lift_drop_frac"] * lw[pk], self.p["lift_floor_m"])
            after = np.flatnonzero(lw[pk:] < thr)
            if len(after):
                j = pk + after[0]
                a = (lw[j - 1] - thr) / max(lw[j - 1] - lw[j], 1e-9)
                cands.append(t[win[j - 1]] + a * (t[win[j]] - t[win[j - 1]]))
        return float(max(cands)) if cands else float(min(t1, t[-1]))


# --------------------------------------------------------------------------- #
# Session: steps, score, mini footwork pad
# --------------------------------------------------------------------------- #
def status_of(offset_ms: float | None) -> str:
    if offset_ms is None:
        return "NO BEAT"
    if abs(offset_ms) <= ON_TIME_MS:
        return "ON TIME"
    return "EARLY" if offset_ms < 0 else "LATE"


def timing_credit(abs_ms: float, half_beat_ms: float) -> float:
    if abs_ms <= ON_TIME_MS:
        return 1.0
    return max(0.0, 1.0 - (abs_ms - ON_TIME_MS) / max(half_beat_ms - ON_TIME_MS, 1.0))


class Coach:
    """Step detection + beat classification + score + mini footwork pad (thread-safe)."""

    def __init__(self, clock: BeatClock | None = None, height_m: float = 1.70, params: dict | None = None,
                 synthetic_time: bool = False):
        self.clock = clock or BeatClock()
        self.det = StepDetector(params)
        self.p = self.det.p
        self.synthetic = synthetic_time
        self.steps: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.t0: float | None = None
        self.last_t: float | None = None
        self.reach, self.half = 0.16 * height_m * 100, 0.055 * height_m * 100
        self.fwd = np.array([0.0, -1.0])  # facing in world (x, z); camera-facing dancer
        self.P: dict[str, np.ndarray] | None = None  # floor positions (cm)
        self.feet = None

    def _now(self, t):
        return t if self.synthetic else None

    def _floor(self, v):  # world (x, z) metres -> (right, forward) cm
        right = np.array([self.fwd[1], -self.fwd[0]])
        return np.array([v @ right, v @ self.fwd]) * 100.0

    def on_pose(self, t: float, world: np.ndarray | None) -> list[dict]:
        with self.lock:
            self.t0 = t if self.t0 is None else self.t0
            self.last_t = t
            if world is not None:
                v = []
                for h, toe in ((L_HEEL, L_TOE), (R_HEEL, R_TOE)):
                    d = world[toe, [0, 2]] - world[h, [0, 2]]
                    n = float(np.linalg.norm(d))
                    if np.isfinite(n) and n > 0.03:
                        v.append(d / n)
                if v and np.linalg.norm(np.mean(v, 0)) > 0.5:
                    f = 0.97 * self.fwd + 0.03 * np.mean(v, 0)
                    self.fwd = f / np.linalg.norm(f)
                sig = foot_signals(world)
                if sig:
                    self.feet = sig
                    if self.P is None:  # floor frame from the starting stance (before any step)
                        half = self._floor(sig[0] - sig[1]) / 2
                        self.P = {"L": half, "R": -half}
            new = []
            for td, foot, det_at in self.det.update(t, world):
                new.append(self._add(td, foot, det_at))
            return new

    def _add(self, td, foot, det_at) -> dict:
        nb = self.clock.nearest(td, self._now(det_at))
        off, beat = (nb if nb else (None, None))
        prev = self.steps[-1] if self.steps else None
        tap = prev is not None and prev["foot"] == foot and td - prev["t"] < self.p["tap_window_s"]
        st = "TAP" if tap else status_of(off)
        if self.feet is not None and self.P is not None:
            other = "R" if foot == "L" else "L"
            d = self._floor((self.feet[0] - self.feet[1]) if foot == "L" else (self.feet[1] - self.feet[0]))
            self.P[foot] = self.P[other] + d
        pos = None if self.P is None else [round(float(self.P[foot][0]), 1), round(float(self.P[foot][1]), 1)]
        s = {"i": len(self.steps), "t": td, "t_rel": round(td - (self.t0 or td), 2), "foot": foot, "beat": beat,
             "offset_ms": None if off is None else round(off, 1), "status": st, "pos": pos,
             "latency_ms": round((det_at - td) * 1000.0)}
        self.steps.append(s)
        return s

    def stats(self) -> dict[str, Any]:
        with self.lock:
            offs = [s["offset_ms"] for s in self.steps if s["status"] != "TAP" and s["offset_ms"] is not None]
            half = self.clock.period * 500
            return {"n_steps": len([s for s in self.steps if s["status"] != "TAP"]),
                    "tempo_score": round(100.0 * float(np.mean([timing_credit(abs(o), half) for o in offs])), 1) if offs else None,
                    "on_tempo_pct": round(100.0 * float(np.mean([abs(o) <= ON_TIME_MS for o in offs])), 1) if offs else None,
                    "mean_offset_ms": round(float(np.mean(offs)), 1) if offs else None}

    def pad(self) -> dict[str, Any]:
        """Mini footwork pad: ideal On-1 target dots + the last steps' floor positions (cm)."""
        with self.lock:
            h, r = self.half, self.reach
            targets = [{"foot": "L", "x": -h, "y": 0, "beats": "3 6"}, {"foot": "R", "x": h, "y": 0, "beats": "2 7"},
                       {"foot": "L", "x": -h, "y": r, "beats": "1"}, {"foot": "R", "x": h, "y": -r, "beats": "5"}]
            cx = cy = 0.0
            if self.P is not None:
                cx, cy = [float(v) for v in 0.5 * (self.P["L"] + self.P["R"])]
            recent = [{"foot": s["foot"], "pos": s["pos"], "status": s["status"], "beat": s["beat"]}
                      for s in self.steps[-8:] if s["pos"] is not None]
            return {"targets": targets, "center": [round(cx, 1), round(cy, 1)], "steps": recent, "reach_cm": round(r, 1)}

    def callouts(self, t: float) -> list[dict]:
        with self.lock:
            out = []
            for s in reversed(self.steps):
                age = t - s["t"]
                if age > CALLOUT_S:
                    break
                if age >= 0 and s["status"] != "TAP":
                    out.append(dict(s, age=age))
            return out

    def last_steps(self, n: int = 8) -> list[dict]:
        with self.lock:
            return [{k: s[k] for k in ("t_rel", "foot", "beat", "offset_ms", "status")} for s in self.steps[-n:]][::-1]


# --------------------------------------------------------------------------- #
# Drawing (OpenCV only; key landmarks only)
# --------------------------------------------------------------------------- #
def _beat_bar(frame, count, progress, flash, s):
    import cv2

    h, w = frame.shape[:2]
    # bottom 8-count bar: 1 - 2 - 3 - [4] - 5 - 6 - 7 - [8]
    bh = int(26 * s)
    y0 = h - bh - int(6 * s)
    roi = frame[y0:y0 + bh, int(6 * s):w - int(6 * s)]
    roi[:] = (roi * 0.35 + np.array(DARK) * 0.65).astype(np.uint8)
    cw = (w - int(12 * s)) / 8
    for i in range(8):
        n = i + 1
        x0, x1 = int(6 * s + i * cw) + 2, int(6 * s + (i + 1) * cw) - 2
        hold = n in (4, 8)
        fg = (110, 110, 110) if hold else (220, 220, 220)
        if count == n:
            cv2.rectangle(frame, (x0, y0 + 2), (x1, y0 + bh - 2), (0, 196, 255) if not hold else (150, 150, 150), -1)
            cv2.rectangle(frame, (x0, y0 + bh - 5), (x0 + int((x1 - x0) * progress), y0 + bh - 2), DARK, -1)
            fg = DARK
        if n in flash:
            a = max(0.0, 1.0 - flash[n] / CALLOUT_S)
            if a > 0:
                sub = frame[y0 + 2:y0 + bh - 2, x0:x1]
                sub[:] = (sub * (1 - 0.85 * a) + np.array(RED) * 0.85 * a).astype(np.uint8)
                fg = WHITE
        label = f"[{n}]" if hold else str(n)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, max(1, int(s)))
        cv2.putText(frame, label, ((x0 + x1 - tw) // 2, y0 + (bh + tht) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, fg,
                    max(1, int(1.5 * s)), cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, kp: np.ndarray | None, count: int | None, progress: float,
                 callouts: list[dict], flash: dict[int, float], mirror: bool = True) -> np.ndarray:
    """Thin upper body, cyan/magenta legs, ankle callouts and the bottom 8-count bar, in place."""
    import cv2

    h, w = frame.shape[:2]
    s = h / 360.0
    _beat_bar(frame, count, progress, flash, s)  # first: legs and callouts stay visible on top of it
    if kp is not None:
        p = kp[:, :2].copy()
        if mirror:
            p[:, 0] = w - 1 - p[:, 0]
        ok = np.isfinite(p).all(1) & (kp[:, 2] >= 0.5)
        P = lambda j: (int(p[j, 0]), int(p[j, 1]))  # noqa: E731
        for a, b in UPPER_LINES:
            if ok[a] and ok[b]:
                cv2.line(frame, P(a), P(b), (190, 170, 120), max(1, int(s)), cv2.LINE_AA)
        for foot, lines in LEG_LINES.items():
            for a, b in lines:
                if ok[a] and ok[b]:
                    cv2.line(frame, P(a), P(b), (10, 10, 10), int(5 * s) + 2, cv2.LINE_AA)
                    cv2.line(frame, P(a), P(b), FOOT_COL[foot], int(3 * s) + 1, cv2.LINE_AA)
        for foot, ids in (("L", (L_KNEE, L_ANK, L_HEEL, L_TOE)), ("R", (R_KNEE, R_ANK, R_HEEL, R_TOE))):
            for j in ids:
                if ok[j]:
                    cv2.circle(frame, P(j), int(4 * s), FOOT_COL[foot], -1, cv2.LINE_AA)
        # callouts beside the ankle that stepped
        for c in callouts:
            j = L_ANK if c["foot"] == "L" else R_ANK
            if not ok[j]:
                continue
            col = STATUS_COL.get(c["status"], GREY)
            x, y = P(j)
            cv2.circle(frame, (x, y), int((16 + 10 * (1 - c["age"] / CALLOUT_S)) * s), col, max(2, int(2 * s)), cv2.LINE_AA)
            parts = ["LEFT" if c["foot"] == "L" else "RIGHT", f"Beat {c['beat']}" if c["beat"] else "Beat -", c["status"]]
            fs, th = 0.45 * s, max(1, int(s))
            sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in parts]
            gap = int(14 * s)
            tw = sum(z[0] for z in sizes) + gap * 2 + int(12 * s)
            th_px = max(z[1] for z in sizes) + int(10 * s)
            side = -1 if x > w * 0.5 else 1  # keep the tag inside the frame
            x0 = x + int(20 * s) if side > 0 else x - int(20 * s) - tw
            y0 = max(0, y - int(34 * s))
            cv2.rectangle(frame, (x0, y0), (x0 + tw, y0 + th_px), col, -1)
            cx = x0 + int(6 * s)
            for k, (t, z) in enumerate(zip(parts, sizes)):
                cv2.putText(frame, t, (cx, y0 + th_px - int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX, fs, DARK if col != RED else WHITE, th, cv2.LINE_AA)
                cx += z[0]
                if k < 2:  # bullet "•"
                    cv2.circle(frame, (cx + gap // 2, y0 + th_px // 2), max(2, int(2 * s)), DARK if col != RED else WHITE, -1, cv2.LINE_AA)
                    cx += gap
    return frame


def placeholder(text: str, w: int = 640, h: int = 360) -> bytes:
    import cv2

    img = np.full((h, w, 3), DARK, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()


# --------------------------------------------------------------------------- #
# Camera & pose backends (injectable for tests)
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 360):
        import cv2

        api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY  # DirectShow: fast startup
        self.cap = cv2.VideoCapture(int(index), api)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise IOError(f"cannot open camera {index}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class MediaPipeLive:
    """Pose Landmarker lite in LIVE_STREAM mode: ``submit`` never blocks; MediaPipe drops frames while busy."""

    def __init__(self, model_path: str | Path, on_result: Callable[[float, np.ndarray | None, np.ndarray | None], None]):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.on_result, self.last_ms, self.size = on_result, -1, (640, 360)

        def cb(result, image, ts_ms):
            kp = world = None
            if result.pose_landmarks:
                w, h = self.size
                kp = np.array([[q.x * w, q.y * h, q.visibility or 0.0] for q in result.pose_landmarks[0]], np.float32)
                if result.pose_world_landmarks:
                    world = np.array([[q.x, q.y, q.z] for q in result.pose_world_landmarks[0]], np.float64)
                    world[kp[:, 2] < 0.3] = np.nan
            self.on_result(ts_ms / 1000.0, kp, world)

        opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                                            running_mode=vision.RunningMode.LIVE_STREAM, num_poses=1, result_callback=cb)
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self.name = Path(model_path).name

    def submit(self, frame_bgr: np.ndarray, ts: float) -> None:
        import cv2
        import mediapipe as mp

        ms = max(int(ts * 1000), self.last_ms + 1)
        self.last_ms = ms
        self.size = (frame_bgr.shape[1], frame_bgr.shape[0])
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.lm.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

    def close(self):
        self.lm.close()


# --------------------------------------------------------------------------- #
# Engine: capture / inference / stream threads
# --------------------------------------------------------------------------- #
class LiveCoach:
    def __init__(self, models_dir: str | Path = HERE / "models", data_dir: str | Path = HERE / "data",
                 camera_factory: Callable[[int], Any] | None = None,
                 backend_factory: Callable[[Callable], Any] | None = None, stream_fps: float = 30.0,
                 height_m: float = 1.70, default_camera: int = 0):
        self.default_camera = default_camera
        self.models_dir, self.data_dir = Path(models_dir), Path(data_dir)
        self.camera_factory = camera_factory or (lambda i: Camera(i))
        self.backend_factory = backend_factory
        self.stream_fps, self.height_m = stream_fps, height_m
        self.clock = BeatClock()
        self.coach = Coach(self.clock, height_m)
        self.running, self.mirror, self.error, self.model = False, True, None, None
        self.cam_latency = PARAMS["camera_latency_ms"] / 1000.0
        self._latest = None  # (seq, frame, ts) — single slot, overwritten
        self._latest_lock = threading.Lock()
        self._new_frame = threading.Event()
        self._pose = (None, None, None)  # (ts, kp, world) of the latest inference result
        self._jpeg = (0, placeholder("Press Start"))
        self._jpeg_cv = threading.Condition()
        self._threads: list[threading.Thread] = []
        self.counters = {"captured": 0, "submitted": 0, "results": 0, "streamed": 0}
        self._fps = {"camera": 0.0, "inference": 0.0, "stream": 0.0}
        self._fps_t: dict[str, deque] = {k: deque(maxlen=30) for k in self._fps}

    def _tick(self, name: str, t: float):
        q = self._fps_t[name]
        q.append(t)
        if len(q) >= 2 and q[-1] > q[0]:
            self._fps[name] = (len(q) - 1) / (q[-1] - q[0])

    # ---- control ----
    def start(self, camera: int = 0, mirror: bool = True, camera_latency_ms: float | None = None) -> dict:
        self.stop()
        self.error = None
        self.mirror = bool(mirror)
        if camera_latency_ms is not None:
            self.cam_latency = float(camera_latency_ms) / 1000.0
        self.coach = Coach(self.clock, self.height_m)
        self.counters = {k: 0 for k in self.counters}
        self._pose = (None, None, None)
        cam = self.camera_factory(int(camera))
        try:
            if self.backend_factory is not None:
                backend = self.backend_factory(self._on_result)
            else:
                backend = MediaPipeLive(ensure_model(self.models_dir), self._on_result)
        except Exception:
            cam.release()
            raise
        self.model = getattr(backend, "name", type(backend).__name__)
        self.running = True
        self._threads = [threading.Thread(target=self._capture, args=(cam,), daemon=True, name="capture"),
                         threading.Thread(target=self._infer, args=(backend,), daemon=True, name="inference"),
                         threading.Thread(target=self._stream, daemon=True, name="stream")]
        for th in self._threads:
            th.start()
        return {"ok": True, "model": self.model}

    def stop(self) -> dict:
        was = self.running
        self.running = False
        self._new_frame.set()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads = []
        if was:
            with self._jpeg_cv:
                self._jpeg = (self._jpeg[0] + 1, placeholder("Stopped - press Start"))
                self._jpeg_cv.notify_all()
        return {"ok": True}

    def sync(self, body: dict) -> dict:
        self.clock.sync(body.get("beat_time"), bool(body.get("playing")), body.get("bpm"))
        return {"ok": True}

    # ---- threads ----
    def _capture(self, cam):
        seq = 0
        try:
            while self.running:
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                seq += 1
                now = time.monotonic()
                with self._latest_lock:
                    self._latest = (seq, frame, now - self.cam_latency)  # overwrite: never queue
                self.counters["captured"] = seq
                self._tick("camera", now)
                self._new_frame.set()
        except Exception as exc:
            self.error = f"camera: {exc}"
        finally:
            cam.release()

    def _infer(self, backend):
        last = 0
        try:
            while self.running:
                self._new_frame.wait(0.2)
                self._new_frame.clear()
                with self._latest_lock:
                    item = self._latest
                if item is None or item[0] == last:
                    continue
                last = item[0]
                backend.submit(item[1], item[2])  # async: busy model -> frame dropped, never queued
                self.counters["submitted"] += 1
        except Exception as exc:
            self.error = f"inference: {exc}"
            self.running = False
        finally:
            try:
                backend.close()
            except Exception:
                pass

    def _on_result(self, ts: float, kp, world):
        self._pose = (ts, kp, world)
        self.counters["results"] += 1
        self._tick("inference", time.monotonic())
        try:
            self.coach.on_pose(ts, world)
        except Exception as exc:  # never kill the pose callback
            self.error = f"step detector: {exc}"

    def _stream(self):
        import cv2

        period = 1.0 / self.stream_fps
        nxt = time.monotonic()
        while self.running:
            nxt += period
            with self._latest_lock:
                item = self._latest
            if item is not None:
                frame = item[1].copy()
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.monotonic() - self.cam_latency
                ts, kp, _ = self._pose
                count, prog = self.clock.current(now)
                calls = self.coach.callouts(now)
                flash = {c["beat"]: c["age"] for c in calls if c["beat"] and c["status"] != "ON TIME"}
                draw_overlay(frame, kp if (ts is not None and now - ts < 0.5) else None, count, prog, calls, flash, self.mirror)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    with self._jpeg_cv:
                        self._jpeg = (self._jpeg[0] + 1, buf.tobytes())
                        self._jpeg_cv.notify_all()
                    self.counters["streamed"] += 1
                    self._tick("stream", time.monotonic())
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:
                nxt = time.monotonic()

    # ---- outputs ----
    def wait_jpeg(self, after: int, timeout: float = 1.0):
        with self._jpeg_cv:
            if self._jpeg[0] == after:
                self._jpeg_cv.wait(timeout)
            return self._jpeg

    def state(self) -> dict[str, Any]:
        now = time.monotonic() - self.cam_latency
        count, prog = self.clock.current(now)
        ts = self._pose[0]
        return {"running": self.running, "error": self.error, "model": self.model, "default_camera": self.default_camera,
                "fps": {k: round(v, 1) for k, v in self._fps.items()}, "counters": dict(self.counters),
                "pose_age_ms": None if ts is None else round((now - ts) * 1000),
                "clock": self.clock.status(), "count": count, "progress": round(prog, 3),
                "stats": self.coach.stats(), "last_steps": self.coach.last_steps(8), "pad": self.coach.pad()}

    def songs(self) -> list[dict[str, Any]]:
        if not self.data_dir.is_dir():
            return []
        files = sorted(p for p in self.data_dir.rglob("*") if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".m4a"))
        return [{"id": i, "name": str(p.relative_to(self.data_dir)), "path": p} for i, p in enumerate(files)]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def make_handler(app: LiveCoach):
    page = PAGE.encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/state":
                return self._json(200, app.state())
            if u.path == "/songs":
                return self._json(200, {"songs": [{"id": s["id"], "name": s["name"]} for s in app.songs()]})
            if u.path == "/song":
                q = parse_qs(u.query)
                match = [s for s in app.songs() if str(s["id"]) == q.get("id", [""])[0]]
                return self._file(match[0]["path"]) if match else self._send(404, b"not found", "text/plain")
            if u.path == "/stream":
                if "once" in parse_qs(u.query):
                    return self._send(200, app.wait_jpeg(-1, 0)[1], "image/jpeg")
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while not getattr(self.server, "closing", False):
                        seq, jpg = app.wait_jpeg(seq, 1.0)
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            self._send(404, b"not found", "text/plain")

        def _file(self, path: Path):
            data = path.read_bytes()
            ctype = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}.get(path.suffix.lower(), "application/octet-stream")
            m = RANGE_RE.fullmatch(self.headers.get("Range", "").strip())
            if m and m.group(1):
                a = int(m.group(1))
                b = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                data = data[a:b + 1]
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            u = urlsplit(self.path)
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"ok": False, "error": "JSON only"})
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host", ""):
                return self._json(403, {"ok": False, "error": "cross-origin refused"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}") if 0 <= n <= 4096 else {}
                if u.path == "/sync":
                    return self._json(200, app.sync(body))
                if u.path == "/start":
                    return self._json(200, app.start(int(body.get("camera", 0)), bool(body.get("mirror", True)),
                                                     body.get("camera_latency_ms")))
                if u.path == "/stop":
                    return self._json(200, app.stop())
            except Exception as exc:
                app.error = str(exc)
                return self._json(500, {"ok": False, "error": str(exc)})
            self._send(404, b"not found", "text/plain")

    return H


def serve(app: LiveCoach, host: str = "127.0.0.1", port: int = 8770) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    return srv


# --------------------------------------------------------------------------- #
# UI (single screen, 100vh, no heavy CSS)
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Salsa Coach</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{height:100vh;overflow:hidden;display:grid;grid-template-columns:65fr 35fr;gap:10px;padding:10px;background:#0b1018;color:#e6edf6;font:13px system-ui,sans-serif}
.l,.r{display:grid;gap:8px;min-height:0;min-width:0}.l{grid-template-rows:minmax(0,1fr) auto}.r{grid-template-rows:auto auto auto minmax(0,1fr)}
.box{background:#121924;border:1px solid #243043;border-radius:8px;padding:8px 10px;min-height:0}
#v{width:100%;height:100%;object-fit:contain;background:#000;border-radius:8px;min-height:0;display:block}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input,button{background:#18212f;color:#e6edf6;border:1px solid #243043;border-radius:6px;padding:5px 8px;font:inherit}
button{cursor:pointer}#go{background:#22c55e;color:#03150a;font-weight:700;border:0;min-width:110px}#go.stop{background:#ef4444;color:#fff}
.m{color:#8b98ab;font-size:12px}h3{margin:0 0 4px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#8b98ab}
.score{font-size:54px;font-weight:800;line-height:1}.kp{display:flex;gap:18px}.kp b{font-size:22px;display:block}
table{width:100%;border-collapse:collapse}td{padding:3px 4px;border-bottom:1px solid #1e293b}
.L{color:#00e5ff;font-weight:700}.R{color:#ff4081;font-weight:700}.ON{color:#22c55e}.EA{color:#f59e0b}.LA{color:#ef4444}
#pad{width:100%;height:160px;display:block}
</style></head><body>
<div class="l"><img id="v" src="/stream?once=1" alt="camera">
<div class="box ctl"><button id="go">Start</button>
<label class="m">Camera <select id="cam"><option>0</option><option>1</option><option>2</option><option>3</option></select></label>
<label class="m"><input type="checkbox" id="mir" checked> Mirror</label>
<label class="m">Music <select id="src"><option value="metro">Salsa clave / cowbell</option></select></label>
<label class="m">BPM <input type="range" id="bpm" min="120" max="200" value="180"> <b id="bv">180</b></label>
<button id="tap" title="Press on a downbeat of the song to set beat 1">Tap 1</button>
<span class="m" id="st">idle</span></div></div>
<div class="r">
<div class="box"><h3>Tempo score</h3><div class="score" id="sc">–</div></div>
<div class="box kp"><div><span class="m">On-tempo</span><b id="ot">–</b></div><div><span class="m">Mean offset</span><b id="mo">–</b></div><div><span class="m">Beat</span><b id="bt">–</b></div></div>
<div class="box"><h3>Footwork pad</h3><canvas id="pad"></canvas></div>
<div class="box" style="overflow:auto"><h3>Last 8 steps</h3><table id="ls"></table></div></div>
<script>
const $=i=>document.getElementById(i);let ctx=null,M=null,au=null,run=false,anchor=0,hb=null;
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
fetch('/songs').then(r=>r.json()).then(j=>{for(const s of j.songs){const o=document.createElement('option');o.value=s.id;o.textContent='♫ '+s.name;$('src').appendChild(o);}});
$('bpm').oninput=()=>{$('bv').textContent=$('bpm').value;if(M)startMetro();sync();};
const lat=()=>ctx?(ctx.outputLatency||ctx.baseLatency||0):0;
function tone(t,f,d,g,ty){const o=ctx.createOscillator(),a=ctx.createGain();o.type=ty;o.frequency.value=f;a.gain.setValueAtTime(1e-4,t);a.gain.exponentialRampToValueAtTime(g,t+.002);a.gain.exponentialRampToValueAtTime(1e-4,t+d);o.connect(a).connect(ctx.destination);o.start(t);o.stop(t+d+.02);}
const CL=[0,1.5,3,5,6];// son clave 3-2
function startMetro(){stopMetro();const p=60/+$('bpm').value;M={t0:ctx.currentTime+.12,p,k:0};M.id=setInterval(()=>{while(M&&M.t0+M.k*M.p<ctx.currentTime+.15){const w=M.t0+M.k*M.p,q=M.k%8;tone(w,587,.08,q?0.12:0.22,'square');tone(w,845,.08,q?0.09:0.18,'square');for(const c of CL)if(Math.floor(c)===q)tone(w+(c-q)*M.p,2500,.04,.3,'sine');M.k++;}},25);}
function stopMetro(){if(M){clearInterval(M.id);M=null;}}
function sync(){let b;if(M)b={playing:true,bpm:+$('bpm').value,beat_time:ctx.currentTime-M.t0-lat()};
 else if(au)b={playing:!au.paused,bpm:+$('bpm').value,beat_time:au.currentTime-anchor-lat()};else b={playing:false};post('/sync',b).catch(()=>{});}
$('tap').onclick=()=>{if(au){anchor=au.currentTime-lat();sync();}else if(M){M.t0=ctx.currentTime;M.k=1;sync();}};
async function start(){ctx=ctx||new AudioContext();await ctx.resume();const s=$('src').value;
 try{const r=await post('/start',{camera:+$('cam').value,mirror:$('mir').checked});if(r.ok===false)throw new Error(r.error);}catch(e){$('st').textContent='error: '+e.message;return;}
 if(s==='metro')startMetro();else{au=new Audio('/song?id='+s);anchor=0;au.onplay=au.onpause=au.onseeked=sync;await au.play();}
 run=true;hb=setInterval(sync,200);sync();$('v').src='/stream?t='+Date.now();$('go').textContent='Stop';$('go').className='stop';}
async function stop(){stopMetro();if(au){au.pause();au=null;}clearInterval(hb);run=false;sync();await post('/stop');$('go').textContent='Start';$('go').className='';}
$('go').onclick=()=>run?stop():start();
const SC={'ON TIME':'ON','EARLY':'EA','LATE':'LA'},PC={'ON TIME':'#22c55e','EARLY':'#f59e0b','LATE':'#ef4444'};
function pad(P){const c=$('pad'),w=c.clientWidth,h=c.clientHeight;if(c.width!==w)c.width=w;if(c.height!==h)c.height=h;const g=c.getContext('2d');g.clearRect(0,0,w,h);
 const k=Math.min(w,h)/(P.reach_cm*2+40),X=x=>w/2+(x-P.center[0])*k,Y=y=>h/2-(y-P.center[1])*k;
 g.strokeStyle='#1e293b';g.beginPath();g.moveTo(w/2,0);g.lineTo(w/2,h);g.moveTo(0,h/2);g.lineTo(w,h/2);g.stroke();
 g.font='10px sans-serif';g.textAlign='center';
 for(const t of P.targets){const x=X(t.x+P.center[0]),y=Y(t.y+P.center[1]);g.strokeStyle=t.foot==='L'?'#00e5ff':'#ff4081';g.setLineDash([3,3]);g.beginPath();g.arc(x,y,9,0,7);g.stroke();g.setLineDash([]);g.fillStyle='#64748b';g.fillText(t.beats,x,y-12);}
 P.steps.forEach((s,i)=>{const x=X(s.pos[0]),y=Y(s.pos[1]);g.globalAlpha=.35+.65*(i+1)/P.steps.length;g.fillStyle=s.foot==='L'?'#00e5ff':'#ff4081';g.beginPath();g.arc(x,y,6,0,7);g.fill();g.strokeStyle=PC[s.status]||'#64748b';g.lineWidth=2;g.stroke();g.lineWidth=1;});
 g.globalAlpha=1;g.fillStyle='#8b98ab';g.fillText('FRONT ▲',w/2,10);}
async function poll(){let S;try{S=await (await fetch('/state')).json();}catch(e){return;}
 const t=S.stats||{};$('sc').textContent=t.tempo_score==null?'–':Math.round(t.tempo_score)+'%';
 $('ot').textContent=t.on_tempo_pct==null?'–':Math.round(t.on_tempo_pct)+'%';$('mo').textContent=t.mean_offset_ms==null?'–':(t.mean_offset_ms>0?'+':'')+Math.round(t.mean_offset_ms)+' ms';
 $('bt').textContent=S.count==null?'–':([4,8].includes(S.count)?'['+S.count+']':S.count);
 $('st').textContent=(S.error?'error: '+S.error+' · ':'')+(S.running?`cam ${S.fps.camera} fps · pose ${S.fps.inference} fps · video ${S.fps.stream} fps · ${S.model||''}`:'idle')+(S.clock.synced?' · beat synced':'');
 $('ls').innerHTML=S.last_steps.map(s=>`<tr><td>${s.t_rel.toFixed(1)}s</td><td class="${s.foot}">${s.foot}</td><td>${s.beat??'–'}</td><td>${s.offset_ms==null?'–':(s.offset_ms>0?'+':'')+Math.round(s.offset_ms)+' ms'}</td><td class="${SC[s.status]||''}">${s.status}</td></tr>`).join('');
 pad(S.pad);}
fetch('/state').then(r=>r.json()).then(S=>{$('cam').value=String(S.default_camera);});
setInterval(poll,100);
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ultra-lightweight real-time webcam salsa coach")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--height", type=float, default=1.70, help="dancer height (m) for the footwork pad")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_model(args.models)  # fetch the lite model up front so Start is instant
    app = LiveCoach(args.models, args.data, height_m=args.height, default_camera=args.camera)
    srv = serve(app, args.host, args.port)
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"Live salsa coach: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.closing = True
        app.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
