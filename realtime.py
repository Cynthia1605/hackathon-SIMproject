"""Phase 5: real-time single-webcam salsa rhythm coach.

The offline two-camera pipeline is untouched; this module only adds a live path. It is
used by ``dashboard.py`` (``/live`` page, ``python dashboard.py --live --camera 0``).

Pieces, from the bottom up:

* ``LiveClock``: maps the browser's audio clock (heartbeats from ``POST /api/live/clock``)
  onto the server's ``time.monotonic()``. The browser plays the song or the clave/cowbell
  metronome and is the master clock, so the audio never stutters.
* ``BeatGrid``: beat times in song time (metronome period, or librosa beats of a song)
  with the 8-count.
* ``StreamingStepDetector``: detects foot plants from MediaPipe ``pose_world_landmarks``
  (hip-centred, metres).
  - The step signal is the left-to-right foot vector (x, z). It does not change when the
    hips move, so a weight change does not look like a step.
  - Motion (or one foot lifting) starts a step; a step ends when the vector is still again.
    The foot that lifted during the motion is the stepping foot.
  - The touchdown time is refined on a rolling ~2 s raw buffer with a centred (lag-free)
    speed estimate. Detection latency is ``still_frames`` frames, about 100-150 ms at
    15-30 fps.
* ``LiveFloor``: top-down floor positions by chaining plants: a stepping foot lands at
  the planted foot plus their measured separation. It uses a facing-aligned frame (+y =
  dancer's front, +x = dancer's right), as in ``footwork.py``.
* ``LiveSession``: steps, beat-offset classification (ON TIME <= 45 ms, EARLY/LATE <= 90 ms,
  OFF-BEAT > 90 ms), live stats, coaching every 2 bars, overlay HUD state, floor-map state
  and export to the recorded-session format.
* ``LiveEngine``: threads for the camera (latest-frame only, so it never lags real time)
  and MediaPipe processing, an MJPEG stream, recording, and Save Session. Camera and pose
  estimator are injectable, so the whole engine is testable without a webcam.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import landmarks as L
from filters import OneEuroFilterStream
from overlay_video import ON_TIME_MS, SLIGHT_MS, classify_offset, is_error, offset_badge

log = logging.getLogger(__name__)

DEFAULT_LIVE_PARAMS: dict[str, Any] = {
    # step detector (metres, seconds)
    "move_speed_mps": 0.25,  # inter-foot vector speed that starts a step
    "still_speed_mps": 0.10,  # ... and that counts as "planted"
    "lift_m": 0.025,  # one foot this much higher than the other also starts a step
    "still_s": 0.03,  # extra stillness needed to confirm a plant (>= 1 frame) -> latency
    "still_window_s": 0.06,  # raw displacement window used for the stillness test (no filter lag)
    "plant_progress": 0.97,  # touchdown = moment the foot covered this fraction of its travel
    "lift_drop_frac": 0.2,  # ... or (in-place steps) the lift fell below this fraction of its peak
    "lift_floor_m": 0.006,
    "min_travel_m": 0.03,  # minimum foot displacement for a step (unless it was lifted)
    "min_step_interval_s": 0.12,
    "buffer_s": 2.0,  # rolling raw buffer used to refine the touchdown time
    "filter_min_cutoff": 3.0,
    "filter_beta": 3.0,
    # session
    "tap_window_s": 1.5,  # same foot twice within this window = TAP
    "active_window_s": 3.0,  # "DANCING" while a step happened within this window
    "callout_s": 0.6,
    "coach_every_beats": 16,  # 2 bars of 8 counts
    "epoch_gap_s": 3.0,
    "camera_latency_ms": 60.0,  # capture pipeline delay subtracted from frame timestamps
    "record_fps": 30.0,
    "jpeg_quality": 80,
    "track_hz": 15.0,
}

LEFT_FOOT = [L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX]
RIGHT_FOOT = [L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX]
EXPECTED_FOOT = {1: "A", 2: "B", 3: "A", 5: "B", 6: "A", 7: "B"}  # A = the foot that steps on 1
MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/"
       f"pose_landmarker_{v}.task" for v in ("lite", "full", "heavy")
}


def _nanmean(a, axis):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


# --------------------------------------------------------------------------- #
# Clock & beat grid
# --------------------------------------------------------------------------- #
class LiveClock:
    """Browser audio clock -> server monotonic clock.

    Each heartbeat gives the audio position the dancer *hears* (``audio_time``, s), sent
    by the browser after subtracting its output latency. Then ``song_time(m) = m + offset``,
    where ``offset`` is the median of recent ``audio_time - receive_time`` samples
    (localhost transport is sub-millisecond). Pauses, seeks and restarts (jumps above
    ``jump_s``) reset the estimate. Heartbeats older than ``stale_s`` make the clock invalid.
    """

    def __init__(self, window: int = 9, jump_s: float = 0.12, stale_s: float = 2.0):
        self.window, self.jump_s, self.stale_s = window, jump_s, stale_s
        self.offsets: deque[float] = deque(maxlen=window)
        self.playing = False
        self.last_update: float | None = None
        self.source: str | None = None
        self.bpm: float | None = None
        self.lock = threading.Lock()

    def update(self, audio_time: float, playing: bool, mono: float | None = None, source: str | None = None,
               bpm: float | None = None) -> None:
        mono = time.monotonic() if mono is None else float(mono)
        with self.lock:
            self.last_update, self.source = mono, source or self.source
            if bpm:
                self.bpm = float(bpm)
            if not playing or audio_time is None or not math.isfinite(audio_time):
                self.playing = False
                self.offsets.clear()
                return
            off = float(audio_time) - mono
            if self.offsets and abs(off - float(np.median(self.offsets))) > self.jump_s:
                self.offsets.clear()  # seek / restart / tempo change
            self.offsets.append(off)
            self.playing = True

    def _offset(self, now: float | None = None) -> float | None:
        if not self.playing or not self.offsets or self.last_update is None:
            return None
        if now is not None and now - self.last_update > self.stale_s:
            return None
        return float(np.median(self.offsets))

    def song_time(self, mono: float, now: float | None = None) -> float | None:
        with self.lock:
            off = self._offset(time.monotonic() if now is None else now)
        return None if off is None else mono + off

    def to_mono(self, song_t: float) -> float | None:
        with self.lock:
            off = self._offset()
        return None if off is None else song_t - off

    def status(self) -> dict[str, Any]:
        with self.lock:
            jitter = float(np.std(self.offsets) * 1000) if len(self.offsets) > 1 else None
            age = None if self.last_update is None else time.monotonic() - self.last_update
            return {"playing": self.playing and (age is not None and age <= self.stale_s), "source": self.source,
                    "bpm": self.bpm, "jitter_ms": jitter, "heartbeat_age_s": age, "samples": len(self.offsets)}


class BeatGrid:
    """Beats on the song timeline with 8-count labels (count of beat k = (k + shift) % 8 + 1)."""

    def __init__(self, period: float, times: np.ndarray | None = None, count_shift: int = 0, bpm: float | None = None):
        self.period = float(period)
        self.times = None if times is None else np.asarray(times, dtype=np.float64)
        self.count_shift = int(count_shift)
        self.bpm = float(bpm) if bpm else 60.0 / self.period

    @classmethod
    def metronome(cls, bpm: float, count_shift: int = 0) -> "BeatGrid":
        return cls(60.0 / float(bpm), None, count_shift, float(bpm))

    @classmethod
    def from_song(cls, beat_times, bpm: float | None = None, count_shift: int = 0) -> "BeatGrid":
        bt = np.asarray(beat_times, dtype=np.float64)
        period = float(np.median(np.diff(bt))) if len(bt) >= 2 else 60.0 / float(bpm or 120)
        return cls(period, bt, count_shift, 60.0 / period)

    def beat_time(self, k: int) -> float:
        if self.times is None:
            return k * self.period
        if 0 <= k < len(self.times):
            return float(self.times[k])
        return float(self.times[0] + k * self.period) if k < 0 else float(self.times[-1] + (k - len(self.times) + 1) * self.period)

    def nearest(self, song_t: float) -> tuple[int, float] | None:
        if self.times is None:
            if song_t < -self.period / 2:
                return None
            k = int(round(song_t / self.period))
            return k, k * self.period
        if len(self.times) == 0 or song_t < self.times[0] - self.period / 2 or song_t > self.times[-1] + self.period / 2:
            return None
        k = int(np.argmin(np.abs(self.times - song_t)))
        return k, float(self.times[k])

    def floor_index(self, song_t: float) -> int | None:
        if self.times is None:
            return int(math.floor(song_t / self.period)) if song_t >= 0 else None
        if len(self.times) == 0 or song_t < self.times[0] or song_t > self.times[-1] + self.period:
            return None
        return int(np.searchsorted(self.times, song_t, side="right")) - 1

    def count_of(self, k: int) -> int:
        return (k + self.count_shift) % 8 + 1

    def current(self, song_t: float | None) -> tuple[int | None, float]:
        """(8-count, progress 0-1 through the current beat) at song time ``song_t``."""
        if song_t is None:
            return None, 0.0
        k = self.floor_index(song_t)
        if k is None:
            return None, 0.0
        t0, t1 = self.beat_time(k), self.beat_time(k + 1)
        return self.count_of(k), float(np.clip((song_t - t0) / max(1e-6, t1 - t0), 0, 1))


# --------------------------------------------------------------------------- #
# Streaming step detection
# --------------------------------------------------------------------------- #
@dataclass
class StepEvent:
    t: float  # touchdown time (same clock as the input frames)
    foot: str  # "L" | "R"
    travel_m: float
    lift_m: float
    detected_at: float  # time of the frame that confirmed the plant (latency = detected_at - t)


def foot_signals(world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Horizontal (x, z) foot centres and heights (+up, m) from hip-centred world landmarks (y down)."""
    lf, rf = world[LEFT_FOOT], world[RIGHT_FOOT]
    if not (np.isfinite(lf).all() and np.isfinite(rf).all()):
        lf_m, rf_m = _nanmean(lf, 0), _nanmean(rf, 0)
        if not (np.isfinite(lf_m).all() and np.isfinite(rf_m).all()):
            return None
    else:
        lf_m, rf_m = lf.mean(0), rf.mean(0)
    return lf_m[[0, 2]], rf_m[[0, 2]], -float(lf_m[1]), -float(rf_m[1])


class StreamingStepDetector:
    def __init__(self, params: dict[str, Any] | None = None):
        self.p = {**DEFAULT_LIVE_PARAMS, **(params or {})}
        self.f_rel = OneEuroFilterStream(self.p["filter_min_cutoff"], self.p["filter_beta"], 1.0, 0.5)
        # the lift signal must react quickly (steps are only ~330 ms apart at 180 BPM)
        self.f_lift = OneEuroFilterStream(3 * self.p["filter_min_cutoff"], 0.0, 1.0, 0.5)
        self.buf: deque = deque()  # raw (t, rel_x, rel_z, hL, hR)
        self.state = "STILL"
        self.prev = None  # (t, rel_f)
        self.still_count = 0
        self.move_start_rel = None
        self.max_lift = 0.0
        self.lift_acc = {"L": 0.0, "R": 0.0}
        self.hip_acc = {"L": 0.0, "R": 0.0}
        self.prev_feet = None
        self.last_event_t = -1e9
        self.move_t0 = None
        self.still_time = 0.0
        self.lift_side = None  # foot currently lifted (above lift_m) during this motion

    def update(self, t: float, world: np.ndarray | None) -> list[StepEvent]:
        sig = None if world is None else foot_signals(np.asarray(world, dtype=np.float64))
        if sig is None:
            return []
        lxz, rxz, hl, hr = sig
        rel = lxz - rxz
        self.buf.append((t, rel[0], rel[1], hl, hr))
        while self.buf and t - self.buf[0][0] > self.p["buffer_s"]:
            self.buf.popleft()
        rel_f = self.f_rel(t, rel)
        lift_f = float(self.f_lift(t, np.array([hl - hr]))[0])
        out: list[StepEvent] = []
        if self.prev is None or t <= self.prev[0]:
            self.prev, self.prev_feet = (t, rel_f), (lxz, rxz)
            return out
        dt = t - self.prev[0]
        speed = float(np.linalg.norm(rel_f - self.prev[1]) / dt)
        lifted = abs(lift_f) > self.p["lift_m"]
        if self.state == "STILL":
            if speed > self.p["move_speed_mps"] or lifted:
                self.state = "MOVING"
                self.move_t0 = self.prev[0]
                self.lift_side = None
                self.move_start_rel = self.prev[1].copy()
                self.max_lift = 0.0
                self.lift_acc = {"L": 0.0, "R": 0.0}
                self.hip_acc = {"L": 0.0, "R": 0.0}
                self.still_count, self.still_time = 0, 0.0
        if self.state == "MOVING":
            # the other foot lifting means the previous step already landed: close it now
            side = ("L" if lift_f > 0 else "R") if lifted else None
            if side is not None and self.lift_side is not None and side != self.lift_side:
                ev = self._emit(t, rel_f, end_t=t - dt)
                if ev is not None:
                    out.append(ev)
                self.state = "MOVING"
                self.move_t0 = self.prev[0]
                self.move_start_rel = self.prev[1].copy()
                self.max_lift, self.lift_acc, self.hip_acc = 0.0, {"L": 0.0, "R": 0.0}, {"L": 0.0, "R": 0.0}
                self.still_count, self.still_time = 0, 0.0
            if side is not None:
                self.lift_side = side
            w = max(speed, 0.05) * dt
            self.lift_acc["L"] += max(lift_f, 0.0) * w
            self.lift_acc["R"] += max(-lift_f, 0.0) * w
            self.max_lift = max(self.max_lift, abs(lift_f))
            self.hip_acc["L"] += float(np.linalg.norm(lxz - self.prev_feet[0]))
            self.hip_acc["R"] += float(np.linalg.norm(rxz - self.prev_feet[1]))
            # stillness on raw samples (displacement over ~66 ms): lag-free, noise-robust
            j = len(self.buf) - 2
            while j > 0 and t - self.buf[j][0] < self.p["still_window_s"]:
                j -= 1
            tj, rxj, rzj = self.buf[j][0], self.buf[j][1], self.buf[j][2]
            raw_speed = float(np.hypot(rel[0] - rxj, rel[1] - rzj) / max(t - tj, 1e-6))
            if raw_speed < self.p["still_speed_mps"] and abs(hl - hr) < 0.5 * self.p["lift_m"]:
                self.still_count += 1
                self.still_time += dt
            else:
                self.still_count, self.still_time = 0, 0.0
            if self.still_count >= 1 and self.still_time >= self.p["still_s"] - 1e-9:
                self.state = "STILL"
                ev = self._emit(t, rel_f)
                if ev is not None:
                    out.append(ev)
        self.prev, self.prev_feet = (t, rel_f), (lxz, rxz)
        return out

    def _emit(self, t: float, rel_f: np.ndarray, end_t: float | None = None) -> StepEvent | None:
        travel = float(np.linalg.norm(rel_f - self.move_start_rel))
        if travel < self.p["min_travel_m"] and self.max_lift < self.p["lift_m"]:
            return None
        if self.lift_acc["L"] + self.lift_acc["R"] > 1e-6:
            foot = "L" if self.lift_acc["L"] >= self.lift_acc["R"] else "R"
        else:
            foot = "L" if self.hip_acc["L"] >= self.hip_acc["R"] else "R"
        t_td = self._touchdown_time(self.move_t0, t if end_t is None else end_t)
        if t_td - self.last_event_t < self.p["min_step_interval_s"]:
            return None
        self.last_event_t = t_td
        return StepEvent(t_td, foot, travel, self.max_lift, t)

    def _touchdown_time(self, t_start: float | None, t_end: float) -> float:
        """Touchdown of the step whose motion started at ``t_start``, from the raw rolling buffer.

        * Travel: the first moment the inter-foot vector has covered ``plant_progress`` (97%)
          of its way from the pre-motion value to the planted value (mean of the samples after
          the plant), interpolated between frames. Raw samples (no smoothing) keep the timing
          independent of the frame rate; the 3% margin is well above landmark noise for real steps.
        * Lift (in-place / replace steps): the first moment after the highest lift that the foot
          is back below ``lift_drop_frac`` (20%) of that peak (at least ``lift_floor_m``).

        The later of the two is the touchdown.
        """
        b = np.asarray(self.buf)
        if len(b) < 3:
            return float(b[-1, 0])
        t, rel, lift = b[:, 0], b[:, 1:3], np.abs(b[:, 3] - b[:, 4])
        lo = (t_start - 0.1) if t_start is not None else t[0]
        win = np.flatnonzero((t >= lo) & (t <= t_end + 1e-9))
        if len(win) < 3:
            return float(min(t_end, t[-1]))
        cands = []
        tail = rel[win[-2:]].mean(axis=0)  # planted value
        start = rel[win[0]]
        d = tail - start
        travel = float(np.linalg.norm(d))
        if travel >= 2 * self.p["min_travel_m"]:
            prog = (rel[win] - start) @ d / (travel * travel)
            thr = self.p["plant_progress"]
            above = np.flatnonzero(prog >= thr)
            if len(above):
                j = above[0]
                if j > 0:
                    a = (thr - prog[j - 1]) / max(prog[j] - prog[j - 1], 1e-9)
                    cands.append(t[win[j - 1]] + a * (t[win[j]] - t[win[j - 1]]))
                else:
                    cands.append(t[win[0]])
        lw = lift[win]
        if lw.max() >= self.p["lift_m"]:
            pk = int(np.argmax(lw))
            half = max(self.p["lift_drop_frac"] * lw[pk], self.p["lift_floor_m"])
            after = np.flatnonzero(lw[pk:] < half)
            if len(after):
                j = pk + after[0]
                a = (lw[j - 1] - half) / max(lw[j - 1] - lw[j], 1e-9)
                cands.append(t[win[j - 1]] + a * (t[win[j]] - t[win[j - 1]]))
        return float(max(cands)) if cands else float(min(t_end, t[-1]))


# --------------------------------------------------------------------------- #
# Floor positions (top-down map)
# --------------------------------------------------------------------------- #
class LiveFloor:
    """Top-down floor positions (cm) by chaining plants in a facing-aligned frame.

    MediaPipe world coordinates have y down, so for a facing vector f = (fx, fz) the
    dancer's right is (fz, -fx). A dancer facing the camera has f = (0, -1) and right =
    (-1, 0), i.e. image-left.
    """

    def __init__(self, height_m: float = 1.70):
        import footwork

        self.geo = footwork.ideal_geometry(height_m)
        self.fwd = np.array([0.0, -1.0])
        self.n_facing = 0
        self.P: dict[str, np.ndarray] | None = None
        self.last_world_feet = None

    def observe(self, world: np.ndarray | None) -> None:
        if world is None:
            return
        v = []
        for heel, toe in ((L.LEFT_HEEL, L.LEFT_FOOT_INDEX), (L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX)):
            d = world[toe, [0, 2]] - world[heel, [0, 2]]
            n = float(np.linalg.norm(d))
            if np.isfinite(n) and n > 0.03:
                v.append(d / n)
        if v:
            m = np.mean(v, axis=0)
            if np.linalg.norm(m) > 0.5:
                m = m / np.linalg.norm(m)
                a = 1.0 / min(self.n_facing + 1, 60)  # running mean, then a slow EMA
                f = (1 - a) * self.fwd + a * m
                self.fwd = f / np.linalg.norm(f)
                self.n_facing += 1
        sig = foot_signals(world)
        if sig is not None:
            self.last_world_feet = (sig[0], sig[1], sig[2], sig[3])

    @property
    def right(self) -> np.ndarray:
        return np.array([self.fwd[1], -self.fwd[0]])

    def to_floor(self, v_xz: np.ndarray) -> np.ndarray:
        return np.array([float(v_xz @ self.right), float(v_xz @ self.fwd)]) * 100.0

    def _init(self):
        lxz, rxz = self.last_world_feet[0], self.last_world_feet[1]
        half = self.to_floor(lxz - rxz) / 2
        self.P = {"L": half, "R": -half}

    def on_step(self, foot: str) -> np.ndarray | None:
        if self.last_world_feet is None:
            return None
        if self.P is None:
            self._init()
        lxz, rxz = self.last_world_feet[0], self.last_world_feet[1]
        other = "R" if foot == "L" else "L"
        d = self.to_floor((lxz - rxz) if foot == "L" else (rxz - lxz))
        self.P[foot] = self.P[other] + d
        return self.P[foot].copy()

    def live_positions(self) -> dict[str, list[float]] | None:
        if self.last_world_feet is None:
            return None
        if self.P is None:
            self._init()
        lxz, rxz, hl, hr = self.last_world_feet
        ref = "L" if hl <= hr else "R"  # the lower foot is the planted reference
        out = {ref: self.P[ref].copy()}
        if ref == "L":
            out["R"] = self.P["L"] + self.to_floor(rxz - lxz)
        else:
            out["L"] = self.P["R"] + self.to_floor(lxz - rxz)
        return {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in out.items()}

    def anchor(self) -> np.ndarray:
        if self.P is None:
            return np.zeros(2)
        return 0.5 * (self.P["L"] + self.P["R"])


# --------------------------------------------------------------------------- #
# Live session
# --------------------------------------------------------------------------- #
@dataclass
class LiveStep:
    i: int
    t: float  # touchdown, server monotonic
    t_rel: float  # seconds since the session started (= cam1 time of the recording)
    foot: str
    song_time: float | None
    beat_song_time: float | None
    offset_ms: float | None
    count: int | None
    category: str
    status: str
    timing_type: str  # QUICK | SLOW | HOLD | TAP | STEP
    label: str
    latency_ms: float
    x: float | None = None
    y: float | None = None
    heading_deg: float = 0.0
    cycle: int = 0
    target_x: float | None = None
    target_y: float | None = None
    err_cm: float | None = None
    len_cm: float | None = None
    fwd_disp_cm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STATUS_EN = {"on_time": "On Time", "early": "Early", "late": "Late", "off_early": "Early (off-beat)",
             "off_late": "Late (off-beat)", "none": "No beat"}
FOOT_WORD = {"L": "left", "R": "right"}


class LiveSession:
    def __init__(self, height_m: float = 1.70, params: dict[str, Any] | None = None, clock: LiveClock | None = None,
                 t0: float | None = None, synthetic_time: bool = False):
        """``synthetic_time``: frame timestamps are not ``time.monotonic()`` (tests / offline replays),
        so clock staleness is judged against the frame time instead of the wall clock."""
        self.p = {**DEFAULT_LIVE_PARAMS, **(params or {})}
        self._synthetic_now = bool(synthetic_time)
        self.height_m = float(height_m)
        self.clock = clock or LiveClock()
        self.grid: BeatGrid | None = None
        self.detector = StreamingStepDetector(self.p)
        self.floor = LiveFloor(height_m)
        self.steps: list[LiveStep] = []
        self.t0 = t0
        self.last_t: float | None = None
        self.cycle = 0
        self.cycle_anchor: dict[int, np.ndarray] = {}
        self.cycle_epoch: dict[int, int] = {}
        self.last_counted: dict[str, LiveStep] = {}
        self.coach = {"text": "Press Start and begin your basic step.", "kind": "info", "bar": None, "updated_at": None}
        self._last_coach_k: int | None = None
        self.beats_seen: dict[int, tuple[float, int]] = {}  # beat index -> (time rel to t0, count)
        self.track: list[tuple[float, list | None, list | None]] = []
        self._last_track_t = -1e9
        self.song0_rel: float | None = None  # session time at which song t=0 happened
        self.lock = threading.RLock()

    # ---- music ----
    def set_grid(self, grid: BeatGrid | None) -> None:
        with self.lock:
            self.grid = grid

    def shift_count(self, delta: int = 1) -> None:
        with self.lock:
            if self.grid is not None:
                self.grid.count_shift = (self.grid.count_shift + int(delta)) % 8

    # ---- frames ----
    def on_frame(self, t: float, kp2d: np.ndarray | None, world: np.ndarray | None) -> list[LiveStep]:
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.last_t = t
            self.floor.observe(world)
            events = self.detector.update(t, world)
            new = [self._add_step(ev) for ev in events]
            self._observe_beats(t)
            if t - self._last_track_t >= 1.0 / self.p["track_hz"]:
                lp = self.floor.live_positions()
                if lp:
                    self.track.append((round(t - self.t0, 3), lp.get("L"), lp.get("R")))
                    self._last_track_t = t
            self._maybe_coach(t)
            return new

    def _observe_beats(self, t: float) -> None:
        if self.grid is None:
            return
        st = self.clock.song_time(t, now=t if self._synthetic_now else None)
        if st is None:
            return
        if self.song0_rel is None:
            self.song0_rel = (t - self.t0) - st
        k = self.grid.floor_index(st)
        if k is not None and k not in self.beats_seen:
            bt = self.grid.beat_time(k)
            self.beats_seen[k] = (t - self.t0 - (st - bt), self.grid.count_of(k))

    def _add_step(self, ev: StepEvent) -> LiveStep:
        st = self.clock.song_time(ev.t, now=ev.detected_at if self._synthetic_now else None)
        off = count = bst = None
        if self.grid is not None and st is not None:
            nb = self.grid.nearest(st)
            if nb is not None:
                k, bst = nb
                off = (st - bst) * 1000.0
                count = self.grid.count_of(k)
        prev = self.steps[-1] if self.steps else None
        tap = prev is not None and prev.foot == ev.foot and ev.t - prev.t < self.p["tap_window_s"]
        if tap:
            ttype = "TAP"
        elif count in (3, 7):
            ttype = "SLOW"
        elif count in (1, 2, 5, 6):
            ttype = "QUICK"
        elif count in (4, 8):
            ttype = "HOLD"
        else:
            ttype = "STEP"
        cat = "none" if (tap or off is None) else classify_offset(off)[0]
        status = "Tap" if tap else ("Stepped on hold" if ttype == "HOLD" else STATUS_EN[cat])
        label = f"{ev.foot}{count if count else ''}" + ("-SLOW" if ttype == "SLOW" else "") + ("-TAP" if tap else "")
        # cycles: a new 8-count phrase starts when the count wraps or after a pause
        last_c = next((s for s in reversed(self.steps) if s.count is not None and s.timing_type != "TAP"), None)
        if prev is not None and (ev.t - prev.t > self.p["epoch_gap_s"] or
                                 (count is not None and last_c is not None and not tap and count <= last_c.count)):
            self.cycle += 1
        if self.cycle not in self.cycle_anchor:
            self.cycle_anchor[self.cycle] = self.floor.anchor()
            self.cycle_epoch[self.cycle] = self._epoch_of(ev.t, prev)
        pos = self.floor.on_step(ev.foot)
        s = LiveStep(len(self.steps), ev.t, ev.t - (self.t0 or ev.t), ev.foot, st, bst, off, count, cat, status, ttype,
                     label, (ev.detected_at - ev.t) * 1000.0, cycle=self.cycle)
        if pos is not None:
            s.x, s.y = round(float(pos[0]), 1), round(float(pos[1]), 1)
            if not tap:
                tg = self._target(count, ev.foot)
                if tg is not None:
                    a = self.cycle_anchor[self.cycle]
                    s.target_x, s.target_y = round(float(a[0] + tg[0]), 1), round(float(a[1] + tg[1]), 1)
                    s.err_cm = round(float(np.hypot(pos[0] - s.target_x, pos[1] - s.target_y)), 1)
                lc = self.last_counted.get(ev.foot)
                if lc is not None and lc.x is not None:
                    s.len_cm = round(float(np.hypot(pos[0] - lc.x, pos[1] - lc.y)), 1)
                    s.fwd_disp_cm = round(float(pos[1] - lc.y), 1)
        if not tap:
            self.last_counted[ev.foot] = s
        self.steps.append(s)
        return s

    def _target(self, count: int | None, foot: str) -> tuple[float, float] | None:
        """Ideal (x, y) offset from the cycle anchor for this count/foot, or None if the foot is unexpected.

        The ideal pattern is defined for a dancer stepping forward with the left foot on 1 (leader's
        On-1 basic). A dancer who steps on 1 with the right foot gets the mirrored pattern
        (feet swapped, x and y negated: right foot back on 1, left forward on 5).
        """
        tg = self.floor.geo["targets"].get(count)
        if tg is None:
            return None
        ideal_foot, x, y = tg
        if self.one_foot() == "L":
            return (x, y) if foot == ideal_foot else None
        return (-x, -y) if foot != ideal_foot else None

    def _epoch_of(self, t: float, prev: LiveStep | None) -> int:
        if prev is None:
            return 0
        e = self.cycle_epoch.get(prev.cycle, 0)
        return e + 1 if t - prev.t > self.p["epoch_gap_s"] else e

    def one_foot(self) -> str:
        """Which foot steps on 1/3/6 (majority vote; default left)."""
        votes = {"L": 0, "R": 0}
        for s in self.steps:
            if s.timing_type in ("QUICK", "SLOW") and s.count in EXPECTED_FOOT:
                a_foot = s.foot if EXPECTED_FOOT[s.count] == "A" else ("R" if s.foot == "L" else "L")
                votes[a_foot] += 1
        return "R" if votes["R"] > votes["L"] else "L"

    # ---- stats & coaching ----
    def stats(self, steps: list[LiveStep] | None = None) -> dict[str, Any]:
        from dashboard import tempo_score

        steps = self.steps if steps is None else steps
        counted = [s for s in steps if s.timing_type != "TAP"]
        offs = [s.offset_ms for s in counted if s.offset_ms is not None]
        one = self.one_foot()
        pat = [s for s in counted if s.count is not None]
        ok_pat = [s for s in pat if s.count in EXPECTED_FOOT and
                  s.foot == (one if EXPECTED_FOOT[s.count] == "A" else ("R" if one == "L" else "L"))]
        qqs = (len(ok_pat) / len(pat)) if pat else None
        period = self.grid.period if self.grid else 0.5
        score = tempo_score(offs, period * 500, qqs)
        return {
            "n_steps": len(counted), "n_taps": len(steps) - len(counted),
            "n_errors": sum(abs(o) > ON_TIME_MS for o in offs),
            "on_tempo_pct": (100.0 * float(np.mean([abs(o) <= ON_TIME_MS for o in offs]))) if offs else None,
            "mean_offset_ms": float(np.mean(offs)) if offs else None,
            "mean_abs_offset_ms": float(np.mean(np.abs(offs))) if offs else None,
            "consistency_ms": float(np.std(offs)) if offs else None,
            "qqs_accuracy_pct": None if qqs is None else 100.0 * qqs,
            "tempo_score": score,
            "one_foot": one,
            "mean_latency_ms": float(np.mean([s.latency_ms for s in steps[-20:]])) if steps else None,
        }

    def _maybe_coach(self, t: float) -> None:
        if self.grid is None:
            return
        st = self.clock.song_time(t, now=t if self._synthetic_now else None)
        if st is None:
            return
        k = self.grid.floor_index(st)
        if k is None:
            return
        every = int(self.p["coach_every_beats"])
        if self._last_coach_k is None:
            self._last_coach_k = k
            return
        if k - self._last_coach_k >= every:
            window_start = self.grid.beat_time(k - every) - self.grid.period / 2
            recent = [s for s in self.steps if s.song_time is not None and s.song_time >= window_start]
            self.coach = dict(coaching_message(recent), bar=(k // 8) + 1, updated_at=round(t - (self.t0 or t), 2))
            self._last_coach_k = k

    # ---- views ----
    def overlay_state(self, t: float) -> dict[str, Any]:
        """HUD state in the format of overlay_video.RhythmTimeline.state_at (for draw_hud)."""
        with self.lock:
            st = self.clock.song_time(t, now=t if self._synthetic_now else None)
            count, prog = (self.grid.current(st) if self.grid is not None else (None, 0.0))
            callouts, flashes = [], {}
            for s in reversed(self.steps):
                age = t - s.t
                if age > self.p["callout_s"]:
                    break
                if age < 0 or s.timing_type == "TAP":
                    continue
                callouts.append({"foot": s.foot, "foot_label": "LEFT FOOT" if s.foot == "L" else "RIGHT FOOT",
                                 "beat": s.count, "offset_ms": s.offset_ms, "category": s.category,
                                 "badge": offset_badge(s.offset_ms), "is_error": is_error(s.offset_ms), "age": age})
                if is_error(s.offset_ms) and s.count:
                    flashes[s.count] = min(age, flashes.get(s.count, age))
            offs = [s.offset_ms for s in self.steps if s.timing_type != "TAP" and s.offset_ms is not None]
            active = bool(self.steps) and t - self.steps[-1].t <= self.p["active_window_s"]
            return {"t": t, "active": active, "status": "DANCING" if active else "IDLE", "count": count,
                    "beat_progress": prog, "callouts": callouts, "error_flash": flashes,
                    "on_time": (sum(abs(o) <= ON_TIME_MS for o in offs), len(offs)),
                    "bpm": self.grid.bpm if self.grid else 0.0}

    def floor_state(self) -> dict[str, Any]:
        with self.lock:
            g = self.floor.geo
            cur = self.cycle
            show = [s for s in self.steps if s.x is not None and s.timing_type != "TAP" and cur - 2 <= s.cycle <= cur]
            a = self.cycle_anchor.get(cur, self.floor.anchor())
            return {
                "ideal": {"reach_cm": round(g["reach_cm"], 1), "half_stance_cm": round(g["half_stance_cm"], 1),
                          "foot_length_cm": round(g["foot_length_cm"], 1), "foot_width_cm": round(g["foot_width_cm"], 1),
                          "one_foot": self.one_foot()},
                "anchor": [round(float(a[0]), 1), round(float(a[1]), 1)],
                "cycle": cur,
                "steps": [{"i": s.i, "x": s.x, "y": s.y, "foot": s.foot, "beat": s.count, "category": s.category,
                           "err_cm": s.err_cm, "target_x": s.target_x, "target_y": s.target_y, "len_cm": s.len_cm,
                           "offset_ms": s.offset_ms, "cycle": s.cycle, "age_cycles": cur - s.cycle} for s in show],
                "live": self.floor.live_positions(),
            }

    def state_dict(self, n_steps: int = 60) -> dict[str, Any]:
        with self.lock:
            t = self.last_t
            ov = self.overlay_state(t) if t is not None else {"count": None, "beat_progress": 0.0}
            return {
                "stats": self.stats(),
                "coach": self.coach,
                "count": ov.get("count"),
                "beat_progress": ov.get("beat_progress"),
                "steps": [_step_view(s) for s in self.steps[-n_steps:]][::-1],
                "floor": self.floor_state(),
                "grid": None if self.grid is None else {"bpm": round(self.grid.bpm, 2), "count_shift": self.grid.count_shift,
                                                         "source": "song" if self.grid.times is not None else "metronome"},
                "clock": self.clock.status(),
                "elapsed_s": None if (t is None or self.t0 is None) else round(t - self.t0, 2),
            }

    # ---- export (recorded-session format) ----
    def to_report(self, subject_id: str, fps: float, duration_s: float, music_source: dict[str, Any]) -> dict[str, Any]:
        """``epochs_and_steps.json``-compatible report, on the recording timeline (cam1 time = t_rel)."""
        import footwork

        with self.lock:
            steps = list(self.steps)
            period = self.grid.period if self.grid else 0.5
            beats = sorted(self.beats_seen.items())
            grid_rel = [round(v[0], 4) for _, v in beats]
            # epochs: groups of steps separated by > epoch_gap_s
            epochs, cur = [], []
            for s in steps:
                if cur and s.t - cur[-1].t > self.p["epoch_gap_s"]:
                    epochs.append(cur)
                    cur = []
                cur.append(s)
            if cur:
                epochs.append(cur)
            ep_of = {}
            ep_out = []
            for e, grp in enumerate(epochs):
                a, b = max(0.0, grp[0].t_rel - 1.0), min(duration_s, grp[-1].t_rel + 1.0)
                ep_out.append({"epoch_id": e, "start_idx": int(round(a * fps)), "end_idx": int(round(b * fps)),
                               "start_sec": a, "end_sec": b, "cam1_start_frame": int(round(a * fps)),
                               "cam1_end_frame": int(round(b * fps)) - 1, "label": "ACTIVE_MOVE", "duration_sec": b - a})
                for s in grp:
                    ep_of[s.i] = e
            st_out = []
            for s in steps:
                nb = None if s.offset_ms is None else s.t_rel - s.offset_ms / 1000.0
                st_out.append({
                    "epoch_id": ep_of.get(s.i, 0), "idx": int(round(s.t_rel * fps)), "cam1_frame": int(round(s.t_rel * fps)),
                    "timestamp_sec": s.t_rel, "foot": s.foot,
                    "timing_type": s.timing_type if s.timing_type in ("QUICK", "SLOW", "TAP") else "QUICK",
                    "beat_count": s.count if s.count in (1, 2, 3, 5, 6, 7) else None,
                    "hold_after": "4-hold" if s.count == 3 else "8-hold" if s.count == 7 else None,
                    "interval_sec": None, "nearest_music_beat_sec": nb, "beat_offset_ms": s.offset_ms,
                    "step_length_cm": s.len_cm or 0.0, "label": s.label, "floor_x_cm": s.x, "floor_y_cm": s.y,
                    "placement_err_cm": s.err_cm, "floor_cycle": s.cycle, "live_status": s.status,
                })
            counted = [x for x in st_out if x["timing_type"] != "TAP"]
            for a, b in zip(counted, counted[1:]):
                a["interval_sec"] = b["timestamp_sec"] - a["timestamp_sec"]
            stats = self.stats()
            qqs = None if stats["qqs_accuracy_pct"] is None else stats["qqs_accuracy_pct"] / 100.0
            # footwork block (same schema as footwork.compute_footwork)
            fsteps = [{"i": s.i, "x": s.x if s.x is not None else 0.0, "y": s.y if s.y is not None else 0.0,
                       "heading_deg": 0.0, "cycle": s.cycle, "epoch": self.cycle_epoch.get(s.cycle, 0),
                       "target_x": s.target_x, "target_y": s.target_y, "err_cm": s.err_cm, "len_cm": s.len_cm,
                       **({"fwd_disp_cm": s.fwd_disp_cm} if s.fwd_disp_cm is not None else {})} for s in steps]
            cyc = {c: {"id": c, "epoch": self.cycle_epoch.get(c, 0), "anchor": np.asarray(a, dtype=float),
                       "t0": min((s.t_rel for s in steps if s.cycle == c), default=0.0),
                       "t1": max((s.t_rel for s in steps if s.cycle == c), default=0.0)} for c, a in self.cycle_anchor.items()}
            geo = self.floor.geo
            report_steps_for_metrics = [dict(x, beat_count=x["beat_count"]) for x in st_out]
            metrics = footwork.spatial_metrics(report_steps_for_metrics, fsteps, cyc, geo)
            fw = {
                "available": bool(steps), "method": ["live_world_landmarks"], "height_m": self.height_m,
                "ideal": {"reach_cm": round(geo["reach_cm"], 1), "half_stance_cm": round(geo["half_stance_cm"], 1),
                          "foot_length_cm": round(geo["foot_length_cm"], 1), "foot_width_cm": round(geo["foot_width_cm"], 1),
                          "targets": [{"beat": b, "foot": f, "x": round(x, 1), "y": round(y, 1), "slow": b in (3, 7)}
                                      for b, (f, x, y) in geo["targets"].items()]},
                "steps": fsteps,
                "cycles": [{"id": c["id"], "epoch": c["epoch"], "t0": c["t0"], "t1": c["t1"],
                            "anchor": [round(float(c["anchor"][0]), 1), round(float(c["anchor"][1]), 1)]}
                           for c in sorted(cyc.values(), key=lambda c: c["id"])],
                "track": {"t": [x[0] for x in self.track], "L": [x[1] for x in self.track], "R": [x[2] for x in self.track]},
                "metrics": metrics,
            }
            bpm = self.grid.bpm if self.grid else None
            return {
                "subject_id": subject_id, "fps": fps, "live": True, "height_meters": self.height_m,
                "sync": {"method": "single_camera", "frame_offset": 0},
                "music": {"bpm": bpm, "beat_period_sec": period, "mp3_offset_sec": music_source.get("mp3_offset_sec", 0.0),
                          "aligned_beat_times_sec": grid_rel, "method": "live", "confidence": 1.0,
                          "source": music_source.get("source"), "beat_times_mp3_sec": [], "warnings": []},
                "beat_grid_sec": grid_rel,
                "epochs": ep_out,
                "steps": st_out,
                "summary": {"beat_period_sec": period, "qqs_pattern_accuracy": qqs, "n_steps": stats["n_steps"],
                            "n_taps": stats["n_taps"], "mean_offset_ms": stats["mean_offset_ms"],
                            "mean_abs_offset_ms": stats["mean_abs_offset_ms"], "std_offset_ms": stats["consistency_ms"],
                            "n_epochs": len(ep_out), "music_bpm": bpm},
                "footwork": fw,
                "warnings": [],
            }


def _step_view(s: LiveStep) -> dict[str, Any]:
    return {"i": s.i, "t": round(s.t_rel, 2), "foot": s.foot, "beat": s.count,
            "offset_ms": None if s.offset_ms is None else round(s.offset_ms, 1), "category": s.category,
            "status": s.status, "type": s.timing_type, "label": s.label, "err_cm": s.err_cm,
            "latency_ms": round(s.latency_ms, 0)}


def coaching_message(steps: list[LiveStep]) -> dict[str, str]:
    """One actionable coaching line from the steps of the last 2 bars."""
    counted = [s for s in steps if s.timing_type != "TAP"]
    if not counted:
        return {"text": "No steps in the last 2 bars — start your basic: 1-2-3, hold, 5-6-7, hold.", "kind": "info"}
    holds = [s for s in counted if s.timing_type == "HOLD"]
    if holds:
        b = holds[-1].count
        return {"text": f"Stepping on the hold (beat {b}) — pause on 4 and 8, step only on 1-2-3 and 5-6-7.",
                "kind": "warn"}
    offs = [s.offset_ms for s in counted if s.offset_ms is not None]
    if not offs:
        return {"text": "Steps detected but no music beat yet — start the song or metronome.", "kind": "info"}
    groups: dict[tuple[int, str], list[float]] = {}
    for s in counted:
        if s.offset_ms is not None and s.count is not None:
            groups.setdefault((s.count, s.foot), []).append(s.offset_ms)
    worst = max(groups.items(), key=lambda kv: abs(np.mean(kv[1])), default=None)
    pct = 100.0 * float(np.mean([abs(o) <= ON_TIME_MS for o in offs]))
    if worst is not None and abs(np.mean(worst[1])) > ON_TIME_MS:
        (b, f), v = worst
        m = float(np.mean(v))
        foot = FOOT_WORD[f]
        if b in (3, 7) and m < 0:
            txt = f"Rushing beat {b} on {foot} foot ({m:+.0f} ms) — hold your slow step."
        elif b in (3, 7):
            txt = f"Late on slow beat {b}, {foot} foot ({m:+.0f} ms) — close the step on time."
        elif m < 0:
            txt = f"Early on beat {b}, {foot} foot ({m:+.0f} ms) — wait for the beat before stepping."
        else:
            txt = f"Late on beat {b}, {foot} foot ({m:+.0f} ms) — shift your weight sooner."
        return {"text": txt, "kind": "warn"}
    if pct >= 80:
        return {"text": f"Great — last 2 bars on tempo ({pct:.0f}% on time). Keep it going!", "kind": "ok"}
    return {"text": f"Keep the pulse — {pct:.0f}% on time in the last 2 bars. Listen for the cowbell on every beat.",
            "kind": "info"}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_live_frame(frame: np.ndarray, kp2d: np.ndarray | None, state: dict[str, Any], mirror: bool = True,
                      footer: str = "") -> np.ndarray:
    """Draw the V2 overlay (faint upper body, cyan/magenta legs, ankle callouts, 8-count strip)."""
    import cv2

    import overlay_video as OV

    if mirror:
        frame = cv2.flip(frame, 1)
        if kp2d is not None:
            kp2d = kp2d.copy()
            kp2d[:, 0] = frame.shape[1] - 1 - kp2d[:, 0]
    # On a webcam the feet are usually near the bottom edge: draw the 8-count strip first and the
    # skeleton + ankle callouts on top of it, so the strip never hides the legs.
    OV.draw_status(frame, state, footer)
    OV.draw_beat_strip(frame, state)
    if kp2d is not None:
        OV.draw_skeleton(frame, kp2d, scale=frame.shape[0] / 720.0)
    OV.draw_callouts(frame, state, kp2d)
    return frame


def placeholder_jpeg(text: str = "Camera idle — press Start", w: int = 960, h: int = 540) -> bytes:
    import cv2

    import overlay_video as OV

    img = np.full((h, w, 3), (24, 18, 11), np.uint8)
    OV.put_label(img, text, w // 2, h // 2 - 14, 26, (200, 210, 225), anchor="ct")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Camera, pose model
# --------------------------------------------------------------------------- #
def choose_model(models_dir: str | Path, prefer: str | None = None) -> Path:
    """lite / full if present (faster on CPU), else heavy (downloaded if missing)."""
    from pose_2d import ensure_model

    d = Path(models_dir)
    if prefer in MODEL_URLS:
        return ensure_model(d / f"pose_landmarker_{prefer}.task", MODEL_URLS[prefer])
    for v in ("lite", "full"):
        p = d / f"pose_landmarker_{v}.task"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return ensure_model(d / "pose_landmarker_heavy.task", MODEL_URLS["heavy"])


class MediaPipeEstimator:
    """PoseLandmarker in VIDEO mode on the latest frame: 2D pixels + 3D world landmarks."""

    def __init__(self, model_path: str | Path):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                                            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self.last_ms = -1
        self.model = Path(model_path).name

    def detect(self, frame_bgr: np.ndarray, t_ms: int):
        import cv2
        import mediapipe as mp

        t_ms = max(int(t_ms), self.last_ms + 1)
        self.last_ms = t_ms
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, t_ms)
        if not res.pose_landmarks:
            return None, None
        kp = np.array([[p.x * w, p.y * h, p.visibility if p.visibility is not None else 1.0]
                       for p in res.pose_landmarks[0]], dtype=np.float32)
        world = None
        if res.pose_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], dtype=np.float64)
            vis = kp[:, 2]
            world[vis < 0.3] = np.nan
        return kp, world

    def close(self):
        self.lm.close()


class CameraSource:
    def __init__(self, index: int = 0, width: int | None = 1280, height: int | None = 720):
        import cv2

        self.cap = cv2.VideoCapture(int(index))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {index}")

    def read(self):
        return self.cap.read()

    def fps(self) -> float:
        import cv2

        return float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def release(self):
        self.cap.release()


def list_cameras(max_index: int = 4) -> list[dict[str, Any]]:
    """Probe camera indices 0..max_index-1 (fast fail; empty on machines without cameras)."""
    import cv2

    out = []
    prev = None
    try:  # probing absent indices makes OpenCV print errors: silence them during the probe
        prev = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    try:
        out = _probe(max_index)
    finally:
        if prev is not None:
            cv2.utils.logging.setLogLevel(prev)
    return out


def _probe(max_index: int) -> list[dict[str, Any]]:
    import cv2

    out = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.append({"index": i, "name": f"Camera {i}", "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        finally:
            cap.release()
    return out


class ClockedWriter:
    """Writes frames at a fixed fps following wall-clock time (duplicates/drops to keep real time)."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        from overlay_video import open_writer

        self.path, self.fps = path, fps
        self.w, self.fourcc = open_writer(path, fps, size, prefer_h264=False)
        self.t0: float | None = None
        self.n = 0
        self.size = size

    def write(self, frame: np.ndarray, t: float) -> None:
        import cv2

        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.t0 is None:
            self.t0 = t
        target = int((t - self.t0) * self.fps) + 1
        while self.n < target:
            self.w.write(frame)
            self.n += 1

    def close(self) -> float:
        self.w.release()
        return self.n / self.fps


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class LiveEngine:
    """Camera + pose + session threads, MJPEG output, recording and Save Session."""

    def __init__(self, config_path: str | None = None, output_root: str | Path = "outputs",
                 models_dir: str | Path = "models", data_dir: str | Path = "data", height_m: float = 1.70,
                 camera: int = 0, source_factory: Callable[[int], Any] | None = None,
                 estimator_factory: Callable[[], Any] | None = None, params: dict[str, Any] | None = None,
                 model: str | None = None):
        self.config_path = config_path
        self.output_root, self.models_dir, self.data_dir = Path(output_root), Path(models_dir), Path(data_dir)
        self.height_m, self.default_camera, self.model_pref = height_m, camera, model
        if config_path and Path(config_path).is_file():
            try:
                from main import load_config

                cfg = load_config(config_path)
                self.output_root = Path(cfg["output_root"])
                self.models_dir = Path(cfg["model_path"]).parent
                self.data_dir = Path(config_path).resolve().parent / "data"
                self._cfg_sessions = cfg["sessions"]
            except Exception as exc:
                log.warning("Live engine could not read %s: %s", config_path, exc)
        self.source_factory = source_factory or (lambda idx: CameraSource(idx))
        self.estimator_factory = estimator_factory
        self.p = {**DEFAULT_LIVE_PARAMS, **(params or {})}
        self.clock = LiveClock()
        self.session: LiveSession | None = None
        self.lock = threading.RLock()
        self.running = False
        self.error: str | None = None
        self.status_msg = "idle"
        self.opts: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []
        self._latest = None  # (seq, frame, ts)
        self._latest_cv = threading.Condition()
        self._jpeg = (0, placeholder_jpeg())
        self._jpeg_cv = threading.Condition()
        self.fps = 0.0
        self.latency_ms = None
        self.model_name = None
        self._rec_dir: Path | None = None
        self._writers: dict[str, ClockedWriter] = {}
        self._rec_t0: float | None = None
        self._rec_end: float | None = None
        self._song_cache: dict[str, Any] = {}
        self.music: dict[str, Any] = {"source": "metronome", "bpm": 180.0, "song": None}
        self.last_saved: dict[str, Any] | None = None

    # ---- catalogue ----
    camera_lister: Callable[[], list[dict[str, Any]]] = staticmethod(list_cameras)
    _cameras_cache: list[dict[str, Any]] | None = None

    def cameras(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Available webcams (probed once, cached). The running camera is always listed."""
        if self._cameras_cache is None or refresh:
            if self.running:  # probing would fight the open camera
                cams = list(self._cameras_cache or [])
            else:
                try:
                    cams = self.camera_lister()
                except Exception as exc:
                    log.warning("camera probe failed: %s", exc)
                    cams = []
            self._cameras_cache = cams
        cams = list(self._cameras_cache)
        cur = self.opts.get("camera", self.default_camera)
        if not any(c["index"] == cur for c in cams):
            cams.insert(0, {"index": cur, "name": f"Camera {cur}" + ("" if self.running else " (default)")})
        return cams

    def songs(self) -> list[dict[str, Any]]:
        seen, out = set(), []
        for s in getattr(self, "_cfg_sessions", []) or []:
            p = Path(s.get("song_mp3", ""))
            if p.is_file() and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()))
                out.append({"id": len(out), "name": f"{p.name} ({s['id']})", "path": str(p)})
        if self.data_dir.is_dir():
            for p in sorted(self.data_dir.rglob("*")):
                if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg") and str(p.resolve()) not in seen:
                    seen.add(str(p.resolve()))
                    out.append({"id": len(out), "name": str(p.relative_to(self.data_dir)), "path": str(p)})
        return out

    def song_path(self, song_id: int) -> Path | None:
        for s in self.songs():
            if s["id"] == int(song_id):
                return Path(s["path"])
        return None

    # ---- control ----
    def start(self, camera: int | None = None, source: str = "metronome", song_id: int | None = None,
              bpm: float = 180.0, mirror: bool = True, height_m: float | None = None,
              camera_latency_ms: float | None = None) -> dict[str, Any]:
        with self.lock:
            if self.running:
                self.stop()
            self.error = None
            cam = self.default_camera if camera is None else int(camera)
            h = float(height_m or self.height_m)
            lat = float(self.p["camera_latency_ms"] if camera_latency_ms is None else camera_latency_ms)
            self.opts = {"camera": cam, "mirror": bool(mirror), "height_m": h, "camera_latency_ms": lat,
                         "source": source, "song_id": song_id, "bpm": float(bpm)}
            self.clock = LiveClock()
            self.session = LiveSession(h, self.p, self.clock)
            self.music = {"source": source, "bpm": float(bpm), "song": None}
            if source == "song" and song_id is not None:
                p = self.song_path(song_id)
                if p is None:
                    raise ValueError(f"unknown song id {song_id}")
                self.music["song"] = str(p)
                self._analyze_song_async(p)
            else:
                self.session.set_grid(BeatGrid.metronome(bpm))
            try:
                src = self.source_factory(cam)
            except Exception as exc:
                self.error = f"Camera {cam} unavailable: {exc}"
                raise
            try:
                est = self.estimator_factory() if self.estimator_factory else self._default_estimator()
            except Exception as exc:
                src.release()
                self.error = f"Pose model unavailable: {exc}"
                raise
            self.model_name = getattr(est, "model", type(est).__name__)
            self._rec_dir = Path(tempfile.mkdtemp(prefix=".live_rec_", dir=self._ensure_out()))
            self._writers, self._rec_t0, self._rec_end = {}, None, None
            self.running = True
            self.status_msg = "running"
            self._threads = [threading.Thread(target=self._capture_loop, args=(src,), daemon=True, name="live-capture"),
                             threading.Thread(target=self._process_loop, args=(est,), daemon=True, name="live-pose")]
            for th in self._threads:
                th.start()
            return {"ok": True, "model": self.model_name, "opts": self.opts}

    def _ensure_out(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self.output_root

    def _default_estimator(self):
        return MediaPipeEstimator(choose_model(self.models_dir, self.model_pref))

    def _analyze_song_async(self, path: Path) -> None:
        key = f"{path.resolve()}:{path.stat().st_mtime}"

        def work():
            try:
                if key not in self._song_cache:
                    self.status_msg = "analysing song beats…"
                    import music_analyzer

                    sa = music_analyzer.analyze_song(str(path))
                    self._song_cache[key] = (sa.beat_times_sec, sa.bpm)
                bt, bpm = self._song_cache[key]
                if self.session is not None:
                    self.session.set_grid(BeatGrid.from_song(bt, bpm))
                    self.music["bpm"] = float(60.0 / self.session.grid.period)
                self.status_msg = "running"
            except Exception as exc:
                self.error = f"Song analysis failed: {exc}"

        threading.Thread(target=work, daemon=True, name="live-song").start()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            was = self.running
            self.running = False
            with self._latest_cv:
                self._latest_cv.notify_all()
            for th in self._threads:
                th.join(timeout=3.0)
            self._threads = []
            self._rec_end = self._rec_end or time.monotonic()
            for w in self._writers.values():
                try:
                    w.close()
                except Exception:
                    pass
            self.status_msg = "stopped" if was else self.status_msg
            with self._jpeg_cv:
                self._jpeg = (self._jpeg[0] + 1, placeholder_jpeg("Stopped — press Start or Save Session"))
                self._jpeg_cv.notify_all()
            return {"ok": True, "was_running": was}

    def clock_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        audio_time = payload.get("audio_time")
        playing = bool(payload.get("playing"))
        source = payload.get("source")
        bpm = payload.get("bpm")
        self.clock.update(None if audio_time is None else float(audio_time), playing, source=source,
                          bpm=None if bpm is None else float(bpm))
        if self.session is not None and source == "metronome" and bpm:
            g = self.session.grid
            if g is None or g.times is not None or abs(g.bpm - float(bpm)) > 1e-6:
                self.session.set_grid(BeatGrid.metronome(float(bpm), 0 if g is None else g.count_shift))
                self.music["bpm"] = float(bpm)
        return {"ok": True, "clock": self.clock.status()}

    def count_shift(self, delta: int = 1) -> dict[str, Any]:
        if self.session is not None:
            self.session.shift_count(delta)
        return {"ok": True, "count_shift": None if not self.session or not self.session.grid else self.session.grid.count_shift}

    # ---- threads ----
    def _capture_loop(self, src) -> None:
        seq = 0
        lat = self.opts.get("camera_latency_ms", 0.0) / 1000.0
        try:
            while self.running:
                ok, frame = src.read()
                now = time.monotonic()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                seq += 1
                ts = now - lat
                if self._rec_t0 is None:
                    self._rec_t0 = ts
                if "raw" not in self._writers and self._rec_dir is not None:
                    self._writers["raw"] = ClockedWriter(self._rec_dir / "raw.mp4", self.p["record_fps"],
                                                         (frame.shape[1], frame.shape[0]))
                if "raw" in self._writers:
                    self._writers["raw"].write(cv2_flip(frame) if self.opts.get("mirror") else frame, ts)
                with self._latest_cv:
                    self._latest = (seq, frame, ts)
                    self._latest_cv.notify_all()
        except Exception as exc:
            self.error = f"Camera error: {exc}"
            log.exception("capture loop")
        finally:
            try:
                src.release()
            except Exception:
                pass

    def _process_loop(self, est) -> None:
        import cv2

        last_seq = 0
        ema_dt = None
        last_t = None
        try:
            while self.running:
                with self._latest_cv:
                    while self.running and (self._latest is None or self._latest[0] == last_seq):
                        self._latest_cv.wait(timeout=0.2)
                    if not self.running:
                        break
                    seq, frame, ts = self._latest  # always the newest frame: never lags behind real time
                last_seq = seq
                kp, world = est.detect(frame, int(ts * 1000))
                sess = self.session
                sess.on_frame(ts, kp, world)
                state = sess.overlay_state(ts)
                footer = f"{self.fps:4.1f} fps  ·  {self.model_name}"
                out = render_live_frame(frame.copy(), kp, state, self.opts.get("mirror", True), footer)
                if "overlay" not in self._writers and self._rec_dir is not None:
                    self._writers["overlay"] = ClockedWriter(self._rec_dir / "overlay.mp4", self.p["record_fps"],
                                                             (out.shape[1], out.shape[0]))
                if "overlay" in self._writers:
                    self._writers["overlay"].write(out, ts)
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.p["jpeg_quality"])])
                now = time.monotonic()
                if last_t is not None:
                    dt = now - last_t
                    ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
                    self.fps = 1.0 / ema_dt if ema_dt else 0.0
                last_t = now
                self.latency_ms = (now - ts) * 1000.0 + self.opts.get("camera_latency_ms", 0.0)
                if ok:
                    with self._jpeg_cv:
                        self._jpeg = (self._jpeg[0] + 1, buf.tobytes())
                        self._jpeg_cv.notify_all()
        except Exception as exc:
            self.error = f"Processing error: {exc}"
            log.exception("process loop")
            self.running = False
        finally:
            try:
                est.close()
            except Exception:
                pass

    # ---- outputs ----
    def wait_jpeg(self, after_seq: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self._jpeg_cv:
            if self._jpeg[0] == after_seq:
                self._jpeg_cv.wait(timeout=timeout)
            return self._jpeg

    def state(self) -> dict[str, Any]:
        base = {"running": self.running, "status": self.status_msg, "error": self.error, "fps": round(self.fps, 1),
                "latency_ms": None if self.latency_ms is None else round(self.latency_ms),
                "model": self.model_name, "opts": self.opts, "music": self.music, "last_saved": self.last_saved}
        if self.session is not None:
            base.update(self.session.state_dict())
        else:
            base.update({"stats": None, "coach": None, "steps": [], "floor": None, "clock": self.clock.status()})
        return base

    def save(self) -> dict[str, Any]:
        """Stop and export ``outputs/live_<timestamp>/`` + a ``sessions.json`` entry for Recorded Sessions."""
        if self.running:
            self.stop()
        if self.session is None or self._rec_dir is None:
            raise ValueError("nothing recorded yet — press Start first")
        sid = "live_" + time.strftime("%Y%m%d_%H%M%S")
        out = self._ensure_out() / sid
        out.mkdir(parents=True, exist_ok=False)
        fps = float(self.p["record_fps"])
        dur = max((w.n / fps for w in self._writers.values()), default=0.0)
        from overlay_video import finalize_with_ffmpeg, find_ffmpeg

        ff = find_ffmpeg()
        for name, dst in (("raw", "cam1.mp4"), ("overlay", "overlay_cam1.mp4")):
            src = self._rec_dir / f"{name}.mp4"
            if not src.is_file():
                continue
            done = False
            if ff:
                try:
                    finalize_with_ffmpeg(src, None, out / dst, ff)
                    done = True
                except Exception as exc:
                    log.warning("ffmpeg re-encode failed for %s: %s", dst, exc)
            if not done:
                shutil.copy2(src, out / dst)
        sess = self.session
        rec_t0 = self._rec_t0 if self._rec_t0 is not None else (sess.t0 or 0.0)
        # session times are relative to the first processed frame; recording starts at the first captured frame
        shift = (sess.t0 or rec_t0) - rec_t0
        for s in sess.steps:
            s.t_rel = s.t - rec_t0
        sess.beats_seen = {k: (v[0] + shift, v[1]) for k, v in sess.beats_seen.items()}
        sess.track = [(round(t + shift, 3), a, b) for t, a, b in sess.track]
        music_src: dict[str, Any] = {"source": self.music.get("source")}
        if self.music.get("source") == "song" and self.music.get("song"):
            song = Path(self.music["song"])
            audio = out / ("song" + song.suffix.lower())
            shutil.copy2(song, audio)
            s0 = None if sess.song0_rel is None else sess.song0_rel + shift
            music_src["mp3_offset_sec"] = round(float(s0), 4) if s0 is not None else 0.0
        else:
            audio = out / "metronome.wav"
            write_click_track(audio, [(t, c) for t, c in sess.beats_seen.values()], max(dur, 1.0))
            music_src["mp3_offset_sec"] = 0.0
        report = sess.to_report(sid, fps, dur, music_src)
        (out / "epochs_and_steps.json").write_text(json.dumps(report, indent=2, default=_json_default))
        (out / "sync.json").write_text(json.dumps({"frame_offset": 0, "offset_sec": 0.0, "method": "single_camera",
                                                   "confidence": 1.0, "warnings": []}, indent=2))
        from main import CSV_COLUMNS

        with open(out / "steps.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for st in report["steps"]:
                row = {k: st.get(k) for k in CSV_COLUMNS if k != "subject_id"}
                row["subject_id"] = sid
                w.writerow(row)
        shutil.rmtree(self._rec_dir, ignore_errors=True)
        self._rec_dir = None
        entry = None
        if self.config_path:
            entry = append_live_session(self.config_path, sid, out / "cam1.mp4", audio, sess.height_m,
                                        music_src["mp3_offset_sec"])
        self.last_saved = {"ok": True, "subject_id": sid, "dir": str(out), "duration_s": round(dur, 2),
                           "n_steps": len(sess.steps), "sessions_json": bool(entry),
                           "url": f"/?subject={sid}" if entry else None}
        return self.last_saved


def cv2_flip(frame):
    import cv2

    return cv2.flip(frame, 1)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_click_track(path: Path, beats: list[tuple[float, int]], duration_s: float, sr: int = 44100) -> Path:
    """Cowbell-like click track (accent on 1) aligned with the recording, for replaying metronome sessions."""
    import soundfile as sf

    n = int(math.ceil(duration_s * sr))
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.06 * sr)) / sr
    env = np.exp(-t * 60)
    for bt, count in beats:
        k = int(round(bt * sr))
        if k < 0 or k >= n:
            continue
        f = 1600.0 if count == 1 else 1000.0
        click = (0.5 if count == 1 else 0.32) * np.sign(np.sin(2 * np.pi * f * t)) * env
        m = min(len(click), n - k)
        y[k:k + m] += click[:m].astype(np.float32)
    sf.write(str(path), np.clip(y, -1, 1), sr)
    return path


def append_live_session(config_path: str | Path, sid: str, cam1: Path, audio: Path, height_m: float,
                        mp3_offset: float) -> dict[str, Any]:
    """Add the saved live session to sessions.json (marked "live": main.py skips it in batch runs)."""
    cfg_path = Path(config_path)
    cfg = json.loads(cfg_path.read_text())
    base = cfg_path.resolve().parent

    def rel(p: Path) -> str:
        try:
            return os.path.relpath(Path(p).resolve(), base)
        except ValueError:
            return str(Path(p).resolve())

    entry = {"id": sid, "live": True, "cam1": rel(cam1), "cam2": rel(cam1), "song_mp3": rel(audio),
             "height_meters": float(height_m), "overrides": {"frame_offset": 0, "mp3_offset_sec": float(mp3_offset)},
             "params": {}}
    cfg.setdefault("sessions", []).append(entry)
    fd, tmp = tempfile.mkstemp(dir=cfg_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, cfg_path)
    return entry
