"""Render the skeleton, foot-specific timing callouts and an 8-count strip onto cam1.mp4.

    python overlay_video.py --config sessions.json --subject subject_01 [--max-seconds 20]

Output: ``outputs/<subject_id>/overlay_cam1.mp4``.

Visual hierarchy:
* Upper body: thin, semi-transparent (alpha 0.35) muted slate-cyan lines.
* Legs, ankles, heels and toes: crisp electric cyan (#00E5FF, left) and magenta-coral
  (#FF4081, right) with glowing joint rings.
* Each footfall: for ~0.6 s, a ring around that foot plus a floating tag beside it.
  The tag reads e.g. "LEFT FOOT • BEAT 3" over a badge: ON TIME (+12ms) in green,
  EARLY / LATE (45-90 ms) in amber, EARLY / LATE / OFF-BEAT (>90 ms) in red.
* Bottom strip: an 8-count ``1 2 3 [4] 5 6 7 [8]`` that lights the active beat with a
  progress fill and flashes red on a beat where a timing error happened.

The video is written with OpenCV, then re-encoded to browser-friendly H.264 with the
original audio when ffmpeg is available (on PATH or bundled with imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import landmarks as L

log = logging.getLogger(__name__)

# MediaPipe Pose landmark connections (same as mediapipe.solutions.pose.POSE_CONNECTIONS)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
LOWER_BODY = set(range(23, 33))
LEFT_LEG = {23, 25, 27, 29, 31}
RIGHT_LEG = {24, 26, 28, 30, 32}
LEG_JOINTS = {25, 26, 27, 28, 29, 30, 31, 32}
FOOT = {"L": [L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX], "R": [L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX]}


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# Palette (BGR)
LEFT_COL = hex_bgr("#00E5FF")  # electric cyan
RIGHT_COL = hex_bgr("#FF4081")  # magenta-coral
HIP_COL = hex_bgr("#E8EEF5")
UPPER_COL = hex_bgr("#78AABE")  # muted slate-cyan
GREEN = hex_bgr("#22C55E")
AMBER = hex_bgr("#F59E0B")
RED = hex_bgr("#EF4444")
GREY = hex_bgr("#64748B")
BEAT_ON = hex_bgr("#FFC400")
DARK = hex_bgr("#0B1220")
WHITE = (255, 255, 255)
FOOT_COL = {"L": LEFT_COL, "R": RIGHT_COL}

ON_TIME_MS = 45.0
SLIGHT_MS = 90.0
CALLOUT_S = 0.6
UPPER_ALPHA = 0.35


# --------------------------------------------------------------------------- #
# Timing classification (shared with dashboard.py)
# --------------------------------------------------------------------------- #
def classify_offset(offset_ms: float | None) -> tuple[str, str]:
    """Return (category, label) for a step's beat offset.

    Categories are on_time, early, late, off_early, off_late, or none.
    """
    if offset_ms is None or not np.isfinite(offset_ms):
        return "none", "NO BEAT REF"
    a = abs(offset_ms)
    if a <= ON_TIME_MS:
        return "on_time", "ON TIME"
    if a <= SLIGHT_MS:
        return ("early", "EARLY") if offset_ms < 0 else ("late", "LATE")
    return ("off_early", "EARLY / OFF-BEAT") if offset_ms < 0 else ("off_late", "LATE / OFF-BEAT")


CATEGORY_BGR = {"on_time": GREEN, "early": AMBER, "late": AMBER, "off_early": RED, "off_late": RED, "none": GREY}


def is_error(offset_ms: float | None) -> bool:
    return offset_ms is not None and np.isfinite(offset_ms) and abs(offset_ms) > ON_TIME_MS


def offset_badge(offset_ms: float | None) -> str:
    cat, label = classify_offset(offset_ms)
    return label if cat == "none" else f"{label} ({offset_ms:+.0f}ms)"


def analysis_beat_grid(report: dict[str, Any]) -> np.ndarray:
    """The beat grid the steps were scored against (cam1 timeline).

    Uses ``beat_grid_sec`` if main.py exported it. For older reports, it rebuilds the
    tempo-octave correction from salsa_rhythm: if the song's beat period is about 2x
    the analysis beat period (half-time beat tracking), midpoints are inserted. The
    raw librosa grid must not be used in that case, or the 8-count runs at half speed.
    """
    g = report.get("beat_grid_sec")
    if g:
        return np.asarray(g, dtype=np.float64)
    music = report.get("music", {}) or {}
    grid = np.asarray(music.get("aligned_beat_times_sec") or [], dtype=np.float64)
    period = (report.get("summary", {}) or {}).get("beat_period_sec")
    if len(grid) >= 2 and period:
        ratio = float(np.median(np.diff(grid))) / float(period)
        if 1.7 < ratio < 2.3:
            grid = np.sort(np.concatenate([grid, 0.5 * (grid[:-1] + grid[1:])]))
    return grid


def _count_phase(grid: np.ndarray, steps: list[dict]) -> int | None:
    """Phase p such that beat-grid index k has 8-count ((k - p) mod 8) + 1, by majority vote."""
    votes = []
    for s in steps:
        bc, nb = s.get("beat_count"), s.get("nearest_music_beat_sec")
        if bc is None or nb is None or len(grid) == 0:
            continue
        k = int(np.argmin(np.abs(grid - nb)))
        votes.append((k - (int(bc) - 1)) % 8)
    if not votes:
        return None
    return int(np.bincount(votes, minlength=8).argmax())


@dataclass
class RhythmTimeline:
    """Answers 'what is the rhythm state at cam1 time t'."""

    epochs: list[tuple[float, float]]
    steps: list[dict]  # sorted by timestamp_sec, TAPs included
    beat_grid: np.ndarray
    beat_period: float
    phase_global: int | None
    phase_by_epoch: dict[int, int | None] = field(default_factory=dict)
    bpm: float = 0.0

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> "RhythmTimeline":
        music = report.get("music", {}) or {}
        grid = analysis_beat_grid(report)
        steps = sorted(report.get("steps", []), key=lambda s: s["timestamp_sec"])
        period = (report.get("summary", {}) or {}).get("beat_period_sec") or music.get("beat_period_sec") or 0.5
        epochs = [(float(e["start_sec"]), float(e["end_sec"])) for e in report.get("epochs", [])]
        by_epoch = {}
        for e in report.get("epochs", []):
            by_epoch[e["epoch_id"]] = _count_phase(grid, [s for s in steps if s.get("epoch_id") == e["epoch_id"]])
        return cls(epochs, steps, grid, float(period), _count_phase(grid, steps), by_epoch,
                   float(music.get("bpm") or 0.0))

    def epoch_at(self, t: float) -> int | None:
        for i, (a, b) in enumerate(self.epochs):
            if a <= t <= b:
                return i
        return None

    def _beat_index(self, t: float) -> int | None:
        g = self.beat_grid
        if len(g) == 0:
            return None
        k = int(np.searchsorted(g, t, side="right")) - 1
        if k < 0 or t - g[k] > 1.5 * self.beat_period:
            return None
        return k

    def beat_count_at(self, t: float) -> int | None:
        k = self._beat_index(t)
        if k is None:
            return None
        ep = self.epoch_at(t)
        phase = self.phase_by_epoch.get(ep) if ep is not None else None
        if phase is None:
            phase = self.phase_global
        if phase is None:
            return None
        return (k - phase) % 8 + 1

    def beat_progress_at(self, t: float) -> float:
        """Fraction (0-1) of the current beat that has elapsed."""
        k = self._beat_index(t)
        if k is None:
            return 0.0
        g = self.beat_grid
        nxt = g[k + 1] if k + 1 < len(g) else g[k] + self.beat_period
        return float(np.clip((t - g[k]) / max(1e-6, nxt - g[k]), 0.0, 1.0))

    def beats_with_counts(self) -> list[dict[str, Any]]:
        return [{"t": float(bt), "count": self.beat_count_at(float(bt) + 1e-6)} for bt in self.beat_grid]

    def recent_steps(self, t: float, window_s: float = CALLOUT_S) -> list[tuple[dict, float]]:
        """Steps whose footfall happened in (t - window, t], with their age in seconds."""
        out = []
        for s in self.steps:
            age = t - s["timestamp_sec"]
            if age < 0:
                break
            if age <= window_s:
                out.append((s, age))
        return out

    def last_step(self, t: float, max_age_s: float = 2.5) -> dict | None:
        last = None
        for s in self.steps:
            if s["timestamp_sec"] <= t:
                last = s
            else:
                break
        if last is None or t - last["timestamp_sec"] > max_age_s:
            return None
        return last

    def on_time_so_far(self, t: float) -> tuple[int, int]:
        n = ok = 0
        for s in self.steps:
            if s["timestamp_sec"] > t:
                break
            off = s.get("beat_offset_ms")
            if s.get("timing_type") == "TAP" or off is None:
                continue
            n += 1
            ok += abs(off) <= ON_TIME_MS
        return ok, n

    def state_at(self, t: float) -> dict[str, Any]:
        ep = self.epoch_at(t)
        step = self.last_step(t)
        count = self.beat_count_at(t)
        if count is None and step is not None and step.get("beat_count"):
            count = int(step["beat_count"])
        callouts = []
        flashes: dict[int, float] = {}
        for s, age in self.recent_steps(t):
            if s.get("timing_type") == "TAP":
                continue
            cat, label = classify_offset(s.get("beat_offset_ms"))
            callouts.append({
                "foot": s.get("foot"),
                "foot_label": "LEFT FOOT" if s.get("foot") == "L" else "RIGHT FOOT",
                "beat": s.get("beat_count"),
                "offset_ms": s.get("beat_offset_ms"),
                "category": cat,
                "badge": offset_badge(s.get("beat_offset_ms")),
                "is_error": is_error(s.get("beat_offset_ms")),
                "age": age,
            })
            if is_error(s.get("beat_offset_ms")) and s.get("beat_count"):
                b = int(s["beat_count"])
                flashes[b] = min(age, flashes.get(b, age))
        ok, n = self.on_time_so_far(t)
        last_cat, last_label = classify_offset(step.get("beat_offset_ms") if step else None)
        return {
            "t": t,
            "active": ep is not None,
            "status": "DANCING" if ep is not None else "IDLE",
            "count": count,
            "beat_progress": self.beat_progress_at(t),
            "callouts": callouts,
            "error_flash": flashes,  # beat count -> age of the error (s)
            "last_step": step,
            "last_category": last_cat if step else None,
            "last_label": last_label if step else None,
            "on_time": (ok, n),
            "bpm": self.bpm,
        }


# --------------------------------------------------------------------------- #
# Text rendering (anti-aliased TrueType via Pillow, cached patches; Hershey fallback)
# --------------------------------------------------------------------------- #
def _font_path() -> str | None:
    try:
        import matplotlib

        p = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return str(p) if p.is_file() else None
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _text_patch(text: str, px: int, fg: tuple, bg: tuple | None, pad: int, radius: int):
    """BGR patch + alpha mask for a (optionally pill-shaped) text label."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(_font_path(), px)
    x0, y0, x1, y1 = font.getbbox(text)
    w, h = x1 - x0 + 2 * pad, (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 2 * pad
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg is not None:
        b, g, r = bg[:3]
        a = bg[3] if len(bg) == 4 else 255
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=(r, g, b, a))
    d.text((pad - x0, pad - font.getbbox("Ag")[1]), text, font=font, fill=(fg[2], fg[1], fg[0], 255))
    arr = np.asarray(img).astype(np.float32)
    return arr[..., [2, 1, 0]], arr[..., 3:4] / 255.0


def put_label(frame: np.ndarray, text: str, x: int, y: int, px: int, fg=WHITE, bg=None, pad: int = 0,
              radius: int = 0, alpha: float = 1.0, anchor: str = "lt") -> tuple[int, int, int, int]:
    """Draw a label; anchor is 'lt' | 'rt' | 'ct' | 'lb' | 'cb'. Returns its (x, y, w, h)."""
    px = max(8, int(px))
    if _font_path() is not None:
        try:
            patch, a = _text_patch(text, px, tuple(fg), tuple(bg) if bg is not None else None, int(pad), int(radius))
        except Exception:
            patch = None
    else:
        patch = None
    if patch is None:  # Hershey fallback (ASCII only)
        import cv2

        sc = px / 30.0
        t = text.replace("•", "-")
        (tw, th), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        w, h = tw + 2 * pad, th + base + 2 * pad
        x, y = _anchor(x, y, w, h, anchor)
        if bg is not None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), tuple(int(c) for c in bg[:3]), -1)
        cv2.putText(frame, t, (x + pad, y + pad + th), cv2.FONT_HERSHEY_SIMPLEX, sc, tuple(fg), 2, cv2.LINE_AA)
        return x, y, w, h
    h, w = patch.shape[:2]
    x, y = _anchor(x, y, w, h, anchor)
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if xb <= xa or yb <= ya:
        return x, y, w, h
    p = patch[ya - y : yb - y, xa - x : xb - x]
    m = a[ya - y : yb - y, xa - x : xb - x] * float(alpha)
    roi = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (roi * (1 - m) + p * m).astype(np.uint8)
    return x, y, w, h


def _anchor(x, y, w, h, anchor):
    if anchor[0] == "r":
        x -= w
    elif anchor[0] == "c":
        x -= w // 2
    if anchor[1] == "b":
        y -= h
    return int(x), int(y)


def _blend_rect(frame, x0, y0, x1, y1, colour, alpha):
    import cv2

    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    col = np.empty_like(roi)
    col[:] = colour
    cv2.addWeighted(col, alpha, roi, 1 - alpha, 0, dst=roi)


# --------------------------------------------------------------------------- #
# Skeleton
# --------------------------------------------------------------------------- #
def _leg_colour(a: int, b: int) -> tuple[int, int, int]:
    if {a, b} == {23, 24}:
        return HIP_COL
    return LEFT_COL if b in LEFT_LEG else RIGHT_COL


def _bbox(pts: np.ndarray, ok: np.ndarray, pad: int, W: int, H: int):
    if not ok.any():
        return None
    p = pts[ok]
    x0, y0 = np.maximum(p.min(axis=0) - pad, 0).astype(int)
    x1, y1 = np.minimum(p.max(axis=0) + pad, [W, H]).astype(int)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, vis_min: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """Draw the skeleton in place: faint upper body, crisp glowing legs/feet."""
    import cv2

    H, W = frame.shape[:2]
    ok = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 2] >= vis_min)
    pts = np.nan_to_num(kp[:, :2]).astype(np.int32)
    s = scale
    box = _bbox(pts.astype(float), ok, int(60 * s), W, H)
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    off = np.array([x0, y0])
    roi = frame[y0:y1, x0:x1]
    P = lambda j: tuple(int(v) for v in pts[j] - off)  # noqa: E731

    # 1) upper body: thin, alpha 0.35
    upper = roi.copy()
    for a, b in POSE_CONNECTIONS:
        if ok[a] and ok[b] and not (a in LOWER_BODY and b in LOWER_BODY):
            cv2.line(upper, P(a), P(b), UPPER_COL, max(1, int(round(2 * s))), cv2.LINE_AA)
    for j in range(0, 23):
        if ok[j] and j >= 11:
            cv2.circle(upper, P(j), max(2, int(3 * s)), UPPER_COL, -1, cv2.LINE_AA)
    cv2.addWeighted(upper, UPPER_ALPHA, roi, 1 - UPPER_ALPHA, 0, dst=roi)

    # 2) leg glow (wide, soft)
    glow = roi.copy()
    for a, b in POSE_CONNECTIONS:
        if ok[a] and ok[b] and a in LOWER_BODY and b in LOWER_BODY and {a, b} != {23, 24}:
            cv2.line(glow, P(a), P(b), _leg_colour(a, b), max(6, int(16 * s)), cv2.LINE_AA)
    for j in LEG_JOINTS:
        if ok[j]:
            cv2.circle(glow, P(j), max(6, int(15 * s)), LEFT_COL if j in LEFT_LEG else RIGHT_COL, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), max(1.0, 4 * s))
    cv2.addWeighted(glow, 0.35, roi, 0.65, 0, dst=roi)

    # 3) crisp legs with dark outline
    for a, b in POSE_CONNECTIONS:
        if ok[a] and ok[b] and a in LOWER_BODY and b in LOWER_BODY:
            th = max(2, int(round(5 * s)))
            cv2.line(roi, P(a), P(b), (15, 15, 15), th + max(2, int(3 * s)), cv2.LINE_AA)
            cv2.line(roi, P(a), P(b), _leg_colour(a, b), th, cv2.LINE_AA)
    # 4) joint rings
    for j in sorted(LOWER_BODY):
        if not ok[j]:
            continue
        col = HIP_COL if j in (23, 24) else (LEFT_COL if j in LEFT_LEG else RIGHT_COL)
        r = max(4, int((8 if j in (27, 28, 29, 30, 31, 32) else 7) * s))
        cv2.circle(roi, P(j), r + max(1, int(2 * s)), (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(roi, P(j), r, col, max(2, int(round(2.5 * s))), cv2.LINE_AA)
        cv2.circle(roi, P(j), max(1, r // 3), col, -1, cv2.LINE_AA)
    return frame


def foot_center(kp: np.ndarray, foot: str, vis_min: float = 0.3) -> np.ndarray | None:
    idx = FOOT[foot]
    p = kp[idx]
    m = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1]) & (p[:, 2] >= vis_min)
    if not m.any():
        return None
    return p[m, :2].mean(axis=0)


# --------------------------------------------------------------------------- #
# HUD pieces
# --------------------------------------------------------------------------- #
def draw_callouts(frame: np.ndarray, state: dict[str, Any], kp: np.ndarray | None) -> list[dict]:
    """Ring + floating tag beside each foot that stepped in the last ~0.6 s. Returns drawn boxes."""
    import cv2

    if kp is None:
        return []
    H, W = frame.shape[:2]
    s = H / 720.0
    hips = kp[[L.LEFT_HIP, L.RIGHT_HIP]]
    hip_ok = np.isfinite(hips[:, 0]).all()
    body_x = float(np.mean(hips[:, 0])) if hip_ok else W / 2
    drawn = []
    for c in state.get("callouts", []):
        ctr = foot_center(kp, c["foot"])
        if ctr is None:
            continue
        age = c["age"]
        fade = 1.0 if age < CALLOUT_S * 0.7 else max(0.0, (CALLOUT_S - age) / (CALLOUT_S * 0.3))
        if fade <= 0:
            continue
        col = CATEGORY_BGR[c["category"]]
        err = c["is_error"]
        cx, cy = int(ctr[0]), int(ctr[1])
        pulse = 1.0 + 0.35 * max(0.0, 1.0 - age / 0.25)
        r = int((34 if err else 26) * s * pulse)
        layer = frame.copy()
        cv2.circle(layer, (cx, cy), r, col, max(2, int((5 if err else 3) * s)), cv2.LINE_AA)
        if err:
            cv2.circle(layer, (cx, cy), r + int(7 * s), col, max(1, int(2 * s)), cv2.LINE_AA)
        # tag placement: outside of the body, above the foot
        side = -1 if ctr[0] < body_x else 1
        tx = cx + side * (r + int(18 * s))
        ty = cy - int(74 * s)
        lead_end = (tx, ty + int(52 * s))
        cv2.line(layer, (cx + side * int(r * 0.7), cy - int(r * 0.7)), lead_end, col, max(1, int(2 * s)), cv2.LINE_AA)
        cv2.addWeighted(layer, fade, frame, 1 - fade, 0, dst=frame)
        anchor = "rt" if side < 0 else "lt"
        beat = f" • BEAT {c['beat']}" if c.get("beat") else ""
        hx, hy, hw, hh = put_label(frame, f"{c['foot_label']}{beat}", tx, ty, int(17 * s), WHITE,
                                   (*DARK, 215), pad=int(7 * s), radius=int(8 * s), alpha=fade, anchor=anchor)
        fg = DARK if c["category"] in ("on_time", "early", "late") else WHITE
        bx, by, bw, bh = put_label(frame, c["badge"], tx, hy + hh + int(4 * s), int(17 * s), fg, (*col, 245),
                                   pad=int(7 * s), radius=int(8 * s), alpha=fade, anchor=anchor)
        drawn.append({"foot": c["foot"], "center": (cx, cy), "ring_r": r, "tag": (hx, hy, hw, hh + bh + int(4 * s)),
                      "category": c["category"]})
    return drawn


def draw_beat_strip(frame: np.ndarray, state: dict[str, Any]) -> tuple[int, int, int, int]:
    """Bottom 8-count strip. Returns the strip box (x, y, w, h)."""
    H, W = frame.shape[:2]
    s = H / 720.0
    margin = int(16 * s)
    sh = int(54 * s)
    x0, y0 = margin, H - margin - sh
    sw = W - 2 * margin
    _blend_rect(frame, x0, y0, x0 + sw, y0 + sh, DARK, 0.62)
    put_label(frame, "COUNT", x0 + int(12 * s), y0 + sh // 2 - int(8 * s), int(13 * s), (170, 180, 190))
    cells_x0 = x0 + int(84 * s)
    gap = int(6 * s)
    cw = (x0 + sw - int(10 * s) - cells_x0 - 7 * gap) // 8
    cur = state.get("count")
    active = state.get("active")
    flashes = state.get("error_flash", {})
    for i in range(8):
        n = i + 1
        hold = n in (4, 8)
        cx0 = cells_x0 + i * (cw + gap)
        cy0, cy1 = y0 + int(8 * s), y0 + sh - int(8 * s)
        base_alpha = 0.55 if active else 0.35
        _blend_rect(frame, cx0, cy0, cx0 + cw, cy1, hex_bgr("#1F2937"), base_alpha)
        flash = 0.92 * max(0.0, 1.0 - flashes[n] / CALLOUT_S) if n in flashes else 0.0
        if cur == n:
            _blend_rect(frame, cx0, cy0, cx0 + cw, cy1, BEAT_ON if not hold else hex_bgr("#94A3B8"),
                        0.92 if active else 0.45)
            prog = state.get("beat_progress", 0.0)
            _blend_rect(frame, cx0, cy1 - int(4 * s), cx0 + int(cw * prog), cy1, DARK, 0.8)
            fg = DARK
        else:
            fg = (110, 118, 128) if hold else ((240, 240, 240) if active else (150, 155, 160))
        if flash > 0:  # a timing error on this beat: red wins over the active highlight
            _blend_rect(frame, cx0, cy0, cx0 + cw, cy1, RED, flash)
            fg = WHITE
        label = f"[{n}]" if hold else str(n)
        put_label(frame, label, cx0 + cw // 2, (cy0 + cy1) // 2 - int(11 * s), int(20 * s), fg, anchor="ct")
    return x0, y0, sw, sh


def draw_status(frame: np.ndarray, state: dict[str, Any], footer: str = "") -> None:
    H, W = frame.shape[:2]
    s = H / 720.0
    x, y = int(16 * s), int(16 * s)
    act = state["active"]
    _, _, w1, h1 = put_label(frame, ("● " if act else "") + state["status"], x, y, int(17 * s),
                             DARK if act else WHITE, (*(GREEN if act else GREY), 240), pad=int(8 * s), radius=int(9 * s))
    xx = x + w1 + int(8 * s)
    if state.get("bpm"):
        _, _, w2, _ = put_label(frame, f"{state['bpm']:.0f} BPM", xx, y, int(17 * s), WHITE, (*DARK, 200),
                                pad=int(8 * s), radius=int(9 * s))
        xx += w2 + int(8 * s)
    ok, n = state.get("on_time", (0, 0))
    if n:
        pct = 100.0 * ok / n
        col = GREEN if pct >= 70 else AMBER if pct >= 45 else RED
        put_label(frame, f"ON TIME {pct:.0f}%  ({ok}/{n})", xx, y, int(17 * s), col, (*DARK, 210),
                  pad=int(8 * s), radius=int(9 * s))
    if footer:
        put_label(frame, footer, x, y + h1 + int(6 * s), int(12 * s), (200, 205, 210), (*DARK, 150),
                  pad=int(5 * s), radius=int(6 * s))


def draw_hud(frame: np.ndarray, state: dict[str, Any], kp: np.ndarray | None = None, footer: str = "") -> np.ndarray:
    draw_status(frame, state, footer)
    draw_callouts(frame, state, kp)
    draw_beat_strip(frame, state)
    return frame


def render_frame(frame: np.ndarray, kp: np.ndarray | None, timeline: RhythmTimeline, t: float,
                 footer: str = "") -> np.ndarray:
    scale = frame.shape[0] / 720.0
    if kp is not None:
        draw_skeleton(frame, kp, scale=scale)
    return draw_hud(frame, timeline.state_at(t), kp, footer)


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def open_writer(path: Path, fps: float, size: tuple[int, int], prefer_h264: bool = True):
    """OpenCV writer trying avc1 (H.264) first, then mp4v. Returns (writer, fourcc_name).

    With ``prefer_h264=False`` it goes straight to mp4v. That is used when ffmpeg
    re-encodes afterwards anyway, and it avoids OpenCV's noisy failed-encoder logs.
    """
    import cv2

    for name in (("avc1", "mp4v") if prefer_h264 else ("mp4v",)):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*name), fps, size)
        if w.isOpened():
            return w, name
        w.release()
    raise RuntimeError("Could not open a VideoWriter with avc1 or mp4v")


def finalize_with_ffmpeg(raw: Path, source_video: Path | None, out: Path, ffmpeg: str) -> dict[str, Any]:
    """Re-encode to H.264/yuv420p (+faststart) and mux the original audio if present."""
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(raw)]
    if source_video is not None and Path(source_video).is_file():
        cmd += ["-i", str(source_video), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
            "-movflags", "+faststart", "-shortest", str(out)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore")[-400:])
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(out)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE).stderr.decode(errors="ignore")
    return {"codec": "h264", "audio": "Audio:" in probe}


def render_overlay(
    video_path: str | Path,
    pose_npz: str | Path,
    report: dict[str, Any] | str | Path,
    out_path: str | Path,
    sync_info: dict[str, Any] | str | Path | None = None,
    max_seconds: float | None = None,
    use_ffmpeg: bool = True,
) -> dict[str, Any]:
    """Render the overlay video. Returns info: path, codec, audio, frames, warnings."""
    import cv2

    from pose_2d import load_npz

    if not isinstance(report, dict):
        report = json.loads(Path(report).read_text())
    if sync_info is not None and not isinstance(sync_info, dict):
        sync_info = json.loads(Path(sync_info).read_text())
    timeline = RhythmTimeline.from_report(report)
    seq, _ = load_npz(pose_npz)
    kps = seq.keypoints

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or seq.fps or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + ".raw.mp4")
    ffmpeg = find_ffmpeg() if use_ffmpeg else None
    writer, fourcc = open_writer(raw_path, fps, (w, h), prefer_h264=ffmpeg is None)

    sync_txt = ""
    if sync_info:
        sync_txt = f"sync: {sync_info.get('method', '?')} {int(sync_info.get('frame_offset', 0)):+d} frames"
    max_frames = int(max_seconds * fps) if max_seconds else None
    warnings: list[str] = []
    if abs(fps - seq.fps) > 0.01 * fps:
        warnings.append(f"Video fps {fps:.3f} differs from cached pose fps {seq.fps:.3f}")
    i = 0
    try:
        while True:
            if max_frames is not None and i >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            kp = kps[i] if i < len(kps) else None
            render_frame(frame, kp, timeline, t, f"t = {t:6.2f} s   {sync_txt}".rstrip())
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        writer.release()
    if i < len(kps) and max_frames is None:
        warnings.append(f"Video has {i} frames but the pose cache has {len(kps)}")

    info: dict[str, Any] = {"path": str(out_path), "frames": i, "fps": fps, "codec": fourcc, "audio": False}
    if ffmpeg:
        try:
            info.update(finalize_with_ffmpeg(raw_path, Path(video_path), out_path, ffmpeg))
            raw_path.unlink(missing_ok=True)
        except Exception as exc:
            warnings.append(f"ffmpeg re-encode failed ({exc}); keeping the OpenCV {fourcc} file")
            os.replace(raw_path, out_path)
    else:
        os.replace(raw_path, out_path)
        if fourcc != "avc1":
            warnings.append("ffmpeg not found: overlay is mp4v (may not play in browsers) and has no audio")
    for wmsg in warnings:
        log.warning(wmsg)
    info["warnings"] = warnings
    log.info("Overlay written: %s (%d frames, %s, audio=%s)", out_path, i, info["codec"], info["audio"])
    return info


def render_for_session(sess: dict[str, Any], cfg: dict[str, Any], max_seconds: float | None = None) -> dict:
    sid = sess["id"]
    out_dir = Path(cfg["output_root"]) / sid
    return render_overlay(
        sess["cam1"],
        Path(cfg["cache_dir"]) / sid / "cam1_pose2d.npz",
        out_dir / "epochs_and_steps.json",
        out_dir / "overlay_cam1.mp4",
        out_dir / "sync.json" if (out_dir / "sync.json").is_file() else None,
        max_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render skeleton + rhythm HUD onto cam1.mp4")
    ap.add_argument("--config", default="sessions.json")
    ap.add_argument("--subject", action="append", help="session id (repeatable; default: all)")
    ap.add_argument("--max-seconds", type=float, help="only render the first N seconds (quick preview)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from main import load_config

    cfg = load_config(args.config)
    sessions = [s for s in cfg["sessions"] if not args.subject or s["id"] in args.subject]
    if not sessions:
        log.error("No matching sessions")
        return 2
    rc = 0
    for sess in sessions:
        try:
            info = render_for_session(sess, cfg, args.max_seconds)
            print(f"{sess['id']}: {info['path']} ({info['frames']} frames, {info['codec']}, audio={info['audio']})")
        except Exception as exc:
            log.error("%s: overlay failed: %s", sess["id"], exc)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
