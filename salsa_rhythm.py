"""Footfall detection and quick-quick-slow (1-2-3, 5-6-7) salsa classification.

Footfalls are local minima of a foot's horizontal (XZ) speed, using
``find_peaks`` on the negated speed with a prominence threshold, while the foot
is near the floor (Y <= ``floor_height_m``). Each event is then moved back to
the touchdown: the first still frame after that foot's movement. This is more
precise than the lowest-speed frame, which can land anywhere in a planted hold.

Counting: in a basic step, beats 1 and 3 are taken by the same foot and 5 and 7
by the other. A QQS group whose SLOW foot matches the beat-1 foot is therefore
labelled 1-2-3, and the other foot's group 5-6-7. So the count re-locks even
after a missed step. The beat-1 foot is the first foot of the first complete
QQS group in the epoch, unless ``count_one_foot`` is set to "L" or "R".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from epoch_segmenter import ActiveEpoch, _fill_nan, foot_centers, horizontal_speed

DEFAULT_RHYTHM_PARAMS: dict[str, Any] = {
    "floor_height_m": 0.05,
    "min_prominence_mps": 0.12,
    "min_step_interval_s": 0.15,
    "still_speed_mps": 0.08,
    "still_speed_frac": 0.3,  # still threshold = max(still_speed_mps, frac * preceding peak speed)
    "slow_ratio": 1.5,
    "speed_smooth_frames": 3,
    "count_one_foot": "auto",  # "auto" | "L" | "R"
}


@dataclass
class Step:
    epoch_id: int
    idx: int  # index into the aligned 3D array
    cam1_frame: int
    timestamp_sec: float
    foot: str  # "L" | "R"
    timing_type: str  # "QUICK" | "SLOW" | "TAP"
    beat_count: int | None
    hold_after: str | None  # "4-hold" | "8-hold"
    interval_sec: float | None  # time to the next counted step
    nearest_music_beat_sec: float | None
    beat_offset_ms: float | None  # step - beat: negative = early, positive = late
    step_length_cm: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RhythmResult:
    steps: list[Step]
    epoch_summaries: list[dict[str, Any]]
    summary: dict[str, Any]
    beat_period_sec: float
    beat_grid_sec: np.ndarray
    foot_speed: dict[str, np.ndarray] = field(default_factory=dict)  # per aligned index
    foot_height: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Footfalls
# --------------------------------------------------------------------------- #
def foot_signals(points3d: np.ndarray, fps: float, smooth: int = 3):
    lf, rf = foot_centers(points3d)
    out_speed, out_height, out_pos = {}, {}, {}
    k = np.ones(max(1, smooth)) / max(1, smooth)
    for name, f in (("L", lf), ("R", rf)):
        sp = horizontal_speed(f, fps)
        if smooth > 1:
            sp = np.convolve(np.pad(sp, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
        out_speed[name] = sp
        out_height[name] = f[:, 1]
        out_pos[name] = _fill_nan(f)
    return out_speed, out_height, out_pos


def detect_footfalls(speed: np.ndarray, height: np.ndarray, fps: float, lo: int, hi: int,
                     params: dict[str, Any]) -> list[tuple[int, float]]:
    """Touchdowns of one foot within [lo, hi) as (frame index, sub-frame index).

    The sub-frame index is where the speed falls through the still threshold,
    linearly interpolated between frames.
    """
    p = params
    seg = speed[lo:hi]
    if len(seg) < 3:
        return []
    # Pad the end with motion so a final plant that lasts until the epoch end
    # still counts as a minimum. The start is not padded: an initial plateau
    # without preceding motion is not a footfall.
    pad = int(round(p["min_step_interval_s"] * fps)) + 1
    padded = np.concatenate([seg, np.full(pad, seg.max())])
    peaks, _ = signal.find_peaks(-padded, prominence=p["min_prominence_mps"],
                                 distance=max(1, int(round(p["min_step_interval_s"] * fps))))
    events = []
    for m in peaks[peaks < len(seg)]:
        w0 = max(0, m - int(fps))  # preceding motion peak within the last ~1 s
        j_peak = w0 + int(np.argmax(seg[w0 : m + 1]))
        peak_speed = float(seg[j_peak])
        if peak_speed < p["min_prominence_mps"]:
            continue
        thr = max(p["still_speed_mps"], p["still_speed_frac"] * peak_speed)
        j = m
        while j - 1 > j_peak and seg[j - 1] < thr:
            j -= 1
        frac = float(j)
        if j - 1 >= 0 and seg[j - 1] > seg[j] and seg[j] < thr:
            frac = (j - 1) + (seg[j - 1] - thr) / (seg[j - 1] - seg[j])
        idx = lo + j
        h = height[idx : idx + 3]
        if np.isfinite(h).any() and np.nanmedian(h) <= p["floor_height_m"]:
            events.append((idx, lo + frac))
    return events


# --------------------------------------------------------------------------- #
# Beat grid helpers
# --------------------------------------------------------------------------- #
def step_interval_period(times: np.ndarray) -> float | None:
    d = np.diff(np.sort(times))
    d = d[d > 0]
    if len(d) == 0:
        return None
    return float(np.median(d[d <= np.percentile(d, 75)]))


def resolve_beat_period(music, step_period: float | None) -> tuple[float, np.ndarray, list[str]]:
    """Beat period and beat grid on the cam1 timeline, with tempo-octave correction."""
    warnings: list[str] = []
    grid = np.asarray(getattr(music, "aligned_beat_times_sec", []) or [], dtype=np.float64)
    method = getattr(music, "method", "unaligned")
    bpm = float(getattr(music, "bpm", 0.0) or 0.0)
    if bpm <= 0 or method == "unaligned":
        if step_period is None:
            raise ValueError("No music tempo and no steps to estimate a beat period")
        warnings.append("Music unaligned: beat period estimated from step intervals")
        return step_period, (grid if method != "unaligned" else np.array([])), warnings
    period = 60.0 / bpm
    if step_period is not None:
        factors = np.array([0.5, 1.0, 2.0])
        best = float(factors[np.argmin(np.abs(np.log(period * factors / step_period)))])
        if best != 1.0:
            warnings.append(f"Tempo octave corrected: song period {period:.3f}s x{best} to match steps")
            if best == 0.5 and len(grid) >= 2:  # half-time tracking: insert midpoints
                mids = 0.5 * (grid[:-1] + grid[1:])
                grid = np.sort(np.concatenate([grid, mids]))
            period *= best
    return period, grid, warnings


def nearest_beat(t: float, grid: np.ndarray, period: float) -> float | None:
    if len(grid) == 0 or t < grid[0] - period / 2 or t > grid[-1] + period / 2:
        return None
    return float(grid[np.argmin(np.abs(grid - t))])


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_epoch_steps(events: list[tuple[int, str]], fps: float, period: float, params: dict[str, Any]):
    """events: sorted (idx, foot). Returns list of dicts with timing/count fields."""
    out = []
    last_foot = None
    for idx, foot in events:
        if foot == last_foot:
            out.append({"idx": idx, "foot": foot, "timing_type": "TAP"})
            continue
        out.append({"idx": idx, "foot": foot, "timing_type": None})
        last_foot = foot
    counted = [s for s in out if s["timing_type"] != "TAP"]
    for i, s in enumerate(counted):
        if i + 1 < len(counted):
            dt = (counted[i + 1]["idx"] - s["idx"]) / fps
            s["interval_sec"] = dt
            s["timing_type"] = "SLOW" if dt > params["slow_ratio"] * period else "QUICK"
        else:
            s["interval_sec"] = None
            prev = [c["timing_type"] for c in counted[max(0, i - 2) : i]]
            s["timing_type"] = "SLOW" if prev == ["QUICK", "QUICK"] else "QUICK"

    # group into ... Q Q S units
    groups, cur = [], []
    for s in counted:
        cur.append(s)
        if s["timing_type"] == "SLOW":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    one_foot = params.get("count_one_foot", "auto")
    if one_foot not in ("L", "R"):
        complete = [g for g in groups if len(g) == 3 and g[-1]["timing_type"] == "SLOW"]
        one_foot = complete[0][0]["foot"] if complete else (counted[0]["foot"] if counted else "L")
    for g in groups:
        for s in g:
            s["beat_count"], s["hold_after"], s["in_qqs"] = None, None, False
        if g[-1]["timing_type"] != "SLOW":
            continue
        first_half = g[-1]["foot"] == one_foot
        labels = (1, 2, 3) if first_half else (5, 6, 7)
        tail = g[-3:]
        for s, b in zip(tail, labels[-len(tail):]):
            s["beat_count"] = b
        g[-1]["hold_after"] = "4-hold" if first_half else "8-hold"
        if len(g) == 3:
            for s in g:
                s["in_qqs"] = True
    for s in out:
        s.setdefault("beat_count", None)
        s.setdefault("hold_after", None)
        s.setdefault("interval_sec", None)
        s.setdefault("in_qqs", False)
    return out


def _stats(offsets: list[float]) -> dict[str, Any]:
    a = np.asarray([o for o in offsets if o is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean_offset_ms": None, "mean_abs_offset_ms": None, "std_offset_ms": None,
                "pct_within_50ms": None, "timing_tendency": "n/a"}
    mean = float(a.mean())
    return {
        "mean_offset_ms": mean,
        "mean_abs_offset_ms": float(np.abs(a).mean()),
        "std_offset_ms": float(a.std()),
        "pct_within_50ms": float(np.mean(np.abs(a) <= 50) * 100),
        "timing_tendency": "early" if mean < -20 else "late" if mean > 20 else "on-beat",
    }


def _summarise(steps: list[Step]) -> dict[str, Any]:
    counted = [s for s in steps if s.timing_type != "TAP"]
    return {
        "n_steps": len(counted),
        "n_taps": len(steps) - len(counted),
        "n_quick": sum(s.timing_type == "QUICK" for s in counted),
        "n_slow": sum(s.timing_type == "SLOW" for s in counted),
        "n_counted": sum(s.beat_count is not None for s in counted),
        **_stats([s.beat_offset_ms for s in counted]),
    }


def analyze_rhythm(
    points3d: np.ndarray,
    fps: float,
    cam1_frames: np.ndarray,
    epochs: list[ActiveEpoch],
    music,
    params: dict[str, Any] | None = None,
) -> RhythmResult:
    p = {**DEFAULT_RHYTHM_PARAMS, **(params or {})}
    cam1_frames = np.asarray(cam1_frames)
    speed, height, pos = foot_signals(points3d, fps, int(p["speed_smooth_frames"]))

    raw: dict[int, list[tuple[int, str]]] = {}
    for ep in epochs:
        ev = [(fi, f) for f in ("L", "R") for _, fi in detect_footfalls(speed[f], height[f], fps,
                                                                        ep.start_idx, ep.end_idx, p)]
        raw[ep.epoch_id] = sorted(ev)

    def to_time(fi: float) -> float:  # fractional aligned index -> cam1 seconds
        return float((cam1_frames[0] + fi) / fps)

    step_period = None
    per_epoch_periods = [step_interval_period(np.array([to_time(fi) for fi, _ in ev]))
                         for ev in raw.values() if len(ev) >= 3]
    per_epoch_periods = [x for x in per_epoch_periods if x]
    if per_epoch_periods:
        step_period = float(np.median(per_epoch_periods))
    period, grid, warnings = resolve_beat_period(music, step_period)

    steps: list[Step] = []
    epoch_summaries = []
    for ep in epochs:
        cls = classify_epoch_steps(raw[ep.epoch_id], fps, period, p)
        ep_steps = []
        prev_idx = ep.start_idx
        for c in cls:
            fi, foot = c["idx"], c["foot"]
            i = min(len(cam1_frames) - 1, int(round(fi)))
            t = to_time(fi)
            nb = nearest_beat(t, grid, period)
            off = (t - nb) * 1000.0 if nb is not None else None
            length = float(np.linalg.norm(pos[foot][i, [0, 2]] - pos[foot][prev_idx, [0, 2]]) * 100.0)
            if c["timing_type"] != "TAP":
                prev_idx = i
            bc = c["beat_count"]
            if c["timing_type"] == "TAP":
                label = f"{foot}-TAP"
            else:
                label = f"{foot}{bc if bc is not None else '?'}" + ("-SLOW" if c["timing_type"] == "SLOW" else "")
            ep_steps.append(Step(ep.epoch_id, i, int(cam1_frames[i]), t, foot, c["timing_type"], bc,
                                 c["hold_after"], c["interval_sec"], nb, off, length, label))
        counted = [s for s, c in zip(ep_steps, cls) if c["timing_type"] != "TAP"]
        in_qqs = [c["in_qqs"] for c in cls if c["timing_type"] != "TAP"]
        summ = _summarise(ep_steps)
        summ["qqs_pattern_accuracy"] = float(np.mean(in_qqs)) if in_qqs else None
        summ["step_tempo_bpm"] = (60.0 / step_interval_period(np.array([s.timestamp_sec for s in counted]))
                                  if len(counted) >= 3 else None)
        summ["mean_step_length_cm"] = float(np.mean([s.step_length_cm for s in counted])) if counted else None
        epoch_summaries.append({"epoch_id": ep.epoch_id, **summ})
        steps.extend(ep_steps)

    overall = _summarise(steps)
    flags = [c for ep in epochs for c in classify_epoch_steps(raw[ep.epoch_id], fps, period, p)
             if c["timing_type"] != "TAP"]
    overall["qqs_pattern_accuracy"] = float(np.mean([c["in_qqs"] for c in flags])) if flags else None
    overall["n_epochs"] = len(epochs)
    overall["beat_period_sec"] = period
    overall["music_bpm"] = float(getattr(music, "bpm", 0.0) or 0.0)
    return RhythmResult(steps, epoch_summaries, overall, period, grid, speed, height, warnings)
