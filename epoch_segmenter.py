"""Active salsa vs. idle segmentation from filtered, floor-aligned 3D feet.

Only foot landmarks (27-32) feed the energy, so upper-body-only motion
(clapping, arm gestures) cannot start an epoch. Pelvis travel damps the energy,
so walking in or out is not mistaken for dancing in place.

Index conventions: ``start_idx``/``end_idx`` index the aligned 3D array
(``end_idx`` exclusive). ``cam1_start_frame``/``cam1_end_frame`` are cam1 video
frames (both inclusive). Times are on the cam1 timeline.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import landmarks as L

DEFAULT_SEGMENTER_PARAMS: dict[str, Any] = {
    "window_s": 1.5,
    "hop_s": 0.25,
    "merge_gap_s": 1.2,
    "min_epoch_s": 2.5,
    # energy = range/range_ref + lift_std/lift_ref + alternation/alt_ref, then damped by travel
    "range_ref_m": 0.10,
    "lift_ref_m": 0.02,
    "alt_ref_mps": 0.15,
    "walk_travel_m": 0.8,
    # adaptive hysteresis: floor/high percentiles of the energy distribution
    "start_frac": 0.4,
    "stop_frac": 0.2,
    "min_start_energy": 1.5,
    "min_stop_energy": 0.8,
    # boundary refinement: trim each candidate epoch to frames where a foot is moving
    "motion_speed_mps": 0.15,
}

LEFT_FOOT = [L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX]
RIGHT_FOOT = [L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX]


@dataclass
class ActiveEpoch:
    epoch_id: int
    start_idx: int
    end_idx: int  # exclusive
    start_sec: float
    end_sec: float
    cam1_start_frame: int
    cam1_end_frame: int  # inclusive

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = "ACTIVE_MOVE"
        d["duration_sec"] = self.duration_sec
        return d


@dataclass
class SegmentationDiagnostics:
    window_centers_sec: np.ndarray  # cam1 timeline
    energy: np.ndarray
    start_threshold: float
    stop_threshold: float
    active_windows: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"start_threshold": self.start_threshold, "stop_threshold": self.stop_threshold,
                "n_windows": int(len(self.energy)), "n_active_windows": int(self.active_windows.sum())}


def _nanmean_joints(points3d: np.ndarray, idx: list[int]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(points3d[:, idx], axis=1)


def foot_centers(points3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) left and right foot centers: mean of ankle, heel and toe."""
    return _nanmean_joints(points3d, LEFT_FOOT), _nanmean_joints(points3d, RIGHT_FOOT)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs along axis 0 (per column)."""
    x = np.array(x, dtype=np.float64)
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        col = x[:, c]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            x[:, c] = np.interp(idx, idx[ok], col[ok])
        elif ok.sum() == 1:
            x[:, c] = col[ok][0]
        else:
            x[:, c] = 0.0
    return x


def horizontal_speed(p: np.ndarray, fps: float) -> np.ndarray:
    v = np.gradient(_fill_nan(p[:, [0, 2]]), axis=0) * fps
    return np.linalg.norm(v, axis=1)


def foot_motion_energy(points3d: np.ndarray, fps: float, params: dict[str, Any]):
    """Per-window foot motion energy. Returns (window start indices, energy)."""
    p = params
    n = len(points3d)
    W = max(2, int(round(p["window_s"] * fps)))
    H = max(1, int(round(p["hop_s"] * fps)))
    lf, rf = foot_centers(points3d)
    lf, rf = _fill_nan(lf), _fill_nan(rf)
    pelvis = _fill_nan(_nanmean_joints(points3d, [L.LEFT_HIP, L.RIGHT_HIP]))
    vl, vr = horizontal_speed(lf, fps), horizontal_speed(rf, fps)

    starts = np.arange(0, max(1, n - W + 1), H)
    energy = np.zeros(len(starts))
    for k, s in enumerate(starts):
        sl = slice(s, min(n, s + W))
        rng = max(np.linalg.norm(np.ptp(f[sl][:, [0, 2]], axis=0)) for f in (lf, rf))
        lift = np.sqrt(np.var(lf[sl, 1]) + np.var(rf[sl, 1]))
        alt = float(np.mean(np.abs(vl[sl] - vr[sl])))
        e = rng / p["range_ref_m"] + lift / p["lift_ref_m"] + alt / p["alt_ref_mps"]
        travel = np.linalg.norm(np.ptp(pelvis[sl][:, [0, 2]], axis=0))
        energy[k] = e / (1.0 + (travel / p["walk_travel_m"]) ** 4)
    return starts, energy, W, np.maximum(vl, vr)


def _hysteresis(energy: np.ndarray, start: float, stop: float) -> np.ndarray:
    active = np.zeros(len(energy), bool)
    on = False
    for i, e in enumerate(energy):
        if not on and e >= start:
            on = True
        elif on and e < stop:
            on = False
        active[i] = on
    return active


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment_epochs(
    points3d: np.ndarray,
    fps: float,
    cam1_frames: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[list[ActiveEpoch], SegmentationDiagnostics]:
    p = {**DEFAULT_SEGMENTER_PARAMS, **(params or {})}
    n = len(points3d)
    cam1_frames = np.arange(n) if cam1_frames is None else np.asarray(cam1_frames)
    starts, energy, W, foot_speed = foot_motion_energy(points3d, fps, p)
    centers = starts + W / 2.0

    lo, hi = np.percentile(energy, 10), np.percentile(energy, 90)
    start_thr = max(lo + p["start_frac"] * (hi - lo), p["min_start_energy"])
    stop_thr = min(max(lo + p["stop_frac"] * (hi - lo), p["min_stop_energy"]), start_thr)
    active_w = _hysteresis(energy, start_thr, stop_thr)

    # window states -> frame mask (each window owns +-hop/2 around its center)
    H = max(1, int(round(p["hop_s"] * fps)))
    mask = np.zeros(n, bool)
    for c, a in zip(centers, active_w):
        if a:
            mask[max(0, int(c - H / 2)) : min(n, int(np.ceil(c + H / 2)))] = True
    if active_w.size and active_w[0]:
        mask[: int(centers[0])] = True
    if active_w.size and active_w[-1]:
        mask[int(centers[-1]) :] = True

    # windows smear boundaries by up to W/2: trim each run to actual foot motion
    moving = foot_speed > p["motion_speed_mps"]
    runs = []
    for s, e in _runs(mask):
        mv = np.flatnonzero(moving[s:e])
        if mv.size:
            runs.append((s + int(mv[0]), s + int(mv[-1]) + 1))
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) / fps < p["merge_gap_s"]:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    runs = [(s, e) for s, e in merged if (e - s) / fps >= p["min_epoch_s"]]

    epochs = []
    for i, (s, e) in enumerate(runs):
        f0, f1 = int(cam1_frames[s]), int(cam1_frames[e - 1])
        epochs.append(ActiveEpoch(i, int(s), int(e), f0 / fps, (f1 + 1) / fps, f0, f1))
    diag = SegmentationDiagnostics(cam1_frames[0] / fps + centers / fps, energy, float(start_thr),
                                   float(stop_thr), active_w)
    return epochs, diag
