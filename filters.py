"""Vectorised 3D One Euro filter for (N, J, 3) trajectories.

Each joint is filtered as a 3D vector: the adaptive cutoff uses the joint's 3D
speed, so all axes share one cutoff. NaN frames stay NaN in the output and
never contaminate the filter state. After a gap longer than
``reset_gap_frames``, the joint's filter restarts from the next valid sample.
Shorter gaps continue with the elapsed time as dt.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_FILTER_PARAMS: dict[str, Any] = {
    "min_cutoff": 1.0,  # Hz: smoothing when a joint is still
    "beta": 2.0,  # 1 / (m/s): how fast the cutoff opens up with speed
    "d_cutoff": 1.0,  # Hz
    "reset_gap_frames": 5,
    # Offline zero-lag mode: average a forward and a time-reversed pass so the
    # One Euro lag cancels and footfall timing is not delayed.
    "bidirectional": True,
}


def _alpha(cutoff: np.ndarray | float, dt: np.ndarray | float) -> np.ndarray:
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=np.float64))
    return 1.0 / (1.0 + tau / np.asarray(dt, dtype=np.float64))


def one_euro_filter(
    x: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 2.0,
    d_cutoff: float = 1.0,
    reset_gap_frames: int = 5,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Filter ``x`` of shape (N, ..., D) along axis 0.

    The last axis is the vector dimension (3 for 3D points). A sample is valid
    only if all D components are finite.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("x must have shape (N, ..., D)")
    n = x.shape[0]
    shape = x.shape
    xr = x.reshape(n, -1, shape[-1])  # (N, J, D)
    out = np.full_like(xr, np.nan)
    if n == 0:
        return out.reshape(shape)
    t = np.arange(n) / fps if timestamps is None else np.asarray(timestamps, dtype=np.float64)
    J = xr.shape[1]
    x_hat = np.full((J, shape[-1]), np.nan)
    dx_hat = np.zeros((J, shape[-1]))
    t_last = np.full(J, np.nan)
    max_dt = (reset_gap_frames + 1.5) / fps

    for i in range(n):
        xi = xr[i]
        valid = np.all(np.isfinite(xi), axis=1)
        if not valid.any():
            continue
        elapsed = t[i] - t_last
        init = valid & (np.isnan(t_last) | (elapsed > max_dt))
        upd = valid & ~init
        if init.any():
            x_hat[init] = xi[init]
            dx_hat[init] = 0.0
        if upd.any():
            dt = np.maximum(elapsed[upd], 1e-6)[:, None]
            dx = (xi[upd] - x_hat[upd]) / dt
            a_d = _alpha(d_cutoff, dt)
            dx_hat[upd] = a_d * dx + (1 - a_d) * dx_hat[upd]
            speed = np.linalg.norm(dx_hat[upd], axis=1, keepdims=True)
            a = _alpha(min_cutoff + beta * speed, dt)
            x_hat[upd] = a * xi[upd] + (1 - a) * x_hat[upd]
        t_last[valid] = t[i]
        out[i, valid] = x_hat[valid]
    return out.reshape(shape)


def one_euro_filter_bidirectional(x: np.ndarray, fps: float, **kw) -> np.ndarray:
    """Mean of forward and backward One Euro passes (zero-lag, offline only)."""
    fwd = one_euro_filter(x, fps, **kw)
    bwd = one_euro_filter(np.asarray(x)[::-1], fps, **kw)[::-1]
    return 0.5 * (fwd + bwd)  # NaN wherever the input was NaN in both passes


def filter_points3d(points3d: np.ndarray, fps: float, params: dict[str, Any] | None = None) -> np.ndarray:
    p = {**DEFAULT_FILTER_PARAMS, **(params or {})}
    kw = dict(min_cutoff=p["min_cutoff"], beta=p["beta"], d_cutoff=p["d_cutoff"],
              reset_gap_frames=int(p["reset_gap_frames"]))
    if p.get("bidirectional", True):
        return one_euro_filter_bidirectional(points3d, fps, **kw)
    return one_euro_filter(points3d, fps, **kw)
