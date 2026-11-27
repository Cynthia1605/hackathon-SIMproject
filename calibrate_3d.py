"""Checkerboard-free stereo calibration from pose landmarks, triangulation,
metric height scaling and floor alignment.

Output world frame (metres): floor plane at Y = 0, +Y up. +X is cam1's image
right projected onto the floor, and +Z = X x Y (roughly pointing back toward
cam1). Input is the sync-aligned keypoints from ``sync.align_sequences``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import landmarks as L

log = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PARAMS: dict[str, Any] = {
    "visibility_min": 0.85,  # correspondences used to estimate F
    "triangulation_visibility_min": 0.5,  # joints triangulated (others -> NaN)
    "focal_factor": 0.85,
    "height_eye_ratio": 0.935,
    "span_percentile": 95,
    "floor_percentile": 20,
    "max_fundamental_points": 20000,
    "ransac_threshold_px": 3.0,
    "ransac_confidence": 0.99,
    "seed": 0,
}

FOOT_IDX = [L.LEFT_HEEL, L.RIGHT_HEEL, L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX]


@dataclass
class Calibration3D:
    K1: np.ndarray
    K2: np.ndarray
    R: np.ndarray  # cam1 -> cam2 rotation
    t: np.ndarray  # cam1 -> cam2 translation (unit norm, pre-scale)
    scale: float
    measured_span: float  # eye-to-heel span (unscaled units) at span_percentile
    floor_R: np.ndarray  # rows = world X, Y(up), Z axes in scaled cam1 coordinates
    floor_origin: np.ndarray  # scaled cam1 coordinates of the world origin
    n_correspondences: int
    n_inliers: int
    reprojection_error_px: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "K1": self.K1.tolist(), "K2": self.K2.tolist(), "R": self.R.tolist(),
            "t": self.t.ravel().tolist(), "scale": self.scale, "measured_span": self.measured_span,
            "floor_R": self.floor_R.tolist(), "floor_origin": self.floor_origin.tolist(),
            "n_correspondences": self.n_correspondences, "n_inliers": self.n_inliers,
            "reprojection_error_px": self.reprojection_error_px, "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def approx_intrinsics(width: int, height: int, focal_factor: float = 0.85) -> np.ndarray:
    f = focal_factor * width
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def _valid_mask(kp: np.ndarray, vis_min: float) -> np.ndarray:
    return np.all(np.isfinite(kp[..., :2]), axis=-1) & (kp[..., L.VIS] >= vis_min)


def estimate_extrinsics(kp1, kp2, K1, K2, params: dict[str, Any]):
    """Estimate R, t (cam1 to cam2) from high-confidence landmark correspondences."""
    import cv2

    m = _valid_mask(kp1, params["visibility_min"]) & _valid_mask(kp2, params["visibility_min"])
    p1 = kp1[..., :2][m].astype(np.float64)
    p2 = kp2[..., :2][m].astype(np.float64)
    if len(p1) < 15:
        raise ValueError(f"Only {len(p1)} high-visibility correspondences; need >= 15")
    if len(p1) > params["max_fundamental_points"]:
        rng = np.random.default_rng(params["seed"])
        sel = rng.choice(len(p1), params["max_fundamental_points"], replace=False)
        p1, p2 = p1[sel], p2[sel]

    F, mask = cv2.findFundamentalMat(
        p1, p2, cv2.FM_RANSAC, params["ransac_threshold_px"], params["ransac_confidence"]
    )
    if F is None or F.shape[0] < 3:
        raise RuntimeError("Fundamental matrix estimation failed")
    F = F[:3]
    inl = mask.ravel().astype(bool)
    # OpenCV's RANSAC F is not re-fitted to its inliers; a normalised 8-point
    # least-squares fit on all inliers is far more accurate.
    if inl.sum() >= 8:
        F8, _ = cv2.findFundamentalMat(p1[inl], p2[inl], cv2.FM_8POINT)
        if F8 is not None and F8.shape == (3, 3):
            F = F8
    E = K2.T @ F @ K1
    n1 = cv2.undistortPoints(p1[inl].reshape(-1, 1, 2), K1, None)
    n2 = cv2.undistortPoints(p2[inl].reshape(-1, 1, 2), K2, None)
    _, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3))
    return R, t.reshape(3, 1), len(p1), int(inl.sum())


def triangulate(kp1, kp2, K1, K2, R, t, vis_min: float = 0.5):
    """Triangulate (N, J) joint pairs. Returns (N, J, 3) in cam1 coordinates and the mean reprojection error."""
    import cv2

    N, J = kp1.shape[:2]
    out = np.full((N, J, 3), np.nan)
    m = _valid_mask(kp1, vis_min) & _valid_mask(kp2, vis_min)
    if not m.any():
        return out, float("nan")
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t])
    a = kp1[..., :2][m].astype(np.float64).T
    b = kp2[..., :2][m].astype(np.float64).T
    Xh = cv2.triangulatePoints(P1, P2, a, b)
    w = Xh[3]
    X = (Xh[:3] / np.where(np.abs(w) > 1e-12, w, np.nan)).T
    out[m] = X

    def reproj(P, pts):
        x = P @ np.vstack([X.T, np.ones(len(X))])
        return np.linalg.norm(x[:2] / x[2] - pts, axis=0)

    with np.errstate(invalid="ignore"):
        err = np.nanmean(np.concatenate([reproj(P1, a), reproj(P2, b)]))
    return out, float(err)


def eye_to_heel_span(points3d: np.ndarray) -> np.ndarray:
    """Per-frame distance from mid-eye (2, 5) to the lowest (farthest) heel (29, 30)."""
    eye = 0.5 * (points3d[:, L.LEFT_EYE] + points3d[:, L.RIGHT_EYE])
    d_l = np.linalg.norm(points3d[:, L.LEFT_HEEL] - eye, axis=1)
    d_r = np.linalg.norm(points3d[:, L.RIGHT_HEEL] - eye, axis=1)
    return np.fmax(d_l, d_r)  # fmax ignores a single NaN heel


def metric_scale(points3d: np.ndarray, height_m: float, eye_ratio: float = 0.935, percentile: float = 95):
    span = eye_to_heel_span(points3d)
    span = span[np.isfinite(span)]
    if span.size == 0:
        raise ValueError("No frames with both eyes and a heel triangulated")
    measured = float(np.percentile(span, percentile))
    return height_m * eye_ratio / measured, measured


def fit_floor(points3d: np.ndarray, floor_percentile: float = 20, iterations: int = 2):
    """Fit the floor with SVD on the lowest heel/toe points.

    Returns ``(R_floor, origin)`` such that ``world = (p - origin) @ R_floor.T``.
    """
    eye = 0.5 * (points3d[:, L.LEFT_EYE] + points3d[:, L.RIGHT_EYE])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heel = np.nanmean(points3d[:, [L.LEFT_HEEL, L.RIGHT_HEEL]], axis=1)
        up = np.nanmean(eye - heel, axis=0)
    if not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-9:
        up = np.array([0.0, -1.0, 0.0])  # image up in cam1
    up /= np.linalg.norm(up)

    foot = points3d[:, FOOT_IDX].reshape(-1, 3)
    foot = foot[np.all(np.isfinite(foot), axis=1)]
    if len(foot) < 3:
        raise ValueError("Not enough heel/toe points to fit the floor")
    n, c = up, foot.mean(axis=0)
    for _ in range(iterations):
        h = foot @ n
        low = foot[h <= np.percentile(h, floor_percentile)]
        if len(low) < 3:
            break
        c = low.mean(axis=0)
        _, _, vt = np.linalg.svd(low - c, full_matrices=False)
        n_new = vt[-1]
        n = n_new if n_new @ up >= 0 else -n_new

    x_ref = np.array([1.0, 0.0, 0.0])
    x_axis = x_ref - (x_ref @ n) * n
    if np.linalg.norm(x_axis) < 1e-6:
        x_ref = np.array([0.0, 0.0, 1.0])
        x_axis = x_ref - (x_ref @ n) * n
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, n)
    return np.stack([x_axis, n, z_axis]), c


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def calibrate_and_triangulate(
    kp1_aligned: np.ndarray,
    kp2_aligned: np.ndarray,
    size1: tuple[int, int],
    size2: tuple[int, int],
    height_m: float,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Calibration3D]:
    """Return metric, floor-aligned (N, 33, 3) points and the calibration.

    ``size1`` and ``size2`` are (width, height) of cam1 and cam2.
    """
    p = {**DEFAULT_CALIBRATION_PARAMS, **(params or {})}
    K1 = approx_intrinsics(*size1, p["focal_factor"])
    K2 = approx_intrinsics(*size2, p["focal_factor"])
    R, t, n_corr, n_inl = estimate_extrinsics(kp1_aligned, kp2_aligned, K1, K2, p)
    pts, err = triangulate(kp1_aligned, kp2_aligned, K1, K2, R, t, p["triangulation_visibility_min"])
    warnings: list[str] = []
    if n_inl < 0.5 * n_corr:
        warnings.append(f"Only {n_inl}/{n_corr} RANSAC inliers; check sync offset")
    if err > 10:
        warnings.append(f"High reprojection error {err:.1f}px")
    for w in warnings:
        log.warning(w)

    scale, span = metric_scale(pts, height_m, p["height_eye_ratio"], p["span_percentile"])
    pts *= scale
    R_f, origin = fit_floor(pts, p["floor_percentile"])
    world = (pts - origin) @ R_f.T
    calib = Calibration3D(K1, K2, R, t, scale, span, R_f, origin, n_corr, n_inl, err, warnings)
    return world, calib


def calibrate_session(pose1, pose2, frame_offset: int, height_m: float, params: dict[str, Any] | None = None):
    """Convenience wrapper: align with the sync offset, then calibrate.

    Returns ``(points3d, calib, cam1_frame_indices)``.
    """
    from sync import align_sequences

    k1, k2, idx = align_sequences(pose1.keypoints, pose2.keypoints, frame_offset)
    pts, calib = calibrate_and_triangulate(
        k1, k2, (pose1.width, pose1.height), (pose2.width, pose2.height), height_m, params
    )
    return pts, calib, idx
