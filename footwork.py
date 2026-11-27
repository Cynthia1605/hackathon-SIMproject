"""Top-down footwork reconstruction: actual floor positions vs. the ideal On-1 salsa basic.

Input: the filtered metric 3D trajectory from the stereo pipeline (``points3d.npz``:
floor at Y = 0, +Y up, X/Z horizontal) and the steps from ``epochs_and_steps.json``.

Floor frame per dance block (epoch), in cm:
* origin: midpoint of both feet in the centre stance just before the block's first step;
* +y: the dancer's facing direction (mean heel->toe vector of both feet). If toes are
  not tracked, falls back to PCA of foot positions, with the sign taken from the
  beat-1 step;
* +x: the dancer's right. So the left foot is on the diagram's left, with FRONT up.

Ideal On-1 basic (scaled by height): stance feet at +/- half stance width; beat 1 = left
foot forward by one reach; 2 = right in place; 3 = left back to centre; 5 = right
backward by one reach; 6 = left in place; 7 = right back to centre.

Placement error is measured per 8-count cycle against the ideal pattern anchored at that
cycle's centre stance. Drift of the anchor between cycles is reported separately, so a
slow drift does not inflate every step's error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import landmarks as L

FOOT_JOINTS = {"L": [L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX], "R": [L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX]}
HEEL = {"L": L.LEFT_HEEL, "R": L.RIGHT_HEEL}
TOE = {"L": L.LEFT_FOOT_INDEX, "R": L.RIGHT_FOOT_INDEX}

REACH_PER_HEIGHT = 0.16  # 1.70 m -> 27 cm step reach
HALF_STANCE_PER_HEIGHT = 0.055  # 1.70 m -> 9.4 cm from centre (19 cm stance)
FOOT_LEN_PER_HEIGHT = 0.15  # 1.70 m -> 25.5 cm shoe
PLANT_FRAMES = 4  # average this many frames from touchdown for a step's position
LIFTOFF_S = 0.3  # a step's foot leaves the floor at most this long before its touchdown


# --------------------------------------------------------------------------- #
# Ideal pattern
# --------------------------------------------------------------------------- #
def ideal_geometry(height_m: float) -> dict[str, Any]:
    reach = float(np.clip(REACH_PER_HEIGHT * height_m, 0.22, 0.32)) * 100
    half = HALF_STANCE_PER_HEIGHT * height_m * 100
    targets = {  # beat -> (foot, x, y) in cm, relative to the cycle's centre stance
        1: ("L", -half, reach), 2: ("R", half, 0.0), 3: ("L", -half, 0.0),
        5: ("R", half, -reach), 6: ("L", -half, 0.0), 7: ("R", half, 0.0),
    }
    return {
        "reach_cm": reach,
        "half_stance_cm": half,
        "foot_length_cm": FOOT_LEN_PER_HEIGHT * height_m * 100,
        "foot_width_cm": 0.38 * FOOT_LEN_PER_HEIGHT * height_m * 100,
        "targets": targets,
        "stance": {"L": (-half, 0.0), "R": (half, 0.0)},
    }


# --------------------------------------------------------------------------- #
# Floor tracks and normalisation
# --------------------------------------------------------------------------- #
def _nanmean(a, axis):
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=axis)


def foot_tracks(points3d: np.ndarray) -> dict[str, np.ndarray]:
    """Horizontal (X, Z) foot-centre tracks in metres, (N, 2) per foot."""
    return {f: _nanmean(points3d[:, idx][:, :, [0, 2]], axis=1) for f, idx in FOOT_JOINTS.items()}


def _fill(track: np.ndarray) -> np.ndarray:
    out = track.copy()
    idx = np.arange(len(out))
    for c in range(out.shape[1]):
        ok = np.isfinite(out[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(idx, idx[ok], out[ok, c])
    return out


def facing_direction(points3d: np.ndarray, sl: slice) -> np.ndarray | None:
    """Unit (X, Z) facing vector: mean heel->toe direction of both feet over ``sl``."""
    vecs = []
    for f in ("L", "R"):
        v = points3d[sl, TOE[f]][:, [0, 2]] - points3d[sl, HEEL[f]][:, [0, 2]]
        n = np.linalg.norm(v, axis=1)
        ok = np.isfinite(n) & (n > 1e-3)
        if ok.any():
            vecs.append(v[ok] / n[ok, None])
    if not vecs:
        return None
    m = np.concatenate(vecs).mean(axis=0)
    nm = np.linalg.norm(m)
    return m / nm if nm > 0.3 else None  # inconsistent directions -> unreliable


def epoch_frame(points3d: np.ndarray, tracks: dict[str, np.ndarray], start: int, end: int,
                first_step_idx: int | None, beat1_step: tuple[str, int] | None = None) -> dict[str, Any]:
    """Origin and axes for one dance block. Returns origin (m), fwd/right unit vectors (X, Z), method."""
    n = len(points3d)
    s0, s1 = max(0, start), min(n, end)
    stance_end = first_step_idx if first_step_idx is not None else s0 + 5
    st = slice(max(s0, stance_end - 15), max(s0 + 1, stance_end - 2))  # ~0.5 s before the first lift-off
    mid = 0.5 * (tracks["L"][st] + tracks["R"][st])
    origin = _nanmean(mid, axis=0)
    if not np.isfinite(origin).all():
        origin = _nanmean(0.5 * (tracks["L"][s0:s1] + tracks["R"][s0:s1]), axis=0)

    fwd = facing_direction(points3d, slice(s0, s1))
    method = "foot_facing"
    if fwd is None:  # PCA of foot positions; the forward axis carries the 1/5 steps
        pts = np.concatenate([tracks["L"][s0:s1], tracks["R"][s0:s1]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 3:
            fwd = np.array([0.0, 1.0])
        else:
            _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
            fwd = vt[0]
            if beat1_step is not None:
                foot, i = beat1_step
                if np.dot(tracks[foot][i] - origin, fwd) < 0:
                    fwd = -fwd
        method = "pca"
    # right = forward x up (right-handed X, Y-up, Z world) -> in (X, Z): (-fz, fx)
    right = np.array([-fwd[1], fwd[0]])
    return {"origin": origin, "fwd": fwd, "right": right, "method": method}


def to_floor_cm(p_xz: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    d = np.asarray(p_xz, dtype=np.float64) - frame["origin"]
    return np.stack([d @ frame["right"], d @ frame["fwd"]], axis=-1) * 100.0


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #
def assign_cycles(steps: list[dict]) -> list[int]:
    """8-count cycle id per step (a new cycle starts when the beat count wraps or at a new epoch)."""
    out, cyc, last_beat, last_ep = [], -1, None, None
    for s in steps:
        b, ep = s.get("beat_count"), s.get("epoch_id")
        if ep != last_ep:
            cyc, last_beat, last_ep = cyc + 1, None, ep
        elif b is not None and last_beat is not None and b <= last_beat:
            cyc += 1
        if b is not None:
            last_beat = b
        out.append(cyc)
    return out


def compute_footwork(points3d: np.ndarray, cam1_frames: np.ndarray, fps: float, report: dict[str, Any],
                     height_m: float, track_hz: float = 15.0) -> dict[str, Any]:
    """Floor map data and spatial metrics. Step entries are keyed by their index in sorted report steps."""
    steps = sorted(report.get("steps", []), key=lambda s: s["timestamp_sec"])
    epochs = report.get("epochs", [])
    geo = ideal_geometry(height_m)
    if not steps or not epochs or points3d is None or len(points3d) == 0:
        return {"available": False, "reason": "no steps or 3D trajectory"}
    n = len(points3d)
    tracks = {f: _fill(t) for f, t in foot_tracks(points3d).items()}
    cam1_frames = np.asarray(cam1_frames)
    f0 = int(cam1_frames[0])

    def idx_of(s):  # aligned index of a step
        i = s.get("idx")
        if i is None:
            i = int(s.get("cam1_frame", 0)) - f0
        return int(np.clip(i, 0, n - 1))

    # per-epoch frames
    frames: dict[int, dict] = {}
    for e in epochs:
        eid = e["epoch_id"]
        es = [s for s in steps if s.get("epoch_id") == eid and s.get("timing_type") != "TAP"]
        first = idx_of(es[0]) if es else None
        b1 = next(((s["foot"], idx_of(s)) for s in es if s.get("beat_count") == 1), None)
        lo = int(e.get("start_idx", int(round(e["start_sec"] * fps)) - f0))
        hi = int(e.get("end_idx", int(round(e["end_sec"] * fps)) - f0))
        if first is not None:
            # look back up to 1 s before the first step for the stance (epoch boundaries are trimmed to motion)
            lo = min(lo, max(0, first - int(fps)))
        frames[eid] = epoch_frame(points3d, tracks, lo, hi, first, b1)

    def planted(foot, i):
        return _nanmean(tracks[foot][i : min(n, i + PLANT_FRAMES)], axis=0)

    def heading_deg(foot, i, frame):
        v = points3d[i : min(n, i + PLANT_FRAMES), TOE[foot]][:, [0, 2]] - points3d[i : min(n, i + PLANT_FRAMES), HEEL[foot]][:, [0, 2]]
        v = _nanmean(v, axis=0)
        if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-3:
            return 0.0
        return float(np.degrees(np.arctan2(v @ frame["right"], v @ frame["fwd"])))  # 0 = facing front

    cycles = assign_cycles(steps)
    cyc_info: dict[int, dict] = {}
    out_steps = []
    prev_counted_idx: dict[int, int] = {}
    last_plant: dict[tuple[int, str], np.ndarray] = {}
    for k, (s, c) in enumerate(zip(steps, cycles)):
        eid = s.get("epoch_id")
        if eid not in frames:
            continue
        fr = frames[eid]
        i = idx_of(s)
        foot = s.get("foot")
        pos = to_floor_cm(planted(foot, i), fr)
        if c not in cyc_info:
            # cycle anchor: centre stance just before this cycle's first lift-off (~0.3 s before its first
            # touchdown, but not before the previous touchdown), so a shuffle during the 4/8 hold counts as
            # drift instead of inflating every step's placement error
            lo_j = prev_counted_idx.get(eid, 0)
            j = max(lo_j, i - int(round(LIFTOFF_S * fps)) - PLANT_FRAMES)
            anchor = 0.5 * (to_floor_cm(planted("L", j), fr) + to_floor_cm(planted("R", j), fr))
            if not np.isfinite(anchor).all():
                anchor = np.zeros(2)
            cyc_info[c] = {"id": c, "epoch": eid, "t0": s["timestamp_sec"], "t1": s["timestamp_sec"],
                           "anchor": anchor}
        cyc_info[c]["t1"] = s["timestamp_sec"]
        entry = {"i": k, "x": round(float(pos[0]), 1), "y": round(float(pos[1]), 1),
                 "heading_deg": round(heading_deg(foot, i, fr), 1), "cycle": c, "epoch": eid,
                 "target_x": None, "target_y": None, "err_cm": None, "len_cm": None}
        if s.get("timing_type") != "TAP":
            b = s.get("beat_count")
            if b in geo["targets"] and geo["targets"][b][0] == foot:
                a = cyc_info[c]["anchor"]
                tx, ty = a[0] + geo["targets"][b][1], a[1] + geo["targets"][b][2]
                entry.update(target_x=round(float(tx), 1), target_y=round(float(ty), 1),
                             err_cm=round(float(np.hypot(pos[0] - tx, pos[1] - ty)), 1))
            key = (eid, foot)
            if key in last_plant:
                entry["len_cm"] = round(float(np.linalg.norm(pos - last_plant[key])), 1)
                entry["fwd_disp_cm"] = round(float(pos[1] - last_plant[key][1]), 1)
            last_plant[key] = pos
            prev_counted_idx[eid] = i
        out_steps.append(entry)

    metrics = spatial_metrics(steps, out_steps, cyc_info, geo)

    # continuous tracks for live playback markers (downsampled)
    stride = max(1, int(round(fps / track_hz)))
    times = cam1_frames[::stride] / fps
    ep_bounds = [(e["epoch_id"], e["start_sec"], e["end_sec"]) for e in epochs if e["epoch_id"] in frames]
    trk = {"t": [round(float(t), 3) for t in times], "L": [], "R": []}
    for t, i in zip(times, range(0, n, stride)):
        eid = min(ep_bounds, key=lambda b: 0 if b[1] <= t <= b[2] else min(abs(t - b[1]), abs(t - b[2])))[0]
        for f in ("L", "R"):
            p = to_floor_cm(tracks[f][i], frames[eid])
            trk[f].append([round(float(p[0]), 1), round(float(p[1]), 1)] if np.isfinite(p).all() else None)

    return {
        "available": True,
        "method": sorted({fr["method"] for fr in frames.values()}),
        "height_m": height_m,
        "ideal": {"reach_cm": round(geo["reach_cm"], 1), "half_stance_cm": round(geo["half_stance_cm"], 1),
                  "foot_length_cm": round(geo["foot_length_cm"], 1), "foot_width_cm": round(geo["foot_width_cm"], 1),
                  "targets": [{"beat": b, "foot": f, "x": round(x, 1), "y": round(y, 1), "slow": b in (3, 7)}
                              for b, (f, x, y) in geo["targets"].items()]},
        "steps": out_steps,
        "cycles": [{"id": c["id"], "epoch": c["epoch"], "t0": c["t0"], "t1": c["t1"],
                    "anchor": [round(float(c["anchor"][0]), 1), round(float(c["anchor"][1]), 1)]}
                   for c in sorted(cyc_info.values(), key=lambda c: c["id"])],
        "track": trk,
        "metrics": metrics,
    }


def spatial_metrics(steps: list[dict], fsteps: list[dict], cycles: dict[int, dict], geo: dict) -> dict[str, Any]:
    by_i = {f["i"]: f for f in fsteps}
    counted = [(s, by_i[k]) for k, s in enumerate(steps) if k in by_i and s.get("timing_type") != "TAP"]

    def mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else None

    lens = {f: mean([fs["len_cm"] for s, fs in counted if s.get("foot") == f]) for f in ("L", "R")}
    # big steps only (beats 1/3/5/7 move the foot by a full reach); replace steps 2/6 are ~0 by design
    big = {f: mean([fs["len_cm"] for s, fs in counted if s.get("foot") == f and s.get("beat_count") in (1, 3, 5, 7)])
           for f in ("L", "R")}
    fwd = [fs.get("fwd_disp_cm") for s, fs in counted if s.get("beat_count") == 1]
    back = [-fs["fwd_disp_cm"] for s, fs in counted if s.get("beat_count") == 5 and fs.get("fwd_disp_cm") is not None]
    fwd_reach, back_reach = mean(fwd), mean(back)
    sym = None
    if fwd_reach is not None and back_reach is not None and max(fwd_reach, back_reach) > 1:
        sym = 100.0 * max(0.0, min(fwd_reach, back_reach)) / max(fwd_reach, back_reach)
    errs = [fs["err_cm"] for s, fs in counted if fs["err_cm"] is not None]
    by_beat = {}
    for b in (1, 2, 3, 5, 6, 7):
        v = mean([fs["err_cm"] for s, fs in counted if s.get("beat_count") == b])
        if v is not None:
            by_beat[b] = round(v, 1)
    # drift of the cycle anchor within each block
    drift = []
    by_epoch: dict[int, list] = {}
    for c in sorted(cycles.values(), key=lambda c: c["id"]):
        by_epoch.setdefault(c["epoch"], []).append(c)
    for eid, cs in by_epoch.items():
        if len(cs) >= 2:
            d = cs[-1]["anchor"] - cs[0]["anchor"]
            drift.append({"epoch": eid, "dx_cm": round(float(d[0]), 1), "dy_cm": round(float(d[1]), 1),
                          "dist_cm": round(float(np.hypot(*d)), 1), "bars": len(cs) - 1})
    foot1 = next((s.get("foot") for s, _ in counted if s.get("beat_count") == 1), "L")
    foot5 = next((s.get("foot") for s, _ in counted if s.get("beat_count") == 5), "R")
    r = lambda v: None if v is None else round(v, 1)  # noqa: E731
    return {
        "step_len_L_cm": r(lens["L"]), "step_len_R_cm": r(lens["R"]),
        "big_step_len_L_cm": r(big["L"]), "big_step_len_R_cm": r(big["R"]),
        "fwd_reach_cm": r(fwd_reach), "back_reach_cm": r(back_reach), "beat1_foot": foot1, "beat5_foot": foot5,
        "fwd_back_symmetry_pct": r(sym),
        "placement_err_cm": r(mean(errs)), "placement_err_by_beat_cm": by_beat,
        "ideal_reach_cm": round(geo["reach_cm"], 1),
        "drift": drift,
    }


FOOT_WORD = {"L": "left", "R": "right"}


def spatial_notes(m: dict[str, Any], max_notes: int = 2) -> list[dict[str, str]]:
    """Up to two concise spatial coaching bullets (kind: ok | warn | info)."""
    notes = []
    worst = max(m.get("drift") or [], key=lambda d: d["dist_cm"], default=None)
    if worst and worst["dist_cm"] >= 15:
        dx, dy = worst["dx_cm"], worst["dy_cm"]
        word = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("forward" if dy > 0 else "backward")
        notes.append({"kind": "warn", "text": f"Dancer drifts {worst['dist_cm']:.0f} cm {word} over {worst['bars']} bars "
                                              f"(block {worst['epoch'] + 1}). Return to the same centre on 4 and 8."})
    fr, br = m.get("fwd_reach_cm"), m.get("back_reach_cm")
    if fr is not None and br is not None:
        f1, f5 = FOOT_WORD.get(m.get("beat1_foot"), "left"), FOOT_WORD.get(m.get("beat5_foot"), "right")
        diff = fr - br
        if abs(diff) >= 5:
            if diff > 0:
                txt = (f"{f5.capitalize()} backward step (beat 5) is {diff:.0f} cm shorter than {f1} forward step "
                       f"(beat 1): {br:.0f} vs {fr:.0f} cm.")
            else:
                txt = (f"{f1.capitalize()} forward step (beat 1) is {-diff:.0f} cm shorter than {f5} backward step "
                       f"(beat 5): {fr:.0f} vs {br:.0f} cm.")
            notes.append({"kind": "warn", "text": txt})
        else:
            notes.append({"kind": "ok", "text": f"Forward (beat 1) and backward (beat 5) steps are balanced: "
                                                f"{fr:.0f} vs {br:.0f} cm."})
    pe = m.get("placement_err_cm")
    if len(notes) < max_notes and pe is not None and pe >= 10 and m.get("placement_err_by_beat_cm"):
        b, v = max(m["placement_err_by_beat_cm"].items(), key=lambda kv: kv[1])
        notes.append({"kind": "info", "text": f"Steps land {pe:.0f} cm from the ideal footprint on average; "
                                              f"largest on beat {b} ({v:.0f} cm)."})
    return notes[:max_notes]
