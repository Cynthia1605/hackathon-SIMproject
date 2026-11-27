"""End-to-end batch runner.

pose_2d, sync, music_analyzer, calibrate_3d, filters, epoch_segmenter,
salsa_rhythm, visualize, then overlay_video (skip with --no-overlay).

    python main.py --config sessions.json [--session subject_01] [--force-pose] [--no-overlay]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import calibrate_3d
import epoch_segmenter
import filters
import music_analyzer
import pose_2d
import salsa_rhythm
import sync
import visualize

log = logging.getLogger("salsa")

MODULE_DEFAULTS = {
    "sync": sync.DEFAULT_SYNC_PARAMS,
    "music": music_analyzer.DEFAULT_MUSIC_PARAMS,
    "calibration": calibrate_3d.DEFAULT_CALIBRATION_PARAMS,
    "filters": filters.DEFAULT_FILTER_PARAMS,
    "segmenter": epoch_segmenter.DEFAULT_SEGMENTER_PARAMS,
    "rhythm": salsa_rhythm.DEFAULT_RHYTHM_PARAMS,
}
REQUIRED_KEYS = ("id", "cam1", "cam2", "song_mp3", "height_meters")
PATH_KEYS = ("cam1", "cam2", "song_mp3")
CSV_COLUMNS = ["subject_id", "epoch_id", "cam1_frame", "timestamp_sec", "nearest_music_beat_sec",
               "beat_offset_ms", "foot", "timing_type", "beat_count", "step_length_cm"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load sessions.json: resolve paths against its directory and merge params.

    Each session gets ``params``: module defaults, then the file's ``defaults``,
    then the session's own ``params``.
    """
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = json.load(fh)
    root = path.parent

    def res(p):
        p = Path(p)
        return str(p if p.is_absolute() else (root / p))

    base_params = deep_merge(MODULE_DEFAULTS, cfg.get("defaults", {}))
    out = {
        "config_path": str(path),
        "output_root": res(cfg.get("output_root", "outputs")),
        "cache_dir": res(cfg.get("cache_dir", "cache")),
        "model_path": res(cfg.get("model_path", "models/pose_landmarker_heavy.task")),
        "sessions": [],
    }
    ids = set()
    for s in cfg.get("sessions", []):
        missing = [k for k in REQUIRED_KEYS if s.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Session {s.get('id', '?')} missing required keys: {missing}")
        if s["id"] in ids:
            raise ValueError(f"Duplicate session id {s['id']}")
        ids.add(s["id"])
        sess = dict(s)
        for k in PATH_KEYS:
            sess[k] = res(s[k])
        sess["height_meters"] = float(s["height_meters"])
        sess["overrides"] = {"frame_offset": None, "mp3_offset_sec": None, **(s.get("overrides") or {})}
        sess["params"] = deep_merge(base_params, s.get("params"))
        out["sessions"].append(sess)
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def analyze_subject(
    subject_id: str,
    pose1,
    pose2,
    music,
    height_m: float,
    params: dict[str, Any],
    out_dir: str | Path,
    video1: str | None = None,
    video2: str | None = None,
    manual_frame_offset: int | None = None,
) -> dict[str, Any]:
    """Everything after 2D pose extraction and music alignment. Writes all outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(pose1.fps)

    sync_res = sync.synchronize(pose1, pose2, video1, video2, manual_frame_offset, params["sync"])
    (out_dir / "sync.json").write_text(json.dumps(sync_res.to_dict(), indent=2, default=_json_default))
    log.info("[%s] sync: offset %+d frames via %s (conf %.2f)", subject_id, sync_res.frame_offset,
             sync_res.method, sync_res.confidence)

    pts, calib, cam1_frames = calibrate_3d.calibrate_session(pose1, pose2, sync_res.frame_offset, height_m,
                                                             params["calibration"])
    log.info("[%s] calibration: scale %.4f, reprojection %.2f px, %d/%d inliers", subject_id, calib.scale,
             calib.reprojection_error_px, calib.n_inliers, calib.n_correspondences)
    pts_f = filters.filter_points3d(pts, fps, params["filters"])
    np.savez_compressed(out_dir / "points3d.npz", points3d=pts_f.astype(np.float32), cam1_frames=cam1_frames,
                        fps=fps)

    epochs, diag = epoch_segmenter.segment_epochs(pts_f, fps, cam1_frames, params["segmenter"])
    log.info("[%s] %d active epochs: %s", subject_id, len(epochs),
             ", ".join(f"{e.start_sec:.1f}-{e.end_sec:.1f}s" for e in epochs))
    rhythm = salsa_rhythm.analyze_rhythm(pts_f, fps, cam1_frames, epochs, music, params["rhythm"])

    import footwork

    step_dicts = [st.to_dict() for st in rhythm.steps]
    fw = footwork.compute_footwork(pts_f, cam1_frames, fps, {"steps": step_dicts, "epochs": [e.to_dict() for e in epochs]},
                                   height_m)
    if fw.get("available"):
        order = sorted(range(len(step_dicts)), key=lambda k: step_dicts[k]["timestamp_sec"])
        for f in fw["steps"]:
            sd = step_dicts[order[f["i"]]]
            sd.update(floor_x_cm=f["x"], floor_y_cm=f["y"], placement_err_cm=f["err_cm"], floor_cycle=f["cycle"])

    with open(out_dir / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for s in rhythm.steps:
            row = {k: getattr(s, k, None) for k in CSV_COLUMNS if k != "subject_id"}
            row["subject_id"] = subject_id
            for k in ("timestamp_sec", "nearest_music_beat_sec"):
                if row[k] is not None:
                    row[k] = f"{row[k]:.4f}"
            for k in ("beat_offset_ms", "step_length_cm"):
                if row[k] is not None:
                    row[k] = f"{row[k]:.1f}"
            w.writerow(row)

    all_warnings = (sync_res.warnings + list(getattr(music, "warnings", [])) + calib.warnings
                    + rhythm.warnings)
    report = {
        "subject_id": subject_id,
        "height_meters": height_m,
        "fps": fps,
        "sync": sync_res.to_dict(),
        "music": music.to_dict(),
        "calibration": calib.to_dict(),
        "segmentation": diag.to_dict(),
        "epochs": [
            {**e.to_dict(), "summary": next((x for x in rhythm.epoch_summaries if x["epoch_id"] == e.epoch_id), {})}
            for e in epochs
        ],
        "steps": step_dicts,
        # top-down footwork (cm, per-block floor frame: +y = facing, +x = dancer's right)
        "footwork": fw,
        # beat grid actually used for scoring (tempo-octave corrected), cam1 timeline
        "beat_grid_sec": np.asarray(rhythm.beat_grid_sec, dtype=np.float64).tolist(),
        "summary": rhythm.summary,
        "warnings": all_warnings,
        "params": params,
    }
    (out_dir / "epochs_and_steps.json").write_text(json.dumps(report, indent=2, default=_json_default))

    visualize.plot_rhythm_analysis(out_dir / "rhythm_analysis.png", subject_id, fps, cam1_frames, epochs, diag,
                                   rhythm, sync_res.clap_frames_cam1)
    return {
        "subject_id": subject_id,
        "status": "ok",
        "bpm": music.bpm,
        "music_method": music.method,
        "sync_method": sync_res.method,
        "frame_offset": sync_res.frame_offset,
        "n_epochs": len(epochs),
        "n_steps": rhythm.summary["n_steps"],
        "n_taps": rhythm.summary["n_taps"],
        "mean_abs_offset_ms": rhythm.summary["mean_abs_offset_ms"],
        "mean_offset_ms": rhythm.summary["mean_offset_ms"],
        "qqs_accuracy": rhythm.summary["qqs_pattern_accuracy"],
        "out_dir": str(out_dir),
        "warnings": all_warnings,
    }


def run_session(sess: dict[str, Any], cfg: dict[str, Any], force_pose: bool = False,
                render_overlay: bool = True) -> dict[str, Any]:
    sid = sess["id"]
    cache = Path(cfg["cache_dir"]) / sid
    out_dir = Path(cfg["output_root"]) / sid
    params = sess["params"]
    log.info("[%s] 2D pose", sid)
    pose1 = pose_2d.load_or_extract(sess["cam1"], cache / "cam1_pose2d.npz", cfg["model_path"], force_pose)
    pose2 = pose_2d.load_or_extract(sess["cam2"], cache / "cam2_pose2d.npz", cfg["model_path"], force_pose)
    log.info("[%s] music", sid)
    music = music_analyzer.align_music(sess["song_mp3"], sess["cam1"], sess["overrides"]["mp3_offset_sec"],
                                       params["music"])
    log.info("[%s] music: %.1f BPM, mp3 offset %.3fs via %s (conf %.2f)", sid, music.bpm, music.mp3_offset_sec,
             music.method, music.confidence)
    row = analyze_subject(sid, pose1, pose2, music, sess["height_meters"], params, out_dir, sess["cam1"],
                          sess["cam2"], sess["overrides"]["frame_offset"])
    row["overlay"] = None
    if render_overlay:
        if not Path(sess["cam1"]).is_file():
            log.info("[%s] overlay skipped: %s not found", sid, sess["cam1"])
        else:
            import overlay_video

            log.info("[%s] rendering overlay video", sid)
            try:
                info = overlay_video.render_for_session(sess, cfg)
                row["overlay"] = info["path"]
                row["warnings"] = row["warnings"] + info["warnings"]
            except Exception as exc:  # the analysis results are still valid
                row["warnings"] = row["warnings"] + [f"overlay rendering failed: {exc}"]
    return row


def format_summary(rows: list[dict[str, Any]]) -> str:
    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    head = ["Subject", "Status", "BPM", "Sync", "Epochs", "Steps", "Taps", "Mean|off| ms", "Bias ms", "QQS acc"]
    lines = []
    for r in rows:
        if r.get("status") != "ok":
            lines.append([r["subject_id"], "FAILED", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        lines.append([r["subject_id"], "ok", f(r["bpm"], ".1f"), r["sync_method"], str(r["n_epochs"]),
                      str(r["n_steps"]), str(r["n_taps"]), f(r["mean_abs_offset_ms"], ".1f"),
                      f(r["mean_offset_ms"], "+.1f"),
                      "-" if r["qqs_accuracy"] is None else f"{100 * r['qqs_accuracy']:.0f}%"])
    widths = [max(len(h), *(len(l[i]) for l in lines)) if lines else len(h) for i, h in enumerate(head)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt_row = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    return "\n".join([sep, fmt_row(head), sep, *map(fmt_row, lines), sep])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dual-camera 3D salsa rhythm pipeline")
    ap.add_argument("--config", default="sessions.json")
    ap.add_argument("--session", action="append", help="only process this session id (repeatable)")
    ap.add_argument("--force-pose", action="store_true", help="ignore cached 2D pose .npz files")
    ap.add_argument("--no-overlay", action="store_true", help="skip rendering overlay_cam1.mp4")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    sessions = cfg["sessions"]
    if args.session:
        sessions = [s for s in sessions if s["id"] in set(args.session)]
        unknown = set(args.session) - {s["id"] for s in sessions}
        if unknown:
            log.error("Unknown session id(s): %s", ", ".join(sorted(unknown)))
            return 2
    rows = []
    for sess in sessions:
        try:
            rows.append(run_session(sess, cfg, args.force_pose, not args.no_overlay))
        except Exception as exc:  # keep going with the other subjects
            log.error("[%s] FAILED: %s", sess["id"], exc)
            log.debug(traceback.format_exc())
            rows.append({"subject_id": sess["id"], "status": "failed", "error": str(exc)})
    print()
    print(format_summary(rows))
    for r in rows:
        for w in r.get("warnings", []):
            print(f"  [{r['subject_id']}] warning: {w}")
        if r.get("status") != "ok":
            print(f"  [{r['subject_id']}] error: {r.get('error')}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
