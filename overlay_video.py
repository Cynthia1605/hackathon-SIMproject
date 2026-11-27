"""Render the MediaPipe skeleton plus a live rhythm HUD onto cam1.mp4.

    python overlay_video.py --config sessions.json --subject subject_01 [--max-seconds 20]

Output: ``outputs/<subject_id>/overlay_cam1.mp4``. The overlay is first written with
OpenCV (``avc1``, falling back to ``mp4v``). When ffmpeg is available (on PATH or
bundled with imageio-ffmpeg), it is then re-encoded to browser-friendly H.264
(yuv420p, faststart) and the original cam1 audio is muxed back in.

HUD text is ASCII only: OpenCV's Hershey fonts cannot draw accented characters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
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

# BGR colours
NEON_LEFT = (255, 255, 0)  # cyan
NEON_RIGHT = (255, 0, 255)  # magenta
NEON_HIPS = (0, 255, 255)  # yellow
UPPER = (235, 235, 235)
GREEN = (80, 200, 60)
YELLOW = (0, 210, 255)
RED = (60, 60, 230)
GREY = (140, 140, 140)

ON_TIME_MS = 45.0
SLIGHT_MS = 90.0


# --------------------------------------------------------------------------- #
# Pure rhythm-state logic (shared with dashboard.py)
# --------------------------------------------------------------------------- #
def classify_offset(offset_ms: float | None) -> tuple[str, str]:
    """Return (category, Spanish ASCII label) for a step's beat offset."""
    if offset_ms is None or not np.isfinite(offset_ms):
        return "none", "SIN REFERENCIA"
    a = abs(offset_ms)
    if a <= ON_TIME_MS:
        return "on_time", "A TIEMPO"
    if a <= SLIGHT_MS:
        return "slight", "LEVEMENTE ADELANTADO" if offset_ms < 0 else "LEVEMENTE ATRASADO"
    return "off", "FUERA DE TEMPO"


CATEGORY_BGR = {"on_time": GREEN, "slight": YELLOW, "off": RED, "none": GREY}


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
    """Everything needed to answer 'what is the rhythm state at cam1 time t'."""

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
        grid = np.asarray(music.get("aligned_beat_times_sec") or [], dtype=np.float64)
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

    def beat_count_at(self, t: float) -> int | None:
        g = self.beat_grid
        if len(g) == 0:
            return None
        k = int(np.searchsorted(g, t, side="right")) - 1
        if k < 0 or t - g[k] > 1.5 * self.beat_period:
            return None
        ep = self.epoch_at(t)
        phase = self.phase_by_epoch.get(ep) if ep is not None else None
        if phase is None:
            phase = self.phase_global
        if phase is None:
            return None
        return (k - phase) % 8 + 1

    def beats_with_counts(self) -> list[dict[str, Any]]:
        out = []
        for k, bt in enumerate(self.beat_grid):
            c = self.beat_count_at(float(bt) + 1e-6)
            out.append({"t": float(bt), "count": c})
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

    def state_at(self, t: float) -> dict[str, Any]:
        ep = self.epoch_at(t)
        step = self.last_step(t)
        count = self.beat_count_at(t)
        if count is None and step is not None and step.get("beat_count"):
            count = int(step["beat_count"])
        cat, label = classify_offset(step.get("beat_offset_ms") if step else None)
        return {
            "t": t,
            "active": ep is not None,
            "status": "BAILANDO / ACTIVO" if ep is not None else "EN PAUSA / IDLE",
            "count": count,
            "last_step": step,
            "feedback_category": cat if step else None,
            "feedback_label": label if step else None,
            "step_flash": step is not None and t - step["timestamp_sec"] <= 0.2,
        }


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _side_colour(j: int) -> tuple[int, int, int]:
    if j in (23, 24):
        return NEON_HIPS
    return NEON_LEFT if j % 2 == 1 else NEON_RIGHT


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, vis_min: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """Draw the 33-landmark skeleton in place. ``kp`` is (33, 3) [x_px, y_px, visibility]."""
    import cv2

    ok = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 2] >= vis_min)
    pts = np.nan_to_num(kp[:, :2]).astype(int)
    thin, thick = max(1, int(2 * scale)), max(2, int(6 * scale))
    for a, b in POSE_CONNECTIONS:
        if not (ok[a] and ok[b]):
            continue
        lower = a in LOWER_BODY and b in LOWER_BODY
        if lower:
            col = NEON_HIPS if {a, b} == {23, 24} else _side_colour(b)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 0, 0), thick + 4, cv2.LINE_AA)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, thick, cv2.LINE_AA)
        else:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (30, 30, 30), thin + 2, cv2.LINE_AA)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), UPPER, thin, cv2.LINE_AA)
    for j in range(len(kp)):
        if not ok[j]:
            continue
        if j in LOWER_BODY:
            r = max(4, int(9 * scale))
            cv2.circle(frame, tuple(pts[j]), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[j]), r, _side_colour(j), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(pts[j]), max(2, int(3 * scale)), UPPER, -1, cv2.LINE_AA)
    return frame


def _text(frame, s, org, scale, colour, thick=2):
    import cv2

    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _badge(frame, s, x, y, scale, bg, fg=(255, 255, 255), thick=2):
    import cv2

    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale / 0.7)
    cv2.rectangle(frame, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), bg, -1)
    cv2.putText(frame, s, (x + pad, y + pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
    return y + th + base + 2 * pad


def draw_hud(frame: np.ndarray, state: dict[str, Any], kp: np.ndarray | None = None,
             footer: str = "") -> np.ndarray:
    """Translucent rhythm HUD in the top-left corner, drawn in place."""
    import cv2

    h, w = frame.shape[:2]
    s = h / 720.0
    fs = 0.75 * s
    th = max(1, int(round(2 * s)))
    x0, y0 = int(16 * s), int(16 * s)
    pw, ph = int(min(w - 2 * x0, 560 * s)), int(230 * s)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    x, y = x0 + int(12 * s), y0 + int(12 * s)
    y = _badge(frame, state["status"], x, y, fs, GREEN if state["active"] else GREY, thick=th) + int(10 * s)

    # 8-count row: 1 - 2 - 3 - (4) - 5 - 6 - 7 - (8)
    tokens = ["1", "2", "3", "(4)", "5", "6", "7", "(8)"]
    cur = state.get("count")
    cx = x
    base_y = y + int(30 * s)
    for i, tok in enumerate(tokens):
        n = i + 1
        is_hold = n in (4, 8)
        big = 1.0 * s
        (tw, tht), _ = cv2.getTextSize(tok, cv2.FONT_HERSHEY_SIMPLEX, big, th + 1)
        if cur == n:
            cv2.rectangle(frame, (cx - int(5 * s), base_y - tht - int(7 * s)),
                          (cx + tw + int(5 * s), base_y + int(7 * s)),
                          (0, 200, 255) if not is_hold else (120, 120, 120), -1)
            col = (0, 0, 0)
        else:
            col = (110, 110, 110) if is_hold else ((235, 235, 235) if state["active"] else (150, 150, 150))
        cv2.putText(frame, tok, (cx, base_y), cv2.FONT_HERSHEY_SIMPLEX, big, col, th + 1, cv2.LINE_AA)
        cx += tw + int(8 * s)
        if i < len(tokens) - 1:
            cv2.putText(frame, "-", (cx, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (120, 120, 120), th,
                        cv2.LINE_AA)
            cx += int(18 * s)
    y = base_y + int(18 * s)

    step = state.get("last_step")
    if step is not None:
        cat = state["feedback_category"]
        y = _badge(frame, state["feedback_label"], x, y, fs, CATEGORY_BGR[cat], thick=th) + int(6 * s)
        off = step.get("beat_offset_ms")
        detail = f"Ultimo paso {step.get('label', '')}"
        if off is not None:
            detail += f"  {off:+.0f} ms"
        _text(frame, detail, (x, y + int(22 * s)), 0.62 * s, (235, 235, 235), th)
        y += int(30 * s)
    else:
        _text(frame, "Ultimo paso: -", (x, y + int(22 * s)), 0.62 * s, (170, 170, 170), th)
        y += int(30 * s)
    if footer:
        _text(frame, footer, (x, y0 + ph - int(10 * s)), 0.5 * s, (200, 200, 200), max(1, th - 1))

    # footfall flash ring on the stepping foot
    if kp is not None and step is not None and state.get("step_flash"):
        j = L.LEFT_ANKLE if step.get("foot") == "L" else L.RIGHT_ANKLE
        if np.isfinite(kp[j, :2]).all():
            c = tuple(int(v) for v in kp[j, :2])
            cv2.circle(frame, c, int(28 * s), CATEGORY_BGR[state["feedback_category"]], max(2, int(4 * s)),
                       cv2.LINE_AA)
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
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22",
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
        sync_txt = f"sync {sync_info.get('method', '?')} {int(sync_info.get('frame_offset', 0)):+d}f"
    bpm_txt = f"BPM {timeline.bpm:.0f}" if timeline.bpm else "BPM -"
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
            render_frame(frame, kp, timeline, t, f"t={t:6.2f}s  {bpm_txt}  {sync_txt}")
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
