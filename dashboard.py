"""Local rhythm dashboard (English UI), served with the standard library.

    python dashboard.py --config sessions.json --subject subject_01

A single-screen widescreen page with no outer scrolling:
* Left: the overlay video with transport, speed and audio-sync calibration, and the
  step error timeline.
* Right: tempo score and KPIs, coaching diagnosis, and a filterable step / error log.

Music sync: the song is served as a WAV copy decoded by ffmpeg when available, so
seeks are sample-accurate. The browser keeps ``audio.currentTime`` locked to
``video.currentTime - mp3_offset_sec`` with a requestAnimationFrame loop.

The page can save a calibrated offset back to ``sessions.json``
(``overrides.mp3_offset_sec``) via ``POST /api/save-offset``.
Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

from overlay_video import ON_TIME_MS, SLIGHT_MS, RhythmTimeline, analysis_beat_grid, classify_offset, find_ffmpeg

log = logging.getLogger(__name__)

BIAS_MS = 25.0  # |mean offset| above this counts as a tendency to rush or drag
MAX_OFFSET_S = 600.0
STATUS_EN = {
    "on_time": "On Time",
    "early": "Early",
    "late": "Late",
    "off_early": "Early (off-beat)",
    "off_late": "Late (off-beat)",
    "none": "No beat ref",
}
TIMING_EN = {"QUICK": "Quick", "SLOW": "Slow", "TAP": "Tap"}


# --------------------------------------------------------------------------- #
# Scoring, verdict and coaching (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def step_category(offset_ms: float | None) -> str:
    return classify_offset(offset_ms)[0]


def timing_credit(abs_ms: float, half_beat_ms: float) -> float:
    """1.0 within ±45 ms, 0.5 at ±90 ms (linear), then down to 0 at half a beat."""
    if abs_ms <= ON_TIME_MS:
        return 1.0
    if abs_ms <= SLIGHT_MS:
        return 1.0 - 0.5 * (abs_ms - ON_TIME_MS) / (SLIGHT_MS - ON_TIME_MS)
    if half_beat_ms <= SLIGHT_MS:
        return 0.0
    return max(0.0, 0.5 * (1.0 - (abs_ms - SLIGHT_MS) / (half_beat_ms - SLIGHT_MS)))


def tempo_score(offsets_ms: list[float], half_beat_ms: float, qqs_accuracy: float | None) -> float | None:
    """0-100: 80% timing credit + 20% quick-quick-slow pattern accuracy (timing only if unknown)."""
    if not offsets_ms:
        return None
    timing = float(np.mean([timing_credit(abs(o), half_beat_ms) for o in offsets_ms]))
    if qqs_accuracy is None:
        return 100.0 * timing
    return 100.0 * (0.8 * timing + 0.2 * float(qqs_accuracy))


def verdict(score: float | None, bias_ms: float | None) -> dict[str, str]:
    if score is None:
        return {"level": "none", "text": "No steps detected",
                "detail": "No counted steps with a music beat reference were found."}
    early = bias_ms is not None and bias_ms < -BIAS_MS
    late = bias_ms is not None and bias_ms > BIAS_MS
    if score >= 85 and not (early or late):
        return {"level": "excellent", "text": "Excellent rhythm — on tempo",
                "detail": "Steps land on the music's pulse consistently."}
    if score >= 70:
        if early:
            return {"level": "good", "text": "Good rhythm, tends to rush",
                    "detail": "Mostly on tempo, but steps tend to land before the beat."}
        if late:
            return {"level": "good", "text": "Good rhythm, tends to drag",
                    "detail": "Mostly on tempo, but steps tend to land after the beat."}
        return {"level": "good", "text": "Good rhythm",
                "detail": "Most steps land on time, with some irregular ones."}
    if score >= 50:
        if early:
            return {"level": "fair", "text": "Tends to rush (early)",
                    "detail": "Many steps land before the beat. Listen for the percussion and wait for it."}
        if late:
            return {"level": "fair", "text": "Tends to drag (late)",
                    "detail": "Many steps land after the beat. Prepare the weight change earlier."}
        return {"level": "fair", "text": "Inconsistent timing",
                "detail": "Some steps are on time and others are not. Work on regularity."}
    return {"level": "poor", "text": "Off tempo",
            "detail": "Most steps do not line up with the music's pulse."}


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _ms(v: float) -> str:
    return f"{v:+.0f} ms"


def coaching_notes(steps: list[dict], period_s: float, n_taps: int = 0) -> list[dict[str, str]]:
    """English coaching notes derived from step timing patterns. kind: ok | warn | info."""
    notes: list[dict[str, str]] = []
    c = [s for s in steps if s.get("timing_type") != "TAP" and s.get("beat_offset_ms") is not None]
    if not c:
        return [{"kind": "info", "text": "Not enough steps with a beat reference to give feedback."}]

    def offs(pred):
        return [s["beat_offset_ms"] for s in c if pred(s)]

    # Strong beats 1 and 5 (judged on |offset| so +/- errors cannot cancel out)
    strong = offs(lambda s: s.get("beat_count") in (1, 5))
    if strong:
        m, ma = _mean(strong), _mean([abs(o) for o in strong])
        if ma <= ON_TIME_MS:
            notes.append({"kind": "ok", "text": f"Beats 1 and 5 land on time (mean {_ms(m)}). Great anchor."})
        elif abs(m) <= BIAS_MS:
            notes.append({"kind": "warn", "text": f"Beats 1 and 5 are irregular — sometimes early, sometimes late "
                                                  f"(mean error {ma:.0f} ms). Mark the 1 clearly."})
        elif m < 0:
            notes.append({"kind": "warn", "text": f"Beats 1 and 5 come early (mean {_ms(m)}). "
                                                  "Wait for the downbeat before stepping."})
        else:
            notes.append({"kind": "warn", "text": f"Beats 1 and 5 come late (mean {_ms(m)}). "
                                                  "Shift your weight earlier so the step lands on 1 and 5."})

    # Slow steps 3 and 7 (and the 4/8 hold)
    slow = [s for s in c if s.get("timing_type") == "SLOW"]
    if slow:
        m = _mean([s["beat_offset_ms"] for s in slow])
        ma = _mean([abs(s["beat_offset_ms"]) for s in slow])
        holds = [s["interval_sec"] for s in slow if s.get("interval_sec")]
        short_hold = bool(holds) and _mean(holds) < 1.8 * period_s
        if ma > SLIGHT_MS and abs(m) <= BIAS_MS and not short_hold:
            notes.append({"kind": "warn", "text": f"Slow steps 3 and 7 miss the pulse (mean error {ma:.0f} ms). "
                                                  "Count 1-2-3, 5-6-7 out loud with the music."})
        elif m < -BIAS_MS or short_hold:
            extra = f"; the hold lasts {_mean(holds):.2f}s of {2 * period_s:.2f}s" if short_hold else ""
            notes.append({"kind": "warn", "text": f"Slow steps 3 and 7 rush (mean {_ms(m)}{extra}). "
                                                  "Hold through beats 4 and 8 before stepping again."})
        elif m > BIAS_MS:
            notes.append({"kind": "warn", "text": f"Slow steps 3 and 7 land late (mean {_ms(m)}). "
                                                  "Close the step on time and use 4 and 8 to pause."})
        else:
            notes.append({"kind": "ok", "text": f"Slow steps 3 and 7 respect the 4/8 hold (mean {_ms(m)})."})

    # Left / right symmetry
    lo = [abs(o) for o in offs(lambda s: s.get("foot") == "L")]
    ro = [abs(o) for o in offs(lambda s: s.get("foot") == "R")]
    if lo and ro:
        ml, mr = _mean(lo), _mean(ro)
        if abs(ml - mr) > 20:
            worse, better = ("Left", "right") if ml > mr else ("Right", "left")
            notes.append({"kind": "warn", "text": f"{worse} foot is less precise than the {better} "
                                                  f"({max(ml, mr):.0f} ms vs {min(ml, mr):.0f} ms mean error)."})
        elif max(ml, mr) > ON_TIME_MS:
            notes.append({"kind": "info", "text": f"Both feet are off by a similar amount (L {ml:.0f} ms, "
                                                  f"R {mr:.0f} ms): this is overall tempo, not one foot."})
        else:
            notes.append({"kind": "ok", "text": f"Good left/right symmetry: L {ml:.0f} ms, R {mr:.0f} ms mean error."})
    ll = [s["step_length_cm"] for s in c if s.get("foot") == "L" and s.get("step_length_cm")]
    rl = [s["step_length_cm"] for s in c if s.get("foot") == "R" and s.get("step_length_cm")]
    if ll and rl:
        a, b = _mean(ll), _mean(rl)
        if max(a, b) > 0 and abs(a - b) / max(a, b) > 0.25:
            big = "Left" if a > b else "Right"
            notes.append({"kind": "info", "text": f"{big}-foot steps are longer ({max(a, b):.0f} cm vs "
                                                  f"{min(a, b):.0f} cm). Smaller steps make it easier to stay on time."})

    # Drift between dance blocks / over time
    by_epoch: dict[int, list[float]] = {}
    for s in c:
        by_epoch.setdefault(s.get("epoch_id", 0), []).append(s["beat_offset_ms"])
    first = last = None
    if len(by_epoch) >= 2:
        ids = sorted(by_epoch)
        first, last, where = _mean(by_epoch[ids[0]]), _mean(by_epoch[ids[-1]]), "from the first to the last dance block"
    elif len(c) >= 12:
        half = len(c) // 2
        first = _mean([s["beat_offset_ms"] for s in c[:half]])
        last = _mean([s["beat_offset_ms"] for s in c[half:]])
        where = "over the course of the dance"
    if first is not None and abs(last - first) > 30:
        word = "later (dragging)" if last > first else "earlier (rushing)"
        notes.append({"kind": "warn", "text": f"Timing drifts {word} {where} ({_ms(first)} → {_ms(last)}). "
                                              "Keep listening to the pulse as you tire."})

    std = float(np.std([s["beat_offset_ms"] for s in c]))
    if std > 60:
        notes.append({"kind": "warn", "text": f"Low consistency (spread {std:.0f} ms). "
                                              "Practise with the metronome at 0.75x speed."})
    if n_taps:
        notes.append({"kind": "info", "text": f"{n_taps} same-foot taps were detected and not counted as steps. "
                                              "Check for incomplete weight changes."})
    return notes


def shift_report(report: dict[str, Any], new_offset: float) -> dict[str, Any]:
    """Copy of the report with the music placed at ``new_offset``.

    The beat grid is shifted and each step's offset is recomputed against the
    nearest shifted beat.
    """
    rep = copy.deepcopy(report)
    music = rep.setdefault("music", {})
    old = float(music.get("mp3_offset_sec") or 0.0)
    d = float(new_offset) - old
    grid = analysis_beat_grid(report) + d  # the (tempo-corrected) grid the steps are scored against
    music["aligned_beat_times_sec"] = (np.asarray(music.get("aligned_beat_times_sec") or [], dtype=np.float64) + d).tolist()
    rep["beat_grid_sec"] = grid.tolist()
    music["mp3_offset_sec"] = float(new_offset)
    period = float((rep.get("summary") or {}).get("beat_period_sec") or music.get("beat_period_sec") or 0.5)
    for s in rep.get("steps", []):
        if len(grid) == 0 or s.get("beat_offset_ms") is None:
            continue
        t = s["timestamp_sec"]
        if t < grid[0] - period / 2 or t > grid[-1] + period / 2:
            s["nearest_music_beat_sec"], s["beat_offset_ms"] = None, None
            continue
        nb = float(grid[np.argmin(np.abs(grid - t))])
        s["nearest_music_beat_sec"], s["beat_offset_ms"] = nb, (t - nb) * 1000.0
    return rep


def build_dashboard_data(report: dict[str, Any], media: dict[str, Any] | None = None,
                         mp3_offset: float | None = None, floor: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the web page needs, derived from epochs_and_steps.json.

    If ``mp3_offset`` is given, the steps are re-scored with the music at that offset.
    ``floor`` is the top-down footwork data from ``footwork.compute_footwork``. It does
    not depend on the music offset.
    """
    analyzed_offset = float((report.get("music") or {}).get("mp3_offset_sec") or 0.0)
    if mp3_offset is not None and abs(float(mp3_offset) - analyzed_offset) > 1e-9:
        report = shift_report(report, float(mp3_offset))
    music = report.get("music", {}) or {}
    summary = report.get("summary", {}) or {}
    period = float(summary.get("beat_period_sec") or music.get("beat_period_sec") or 0.5)
    half_ms = period * 500.0
    steps_all = sorted(report.get("steps", []), key=lambda s: s["timestamp_sec"])
    counted = [s for s in steps_all if s.get("timing_type") != "TAP"]
    offs = [s["beat_offset_ms"] for s in counted if s.get("beat_offset_ms") is not None]
    qqs = summary.get("qqs_pattern_accuracy")
    score = tempo_score(offs, half_ms, qqs)
    bias = _mean(offs) if offs else None
    n_taps = len(steps_all) - len(counted)

    warnings = list(report.get("warnings", []))
    if len(offs) >= 8:
        # |offset| clusters tightly near half a beat: the beat grid is probably on the
        # off-beat (beat-phase error), not the dancer. Flag it; never auto-correct.
        absd = np.abs(np.asarray(offs, dtype=np.float64))
        if absd.mean() > 0.3 * period * 1000 and absd.std() < 0.15 * period * 1000:
            warnings.append(
                f"Steps land very regularly ~{absd.mean():.0f} ms from the beat (almost half a beat). "
                "The song's beat grid is probably on the off-beat: check with the metronome, "
                "then use 'Tap Beat 1' or the offset controls and save.")

    timeline = RhythmTimeline.from_report(report)
    steps_out = []
    for i, s in enumerate(steps_all):
        tap = s.get("timing_type") == "TAP"
        cat = "none" if tap else step_category(s.get("beat_offset_ms"))
        off = s.get("beat_offset_ms")
        steps_out.append({
            "i": i, "t": s["timestamp_sec"], "foot": s.get("foot"), "label": s.get("label"),
            "timing_type": s.get("timing_type"), "type": TIMING_EN.get(s.get("timing_type"), s.get("timing_type")),
            "beat": s.get("beat_count"), "offset_ms": off, "category": cat,
            "status": "Tap" if tap else STATUS_EN[cat],
            "is_error": (not tap) and off is not None and abs(off) > ON_TIME_MS,
            "epoch_id": s.get("epoch_id"), "step_length_cm": s.get("step_length_cm"),
        })
    coaching = coaching_notes(steps_all, period, n_taps)
    if floor and floor.get("available"):
        import footwork

        # spatial bullets right after the first (headline) timing note, so they are visible without scrolling
        coaching = coaching[:1] + footwork.spatial_notes(floor.get("metrics", {})) + coaching[1:]
        by_i = {f["i"]: f for f in floor.get("steps", [])}
        for so in steps_out:
            f = by_i.get(so["i"])
            if f:
                so["floor"] = {"x": f["x"], "y": f["y"], "err_cm": f["err_cm"], "len_cm": f["len_cm"], "cycle": f["cycle"]}
    epochs = [{"id": e["epoch_id"], "start": e["start_sec"], "end": e["end_sec"]} for e in report.get("epochs", [])]
    t_end = max([e["end"] for e in epochs] + [s["t"] for s in steps_out] + [0.0])
    n_err = sum(s["is_error"] for s in steps_out)
    return {
        "subject_id": report.get("subject_id", ""),
        "fps": report.get("fps"),
        "bpm": music.get("bpm"),
        "beat_period_ms": period * 1000.0,
        "mp3_offset_sec": float(music.get("mp3_offset_sec") or 0.0),
        "mp3_offset_analyzed_sec": analyzed_offset,
        "music_method": music.get("method"),
        "music_confidence": music.get("confidence"),
        "score": score,
        "verdict": verdict(score, bias),
        "kpis": {
            "bpm": music.get("bpm"),
            "on_tempo_pct": (100.0 * float(np.mean([abs(o) <= ON_TIME_MS for o in offs]))) if offs else None,
            "mean_offset_ms": bias,
            "mean_abs_offset_ms": _mean([abs(o) for o in offs]) if offs else None,
            "consistency_ms": float(np.std(offs)) if offs else None,
            "qqs_accuracy_pct": (100.0 * qqs) if qqs is not None else None,
            "n_steps": len(counted),
            "n_errors": n_err,
            "n_taps": n_taps,
            "n_blocks": len(epochs),
        },
        "coaching": coaching,
        "steps": steps_out,
        "floor": floor if floor and floor.get("available") else {"available": False,
                                                                  "reason": (floor or {}).get("reason", "no 3D data")},
        "epochs": epochs,
        "beats": timeline.beats_with_counts(),
        "duration": t_end,
        "thresholds": {"on_time_ms": ON_TIME_MS, "slight_ms": SLIGHT_MS},
        "warnings": warnings,
        "media": media or {"video": False, "video_is_overlay": False, "audio": False},
    }


# --------------------------------------------------------------------------- #
# sessions.json persistence
# --------------------------------------------------------------------------- #
def save_offset_to_config(config_path: str | Path, subject_id: str, offset_sec: float) -> dict[str, Any]:
    """Write ``overrides.mp3_offset_sec`` for one session, keeping the rest of the file intact."""
    v = float(offset_sec)
    if not math.isfinite(v) or abs(v) > MAX_OFFSET_S:
        raise ValueError(f"mp3_offset_sec must be a finite number within ±{MAX_OFFSET_S:.0f} s")
    path = Path(config_path)
    cfg = json.loads(path.read_text())
    match = [s for s in cfg.get("sessions", []) if s.get("id") == subject_id]
    if not match:
        raise KeyError(f"session '{subject_id}' not found in {path}")
    overrides = match[0].get("overrides") or {}
    previous = overrides.get("mp3_offset_sec")
    overrides["mp3_offset_sec"] = round(v, 4)
    match[0]["overrides"] = overrides
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)  # atomic
    return {"ok": True, "subject_id": subject_id, "mp3_offset_sec": round(v, 4), "previous": previous,
            "config": str(path)}


# --------------------------------------------------------------------------- #
# Sync-accurate audio
# --------------------------------------------------------------------------- #
def prepare_sync_audio(song: Path, out_dir: Path) -> Path:
    """Decode the song to a PCM WAV next to the outputs (cached).

    Browsers seek WAV sample-accurately, unlike MP3. Falls back to the original song
    if ffmpeg is not available.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        log.warning("ffmpeg not found: serving the original .mp3 (seeking may be less precise)")
        return song
    wav = out_dir / "song_sync.wav"
    if wav.is_file() and wav.stat().st_mtime >= song.stat().st_mtime:
        return wav
    proc = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(song), "-vn", "-ac", "2", "-ar", "44100",
                           "-c:a", "pcm_s16le", str(wav)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not wav.is_file():
        log.warning("Could not decode %s to WAV; serving the original", song)
        return song
    return wav


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def make_handler(report: dict[str, Any], media: dict[str, Any], files: dict[str, Path | None],
                 save_cb: Callable[[float], dict[str, Any]] | None, state: dict[str, Any],
                 floor: dict[str, Any] | None = None):
    page = INDEX_HTML.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj: Any):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: Path):
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            m = RANGE_RE.fullmatch(self.headers.get("Range", "").strip())
            code = 200
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = length
                try:
                    while remaining > 0:
                        chunk = fh.read(min(1 << 16, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser cancelled a range request while seeking

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            parts = urlsplit(self.path)
            route = parts.path
            if route in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if route == "/api/data":
                q = parse_qs(parts.query)
                off = None
                if "mp3_offset" in q:
                    try:
                        off = float(q["mp3_offset"][0])
                        if not math.isfinite(off) or abs(off) > MAX_OFFSET_S:
                            raise ValueError
                    except ValueError:
                        return self._json(400, {"ok": False, "error": "invalid mp3_offset"})
                data = build_dashboard_data(report, dict(media, can_save=save_cb is not None,
                                                         saved_offset=state.get("saved_offset")), off, floor)
                return self._json(200, data)
            if route in ("/media/video", "/media/audio"):
                f = files.get(route.rsplit("/", 1)[-1])
                if f is not None and Path(f).is_file():
                    return self._file(Path(f))
            return self._send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if urlsplit(self.path).path != "/api/save-offset":
                return self._send(404, b"Not found", "text/plain; charset=utf-8")
            # CSRF guard: JSON only (forces a CORS preflight that we never approve) and same-origin.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"ok": False, "error": "Content-Type must be application/json"})
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and urlsplit(origin).netloc != host:
                return self._json(403, {"ok": False, "error": "cross-origin request refused"})
            if save_cb is None:
                return self._json(403, {"ok": False, "error": "saving is disabled"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > 4096:
                    raise ValueError("bad body size")
                body = json.loads(self.rfile.read(n))
                res = save_cb(float(body["mp3_offset_sec"]))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            state["saved_offset"] = res["mp3_offset_sec"]
            return self._json(200, res)

    return Handler


def make_server(report: dict[str, Any], media: dict[str, Any], video: Path | None, audio: Path | None,
                host: str = "127.0.0.1", port: int = 8765,
                save_cb: Callable[[float], dict[str, Any]] | None = None,
                floor: dict[str, Any] | None = None) -> ThreadingHTTPServer:
    mimetypes.add_type("audio/mpeg", ".mp3")
    mimetypes.add_type("audio/wav", ".wav")
    mimetypes.add_type("video/mp4", ".mp4")
    state: dict[str, Any] = {"saved_offset": None}
    handler = make_handler(report, media, {"video": video, "audio": audio}, save_cb, state, floor)
    return ThreadingHTTPServer((host, port), handler)


def load_floor(out_dir: Path, report: dict[str, Any], height_m: float) -> dict[str, Any]:
    """Top-down footwork data, recomputed from points3d.npz if present, else read from the report."""
    import footwork

    p = out_dir / "points3d.npz"
    if p.is_file():
        try:
            with np.load(p) as z:
                return footwork.compute_footwork(z["points3d"], z["cam1_frames"], float(z["fps"]), report, height_m)
        except Exception as exc:  # corrupt / incompatible file
            log.warning("Could not compute footwork from %s: %s", p, exc)
    fw = report.get("footwork")
    if fw and fw.get("available"):
        return fw
    return {"available": False, "reason": f"{p.name} not found: re-run main.py for this session"}


def load_subject(config: str, subject: str | None, prepare_audio: bool = True):
    """Return (report, media, video_path, audio_path, notes, session_id, floor)."""
    from main import load_config

    cfg = load_config(config)
    sessions = cfg["sessions"]
    if not sessions:
        raise SystemExit("sessions.json has no sessions")
    if subject is None:
        sess = sessions[0]
    else:
        match = [s for s in sessions if s["id"] == subject]
        if not match:
            raise SystemExit(f"Subject '{subject}' not found. Available: {', '.join(s['id'] for s in sessions)}")
        sess = match[0]
    out_dir = Path(cfg["output_root"]) / sess["id"]
    report_path = out_dir / "epochs_and_steps.json"
    if not report_path.is_file():
        raise SystemExit(f"{report_path} not found. Run: python main.py --config {config} --session {sess['id']}")
    report = json.loads(report_path.read_text())
    notes = []
    overlay = out_dir / "overlay_cam1.mp4"
    if overlay.is_file():
        video, is_overlay = overlay, True
    elif Path(sess["cam1"]).is_file():
        video, is_overlay = Path(sess["cam1"]), False
        notes.append("Showing the original video (no skeleton). Render the overlay with: "
                     f"python overlay_video.py --config {config} --subject {sess['id']}")
    else:
        video, is_overlay = None, False
        notes.append("cam1 video not found.")
    song = Path(sess["song_mp3"])
    audio = None
    if song.is_file():
        audio = prepare_sync_audio(song, out_dir) if prepare_audio else song
    else:
        notes.append("Song (.mp3) not found; the metronome still works.")
    media = {"video": video is not None, "video_is_overlay": is_overlay, "audio": audio is not None,
             "audio_format": audio.suffix.lstrip(".") if audio else None, "notes": notes}
    floor = load_floor(out_dir, report, float(sess["height_meters"]))
    if not floor.get("available"):
        notes.append(f"Footwork map unavailable: {floor.get('reason')}")
    return report, media, video, audio, notes, sess["id"], floor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local rhythm / tempo dashboard")
    ap.add_argument("--config", default="sessions.json")
    ap.add_argument("--subject", help="session id (default: first session)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--read-only", action="store_true", help="disable 'Save offset to sessions.json'")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report, media, video, audio, notes, sid, floor = load_subject(args.config, args.subject)
    for n in notes:
        log.warning(n)
    save_cb = None if args.read_only else (lambda v: save_offset_to_config(args.config, sid, v))
    server = make_server(report, media, video, audio, args.host, args.port, save_cb, floor)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Rhythm dashboard for {sid}: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# --------------------------------------------------------------------------- #
# Page (self-contained: no external scripts, fonts or CDNs)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rhythm Dashboard</title>
<style>
:root{
  --bg:#0b1018; --panel:#121924; --panel2:#18212f; --line:#243043; --text:#e6edf6; --muted:#8b98ab;
  --left:#00e5ff; --right:#ff4081; --green:#22c55e; --amber:#f59e0b; --red:#ef4444; --accent:#ffc400; --blue:#60a5fa;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;height:100vh;overflow:hidden;display:grid;grid-template-rows:auto minmax(0,1fr);
  background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:8px 16px;border-bottom:1px solid var(--line);min-width:0}
header h1{font-size:15px;margin:0;font-weight:650;white-space:nowrap}
header .meta{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.legend{display:flex;gap:12px;font-size:12px;color:var(--muted);white-space:nowrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:0}
main{display:grid;grid-template-columns:minmax(0,55fr) minmax(0,45fr);gap:12px;padding:10px 12px 12px;min-height:0}
.col{display:grid;gap:8px;min-height:0;min-width:0}
.left{grid-template-rows:auto minmax(0,1fr) auto auto clamp(120px,22vh,210px)}
.viewbar{display:flex;align-items:center;gap:8px;min-width:0}
.viewbar .seg .btn{padding:3px 10px}
.stage{position:relative;display:grid;gap:8px;min-height:0;grid-template-columns:1fr}
.stage.mode-split{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
.stage.mode-camera .mapwrap{display:none}
.stage.mode-map .videowrap{position:absolute;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;border:0}
.mapwrap{position:relative;min-height:0;min-width:0;background:#0d141f;border:1px solid var(--line);border-radius:10px;overflow:hidden}
#map{position:absolute;inset:0;width:100%;height:100%;display:block}
#map .fp{cursor:pointer}
.mapcard{position:absolute;background:rgba(11,18,32,.86);border:1px solid #273449;border-radius:8px;padding:6px 9px;font-size:11.5px;line-height:1.45;pointer-events:none;max-width:62%}
.mapcard b{color:var(--text);font-weight:650} .mapcard .k{color:var(--muted)}
#mapMetrics{left:8px;top:8px} #mapSel{left:8px;bottom:8px;display:none;pointer-events:auto;max-width:55%} #mapLegend{right:8px;bottom:8px;text-align:right;max-width:44%;font-size:10.5px}
.mapempty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);padding:20px;text-align:center}
.right{grid-template-rows:auto auto minmax(0,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-height:0;min-width:0}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 6px;font-weight:650;display:flex;align-items:center;gap:8px}
.videowrap{position:relative;min-height:0;background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--line)}
.videowrap video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000}
.bar{display:flex;align-items:center;gap:8px;flex-wrap:nowrap;min-width:0}
.bar .grow{flex:1;min-width:60px}
.bar.wrap{flex-wrap:wrap;row-gap:6px}
.btn{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:4px 9px;cursor:pointer;font:inherit;white-space:nowrap;line-height:1.3}
.btn:hover{border-color:#3b4a61}
.btn.on{background:var(--blue);border-color:var(--blue);color:#05101f;font-weight:650}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#1a1200;font-weight:650}
.btn.icon{width:34px;padding:4px 0;text-align:center}
.seg{display:flex}
.seg .btn{border-radius:0;margin-left:-1px}
.seg .btn:first-child{border-radius:7px 0 0 7px;margin-left:0} .seg .btn:last-child{border-radius:0 7px 7px 0}
input[type=range]{accent-color:var(--accent)}
input[type=number]{width:86px;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:4px 6px;font:inherit}
.lbl{color:var(--muted);font-size:12px;white-space:nowrap}
.mono{font-variant-numeric:tabular-nums}
.chk{display:flex;gap:4px;align-items:center;cursor:pointer;white-space:nowrap;color:var(--muted)}
.drift{font-size:12px;color:var(--muted);white-space:nowrap;min-width:84px;text-align:right}
.drift.bad{color:var(--amber)}
#tl{display:grid;grid-template-rows:auto minmax(0,1fr)}
#tlsvg{width:100%;height:100%;display:block;min-height:0}
#tlsvg .pt{cursor:pointer}
.tip{position:fixed;pointer-events:none;background:#0b1220;border:1px solid #334155;border-radius:7px;padding:6px 9px;font-size:12px;z-index:9;display:none;white-space:nowrap;box-shadow:0 6px 20px rgba(0,0,0,.4)}
.scorerow{display:grid;grid-template-columns:auto minmax(0,1fr);gap:12px;align-items:center}
.ring{position:relative;width:104px;height:104px}
.ring svg{transform:rotate(-90deg)}
.ring .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring .val b{font-size:24px;line-height:1}
.ring .val span{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.06em}
.verdict{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:650;font-size:14px}
.v-excellent{background:rgba(34,197,94,.15);color:var(--green)} .v-good{background:rgba(96,165,250,.15);color:var(--blue)}
.v-fair{background:rgba(245,158,11,.15);color:var(--amber)} .v-poor{background:rgba(239,68,68,.15);color:var(--red)}
.v-none{background:var(--panel2);color:var(--muted)}
.vdetail{color:var(--muted);margin-top:4px;font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:8px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:6px 8px;min-width:0}
.kpi .k{color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .v{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
.kpi .h{color:var(--muted);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#coachCard ul{list-style:none;margin:0;padding:0;display:grid;gap:5px;max-height:clamp(70px,20vh,190px);overflow:auto}
#coachCard li{padding:6px 9px;border-radius:7px;border-left:3px solid var(--line);background:var(--panel2);font-size:12.5px}
#coachCard li.ok{border-left-color:var(--green)} #coachCard li.warn{border-left-color:var(--amber)} #coachCard li.info{border-left-color:var(--blue)}
.warnline{color:var(--amber);font-size:12px;margin-top:6px}
#logCard{display:grid;grid-template-rows:auto minmax(0,1fr)}
.tablewrap{overflow:auto;min-height:0;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:600;font-size:11.5px;z-index:1}
tbody tr{cursor:pointer} tbody tr:hover{background:#1b2534} tbody tr.cur{background:rgba(255,196,0,.13)}
.foot-L{color:var(--left);font-weight:650} .foot-R{color:var(--right);font-weight:650}
.tag{padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:650}
.c-on_time{background:rgba(34,197,94,.15);color:var(--green)} .c-early,.c-late{background:rgba(245,158,11,.15);color:var(--amber)}
.c-off_early,.c-off_late{background:rgba(239,68,68,.17);color:var(--red)} .c-none{background:var(--panel2);color:var(--muted)}
.toast{position:fixed;right:16px;bottom:16px;background:#0b1220;border:1px solid #334155;border-radius:8px;padding:9px 12px;display:none;max-width:460px;z-index:10;font-size:12.5px}
.spacer{flex:1}
</style>
</head>
<body>
<header>
  <h1>Rhythm Dashboard · <span id="subj"></span></h1>
  <div class="meta" id="meta"></div>
  <div class="legend">
    <span><i class="dot" style="background:var(--left)"></i>Left foot</span>
    <span><i class="dot" style="background:var(--right)"></i>Right foot</span>
    <span><i class="dot" style="background:var(--green)"></i>On time ≤ ±45 ms</span>
    <span><i class="dot" style="background:var(--amber)"></i>Early/Late ≤ 90 ms</span>
    <span><i class="dot" style="background:var(--red)"></i>Off-beat &gt; 90 ms</span>
  </div>
</header>
<main>
  <div class="col left">
    <div class="viewbar">
      <div class="seg" id="views">
        <button class="btn on" data-view="camera">Camera + Skeleton</button><button class="btn" data-view="map">Top-Down Footwork Map</button><button class="btn" data-view="split">Split View</button>
      </div>
      <span class="spacer"></span>
      <div class="seg" id="mapModes">
        <button class="btn on" data-mapmode="cycle" title="Only the active 8-count phrase, with faded trails of the previous ones">Current 8-Count Cycle</button><button class="btn" data-mapmode="all" title="Every detected footstep at once">All Steps Overlay</button>
      </div>
    </div>
    <div class="stage mode-camera" id="stage">
      <div class="videowrap"><video id="video" src="media/video" preload="auto" playsinline muted></video></div>
      <div class="mapwrap" id="mapwrap">
        <svg id="map"></svg>
        <div class="mapcard" id="mapMetrics"></div>
        <div class="mapcard" id="mapLegend"></div>
        <div class="mapcard" id="mapSel"></div>
      </div>
    </div>
    <div class="card bar" style="padding:6px 10px">
      <button class="btn icon" id="play" title="Play / pause (space)">▶</button>
      <span class="mono lbl" id="time">0.00 / 0.00 s</span>
      <input type="range" class="grow" id="scrub" min="0" max="1000" value="0" step="1">
      <div class="seg" id="speeds">
        <button class="btn" data-rate="0.5">0.5x</button><button class="btn" data-rate="0.75">0.75x</button><button class="btn on" data-rate="1">1.0x</button>
      </div>
      <label class="chk"><input type="checkbox" id="musicOn" checked>Music</label>
      <label class="chk"><input type="checkbox" id="camAudio">Cam audio</label>
      <label class="chk"><input type="checkbox" id="metroOn">Metronome</label>
    </div>
    <div class="card bar wrap" style="padding:6px 10px">
      <span class="lbl">Music offset</span>
      <input type="number" id="offset" step="0.001" title="mp3_offset_sec: cam1 time at which the song starts (s)">
      <span class="lbl">s</span>
      <button class="btn" data-nudge="-50">−50ms</button><button class="btn" data-nudge="-10">−10</button>
      <button class="btn" data-nudge="10">+10</button><button class="btn" data-nudge="50">+50ms</button>
      <button class="btn primary" id="tap" title="Press exactly when you hear/see beat 1 in the video">Tap Beat 1 Now</button>
      <button class="btn" id="reset" title="Back to the analysed offset">Reset</button>
      <span class="spacer"></span>
      <button class="btn" id="save">Save Offset to sessions.json</button>
      <span class="drift mono" id="drift">sync –</span>
    </div>
    <div class="card" id="tl">
      <h2>Step error timeline <span class="lbl" style="text-transform:none;letter-spacing:0">● left · ◆ right · click a step to replay from 1 s before</span></h2>
      <svg id="tlsvg"></svg>
    </div>
  </div>

  <div class="col right">
    <div class="card">
      <div class="scorerow">
        <div class="ring">
          <svg width="104" height="104" viewBox="0 0 104 104">
            <circle cx="52" cy="52" r="45" fill="none" stroke="#243043" stroke-width="8"/>
            <circle id="arc" cx="52" cy="52" r="45" fill="none" stroke="#22c55e" stroke-width="8" stroke-linecap="round" stroke-dasharray="0 999"/>
          </svg>
          <div class="val"><b id="score">–</b><span>Tempo</span></div>
        </div>
        <div style="min-width:0">
          <div class="verdict" id="verdict"></div>
          <div class="vdetail" id="vdetail"></div>
        </div>
      </div>
      <div class="kpis" id="kpis"></div>
    </div>
    <div class="card" id="coachCard">
      <h2>Coaching diagnosis</h2>
      <ul id="coach"></ul>
      <div class="warnline" id="warn"></div>
    </div>
    <div class="card" id="logCard">
      <h2>Rhythm error &amp; step log <span class="spacer"></span>
        <span class="seg"><button class="btn on" id="fAll">All steps</button><button class="btn" id="fErr">Errors only</button></span>
      </h2>
      <div class="tablewrap">
        <table>
          <thead><tr><th>Time (s)</th><th>Foot</th><th>Beat</th><th>Error (ms)</th><th>Status</th></tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<div class="tip" id="tip"></div>
<div class="toast" id="toast"></div>

<script>
(async function(){
const $ = id => document.getElementById(id);
let D = await (await fetch('api/data')).json();
const video = $('video');
const audio = D.media.audio ? new Audio('media/audio') : null;
if (audio){ audio.preload = 'auto'; audio.preservesPitch = true; }
const ANALYZED = D.mp3_offset_analyzed_sec;
let offset = D.mp3_offset_sec;           // current mp3_offset_sec (cam1 time of song t=0)
let filterErrors = false;
const fmt = (v, d=0) => (v == null || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);
const sgn = (v, d=0) => v == null ? '–' : (v > 0 ? '+' : '') + Number(v).toFixed(d);
const COL = {on_time:'#22c55e', early:'#f59e0b', late:'#f59e0b', off_early:'#ef4444', off_late:'#ef4444', none:'#64748b'};
const FOOT = {L:'#00e5ff', R:'#ff4081'};
function toast(msg, ms=4500){ const t = $('toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = 'none', ms); }

// ======================= rendering of analysis data =======================
function renderHeader(){
  $('subj').textContent = D.subject_id;
  $('meta').textContent = `${fmt(D.bpm,1)} BPM · beat ${fmt(D.beat_period_ms,0)} ms · music alignment: ${D.music_method || '–'}` +
    (D.music_confidence != null ? ` (conf ${fmt(D.music_confidence,2)})` : '') +
    ` · analysed offset ${fmt(ANALYZED,3)} s` + (D.media.audio_format ? ` · audio: ${D.media.audio_format}` : '');
}
function renderScore(){
  const v = D.verdict, sc = D.score;
  $('verdict').textContent = v.text; $('verdict').className = 'verdict v-' + v.level; $('vdetail').textContent = v.detail;
  $('score').textContent = sc == null ? '–' : Math.round(sc) + '%';
  const C = 2*Math.PI*45, col = {excellent:'#22c55e', good:'#60a5fa', fair:'#f59e0b', poor:'#ef4444'}[v.level] || '#64748b';
  $('arc').setAttribute('stroke-dasharray', `${C*(sc||0)/100} ${C}`); $('arc').setAttribute('stroke', col);
  const K = D.kpis, mo = K.mean_offset_ms;
  const items = [
    ['BPM', fmt(K.bpm,1), `beat ${fmt(D.beat_period_ms,0)} ms`],
    ['On-tempo', K.on_tempo_pct == null ? '–' : fmt(K.on_tempo_pct,0)+'%', `${K.n_steps - K.n_errors}/${K.n_steps} steps ≤ ±45 ms`],
    ['Mean offset', mo == null ? '–' : sgn(mo,0)+' ms', mo == null ? '' : `${mo < -25 ? 'early' : mo > 25 ? 'late' : 'centred'} · spread ±${fmt(K.consistency_ms,0)}`],
    ['QQS accuracy', K.qqs_accuracy_pct == null ? '–' : fmt(K.qqs_accuracy_pct,0)+'%', `1-2-3 / 5-6-7 · ${K.n_blocks} blocks`],
  ];
  $('kpis').innerHTML = '';
  for (const [k, val, h] of items){ const d = document.createElement('div'); d.className = 'kpi';
    for (const [cls, txt] of [['k',k],['v',val],['h',h]]){ const e = document.createElement('div'); e.className = cls; e.textContent = txt; d.appendChild(e); }
    $('kpis').appendChild(d); }
  $('coach').innerHTML = '';
  for (const n of D.coaching){ const li = document.createElement('li'); li.className = n.kind; li.textContent = n.text; $('coach').appendChild(li); }
  $('warn').textContent = [...(D.media.notes || []), ...(D.warnings || [])].join(' · ');
}
let rows = [];
function renderTable(){
  const tb = $('tbody'); tb.innerHTML = ''; rows = [];
  for (const s of D.steps){
    if (filterErrors && !s.is_error) continue;
    const tr = document.createElement('tr'); tr.dataset.i = s.i;
    const cells = [fmt(s.t,2), s.foot, s.beat ?? '–', s.offset_ms == null || s.timing_type === 'TAP' ? '–' : sgn(s.offset_ms,0), ''];
    cells.forEach((c, j) => { const td = document.createElement('td'); td.textContent = c; if (j===1) td.className = 'foot-'+s.foot; tr.appendChild(td); });
    const tag = document.createElement('span'); tag.className = 'tag c-'+s.category; tag.textContent = s.status; tr.children[4].appendChild(tag);
    tr.addEventListener('click', () => { seek(s.t); selectStep(s.i); });
    tb.appendChild(tr); rows.push([s, tr]);
  }
  $('fAll').classList.toggle('on', !filterErrors); $('fErr').classList.toggle('on', filterErrors);
  $('fErr').textContent = `Errors only (${D.kpis.n_errors})`;
  lastCur = null;
}
$('fAll').onclick = () => { filterErrors = false; renderTable(); };
$('fErr').onclick = () => { filterErrors = true; renderTable(); };

// ---------- timeline (redrawn at real pixel size) ----------
const svg = $('tlsvg'), NS = 'http://www.w3.org/2000/svg', tip = $('tip');
let TL = null, playhead = null;
function el(tag, attrs, parent=svg){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
function drawTimeline(){
  const W = svg.clientWidth, H = svg.clientHeight; if (W < 50 || H < 40) return;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
  const PL = 44, PR = 8, PT = 6, PB = 18, half = D.beat_period_ms / 2;
  const t1 = Math.max(video.duration || 0, D.duration + 1, 1), yMax = Math.max(120, Math.min(half, 250));
  const X = t => PL + t / t1 * (W - PL - PR);
  const Y = o => PT + (1 - (Math.max(-yMax, Math.min(yMax, o)) + yMax) / (2*yMax)) * (H - PT - PB);
  TL = {X, PL, PR, W, H, t1, PT, PB};
  for (const e of D.epochs) el('rect', {x:X(e.start), y:PT, width:Math.max(1, X(e.end)-X(e.start)), height:H-PT-PB, fill:'rgba(96,165,250,.06)'});
  el('rect', {x:PL, y:Y(90), width:W-PL-PR, height:Y(-90)-Y(90), fill:'rgba(245,158,11,.06)'});
  el('rect', {x:PL, y:Y(45), width:W-PL-PR, height:Y(-45)-Y(45), fill:'rgba(34,197,94,.17)'});
  for (const o of [-yMax, -90, -45, 0, 45, 90, yMax]){
    el('line', {x1:PL, x2:W-PR, y1:Y(o), y2:Y(o), stroke: o===0 ? '#475569' : '#1e293b'});
    if (Math.abs(o) !== 45) { const tx = el('text', {x:PL-5, y:Y(o)+4, 'text-anchor':'end', fill:'#8b98ab', 'font-size':10}); tx.textContent = (o>0?'+':'')+Math.round(o); }
  }
  const tx45 = el('text', {x:PL+4, y:Y(45)-3, fill:'#22c55e', 'font-size':10}); tx45.textContent = '±45 ms on tempo';
  const stepT = t1 > 300 ? 60 : t1 > 120 ? 20 : t1 > 40 ? 10 : 5;
  for (let t = 0; t <= t1; t += stepT){ const tx = el('text', {x:X(t), y:H-4, 'text-anchor':'middle', fill:'#8b98ab', 'font-size':10}); tx.textContent = t+'s'; }
  for (const s of D.steps){
    if (s.timing_type === 'TAP' || s.offset_ms == null) continue;
    const x = X(s.t), y = Y(s.offset_ms), off = Math.abs(s.offset_ms) > 90, slight = s.is_error && !off;
    const fill = off ? '#ef4444' : slight ? '#f59e0b' : FOOT[s.foot], r = off ? 5.5 : 4;
    const p = s.foot === 'L'
      ? el('circle', {cx:x, cy:y, r, fill, stroke: off ? FOOT.L : '#0b1018', 'stroke-width': off ? 2 : 1, class:'pt'})
      : el('rect', {x:x-r, y:y-r, width:2*r, height:2*r, transform:`rotate(45 ${x} ${y})`, fill, stroke: off ? FOOT.R : '#0b1018', 'stroke-width': off ? 2 : 1, class:'pt'});
    p.addEventListener('mousemove', ev => { tip.style.display = 'block'; tip.style.left = (ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY - 34) + 'px';
      tip.innerHTML = ''; const b = document.createElement('b'); b.style.color = FOOT[s.foot]; b.textContent = s.foot === 'L' ? 'Left foot' : 'Right foot'; tip.appendChild(b);
      tip.appendChild(document.createTextNode(` · ${s.t.toFixed(2)} s · Beat ${s.beat ?? '–'} · ${sgn(s.offset_ms,0)} ms · ${s.status}`)); });
    p.addEventListener('mouseleave', () => tip.style.display = 'none');
    p.addEventListener('click', ev => { ev.stopPropagation(); seek(Math.max(0, s.t - 1.0), true); selectStep(s.i); });
  }
  playhead = el('line', {x1:X(0), x2:X(0), y1:PT, y2:H-PB, stroke:'#ffc400', 'stroke-width':1.5});
}
svg.addEventListener('click', ev => { if (!TL) return; const r = svg.getBoundingClientRect(), x = ev.clientX - r.left;
  if (x >= TL.PL && x <= TL.W - TL.PR) seek((x - TL.PL) / (TL.W - TL.PL - TL.PR) * TL.t1); });
new ResizeObserver(() => drawTimeline()).observe(svg);
video.addEventListener('loadedmetadata', drawTimeline);

function renderAll(){ renderHeader(); renderScore(); renderTable(); drawTimeline(); $('offset').value = offset.toFixed(3); MAP.redraw(); }

// ======================= audio sync engine =======================
// target_audio_time = video.currentTime - mp3_offset_sec. Enforced on play/pause/seeked/ratechange
// and continuously in a requestAnimationFrame loop:
//   |drift| > 40 ms  -> corrected; small drift by a gentle playbackRate trim (no audible seek),
//   |drift| > 150 ms -> hard seek (rate-limited so we never seek every frame).
const SYNC = {tol: 0.04, settleTol: 0.01, settleMs: 1500, hard: 0.15, lastSeek: 0, settleUntil: 0, stalled: false, drift: null};
// Smooth video clock: video.currentTime only advances once per decoded frame (33 ms steps at 30 fps),
// which would look like +/-33 ms of "drift". requestVideoFrameCallback gives the presented frame's
// mediaTime + its display time, so we extrapolate between frames.
const VC = {media: null, at: 0};
function onVideoFrame(now, meta){ VC.media = meta.mediaTime; VC.at = meta.expectedDisplayTime || now; video.requestVideoFrameCallback(onVideoFrame); }
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) video.requestVideoFrameCallback(onVideoFrame);
function videoClock(){
  if (VC.media == null || video.paused || video.seeking) return video.currentTime;
  const dt = (performance.now() - VC.at) / 1000 * video.playbackRate;
  if (dt < 0 || dt > 0.25 || Math.abs(VC.media - video.currentTime) > 0.25) return video.currentTime;  // stale: fall back
  return VC.media + dt;
}
video.addEventListener('seeked', () => { VC.media = null; });
function musicOn(){ return audio && $('musicOn').checked; }
function targetTime(){ return videoClock() - offset; }
function inRange(t){ return t >= 0 && (!isFinite(audio.duration) || t < audio.duration); }
function hardSync(){
  if (!audio) return;
  const t = targetTime();
  audio.playbackRate = video.playbackRate;
  if (!musicOn() || video.paused || video.seeking || SYNC.stalled || !inRange(t)){
    if (!audio.paused) audio.pause();
    if (inRange(t)) audio.currentTime = t;     // pre-position so the next play starts in sync
    return;
  }
  audio.currentTime = t; SYNC.lastSeek = performance.now(); SYNC.settleUntil = SYNC.lastSeek + SYNC.settleMs;
  if (audio.paused) audio.play().catch(() => {});
}
function syncTick(){
  if (!audio) return;
  if (!musicOn() || video.paused || video.seeking || SYNC.stalled){ if (!audio.paused) audio.pause(); SYNC.drift = null; return; }
  const t = targetTime();
  if (!inRange(t)){ if (!audio.paused) audio.pause(); SYNC.drift = null; return; }  // before song start / after its end
  if (audio.paused || audio.seeking){ if (audio.paused && performance.now() - SYNC.lastSeek > 250) hardSync(); return; }
  const drift = audio.currentTime - t; SYNC.drift = drift;
  const now = performance.now();
  if (Math.abs(drift) > SYNC.hard){
    if (now - SYNC.lastSeek > 400){ audio.currentTime = t; SYNC.lastSeek = now; audio.playbackRate = video.playbackRate; }
  } else if (Math.abs(drift) > (now < SYNC.settleUntil ? SYNC.settleTol : SYNC.tol)){
    // steady state: re-sync above 40 ms; for 1.5 s after a (re)start also remove the play() start-up latency
    const trim = Math.max(-0.08, Math.min(0.08, drift * 1.2));      // ahead -> slow down, behind -> speed up
    audio.playbackRate = video.playbackRate * (1 - trim);
  } else if (Math.abs(drift) < 0.01 && audio.playbackRate !== video.playbackRate){
    audio.playbackRate = video.playbackRate;
  }
}
['play','pause','seeked','ratechange'].forEach(ev => video.addEventListener(ev, () => { metro.clear(); hardSync(); updatePlayBtn(); }));
video.addEventListener('seeking', () => { if (audio && !audio.paused) audio.pause(); });
video.addEventListener('waiting', () => { SYNC.stalled = true; if (audio && !audio.paused) audio.pause(); });
video.addEventListener('playing', () => { SYNC.stalled = false; hardSync(); });
video.addEventListener('ended', () => { if (audio) audio.pause(); updatePlayBtn(); });
$('musicOn').addEventListener('change', hardSync);
$('camAudio').addEventListener('change', e => { video.muted = !e.target.checked; });

// ---------- transport ----------
function updatePlayBtn(){ $('play').textContent = video.paused ? '▶' : '❚❚'; }
$('play').onclick = () => { if (video.paused) video.play(); else video.pause(); };
function seek(t, autoplay=false){ video.currentTime = Math.max(0, Math.min(t, (video.duration || t))); if (autoplay && video.paused) video.play(); }
let scrubbing = false;
$('scrub').addEventListener('input', e => { scrubbing = true; if (video.duration) seek(e.target.value / 1000 * video.duration); });
$('scrub').addEventListener('change', () => { scrubbing = false; });
document.querySelectorAll('[data-rate]').forEach(b => b.addEventListener('click', () => {
  video.playbackRate = +b.dataset.rate; document.querySelectorAll('[data-rate]').forEach(x => x.classList.toggle('on', x === b)); }));
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space'){ e.preventDefault(); $('play').click(); }
  if (e.code === 'ArrowLeft') seek(video.currentTime - (e.shiftKey ? 0.1 : 1));
  if (e.code === 'ArrowRight') seek(video.currentTime + (e.shiftKey ? 0.1 : 1));
});

// ---------- offset calibration ----------
let refetchTimer = null;
function setOffset(v, {refetch=true} = {}){
  if (!isFinite(v)) return;
  offset = Math.round(v * 1000) / 1000; $('offset').value = offset.toFixed(3);
  metro.clear(); hardSync();
  if (refetch){ clearTimeout(refetchTimer); refetchTimer = setTimeout(async () => {
    const r = await fetch('api/data?mp3_offset=' + encodeURIComponent(offset)); if (r.ok){ D = await r.json(); renderAll(); } }, 180); }
}
$('offset').addEventListener('change', e => setOffset(parseFloat(e.target.value)));
document.querySelectorAll('[data-nudge]').forEach(b => b.addEventListener('click', () => setOffset(offset + (+b.dataset.nudge) / 1000)));
$('reset').onclick = () => setOffset(ANALYZED);
$('tap').onclick = () => {
  // The current video moment should be a beat 1 of the song: snap the nearest song "1" (in mp3 time) onto it.
  const tv = video.currentTime, m = tv - offset;
  let ones = D.beats.filter(b => b.count === 1).map(b => b.t - D.mp3_offset_sec);
  if (!ones.length) ones = D.beats.map(b => b.t - D.mp3_offset_sec);
  if (!ones.length){ toast('No beat grid available for this song.'); return; }
  let best = ones[0]; for (const b of ones) if (Math.abs(b - m) < Math.abs(best - m)) best = b;
  setOffset(tv - best);
  toast(`Beat 1 placed at ${tv.toFixed(3)} s → mp3_offset_sec = ${(tv - best).toFixed(3)} s (${sgn((tv - best - ANALYZED)*1000,0)} ms vs analysis)`);
};
$('save').onclick = async () => {
  if (!D.media.can_save){ toast('Saving is disabled (read-only mode).'); return; }
  const r = await fetch('api/save-offset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mp3_offset_sec: offset})});
  const j = await r.json().catch(() => ({}));
  if (r.ok && j.ok) toast(`Saved mp3_offset_sec = ${j.mp3_offset_sec} s to ${j.config}. Re-run main.py for this session to re-score the steps and re-render the overlay.`, 8000);
  else toast('Save failed: ' + (j.error || r.status));
};

// ---------- metronome (Web Audio, beats 1 2 3 5 6 7, follows the current offset) ----------
let actx = null;
const metro = {done: new Set(), clear(){ this.done.clear(); }};
$('metroOn').addEventListener('change', e => { if (e.target.checked && !actx) actx = new (window.AudioContext || window.webkitAudioContext)(); metro.clear(); });
function click(when, accent){
  const o = actx.createOscillator(), g = actx.createGain();
  o.frequency.value = accent ? 1760 : 1100; o.type = 'square';
  g.gain.setValueAtTime(0.0001, when); g.gain.exponentialRampToValueAtTime(accent ? 0.35 : 0.2, when + 0.002);
  g.gain.exponentialRampToValueAtTime(0.0001, when + 0.05);
  o.connect(g).connect(actx.destination); o.start(when); o.stop(when + 0.06);
}
function metronomeTick(){
  if (!actx || !$('metroOn').checked || video.paused) return;
  const vt = video.currentTime, rate = video.playbackRate || 1, shift = offset - D.mp3_offset_sec;
  for (let k = 0; k < D.beats.length; k++){
    const b = D.beats[k]; if (![1,2,3,5,6,7].includes(b.count)) continue;
    const bt = b.t + shift; if (bt < vt - 0.01) continue; if (bt - vt > 0.15 * rate) break;
    if (metro.done.has(k)) continue; metro.done.add(k);
    click(actx.currentTime + Math.max(0, (bt - vt) / rate), b.count === 1);
  }
}


// ======================= top-down footwork map =======================
// Floor frame (cm): +y = dancer's facing ("FRONT"), +x = dancer's right, origin = centre stance of the block.
const SOLE = [[0,-.5],[.12,-.47],[.16,-.36],[.14,-.12],[.19,.12],[.2,.3],[.15,.44],[.06,.5],[-.05,.49],[-.14,.42],[-.18,.26],[-.17,.06],[-.13,-.14],[-.15,-.35],[-.11,-.47]];
const STATUS_COL = {on_time:'#22c55e', early:'#f59e0b', late:'#f59e0b', off_early:'#ef4444', off_late:'#ef4444', none:'#64748b'};
const MAP = (() => {
  const svg = $('map'); let mode = 'cycle', view = 'camera', sel = null, pin = null, shownCycle = null, live = null, geom = null;
  const stepByI = () => { const m = new Map(); for (const s of D.steps) m.set(s.i, s); return m; };
  const F = () => D.floor || {available:false};
  function fstep(i){ return (F().steps || []).find(f => f.i === i); }
  function cycleAt(t){
    const fl = F(); if (!fl.available || !fl.cycles.length) return null;
    if (pin != null){ const c = fl.cycles.find(c => c.id === pin); if (c && t >= c.t0 - 1.6 && t <= c.t1 + 1.2) return pin; pin = null; }
    let cur = fl.cycles[0].id; for (const c of fl.cycles){ if (c.t0 <= t + 0.05) cur = c.id; else break; }
    return cur;
  }
  function footPts(x, y, headDeg, foot, Lf){
    const h = headDeg * Math.PI / 180, fx = Math.sin(h), fy = Math.cos(h), rx = Math.cos(h), ry = -Math.sin(h);
    const W = Lf * 1.0, mir = foot === 'L' ? 1 : -1;   // arch on the inner side
    return SOLE.map(([u, v]) => { const uu = u * mir * W, vv = v * Lf; return geom.px(x + vv*fx + uu*rx, y + vv*fy + uu*ry); });
  }
  function el(tag, attrs, parent = svg){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
  function text(x, y, str, attrs = {}, parent = svg){ const t = el('text', Object.assign({x, y, fill:'#cbd5e1', 'font-size':11, 'text-anchor':'middle'}, attrs), parent); t.textContent = str; return t; }
  function setGeom(bounds){
    const W = svg.clientWidth, H = svg.clientHeight; if (W < 40 || H < 40) return false;
    const [x0, x1, y0, y1] = bounds, pad = 26, k = Math.min((W - 2*pad) / (x1 - x0), (H - 2*pad - 14) / (y1 - y0));
    const cx = W / 2, cy = H / 2 + 7, xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
    geom = {W, H, k, px: (x, y) => [cx + (x - xm) * k, cy - (y - ym) * k], bounds};
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); return true;
  }
  function arrow(from, to, col, bend, label){
    const [ax, ay] = geom.px(...from), [bx, by] = geom.px(...to), mx = (ax + bx) / 2, my = (ay + by) / 2;
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
    const qx = mx + nx * bend, qy = my + ny * bend;
    el('path', {d:`M${ax} ${ay} Q${qx} ${qy} ${bx} ${by}`, fill:'none', stroke:col, 'stroke-width':1.4, 'stroke-dasharray':'5 4', 'stroke-opacity':.75, 'marker-end':`url(#arr-${col.slice(1)})`});
    if (label) text(qx + nx * 9, qy + ny * 9 + 4, label, {fill:col, 'font-size':11, 'font-weight':700});
  }
  function drawIdeal(anchor){
    const fl = F(), id = fl.ideal, Lf = id.foot_length_cm, [ax, ay] = anchor, hs = id.half_stance_cm, R = id.reach_cm;
    const spots = [['L', -hs, 0, '3 · 6'], ['R', hs, 0, '2 · 7'], ['L', -hs, R, '1'], ['R', hs, -R, '5']];
    for (const [foot, x, y, lab] of spots){
      const col = foot === 'L' ? '#00e5ff' : '#ff4081', pts = footPts(ax + x, ay + y, 0, foot, Lf);
      el('polygon', {points: pts.map(p => p.join(',')).join(' '), fill:'none', stroke:col, 'stroke-opacity':.55, 'stroke-width':1.6, 'stroke-dasharray':'4 3', 'stroke-linejoin':'round'});
      const [lx, ly] = geom.px(ax + x + (foot === 'L' ? -1 : 1) * (id.foot_width_cm * .5 + 7), ay + y);
      text(lx, ly + 4, lab, {fill:col, 'fill-opacity':.8, 'font-size':12, 'font-weight':700});
    }
    arrow([ax - hs - 3, ay + Lf * .55], [ax - hs - 3, ay + R - Lf * .55], '#00e5ff', -14, '1');
    arrow([ax - hs + 3, ay + R - Lf * .55], [ax - hs + 3, ay + Lf * .55], '#00e5ff', -14, '3');
    arrow([ax + hs - 3, ay - Lf * .55], [ax + hs - 3, ay - R + Lf * .55], '#ff4081', -14, '5');
    arrow([ax + hs + 3, ay - R + Lf * .55], [ax + hs + 3, ay - Lf * .55], '#ff4081', -14, '7');
    const [sx, sy] = geom.px(ax, ay - Lf * .62);
    text(sx, sy + 12, '4 & 8: feet together', {fill:'#94a3b8', 'font-size':10});
  }
  function drawGrid(){
    const [x0, x1, y0, y1] = geom.bounds;
    for (let v = Math.ceil(x0 / 10) * 10; v <= x1; v += 10){ const [a] = geom.px(v, 0); el('line', {x1:a, x2:a, y1:geom.px(0, y1)[1], y2:geom.px(0, y0)[1], stroke: v % 50 === 0 ? '#2a3a52' : '#172131', 'stroke-width': v % 50 === 0 ? 1.2 : 1}); }
    for (let v = Math.ceil(y0 / 10) * 10; v <= y1; v += 10){ const [, b] = geom.px(0, v); el('line', {y1:b, y2:b, x1:geom.px(x0, 0)[0], x2:geom.px(x1, 0)[0], stroke: v % 50 === 0 ? '#2a3a52' : '#172131', 'stroke-width': v % 50 === 0 ? 1.2 : 1}); }
    for (let v = Math.ceil(x0 / 50) * 50; v <= x1; v += 50) if (v){ const [a, b] = geom.px(v, y0); text(a, b - 4, `${v} cm`, {fill:'#5b6b82', 'font-size':9.5}); }
    for (let v = Math.ceil(y0 / 50) * 50; v <= y1; v += 50) if (v){ const [a, b] = geom.px(x0, v); text(a + 4, b - 3, `${v} cm`, {fill:'#5b6b82', 'font-size':9.5, 'text-anchor':'start'}); }
    const [ox, oy] = geom.px(0, 0);
    el('line', {x1:ox - 9, x2:ox + 9, y1:oy, y2:oy, stroke:'#94a3b8', 'stroke-width':1.5});
    el('line', {x1:ox, x2:ox, y1:oy - 9, y2:oy + 9, stroke:'#94a3b8', 'stroke-width':1.5});
    text(ox + 12, oy + 14, '0,0', {fill:'#64748b', 'font-size':9.5, 'text-anchor':'start'});
    const [fx, fy] = geom.px((x0 + x1) / 2, y1);
    el('path', {d:`M${fx} ${fy - 6} l-7 11 h14 z`, fill:'#e2e8f0'});
    text(fx, fy + 18, 'FRONT / FACING', {fill:'#e2e8f0', 'font-size':10.5, 'font-weight':700, 'letter-spacing':'.06em'});
  }
  function drawActual(fs, s, alpha, opts = {}){
    const fl = F(), Lf = fl.ideal.foot_length_cm, col = s.foot === 'L' ? '#00e5ff' : '#ff4081';
    const g = el('g', {class:'fp', opacity: alpha});
    const pts = footPts(fs.x, fs.y, fs.heading_deg || 0, s.foot, Lf).map(p => p.join(',')).join(' ');
    const [cx, cy] = geom.px(fs.x, fs.y), st = STATUS_COL[s.category] || '#64748b';
    if (!opts.faint){
      // status ring = thick timing-coloured contour around the footprint, then the foot-coloured sole
      el('polygon', {points: pts, fill:'none', stroke:st, 'stroke-width':6, 'stroke-linejoin':'round', 'stroke-opacity':.95}, g);
      el('polygon', {points: pts, fill:col, 'fill-opacity':.85, stroke:'#0b1220', 'stroke-width':1.2, 'stroke-linejoin':'round'}, g);
    } else {
      el('polygon', {points: pts, fill:col, 'fill-opacity':.14, stroke:col, 'stroke-opacity':.35, 'stroke-width':1, 'stroke-linejoin':'round'}, g);
      el('circle', {cx, cy, r: 3.2, fill: st}, g);
    }
    if (!opts.faint && s.beat) text(cx, cy + 4, String(s.beat), {fill:'#0b1220', 'font-size':11, 'font-weight':800}, g);
    g.addEventListener('mousemove', ev => { tip.style.display = 'block'; tip.style.left = (ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY - 34) + 'px';
      tip.textContent = `${s.foot === 'L' ? 'Left' : 'Right'} foot · Beat ${s.beat ?? '–'} · ${s.t.toFixed(2)} s · ${sgn(s.offset_ms,0)} ms ${s.status}` + (fs.err_cm != null ? ` · offset ${fs.err_cm.toFixed(0)} cm` : ''); });
    g.addEventListener('mouseleave', () => tip.style.display = 'none');
    g.addEventListener('click', ev => { ev.stopPropagation(); seek(s.t); selectStep(s.i); });
    return g;
  }
  function errVector(fs, strong){
    if (fs.target_x == null || fs.err_cm == null) return;
    const [ax, ay] = geom.px(fs.x, fs.y), [bx, by] = geom.px(fs.target_x, fs.target_y);
    el('line', {x1:ax, y1:ay, x2:bx, y2:by, stroke: strong ? '#f8fafc' : '#94a3b8', 'stroke-width': strong ? 1.6 : 1.1, 'stroke-dasharray':'2 3', 'stroke-opacity': strong ? .95 : .7});
    el('circle', {cx:bx, cy:by, r:2.4, fill: strong ? '#f8fafc' : '#94a3b8'});
    if (fs.err_cm >= 3 || strong) text((ax + bx) / 2 + 12, (ay + by) / 2, `${fs.err_cm.toFixed(0)} cm`, {fill: strong ? '#f8fafc' : '#94a3b8', 'font-size':10, 'font-weight':600});
  }
  function defs(){
    const d = el('defs', {});
    for (const c of ['#00e5ff', '#ff4081']){ const m = el('marker', {id:'arr-' + c.slice(1), viewBox:'0 0 10 10', refX:8, refY:5, markerWidth:7, markerHeight:7, orient:'auto-start-reverse'}, d);
      el('path', {d:'M0 0 L10 5 L0 10 z', fill:c, 'fill-opacity':.8}, m); }
    const f = el('filter', {id:'glow', x:'-50%', y:'-50%', width:'200%', height:'200%'}, d);
    el('feGaussianBlur', {stdDeviation:3, result:'b'}, f); const mg = el('feMerge', {}, f); el('feMergeNode', {in:'b'}, mg); el('feMergeNode', {in:'SourceGraphic'}, mg);
  }
  function redraw(){
    if (view === 'camera') return;
    const fl = F(); svg.innerHTML = ''; $('mapSel').style.display = 'none';
    const empty = $('mapwrap').querySelector('.mapempty'); if (empty) empty.remove();
    if (!fl.available){ const d = document.createElement('div'); d.className = 'mapempty'; d.textContent = 'Footwork map unavailable: ' + (fl.reason || 'no 3D data') + '.'; $('mapwrap').appendChild(d);
      $('mapMetrics').style.display = $('mapLegend').style.display = 'none'; return; }
    $('mapMetrics').style.display = $('mapLegend').style.display = '';
    const S = stepByI(), all = fl.steps.filter(f => S.has(f.i) && S.get(f.i).timing_type !== 'TAP');
    const t = video.currentTime || 0; shownCycle = cycleAt(t);
    const cyc = fl.cycles.find(c => c.id === shownCycle), anchor = (mode === 'cycle' && cyc) ? cyc.anchor : [0, 0];
    let b;
    if (mode === 'cycle'){ const r = Math.max(60, fl.ideal.reach_cm + 30); b = [anchor[0] - r, anchor[0] + r, anchor[1] - r, anchor[1] + r]; }
    else { const xs = all.map(f => f.x).concat([-60, 60]), ys = all.map(f => f.y).concat([-60, 60]);
      b = [Math.min(...xs) - 18, Math.max(...xs) + 18, Math.min(...ys) - 22, Math.max(...ys) + 22]; }
    if (!setGeom(b)) return;
    defs(); drawGrid(); drawIdeal(anchor);
    if (mode === 'cycle'){
      const ids = fl.cycles.filter(c => c.epoch === (cyc ? cyc.epoch : null)).map(c => c.id), pos = ids.indexOf(shownCycle);
      for (const [back, a] of [[2, .16], [1, .3]]){ const cid = ids[pos - back]; if (cid == null) continue;
        for (const f of all.filter(f => f.cycle === cid)) drawActual(f, S.get(f.i), a, {faint:true}); }
      const cur = all.filter(f => f.cycle === shownCycle);
      for (const f of cur) errVector(f, false);
      for (const f of cur) drawActual(f, S.get(f.i), 1);
    } else {
      // all steps: faint footprints (overlaps build up a heatmap) + anchor drift path per block
      for (const f of all) drawActual(f, S.get(f.i), .9, {faint:true});
      for (const e of D.epochs){ const cs = fl.cycles.filter(c => c.epoch === e.id); if (cs.length < 2) continue;
        el('polyline', {points: cs.map(c => geom.px(...c.anchor).join(',')).join(' '), fill:'none', stroke:'#facc15', 'stroke-width':1.6, 'stroke-dasharray':'1 4', 'stroke-linecap':'round'});
        const [lx, ly] = geom.px(...cs[cs.length - 1].anchor); el('circle', {cx:lx, cy:ly, r:3.5, fill:'#facc15'}); }
    }
    if (sel != null){ const f = fstep(sel), s = S.get(sel);
      if (f && s && (mode === 'all' || f.cycle === shownCycle)){
        if (mode === 'all') drawActual(f, s, 1);
        errVector(f, true);
        const [cx, cy] = geom.px(f.x, f.y);
        const ring = el('polygon', {points: footPts(f.x, f.y, f.heading_deg || 0, s.foot, fl.ideal.foot_length_cm * 1.22).map(p => p.join(',')).join(' '), fill:'none', stroke:'#fff', 'stroke-width':2.2, filter:'url(#glow)'});
        el('animate', {attributeName:'stroke-opacity', values:'1;.25;1', dur:'1.2s', repeatCount:'indefinite'}, ring);
        showSel(f, s);
      }
    }
    live = {L: el('circle', {r:6, fill:'#00e5ff', stroke:'#0b1220', 'stroke-width':1.5, filter:'url(#glow)', opacity:0}),
            R: el('circle', {r:6, fill:'#ff4081', stroke:'#0b1220', 'stroke-width':1.5, filter:'url(#glow)', opacity:0})};
    metrics(); tick(t, true);
  }
  function showSel(f, s){
    const c = $('mapSel'); c.style.display = 'block'; c.innerHTML = '';
    const add = (k, v) => { const d = document.createElement('div'); const a = document.createElement('span'); a.className = 'k'; a.textContent = k + ' ';
      const b = document.createElement('b'); b.textContent = v; d.appendChild(a); d.appendChild(b); c.appendChild(d); };
    const h = document.createElement('div'); h.style.fontWeight = 700; h.style.color = s.foot === 'L' ? '#00e5ff' : '#ff4081';
    h.textContent = `${s.label} · ${s.foot === 'L' ? 'Left' : 'Right'} foot · Beat ${s.beat ?? '–'} · ${s.t.toFixed(2)} s`; c.appendChild(h);
    add('Timing error', s.offset_ms == null ? '–' : `${sgn(s.offset_ms,0)} ms (${s.status})`);
    add('Step length', f.len_cm == null ? '–' : `${f.len_cm.toFixed(0)} cm`);
    add('Position offset', f.err_cm == null ? '– (no ideal target)' : `${f.err_cm.toFixed(0)} cm from ideal footprint`);
  }
  function metrics(){
    const m = F().metrics || {}, c = $('mapMetrics'); c.innerHTML = '';
    const row = (k, v) => { const d = document.createElement('div'); const a = document.createElement('span'); a.className = 'k'; a.textContent = k + ' ';
      const b = document.createElement('b'); b.textContent = v; d.appendChild(a); d.appendChild(b); c.appendChild(d); };
    const cm = v => v == null ? '–' : v.toFixed(0) + ' cm';
    row('Step length L / R', `${cm(m.big_step_len_L_cm)} / ${cm(m.big_step_len_R_cm)}`);
    row('Fwd / back symmetry', m.fwd_back_symmetry_pct == null ? '–' : `${m.fwd_back_symmetry_pct.toFixed(0)}% (${cm(m.fwd_reach_cm)} / ${cm(m.back_reach_cm)})`);
    row('Placement error', cm(m.placement_err_cm));
    const dr = (m.drift || []).reduce((a, d) => d.dist_cm > (a ? a.dist_cm : -1) ? d : a, null);
    row('Max drift', dr ? `${cm(dr.dist_cm)} over ${dr.bars} bars` : '–');
    const L = $('mapLegend'); L.innerHTML = '';
    const lg = (html) => { const d = document.createElement('div'); d.innerHTML = html; L.appendChild(d); };
    lg('<span style="color:#94a3b8">┄ ideal On-1 basic</span> · <span style="color:#e2e8f0">solid = actual</span>');
    lg('<span style="color:#00e5ff">■ left</span> <span style="color:#ff4081">■ right</span> · ring: <span style="color:#22c55e">on time</span> <span style="color:#f59e0b">early/late</span> <span style="color:#ef4444">off-beat</span>');
    lg(mode === 'cycle' ? `<span class="k">cycle ${shownCycle == null ? '–' : shownCycle + 1} of ${F().cycles.length}</span>` : '<span class="k">all steps · <span style="color:#facc15">┈ centre drift</span></span>');
  }
  function lerpTrack(t){
    const tr = F().track; if (!tr || !tr.t.length) return null;
    let lo = 0, hi = tr.t.length - 1; if (t < tr.t[0] || t > tr.t[hi]) return null;
    while (hi - lo > 1){ const m = (lo + hi) >> 1; if (tr.t[m] <= t) lo = m; else hi = m; }
    const a = (t - tr.t[lo]) / Math.max(1e-6, tr.t[hi] - tr.t[lo]), out = {};
    for (const f of ['L', 'R']){ const p = tr[f][lo], q = tr[f][hi]; out[f] = (p && q) ? [p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a] : (p || q); }
    return out;
  }
  function tick(t, force){
    if (view === 'camera' || !F().available) return;
    if (!force && mode === 'cycle' && cycleAt(t) !== shownCycle){ redraw(); return; }
    if (!live || !geom) return;
    const p = lerpTrack(t);
    for (const f of ['L', 'R']){ const q = p && p[f];
      if (!q){ live[f].setAttribute('opacity', 0); continue; }
      const [x, y] = geom.px(q[0], q[1]); live[f].setAttribute('cx', x); live[f].setAttribute('cy', y); live[f].setAttribute('opacity', .95); }
  }
  function setView(v){ view = v; $('stage').className = 'stage mode-' + v;
    document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('on', b.dataset.view === v));
    $('mapModes').style.visibility = v === 'camera' ? 'hidden' : 'visible'; requestAnimationFrame(redraw); }
  function setMode(m){ mode = m; document.querySelectorAll('[data-mapmode]').forEach(b => b.classList.toggle('on', b.dataset.mapmode === m)); redraw(); }
  function select(i){ sel = i; const f = fstep(i); if (f) pin = f.cycle; redraw(); }
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  document.querySelectorAll('[data-mapmode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mapmode)));
  svg.addEventListener('click', () => { sel = null; pin = null; redraw(); });
  new ResizeObserver(() => redraw()).observe($('mapwrap'));
  $('mapModes').style.visibility = 'hidden';
  return {redraw, tick, select, setView, setMode, get state(){ return {view, mode, sel, shownCycle}; }};
})();
function selectStep(i){
  MAP.select(i);
  for (const [s, tr] of rows) tr.style.outline = s.i === i ? '1px solid #ffc400' : '';
}

// ======================= main loop =======================
let lastCur = null, lastDriftUi = 0;
function frame(now){
  syncTick();
  metronomeTick();
  const t = video.currentTime || 0, dur = video.duration || 0;
  $('time').textContent = `${t.toFixed(2)} / ${dur.toFixed(2)} s`;
  if (!scrubbing && dur) $('scrub').value = Math.round(t / dur * 1000);
  if (playhead && TL){ const x = TL.X(Math.min(t, TL.t1)); playhead.setAttribute('x1', x); playhead.setAttribute('x2', x); }
  if (now - lastDriftUi > 200){
    lastDriftUi = now; const d = $('drift');
    if (!audio) d.textContent = 'no song';
    else if (SYNC.drift == null) d.textContent = (targetTime() < 0 && !video.paused) ? 'music starts in ' + (-targetTime()).toFixed(1) + ' s' : 'sync idle';
    else { d.textContent = 'drift ' + sgn(SYNC.drift * 1000, 0) + ' ms'; d.classList.toggle('bad', Math.abs(SYNC.drift) > SYNC.tol); }
  }
  MAP.tick(t);
  let cur = null; for (const [s, tr] of rows){ if (s.t <= t + 1e-3) cur = tr; else break; }
  if (cur !== lastCur){ if (lastCur) lastCur.classList.remove('cur'); if (cur){ cur.classList.add('cur'); if (!video.paused) cur.scrollIntoView({block:'nearest'}); } lastCur = cur; }
  requestAnimationFrame(frame);
}
if (!D.media.video){ $('play').disabled = true; toast('Video not available.'); }
window.__dash = {get offset(){ return offset; }, sync: SYNC, video, audio, setOffset, videoClock, map: MAP, selectStep, get data(){ return D; }};
renderAll(); updatePlayBtn(); requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
