"""Local tempo-evaluation dashboard (Spanish UI), served with the standard library.

    python dashboard.py --config sessions.json --subject subject_01

It serves the overlay video, the clean song, and an interactive page with the
tempo scorecard, coaching notes, a clickable step timeline and a step table.
It only binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from overlay_video import ON_TIME_MS, SLIGHT_MS, RhythmTimeline, classify_offset

log = logging.getLogger(__name__)

BIAS_MS = 25.0  # |mean offset| above this counts as a tendency to rush or drag
CATEGORY_ES = {
    "on_time": "A tiempo",
    "slight_early": "Levemente adelantado",
    "slight_late": "Levemente atrasado",
    "off": "Fuera de tempo",
    "none": "Sin referencia",
}
TIMING_ES = {"QUICK": "Rápido", "SLOW": "Lento", "TAP": "Toque"}


# --------------------------------------------------------------------------- #
# Scoring, verdict and coaching (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def step_category(offset_ms: float | None) -> str:
    cat, _ = classify_offset(offset_ms)
    if cat == "slight":
        return "slight_early" if offset_ms < 0 else "slight_late"
    return cat


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
        return {"nivel": "sin_datos", "texto": "Sin pasos detectados",
                "detalle": "No se detectaron pasos contados con referencia musical."}
    early = bias_ms is not None and bias_ms < -BIAS_MS
    late = bias_ms is not None and bias_ms > BIAS_MS
    if score >= 85 and not (early or late):
        return {"nivel": "excelente", "texto": "¡Excelente ritmo! En tempo",
                "detalle": "Los pasos caen sobre el pulso de la música de forma estable."}
    if score >= 70:
        if early:
            return {"nivel": "bueno", "texto": "Buen ritmo, con tendencia a adelantarse",
                    "detalle": "Vas en tempo, pero tiendes a pisar antes del pulso."}
        if late:
            return {"nivel": "bueno", "texto": "Buen ritmo, con tendencia a atrasarse",
                    "detalle": "Vas en tempo, pero tiendes a pisar después del pulso."}
        return {"nivel": "bueno", "texto": "Buen ritmo",
                "detalle": "La mayoría de los pasos caen a tiempo; hay algo de irregularidad."}
    if score >= 50:
        if early:
            return {"nivel": "regular", "texto": "Tendencia a adelantarse",
                    "detalle": "Muchos pasos llegan antes del pulso. Escucha la percusión y espera el tiempo."}
        if late:
            return {"nivel": "regular", "texto": "Tendencia a atrasarse",
                    "detalle": "Muchos pasos llegan después del pulso. Anticipa el movimiento del pie."}
        return {"nivel": "regular", "texto": "Ritmo inconsistente",
                "detalle": "Algunos pasos caen a tiempo y otros no; trabaja la regularidad."}
    return {"nivel": "malo", "texto": "Fuera de tiempo",
            "detalle": "La mayoría de los pasos no coinciden con el pulso de la música."}


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _fmt_ms(v: float) -> str:
    return f"{v:+.0f} ms"


def coaching_notes(steps: list[dict], period_s: float, n_taps: int = 0) -> list[dict[str, str]]:
    """Spanish coaching notes derived from step timing patterns."""
    notes: list[dict[str, str]] = []
    c = [s for s in steps if s.get("timing_type") != "TAP" and s.get("beat_offset_ms") is not None]
    if not c:
        return [{"tipo": "info", "texto": "No hay suficientes pasos con referencia musical para dar recomendaciones."}]

    def offs(pred):
        return [s["beat_offset_ms"] for s in c if pred(s)]

    # Strong beats 1 and 5
    strong = offs(lambda s: s.get("beat_count") in (1, 5))
    if strong:
        m, ma = _mean(strong), _mean([abs(o) for o in strong])
        if ma <= ON_TIME_MS:
            notes.append({"tipo": "ok", "texto": f"Los tiempos fuertes 1 y 5 caen a tiempo (promedio {_fmt_ms(m)}). ¡Muy bien!"})
        elif abs(m) <= BIAS_MS:
            notes.append({"tipo": "warn", "texto": f"Los tiempos 1 y 5 son irregulares: a veces antes y a veces después "
                                                   f"del pulso (desfase medio {ma:.0f} ms). Marca el 1 con claridad."})
        elif m < 0:
            notes.append({"tipo": "warn", "texto": f"Los tiempos 1 y 5 llegan adelantados (promedio {_fmt_ms(m)}). "
                                                   "Espera el golpe fuerte de la música antes de pisar."})
        else:
            notes.append({"tipo": "warn", "texto": f"Los tiempos 1 y 5 llegan atrasados (promedio {_fmt_ms(m)}). "
                                                   "Prepara el peso para pisar justo en el 1 y el 5."})

    # Slow steps 3 and 7 (and the 4/8 hold)
    slow = [s for s in c if s.get("timing_type") == "SLOW"]
    if slow:
        m = _mean([s["beat_offset_ms"] for s in slow])
        ma = _mean([abs(s["beat_offset_ms"]) for s in slow])
        holds = [s["interval_sec"] for s in slow if s.get("interval_sec")]
        short_hold = holds and _mean(holds) < 1.8 * period_s
        if ma > SLIGHT_MS and abs(m) <= BIAS_MS and not short_hold:
            notes.append({"tipo": "warn", "texto": f"Los pasos lentos 3 y 7 no caen con el pulso (desfase medio {ma:.0f} ms). "
                                                   "Cuenta en voz alta 1-2-3, 5-6-7 con la música."})
        elif m < -BIAS_MS or short_hold:
            extra = f" y la pausa dura {_mean(holds):.2f} s de {2 * period_s:.2f} s" if short_hold else ""
            notes.append({"tipo": "warn", "texto": f"Los pasos lentos 3 y 7 se apresuran (promedio {_fmt_ms(m)}{extra}). "
                                                   "Sostén la pausa del 4 y del 8 antes de volver a pisar."})
        elif m > BIAS_MS:
            notes.append({"tipo": "warn", "texto": f"Los pasos lentos 3 y 7 llegan tarde (promedio {_fmt_ms(m)}). "
                                                   "Cierra el paso a tiempo y usa el 4 y el 8 para pausar."})
        else:
            notes.append({"tipo": "ok", "texto": f"Los pasos lentos 3 y 7 respetan la pausa del 4 y del 8 (promedio {_fmt_ms(m)})."})

    # Left / right symmetry
    lo = [abs(o) for o in offs(lambda s: s.get("foot") == "L")]
    ro = [abs(o) for o in offs(lambda s: s.get("foot") == "R")]
    if lo and ro:
        ml, mr = _mean(lo), _mean(ro)
        if abs(ml - mr) > 20:
            worse, better = ("izquierdo", "derecho") if ml > mr else ("derecho", "izquierdo")
            notes.append({"tipo": "warn", "texto": f"El pie {worse} es menos preciso que el {better} "
                                                   f"({max(ml, mr):.0f} ms vs {min(ml, mr):.0f} ms de desfase medio)."})
        elif max(ml, mr) > ON_TIME_MS:
            notes.append({"tipo": "info", "texto": f"Ambos pies tienen un desfase parecido (izquierdo {ml:.0f} ms, "
                                                   f"derecho {mr:.0f} ms): el problema es de tempo general, no de un pie."})
        else:
            notes.append({"tipo": "ok", "texto": f"Buena simetría entre pies: izquierdo {ml:.0f} ms, derecho {mr:.0f} ms de desfase medio."})
    ll = [s["step_length_cm"] for s in c if s.get("foot") == "L" and s.get("step_length_cm")]
    rl = [s["step_length_cm"] for s in c if s.get("foot") == "R" and s.get("step_length_cm")]
    if ll and rl:
        a, b = _mean(ll), _mean(rl)
        if max(a, b) > 0 and abs(a - b) / max(a, b) > 0.25:
            big = "izquierdo" if a > b else "derecho"
            notes.append({"tipo": "info", "texto": f"Los pasos del pie {big} son más largos ({max(a, b):.0f} cm vs "
                                                   f"{min(a, b):.0f} cm). Pasos más cortos ayudan a llegar a tiempo."})

    # Drift between dance blocks / over time
    by_epoch: dict[int, list[float]] = {}
    for s in c:
        by_epoch.setdefault(s.get("epoch_id", 0), []).append(s["beat_offset_ms"])
    if len(by_epoch) >= 2:
        ids = sorted(by_epoch)
        first, last = _mean(by_epoch[ids[0]]), _mean(by_epoch[ids[-1]])
        if abs(last - first) > 30:
            word = "atrasarte" if last > first else "adelantarte"
            notes.append({"tipo": "warn", "texto": f"Del primer al último bloque de baile tiendes a {word} "
                                                   f"({_fmt_ms(first)} → {_fmt_ms(last)}). Mantén la concentración en el pulso."})
    elif len(c) >= 12:
        half = len(c) // 2
        first, last = _mean([s["beat_offset_ms"] for s in c[:half]]), _mean([s["beat_offset_ms"] for s in c[half:]])
        if abs(last - first) > 30:
            word = "atrasarte" if last > first else "adelantarte"
            notes.append({"tipo": "warn", "texto": f"Con el paso del tiempo tiendes a {word} ({_fmt_ms(first)} → {_fmt_ms(last)})."})

    std = float(np.std([s["beat_offset_ms"] for s in c]))
    if std > 60:
        notes.append({"tipo": "warn", "texto": f"La regularidad es baja (dispersión de {std:.0f} ms). "
                                               "Practica con el metrónomo a velocidad lenta."})
    if n_taps:
        notes.append({"tipo": "info", "texto": f"Se detectaron {n_taps} toques (TAP) del mismo pie seguidos; "
                                               "no cuentan como pasos. Revisa si hubo cambios de peso incompletos."})
    return notes


def build_dashboard_data(report: dict[str, Any], media: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the web page needs, derived from epochs_and_steps.json."""
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
        # Steps cluster tightly around half a beat: the detected beat grid is probably
        # on the off-beat (beat-phase error), not the dancer. Flag it; never auto-correct.
        # Use |offset|: near half a beat, offsets wrap between +P/2 and -P/2.
        absd = np.abs(np.asarray(offs, dtype=np.float64))
        if absd.mean() > 0.3 * period * 1000 and absd.std() < 0.15 * period * 1000:
            warnings.append(
                f"Los pasos caen de forma muy regular a ~{absd.mean():.0f} ms del pulso (casi medio tiempo). "
                "Es posible que la detección de beats de la canción esté en el contratiempo. "
                "Verifica con el metrónomo y, si es así, corrige overrides.mp3_offset_sec.")

    timeline = RhythmTimeline.from_report(report)
    steps_out = []
    for i, s in enumerate(steps_all):
        cat = step_category(s.get("beat_offset_ms")) if s.get("timing_type") != "TAP" else "none"
        steps_out.append({
            "i": i, "t": s["timestamp_sec"], "foot": s.get("foot"), "label": s.get("label"),
            "timing_type": s.get("timing_type"), "tipo": TIMING_ES.get(s.get("timing_type"), s.get("timing_type")),
            "beat_count": s.get("beat_count"), "offset_ms": s.get("beat_offset_ms"),
            "category": cat, "category_es": CATEGORY_ES[cat], "epoch_id": s.get("epoch_id"),
            "step_length_cm": s.get("step_length_cm"), "cam1_frame": s.get("cam1_frame"),
        })
    epochs = [{"id": e["epoch_id"], "start": e["start_sec"], "end": e["end_sec"]} for e in report.get("epochs", [])]
    t_end = max([e["end"] for e in epochs] + [s["t"] for s in steps_out] + [0.0])
    return {
        "subject_id": report.get("subject_id", ""),
        "fps": report.get("fps"),
        "bpm": music.get("bpm"),
        "beat_period_ms": period * 1000.0,
        "mp3_offset_sec": float(music.get("mp3_offset_sec") or 0.0),
        "music_method": music.get("method"),
        "music_confidence": music.get("confidence"),
        "score": score,
        "verdict": verdict(score, bias),
        "kpis": {
            "bpm": music.get("bpm"),
            "step_bpm": (60.0 / period) if period else None,
            "pct_a_tiempo": (100.0 * float(np.mean([abs(o) <= ON_TIME_MS for o in offs]))) if offs else None,
            "desfase_promedio_ms": bias,
            "desfase_abs_ms": _mean([abs(o) for o in offs]) if offs else None,
            "consistencia_ms": float(np.std(offs)) if offs else None,
            "precision_qqs": (100.0 * qqs) if qqs is not None else None,
            "n_pasos": len(counted),
            "n_taps": n_taps,
            "n_bloques": len(epochs),
        },
        "coaching": coaching_notes(steps_all, period, n_taps),
        "steps": steps_out,
        "epochs": epochs,
        "beats": timeline.beats_with_counts(),
        "duration": t_end,
        "thresholds": {"on_time_ms": ON_TIME_MS, "slight_ms": SLIGHT_MS},
        "warnings": warnings,
        "media": media or {"video": False, "video_is_overlay": False, "audio": False},
    }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def make_handler(data: dict[str, Any], files: dict[str, Path | None]):
    page = INDEX_HTML.encode("utf-8")
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

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
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if route == "/api/data":
                return self._send(200, payload, "application/json; charset=utf-8")
            if route in ("/media/video", "/media/audio"):
                f = files.get(route.rsplit("/", 1)[-1])
                if f is not None and Path(f).is_file():
                    return self._file(Path(f))
            return self._send(404, "No encontrado".encode("utf-8"), "text/plain; charset=utf-8")

    return Handler


def make_server(data: dict[str, Any], video: Path | None, audio: Path | None,
                host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    mimetypes.add_type("audio/mpeg", ".mp3")
    mimetypes.add_type("video/mp4", ".mp4")
    return ThreadingHTTPServer((host, port), make_handler(data, {"video": video, "audio": audio}))


def load_subject(config: str, subject: str | None) -> tuple[dict, Path | None, Path | None, list[str]]:
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
        notes.append("Se muestra el video original (sin esqueleto). Genera el overlay con: "
                     f"python overlay_video.py --config {config} --subject {sess['id']}")
    else:
        video, is_overlay = None, False
        notes.append("No se encontró el video de la cámara 1.")
    audio = Path(sess["song_mp3"]) if Path(sess["song_mp3"]).is_file() else None
    if audio is None:
        notes.append("No se encontró la canción (.mp3); el metrónomo sigue disponible.")
    data = build_dashboard_data(report, {"video": video is not None, "video_is_overlay": is_overlay,
                                         "audio": audio is not None, "notes": notes})
    return data, video, audio, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Panel local de evaluación de tempo (salsa)")
    ap.add_argument("--config", default="sessions.json")
    ap.add_argument("--subject", help="session id (default: first session)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data, video, audio, notes = load_subject(args.config, args.subject)
    for n in notes:
        log.warning(n)
    server = make_server(data, video, audio, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Panel de tempo para {data['subject_id']}: {url}  (Ctrl+C para salir)")
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
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panel de Tempo · Salsa</title>
<style>
:root{
  --bg:#0f1218; --panel:#171b23; --panel2:#1e2430; --line:#2a3140; --text:#e8ecf3; --muted:#9aa4b5;
  --green:#3ecf6e; --green-bg:rgba(62,207,110,.14); --yellow:#f2c230; --yellow-bg:rgba(242,194,48,.14);
  --red:#f0525c; --red-bg:rgba(240,82,92,.14); --blue:#4aa8ff; --left:#22d3ee; --right:#e879f9; --accent:#ffb020;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.2px}
header .sub{color:var(--muted);font-size:13px}
main{max-width:1400px;margin:0 auto;padding:18px 24px 40px;display:grid;gap:18px}
.grid-top{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px}
@media (max-width:1000px){.grid-top{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px;font-weight:600}
video{width:100%;border-radius:8px;background:#000;display:block;max-height:62vh}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-top:12px}
.btn{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}
.btn:hover{border-color:#3b4558}
.btn.on{background:var(--blue);border-color:var(--blue);color:#06121f;font-weight:600}
.group{display:flex;gap:6px;align-items:center}
.lbl{color:var(--muted);font-size:12px}
input[type=range]{width:180px;accent-color:var(--accent)}
label.chk{display:flex;gap:6px;align-items:center;cursor:pointer}
.score-wrap{display:flex;gap:18px;align-items:center}
.ring{position:relative;width:132px;height:132px;flex:none}
.ring svg{transform:rotate(-90deg)}
.ring .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring .val b{font-size:32px;line-height:1}
.ring .val span{font-size:11px;color:var(--muted);margin-top:4px}
.verdict{display:inline-block;padding:6px 12px;border-radius:999px;font-weight:650;font-size:15px}
.v-excelente{background:var(--green-bg);color:var(--green)} .v-bueno{background:rgba(74,168,255,.15);color:var(--blue)}
.v-regular{background:var(--yellow-bg);color:var(--yellow)} .v-malo{background:var(--red-bg);color:var(--red)}
.v-sin_datos{background:var(--panel2);color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .k{color:var(--muted);font-size:12px}
.kpi .v{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.kpi .h{color:var(--muted);font-size:11px}
ul.coach{list-style:none;margin:0;padding:0;display:grid;gap:8px}
ul.coach li{padding:9px 12px;border-radius:8px;border-left:4px solid var(--line);background:var(--panel2)}
ul.coach li.ok{border-left-color:var(--green)} ul.coach li.warn{border-left-color:var(--yellow)} ul.coach li.info{border-left-color:var(--blue)}
.adj{margin-top:10px;color:var(--muted);font-size:12px;min-height:1em}
#chart{width:100%;height:auto;display:block;cursor:crosshair;user-select:none}
#chart .pt{cursor:pointer}
#chart .pt:hover{stroke:#fff;stroke-width:2}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.tablewrap{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:600;font-size:12px;z-index:1}
tbody tr{cursor:pointer} tbody tr:hover{background:#202838} tbody tr.cur{background:rgba(255,176,32,.14)}
.tag{padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}
.c-on_time{background:var(--green-bg);color:var(--green)} .c-slight_early,.c-slight_late{background:var(--yellow-bg);color:var(--yellow)}
.c-off{background:var(--red-bg);color:var(--red)} .c-none{background:var(--panel2);color:var(--muted)}
.foot-L{color:var(--left);font-weight:600} .foot-R{color:var(--right);font-weight:600}
.notes{color:var(--yellow);font-size:12px;margin-top:8px}
.beatbar{display:flex;gap:6px;margin-top:10px}
.beatbar span{flex:1;text-align:center;padding:6px 0;border-radius:6px;background:var(--panel2);color:var(--muted);font-weight:650}
.beatbar span.hold{opacity:.55}
.beatbar span.cur{background:var(--accent);color:#1a1200}
</style>
</head>
<body>
<header>
  <div><h1>Panel de Tempo · <span id="subj"></span></h1><div class="sub" id="hdrsub"></div></div>
  <div class="sub">Tiempos: <b style="color:var(--green)">a tiempo ≤ ±45 ms</b> · <b style="color:var(--yellow)">leve ≤ ±90 ms</b> · <b style="color:var(--red)">fuera &gt; 90 ms</b></div>
</header>
<main>
  <div class="grid-top">
    <section class="card">
      <h2>Video y música sincronizados</h2>
      <video id="video" src="media/video" preload="auto" playsinline controls muted></video>
      <div class="beatbar" id="beatbar"></div>
      <div class="controls">
        <div class="group"><span class="lbl">Velocidad</span>
          <button class="btn" data-rate="0.5">0.5× cámara lenta</button>
          <button class="btn" data-rate="0.75">0.75×</button>
          <button class="btn on" data-rate="1">1.0× normal</button>
        </div>
        <label class="chk"><input type="checkbox" id="musicOn" checked> Música (.mp3)</label>
        <label class="chk"><input type="checkbox" id="camAudio"> Audio de la cámara</label>
        <label class="chk"><input type="checkbox" id="metroOn"> Metrónomo (1-2-3, 5-6-7)</label>
      </div>
      <div class="controls">
        <div class="group"><span class="lbl">Ajuste de audio</span>
          <button class="btn" id="dminus">−10 ms</button>
          <input type="range" id="delta" min="-500" max="500" step="5" value="0">
          <button class="btn" id="dplus">+10 ms</button>
          <button class="btn" id="dreset">Restablecer</button>
        </div>
        <span class="lbl" id="deltaTxt"></span>
      </div>
      <div class="notes" id="notes"></div>
    </section>
    <section class="card">
      <h2>Veredicto general de tempo</h2>
      <div class="score-wrap">
        <div class="ring">
          <svg width="132" height="132" viewBox="0 0 132 132">
            <circle cx="66" cy="66" r="56" fill="none" stroke="#2a3140" stroke-width="12"/>
            <circle id="ringArc" cx="66" cy="66" r="56" fill="none" stroke="#3ecf6e" stroke-width="12" stroke-linecap="round" stroke-dasharray="0 999"/>
          </svg>
          <div class="val"><b id="score">–</b><span>Puntuación</span></div>
        </div>
        <div>
          <div class="verdict" id="verdict"></div>
          <p id="verdictDet" style="color:var(--muted);margin:10px 0 0"></p>
        </div>
      </div>
      <div class="kpis" id="kpis"></div>
      <div class="adj" id="adjTxt"></div>
      <h2 style="margin-top:18px">Recomendaciones</h2>
      <ul class="coach" id="coach"></ul>
    </section>
  </div>

  <section class="card">
    <h2>Línea de tiempo de pasos · desfase respecto al pulso (clic para saltar al video)</h2>
    <svg id="chart" viewBox="0 0 1000 300" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:var(--green)"></i>A tiempo</span>
      <span><i style="background:var(--yellow)"></i>Levemente adelantado / atrasado</span>
      <span><i style="background:var(--red)"></i>Fuera de tempo</span>
      <span>● Paso rápido &nbsp; ■ Paso lento &nbsp; ✕ Toque</span>
      <span>Negativo = adelantado · Positivo = atrasado</span>
    </div>
  </section>

  <section class="card">
    <h2>Pasos detectados (clic en una fila para saltar al video)</h2>
    <div class="tablewrap">
      <table>
        <thead><tr><th>#</th><th>Tiempo</th><th>Bloque</th><th>Pie</th><th>Paso</th><th>Tipo</th><th>Cuenta</th><th>Desfase</th><th>Veredicto</th><th>Long. paso</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </section>
</main>

<script>
(async function(){
const D = await (await fetch('api/data')).json();
const $ = id => document.getElementById(id);
const video = $('video');
const audio = D.media.audio ? new Audio('media/audio') : null;
if (audio) audio.preload = 'auto';
const periodMs = D.beat_period_ms, halfMs = periodMs / 2, TH = D.thresholds;
let delta = 0; // user audio offset adjustment (s); positive = music later

// ---------- header, notes ----------
$('subj').textContent = D.subject_id;
$('hdrsub').textContent = `BPM ${fmt(D.bpm,1)} · alineación musical: ${D.music_method || '–'}` +
  (D.music_confidence != null ? ` (confianza ${fmt(D.music_confidence,2)})` : '') + ` · offset mp3 ${fmt(D.mp3_offset_sec,3)} s`;
$('notes').textContent = [...(D.media.notes || []), ...(D.warnings || [])].join(' · ');
if (!D.media.video) video.replaceWith(Object.assign(document.createElement('div'), {textContent: 'Video no disponible', className: 'notes'}));

function fmt(v, d=0){ return (v == null || Number.isNaN(v)) ? '–' : Number(v).toFixed(d); }
function sgn(v, d=0){ return v == null ? '–' : (v > 0 ? '+' : '') + Number(v).toFixed(d); }
function mmss(t){ const m = Math.floor(t/60), s = t - 60*m; return `${m}:${s.toFixed(2).padStart(5,'0')}`; }
function catOf(off){ if (off == null) return 'none'; const a = Math.abs(off);
  return a <= TH.on_time_ms ? 'on_time' : a <= TH.slight_ms ? (off < 0 ? 'slight_early' : 'slight_late') : 'off'; }
const CAT_ES = {on_time:'A tiempo', slight_early:'Levemente adelantado', slight_late:'Levemente atrasado', off:'Fuera de tempo', none:'Sin referencia'};
const CAT_COL = {on_time:'#3ecf6e', slight_early:'#f2c230', slight_late:'#f2c230', off:'#f0525c', none:'#7b8494'};
// offset after the live audio adjustment, wrapped to the nearest beat
function adjOff(off){ if (off == null) return null; let o = off - delta*1000;
  o = ((o + halfMs) % periodMs + periodMs) % periodMs - halfMs; return o; }

// ---------- scorecard ----------
const v = D.verdict;
$('verdict').textContent = v.texto; $('verdict').className = 'verdict v-' + v.nivel;
$('verdictDet').textContent = v.detalle;
const sc = D.score;
$('score').textContent = sc == null ? '–' : Math.round(sc) + '%';
const C = 2*Math.PI*56, arcCol = v.nivel === 'excelente' ? '#3ecf6e' : v.nivel === 'bueno' ? '#4aa8ff' : v.nivel === 'regular' ? '#f2c230' : '#f0525c';
$('ringArc').setAttribute('stroke-dasharray', `${C*(sc||0)/100} ${C}`); $('ringArc').setAttribute('stroke', arcCol);
const K = D.kpis;
const kpis = [
  ['BPM detectado', fmt(K.bpm,1), `pulso ${fmt(periodMs,0)} ms`],
  ['Pasos a tiempo', K.pct_a_tiempo == null ? '–' : fmt(K.pct_a_tiempo,0)+'%', `${K.n_pasos} pasos · ±${TH.on_time_ms} ms`],
  ['Desfase promedio', K.desfase_promedio_ms == null ? '–' : sgn(K.desfase_promedio_ms,0)+' ms',
     K.desfase_promedio_ms == null ? '' : (K.desfase_promedio_ms < -25 ? 'adelantado' : K.desfase_promedio_ms > 25 ? 'atrasado' : 'centrado') + ` · |medio| ${fmt(K.desfase_abs_ms,0)} ms`],
  ['Consistencia', K.consistencia_ms == null ? '–' : '±'+fmt(K.consistencia_ms,0)+' ms', 'desviación estándar'],
  ['Precisión 1-2-3 / 5-6-7', K.precision_qqs == null ? '–' : fmt(K.precision_qqs,0)+'%', `${K.n_bloques} bloques · ${K.n_taps} toques`],
];
$('kpis').innerHTML = '';
for (const [k, val, h] of kpis){ const d = document.createElement('div'); d.className='kpi';
  d.innerHTML = '<div class="k"></div><div class="v"></div><div class="h"></div>';
  d.children[0].textContent = k; d.children[1].textContent = val; d.children[2].textContent = h; $('kpis').appendChild(d); }
for (const n of D.coaching){ const li = document.createElement('li'); li.className = n.tipo; li.textContent = n.texto; $('coach').appendChild(li); }

// ---------- 8-count bar ----------
const tokens = ['1','2','3','(4)','5','6','7','(8)'];
tokens.forEach((t,i) => { const s = document.createElement('span'); s.textContent = t; if (i===3||i===7) s.classList.add('hold'); $('beatbar').appendChild(s); });
const beats = D.beats || [];
function countAt(t){ // t in cam1 time, with audio adjustment
  let lo = 0, hi = beats.length - 1, k = -1;
  while (lo <= hi){ const m = (lo+hi)>>1; if (beats[m].t + delta <= t){ k = m; lo = m+1; } else hi = m-1; }
  if (k < 0 || t - (beats[k].t + delta) > 1.5*periodMs/1000) return null;
  return beats[k].count;
}

// ---------- table ----------
const tbody = $('tbody');
const rows = D.steps.map(s => {
  const tr = document.createElement('tr');
  const cells = [s.i+1, mmss(s.t), s.epoch_id != null ? '#'+s.epoch_id : '–', s.foot === 'L' ? 'Izq.' : 'Der.', s.label, s.tipo,
                 s.beat_count ?? '–', '', '', s.step_length_cm != null ? fmt(s.step_length_cm,0)+' cm' : '–'];
  cells.forEach((c, j) => { const td = document.createElement('td'); td.textContent = c; if (j===3) td.className = 'foot-'+s.foot; tr.appendChild(td); });
  tr.addEventListener('click', () => seek(s.t));
  tbody.appendChild(tr); return tr;
});
function refreshTable(){
  D.steps.forEach((s, i) => {
    const off = s.timing_type === 'TAP' ? null : adjOff(s.offset_ms), cat = s.timing_type === 'TAP' ? 'none' : catOf(off);
    rows[i].children[7].textContent = off == null ? '–' : sgn(off,0)+' ms';
    rows[i].children[8].innerHTML = ''; const tag = document.createElement('span'); tag.className = 'tag c-'+cat;
    tag.textContent = s.timing_type === 'TAP' ? 'Toque' : CAT_ES[cat]; rows[i].children[8].appendChild(tag);
  });
}

// ---------- timeline chart ----------
const svg = $('chart'), W = 1000, H = 300, PL = 52, PR = 12, PT = 12, PB = 30;
const t0 = Math.max(0, Math.min(...D.epochs.map(e=>e.start), ...D.steps.map(s=>s.t), D.duration) - 1);
const t1 = Math.max(...D.epochs.map(e=>e.end), ...D.steps.map(s=>s.t), 1) + 1;
const yMax = Math.max(120, Math.min(halfMs, 250));
const X = t => PL + (t - t0) / (t1 - t0) * (W - PL - PR);
const Y = o => PT + (1 - (Math.max(-yMax, Math.min(yMax, o)) + yMax) / (2*yMax)) * (H - PT - PB);
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent=svg){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
let playhead;
function drawChart(){
  svg.innerHTML = '';
  for (const e of D.epochs) el('rect', {x:X(e.start), y:PT, width:Math.max(1,X(e.end)-X(e.start)), height:H-PT-PB, fill:'rgba(74,168,255,.07)'});
  el('rect', {x:PL, y:Y(TH.slight_ms), width:W-PL-PR, height:Y(-TH.slight_ms)-Y(TH.slight_ms), fill:'rgba(242,194,48,.08)'});
  el('rect', {x:PL, y:Y(TH.on_time_ms), width:W-PL-PR, height:Y(-TH.on_time_ms)-Y(TH.on_time_ms), fill:'rgba(62,207,110,.16)'});
  for (const o of [-yMax, -TH.slight_ms, -TH.on_time_ms, 0, TH.on_time_ms, TH.slight_ms, yMax]){
    el('line', {x1:PL, x2:W-PR, y1:Y(o), y2:Y(o), stroke: o===0 ? '#5b6577' : '#262d3a', 'stroke-width': o===0?1.2:1});
    const tx = el('text', {x:PL-6, y:Y(o)+4, 'text-anchor':'end', fill:'#9aa4b5', 'font-size':11}); tx.textContent = (o>0?'+':'')+Math.round(o);
  }
  const span = t1 - t0, stepT = span > 300 ? 60 : span > 120 ? 20 : span > 40 ? 10 : 5;
  for (let t = Math.ceil(t0/stepT)*stepT; t <= t1; t += stepT){
    el('line', {x1:X(t), x2:X(t), y1:H-PB, y2:H-PB+4, stroke:'#5b6577'});
    const tx = el('text', {x:X(t), y:H-PB+17, 'text-anchor':'middle', fill:'#9aa4b5', 'font-size':11}); tx.textContent = mmss(t).replace(/\.\d+$/,'');
  }
  const yl = el('text', {x:14, y:(H-PB)/2, fill:'#9aa4b5', 'font-size':11, transform:`rotate(-90 14 ${(H-PB)/2})`, 'text-anchor':'middle'}); yl.textContent = 'desfase (ms)';
  // per-epoch rolling mean
  for (const e of D.epochs){
    const pts = D.steps.filter(s => s.epoch_id === e.id && s.timing_type !== 'TAP' && s.offset_ms != null).map(s => [s.t, adjOff(s.offset_ms)]);
    if (pts.length < 5) continue; let d = '';
    for (let i = 4; i < pts.length; i++){ const m = (pts[i][1]+pts[i-1][1]+pts[i-2][1]+pts[i-3][1]+pts[i-4][1])/5; d += (d?'L':'M') + X(pts[i][0]) + ' ' + Y(m); }
    el('path', {d, fill:'none', stroke:'#cfd6e3', 'stroke-width':1.5, 'stroke-opacity':.7});
  }
  for (const s of D.steps){
    const off = s.timing_type === 'TAP' ? 0 : adjOff(s.offset_ms); if (off == null) continue;
    const cat = s.timing_type === 'TAP' ? 'none' : catOf(off), x = X(s.t), y = Y(off), col = CAT_COL[cat];
    let p;
    if (s.timing_type === 'SLOW') p = el('rect', {x:x-5, y:y-5, width:10, height:10, fill:col, class:'pt'});
    else if (s.timing_type === 'TAP') p = el('path', {d:`M${x-5} ${y-5}L${x+5} ${y+5}M${x+5} ${y-5}L${x-5} ${y+5}`, stroke:col, 'stroke-width':2, class:'pt'});
    else p = el('circle', {cx:x, cy:y, r:4.5, fill:col, class:'pt'});
    const tt = document.createElementNS(NS, 'title');
    tt.textContent = `${s.label} · ${mmss(s.t)} · ${s.timing_type==='TAP' ? 'Toque' : sgn(off,0)+' ms · '+CAT_ES[cat]}`; p.appendChild(tt);
    p.addEventListener('click', ev => { ev.stopPropagation(); seek(s.t); });
  }
  playhead = el('line', {x1:PL, x2:PL, y1:PT, y2:H-PB, stroke:'#ffb020', 'stroke-width':1.5});
}
svg.addEventListener('click', ev => { // click on empty chart area also seeks
  const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
  if (x >= PL && x <= W-PR) seek(t0 + (x - PL) / (W - PL - PR) * (t1 - t0));
});

// ---------- live adjustment summary ----------
function refreshAdj(){
  $('deltaTxt').textContent = `${delta >= 0 ? '+' : ''}${Math.round(delta*1000)} ms · offset total ${(D.mp3_offset_sec + delta).toFixed(3)} s`;
  if (Math.abs(delta) < 1e-9){ $('adjTxt').textContent = ''; return; }
  const offs = D.steps.filter(s => s.timing_type !== 'TAP' && s.offset_ms != null).map(s => adjOff(s.offset_ms));
  if (!offs.length) return;
  const pct = 100 * offs.filter(o => Math.abs(o) <= TH.on_time_ms).length / offs.length;
  const mean = offs.reduce((a,b)=>a+b,0) / offs.length;
  $('adjTxt').textContent = `Con el ajuste de audio (${sgn(delta*1000,0)} ms): ${pct.toFixed(0)}% de pasos a tiempo, desfase promedio ${sgn(mean,0)} ms. La puntuación general usa la alineación original.`;
}
function setDelta(ms){ delta = Math.max(-500, Math.min(500, ms)) / 1000; $('delta').value = Math.round(delta*1000);
  refreshAdj(); refreshTable(); drawChart(); syncAudio(true); scheduled.clear(); }
$('delta').addEventListener('input', e => setDelta(+e.target.value));
$('dminus').onclick = () => setDelta(delta*1000 - 10); $('dplus').onclick = () => setDelta(delta*1000 + 10); $('dreset').onclick = () => setDelta(0);

// ---------- player sync ----------
function musicTime(){ return video.currentTime - (D.mp3_offset_sec + delta); }
function syncAudio(force){
  if (!audio) return;
  const target = musicTime(), dur = audio.duration || Infinity;
  if (video.paused || !$('musicOn').checked || target < 0 || target >= dur){
    if (!audio.paused) audio.pause();
    if (target >= 0 && target < dur && Math.abs(audio.currentTime - target) > 0.02) audio.currentTime = target;
    return;
  }
  audio.playbackRate = video.playbackRate;
  if (force || Math.abs(audio.currentTime - target) > 0.06) audio.currentTime = target;
  if (audio.paused) audio.play().catch(() => {});
}
function seek(t){ video.currentTime = Math.max(0, t); scheduled.clear(); syncAudio(true); }
['play','pause','seeked','ratechange'].forEach(ev => video.addEventListener(ev, () => { scheduled.clear(); syncAudio(true); }));
setInterval(() => syncAudio(false), 250);
$('musicOn').addEventListener('change', () => syncAudio(true));
$('camAudio').addEventListener('change', e => { video.muted = !e.target.checked; });
document.querySelectorAll('[data-rate]').forEach(b => b.addEventListener('click', () => {
  video.playbackRate = +b.dataset.rate; document.querySelectorAll('[data-rate]').forEach(x => x.classList.toggle('on', x === b));
}));

// ---------- metronome (Web Audio, beats 1 2 3 5 6 7) ----------
let actx = null; const scheduled = new Set();
$('metroOn').addEventListener('change', e => { if (e.target.checked && !actx) actx = new (window.AudioContext || window.webkitAudioContext)(); scheduled.clear(); });
function click(when, accent){
  const o = actx.createOscillator(), g = actx.createGain();
  o.frequency.value = accent ? 1760 : 1100; o.type = 'square';
  g.gain.setValueAtTime(0.0001, when); g.gain.exponentialRampToValueAtTime(accent ? 0.35 : 0.2, when + 0.002);
  g.gain.exponentialRampToValueAtTime(0.0001, when + 0.05);
  o.connect(g).connect(actx.destination); o.start(when); o.stop(when + 0.06);
}
function metronomeTick(){
  if (!actx || !$('metroOn').checked || video.paused) return;
  const vt = video.currentTime, rate = video.playbackRate || 1;
  for (let k = 0; k < beats.length; k++){
    const b = beats[k]; if (![1,2,3,5,6,7].includes(b.count)) continue;
    const bt = b.t + delta; if (bt < vt - 0.01) continue; if (bt - vt > 0.15 * rate) break;
    if (scheduled.has(k)) continue; scheduled.add(k);
    click(actx.currentTime + Math.max(0, (bt - vt) / rate), b.count === 1);
  }
}

// ---------- animation loop ----------
let lastCur = -1;
function frame(){
  const t = video.currentTime || 0;
  if (playhead){ const x = Math.max(PL, Math.min(W-PR, X(t))); playhead.setAttribute('x1', x); playhead.setAttribute('x2', x); }
  const c = countAt(t); const spans = $('beatbar').children;
  for (let i = 0; i < 8; i++) spans[i].classList.toggle('cur', c === i+1);
  let cur = -1; for (let i = 0; i < D.steps.length && D.steps[i].t <= t + 1e-3; i++) cur = i;
  if (cur !== lastCur){ if (lastCur >= 0) rows[lastCur].classList.remove('cur');
    if (cur >= 0){ rows[cur].classList.add('cur'); if (!video.paused) rows[cur].scrollIntoView({block:'nearest'}); } lastCur = cur; }
  metronomeTick();
  requestAnimationFrame(frame);
}
refreshAdj(); refreshTable(); drawChart(); requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
