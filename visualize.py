"""Diagnostic plots: rhythm_analysis.png (timeline, foot signals, beat offsets)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

COLORS = {"L": "#1f77b4", "R": "#d62728", "active": "#2ca02c", "beat": "#7f7f7f", "clap": "#9467bd"}


def plot_rhythm_analysis(
    out_path: str | Path,
    subject_id: str,
    fps: float,
    cam1_frames: np.ndarray,
    epochs: Sequence,
    diag,
    rhythm,
    clap_frames_cam1: Sequence[int] = (),
    zoom_s: float = 6.0,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(cam1_frames) / fps
    steps = [s for s in rhythm.steps]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1.4, 1]})
    fig.suptitle(f"{subject_id}: salsa rhythm analysis  "
                 f"(beat period {rhythm.beat_period_sec * 1000:.0f} ms, "
                 f"{len(epochs)} active epochs, {rhythm.summary.get('n_steps', 0)} steps)", fontsize=13)

    # ---- Panel 1: timeline ------------------------------------------------
    ax = axes[0]
    ax.plot(diag.window_centers_sec, diag.energy, color="k", lw=1.2, label="foot motion energy")
    ax.axhline(diag.start_threshold, color="orange", ls="--", lw=1, label="start threshold")
    ax.axhline(diag.stop_threshold, color="orange", ls=":", lw=1, label="stop threshold")
    emax = float(np.nanmax(diag.energy)) if len(diag.energy) else 1.0
    ax.set_ylim(0, max(emax, diag.start_threshold) * 1.35)
    for ep in epochs:
        ax.axvspan(ep.start_sec, ep.end_sec, color=COLORS["active"], alpha=0.18)
        ax.text((ep.start_sec + ep.end_sec) / 2, 0.04 * ax.get_ylim()[1], f"ACTIVE_MOVE #{ep.epoch_id}",
                ha="center", va="bottom", fontsize=8, color="darkgreen")
    for k, f in enumerate(clap_frames_cam1):
        ax.axvline(f / fps, color=COLORS["clap"], lw=1.5, label="clap (sync)" if k == 0 else None)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("energy")
    ax.set_title("Recording timeline: ACTIVE_MOVE (green) vs IDLE")
    ax.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.9)

    # ---- Panel 2: zoomed foot signals ---------------------------------------
    ax = axes[1]
    if epochs:
        z0 = epochs[0].start_sec
        z1 = min(epochs[0].end_sec, z0 + zoom_s)
    else:
        z0, z1 = t[0], min(t[-1], t[0] + zoom_s)
    zm = (t >= z0) & (t <= z1)
    for f in ("L", "R"):
        ax.plot(t[zm], rhythm.foot_speed[f][zm], color=COLORS[f], lw=1.4, label=f"{f} horizontal speed (m/s)")
    ax.set_ylabel("XZ speed (m/s)")
    ax2 = ax.twinx()
    for f in ("L", "R"):
        ax2.plot(t[zm], rhythm.foot_height[f][zm] * 100, color=COLORS[f], lw=1, ls="--", alpha=0.6,
                 label=f"{f} lift Y (cm)")
    ax2.set_ylabel("foot lift Y (cm)")
    for k, b in enumerate(g for g in rhythm.beat_grid_sec if z0 <= g <= z1):
        ax.axvline(b, color=COLORS["beat"], lw=0.8, alpha=0.6, label="music beat" if k == 0 else None)
    ymax = max([rhythm.foot_speed[f][zm].max() for f in ("L", "R")] + [0.1]) if zm.any() else 1.0
    for s in steps:
        if z0 <= s.timestamp_sec <= z1:
            ax.plot(s.timestamp_sec, 0, marker="v" if s.timing_type != "TAP" else "x", ms=9,
                    color=COLORS[s.foot])
            ax.annotate(s.label, (s.timestamp_sec, 0), xytext=(0, 12 + (s.foot == "R") * 14),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS[s.foot],
                        fontweight="bold" if s.timing_type == "SLOW" else "normal")
    ax.set_ylim(-0.05 * ymax, ymax * 1.25)
    ax.set_xlim(z0, z1)
    ax.set_title(f"Zoom {z0:.1f}-{z1:.1f}s: footfalls vs. music beat grid")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=3)

    # ---- Panel 3: beat offsets ------------------------------------------------
    ax = axes[2]
    for ep in epochs:
        ax.axvspan(ep.start_sec, ep.end_sec, color=COLORS["active"], alpha=0.08)
    for f in ("L", "R"):
        for kind, marker in (("QUICK", "o"), ("SLOW", "s")):
            pts = [(s.timestamp_sec, s.beat_offset_ms) for s in steps
                   if s.foot == f and s.timing_type == kind and s.beat_offset_ms is not None]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, color=COLORS[f], marker=marker, s=22, label=f"{f} {kind}")
    k, labelled = 5, False
    for ep in epochs:  # rolling mean per epoch, so the line does not bridge idle gaps
        counted = sorted((s.timestamp_sec, s.beat_offset_ms) for s in steps if s.epoch_id == ep.epoch_id
                         and s.timing_type != "TAP" and s.beat_offset_ms is not None)
        if len(counted) >= k:
            x, y = map(np.asarray, zip(*counted))
            ax.plot(x[k - 1 :], np.convolve(y, np.ones(k) / k, mode="valid"), color="k", lw=1.2,
                    label=None if labelled else f"rolling mean ({k} steps)")
            labelled = True
    ax.axhline(0, color="k", lw=0.8)
    half = rhythm.beat_period_sec * 500
    ax.set_ylim(-half, half)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("cam1 time (s)")
    ax.set_ylabel("beat offset (ms)\n(- early / + late)")
    m = rhythm.summary.get("mean_abs_offset_ms")
    ax.set_title("Step-to-music timing offset" + (f"  (mean |offset| {m:.0f} ms)" if m is not None else ""))
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
