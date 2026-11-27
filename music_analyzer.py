"""Beat grid of the clean song .mp3 and its alignment to the cam1 timeline.

``mp3_offset_sec`` is the cam1 time (s) at which the song's t=0 occurs::

    cam1_time = mp3_time + mp3_offset_sec

It is negative when the song was already playing before cam1 started recording.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import signal

log = logging.getLogger(__name__)

DEFAULT_MUSIC_PARAMS: dict[str, Any] = {
    "bandpass_hz": [60, 250],
    "sr": 22050,
    "hop_length": 512,
    "max_offset_s": None,  # optionally limit |mp3_offset_sec| searched
}


@dataclass
class SongAnalysis:
    bpm: float
    beat_times_sec: np.ndarray  # beat times on the mp3 timeline
    onset_env: np.ndarray  # percussive onset strength envelope
    sr: int
    hop_length: int
    duration_sec: float


@dataclass
class MusicAlignment:
    bpm: float
    beat_period_sec: float
    mp3_offset_sec: float
    aligned_beat_times_sec: list[float]  # beat times on the cam1 timeline
    method: str  # "manual" | "xcorr" | "unaligned"
    confidence: float
    beat_times_mp3_sec: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Song analysis
# --------------------------------------------------------------------------- #
def analyze_song_audio(y: np.ndarray, sr: int = 22050, hop_length: int = 512) -> SongAnalysis:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    _, y_perc = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="time"
    )
    bpm = float(np.atleast_1d(tempo)[0])
    return SongAnalysis(
        bpm=bpm,
        beat_times_sec=np.asarray(beats, dtype=np.float64),
        onset_env=np.asarray(onset_env, dtype=np.float64),
        sr=sr,
        hop_length=hop_length,
        duration_sec=len(y) / sr,
    )


def analyze_song(mp3_path: str, sr: int = 22050, hop_length: int = 512) -> SongAnalysis:
    import librosa

    y, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    return analyze_song_audio(y, sr, hop_length)


# --------------------------------------------------------------------------- #
# Video-audio onset envelope & cross-correlation
# --------------------------------------------------------------------------- #
def bandpass(audio: np.ndarray, sr: int, band_hz=(60, 250), order: int = 4) -> np.ndarray:
    lo, hi = float(band_hz[0]), min(float(band_hz[1]), 0.45 * sr)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float64))


def video_onset_envelope(audio: np.ndarray, sr: int, band_hz=(60, 250), hop_length: int = 512) -> np.ndarray:
    import librosa

    y = bandpass(audio, sr, band_hz).astype(np.float32)
    return np.asarray(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length), dtype=np.float64)


def _norm_env(env: np.ndarray, median_frames: int = 17) -> np.ndarray:
    """Log-compress, remove the local median (sustained noise / loudness drift), half-wave rectify, z-score."""
    from scipy.ndimage import median_filter

    e = np.log1p(np.maximum(np.nan_to_num(np.asarray(env, dtype=np.float64)), 0.0))
    if len(e) > median_frames:
        e = np.maximum(e - median_filter(e, size=median_frames, mode="nearest"), 0.0)
    sd = e.std()
    return (e - e.mean()) / sd if sd > 1e-12 else e * 0.0


def estimate_offset_from_envelopes(
    song_env: np.ndarray,
    video_env: np.ndarray,
    sr: int,
    hop_length: int,
    max_offset_s: float | None = None,
) -> tuple[float, float]:
    """Return ``(mp3_offset_sec, confidence)``.

    The lag k maximises ``corr(video_env[n + k], song_env[n])``. A parabolic
    fit around the peak gives sub-hop precision. The confidence is the Pearson
    correlation of the two envelopes over their overlap at that lag.
    """
    s, v = _norm_env(song_env), _norm_env(video_env)
    c = signal.correlate(v, s, mode="full", method="fft")
    lags = signal.correlation_lags(len(v), len(s), mode="full")
    if max_offset_s is not None:
        m = np.abs(lags) <= int(np.ceil(max_offset_s * sr / hop_length))
        c, lags = c[m], lags[m]
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if abs(den) > 1e-12:
            frac = float(np.clip(0.5 * (c[k - 1] - c[k + 1]) / den, -0.5, 0.5))
    lag = int(lags[k])

    # Pearson over overlap
    n0, n1 = max(0, -lag), min(len(s), len(v) - lag)
    conf = 0.0
    if n1 - n0 > 2:
        a, b = s[n0:n1], v[n0 + lag : n1 + lag]
        if a.std() > 0 and b.std() > 0:
            conf = float(np.corrcoef(a, b)[0, 1])
    return (lag + frac) * hop_length / sr, conf


def estimate_mp3_offset(
    song: SongAnalysis, video_audio: np.ndarray, sr: int, band_hz=(60, 250), max_offset_s: float | None = None
) -> tuple[float, float]:
    venv = video_onset_envelope(video_audio, sr, band_hz, song.hop_length)
    return estimate_offset_from_envelopes(song.onset_env, venv, sr, song.hop_length, max_offset_s)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_alignment(song: SongAnalysis, offset: float, method: str, confidence: float,
                    warnings: list[str] | None = None) -> MusicAlignment:
    bpm = song.bpm
    if len(song.beat_times_sec) >= 2:
        period = float(np.median(np.diff(song.beat_times_sec)))
    else:
        period = 60.0 / bpm if bpm > 0 else float("nan")
    return MusicAlignment(
        bpm=bpm,
        beat_period_sec=period,
        mp3_offset_sec=float(offset),
        aligned_beat_times_sec=(song.beat_times_sec + offset).tolist(),
        method=method,
        confidence=float(confidence),
        beat_times_mp3_sec=song.beat_times_sec.tolist(),
        warnings=list(warnings or []),
    )


def align_music(
    song_mp3: str,
    cam1_video: str | None,
    manual_offset_sec: float | None = None,
    params: dict[str, Any] | None = None,
    song: SongAnalysis | None = None,
) -> MusicAlignment:
    """Analyse the song and place its beat grid on the cam1 timeline."""
    p = {**DEFAULT_MUSIC_PARAMS, **(params or {})}
    sr, hop = int(p["sr"]), int(p["hop_length"])
    if song is None:
        song = analyze_song(song_mp3, sr, hop)

    if manual_offset_sec is not None:
        return build_alignment(song, float(manual_offset_sec), "manual", 1.0)

    if not cam1_video:
        msg = "No cam1 video given; mp3 offset assumed 0"
        log.warning(msg)
        return build_alignment(song, 0.0, "unaligned", 0.0, [msg])

    try:
        from sync import load_video_audio

        video_audio = load_video_audio(cam1_video, sr)
        if video_audio.size < sr or not np.any(np.abs(video_audio) > 1e-6):
            raise RuntimeError("video audio is empty or silent")
    except Exception as exc:
        msg = f"cam1 audio unavailable ({exc}); mp3 offset assumed 0. Set overrides.mp3_offset_sec."
        log.warning(msg)
        return build_alignment(song, 0.0, "unaligned", 0.0, [msg])

    offset, conf = estimate_mp3_offset(song, video_audio, sr, p["bandpass_hz"], p["max_offset_s"])
    warnings = []
    if conf < 0.2:
        warnings.append(f"Low music alignment confidence ({conf:.2f}); consider overrides.mp3_offset_sec")
        log.warning(warnings[-1])
    return build_alignment(song, offset, "xcorr", conf, warnings)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Song beat grid + alignment to cam1.")
    ap.add_argument("song_mp3")
    ap.add_argument("--cam1")
    ap.add_argument("--offset", type=float, help="manual mp3_offset_sec override")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = align_music(args.song_mp3, args.cam1, args.offset)
    text = json.dumps(res.to_dict(), indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
