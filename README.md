# Dual-Camera 3D MediaPipe Salsa Rhythm Pipeline

Reconstructs metric 3D pose from two angled cameras (no checkerboard), segments active dancing from idle time, aligns the song `.mp3`, and scores `1-2-3, 5-6-7` (quick-quick-slow) salsa step timing.

## Setup
```bash
pip install -r requirements.txt   # Python 3.10+
```
On headless Linux, mediapipe also needs `libegl1` and `libgles2` (`apt-get install libegl1 libgles2`).

## Layout
| Module | Role | Status |
|---|---|---|
| `landmarks.py` | MediaPipe 33-landmark index constants | done |
| `pose_2d.py` | PoseLandmarker (heavy, VIDEO mode) to `(N,33,3)` `[x_px, y_px, vis]`, cached as `.npz` | done |
| `sync.py` | cam1/cam2 frame offset: visual clap, >2.5 kHz audio check, hip/ankle motion xcorr fallback, manual override | done |
| `music_analyzer.py` | librosa HPSS beat grid + mp3 to cam1 alignment (`cam1_time = mp3_time + mp3_offset_sec`) | done |
| `calibrate_3d.py` | F-matrix/recoverPose extrinsics, triangulation, height scale, floor at Y=0 (+Y up) | done |
| `filters.py` | vectorised 3D One Euro filter, NaN-gap safe | done |
| `epoch_segmenter.py` | active vs. idle epochs | todo |
| `salsa_rhythm.py`, `visualize.py` | footfalls, QUICK/SLOW, beat offsets, plots | todo |
| `main.py` | batch runner over `sessions.json` | todo |

Copy `sessions.example.json` to `sessions.json` and edit the paths and heights. Inputs go in `data/<id>/`, caches in `cache/`, results in `outputs/<id>/`.

**Offset convention:** `cam2_frame = cam1_frame + frame_offset` (cam1 is the master timeline).

## CLI
```bash
python pose_2d.py data/subject_01/cam1.mp4 --cache cache/subject_01/cam1_pose2d.npz
python sync.py cache/subject_01/cam1_pose2d.npz cache/subject_01/cam2_pose2d.npz \
    --video1 data/subject_01/cam1.mp4 --video2 data/subject_01/cam2.mp4 --out outputs/subject_01/sync.json
```

```bash
python music_analyzer.py data/subject_01/song.mp3 --cam1 data/subject_01/cam1.mp4 --out outputs/subject_01/music.json
```

## Tests
```bash
pytest -q
```
