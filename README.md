# Dual-Camera 3D MediaPipe Salsa Rhythm Pipeline

Reconstructs metric 3D pose from two angled phone/cameras (no checkerboard), finds the stretches where the
subject is actually dancing, lines the clean song `.mp3` up with the video, and scores the timing of every
`1-2-3, 5-6-7` (quick-quick-slow) salsa basic step against the music.

## 1. Setup
```bash
pip install -r requirements.txt   # Python 3.10+
```
On headless Linux, mediapipe also needs the EGL/GLES libraries:
```bash
apt-get install -y libegl1 libgles2
```
The MediaPipe `pose_landmarker_heavy.task` model (~30 MB) downloads automatically to `models/` on first run.

## 2. Recording checklist
- Two cameras, front-left (`cam1`) and front-right (`cam2`), 60–90° apart, both seeing the whole body including the feet, for the whole take.
- **Clap once or twice** with the hands clearly visible, while standing still near the start. That clap synchronises the cameras.
- Flat shoes, and a measured standing height for each subject.
- Idle time (walking in, standing, resting) is fine. It is detected and excluded automatically.

## 3. Configure `sessions.json`
```bash
cp sessions.example.json sessions.json
```
Edit one entry per subject. Relative paths are resolved against the folder that holds `sessions.json`.
```json
{
  "id": "subject_01",
  "cam1": "data/subject_01/cam1.mp4",
  "cam2": "data/subject_01/cam2.mp4",
  "song_mp3": "data/subject_01/song.mp3",
  "height_meters": 1.68,
  "overrides": {"frame_offset": null, "mp3_offset_sec": null},
  "params": {}
}
```
- `defaults` holds every tunable parameter, grouped by module (`sync`, `music`, `calibration`, `filters`, `segmenter`, `rhythm`).
- `params` in a session overrides any of them for that subject only, deep-merged. For example: `{"rhythm": {"count_one_foot": "R"}}`.
- `overrides.frame_offset`: force the camera sync if clap detection gets it wrong. Convention: `cam2_frame = cam1_frame + frame_offset`.
- `overrides.mp3_offset_sec`: force the music alignment. Convention: `cam1_time = mp3_time + mp3_offset_sec`, negative if the song was already playing when cam1 started.

## 4. Run
```bash
python main.py --config sessions.json
```
To process only one subject:
```bash
python main.py --config sessions.json --session subject_01
```
To re-run 2D pose extraction and ignore the `cache/*.npz` files:
```bash
python main.py --config sessions.json --force-pose
```
Pose extraction is the slow part, roughly real-time on a CPU. It is cached per video, so re-runs after changing analysis parameters take seconds.

The run ends with a comparison table:
```
+-----------+--------+-------+------+--------+-------+------+--------------+---------+---------+
| Subject   | Status | BPM   | Sync | Epochs | Steps | Taps | Mean|off| ms | Bias ms | QQS acc |
+-----------+--------+-------+------+--------+-------+------+--------------+---------+---------+
| subject_A | ok     | 180.0 | clap | 2      | 72    | 0    | 42.1         | +41.8   | 100%    |
| subject_B | ok     | 180.0 | clap | 2      | 72    | 0    | 43.1         | +41.7   | 100%    |
+-----------+--------+-------+------+--------+-------+------+--------------+---------+---------+
```
(That example is the synthetic test run, where the second epoch is deliberately danced 60 ms late.)
Any warnings follow the table: an audio/visual sync disagreement, low music-alignment confidence, few RANSAC inliers, or a tempo octave correction.

## 5. Outputs (`outputs/<subject_id>/`)
| File | Contents |
|---|---|
| `sync.json` | `frame_offset`, `method` (`manual` / `clap` / `clap+audio` / `motion_xcorr`), confidence, clap frames in each camera, audio cross-check |
| `epochs_and_steps.json` | Everything: sync, music alignment (BPM, offset, beat grid), calibration (K, R, t, scale, reprojection error), segmentation thresholds, epochs with per-epoch summaries, every step, overall summary, warnings, the merged parameters used |
| `steps.csv` | One row per footfall: `subject_id, epoch_id, cam1_frame, timestamp_sec, nearest_music_beat_sec, beat_offset_ms, foot, timing_type, beat_count, step_length_cm` |
| `rhythm_analysis.png` | Diagnostic plot (below) |
| `points3d.npz` | Filtered metric 3D landmarks `(N, 33, 3)` plus the `cam1_frames` they belong to |

**Reading the results**
- `beat_offset_ms` = step time minus the nearest music beat. **Negative = early, positive = late.** At 180 BPM a beat is 333 ms, so offsets are bounded to ±167 ms.
- `timing_type` is `QUICK` (beats 1, 2, 5, 6), `SLOW` (beats 3 and 7, followed by the 4/8 hold), or `TAP`. A `TAP` is a same-foot double trigger and is excluded from counting and statistics.
- `beat_count` is the 8-count position. `hold_after` marks the `4-hold` / `8-hold` after a SLOW step.
- In the summaries:
  - `mean_abs_offset_ms`: overall timing accuracy.
  - `mean_offset_ms`: bias; `timing_tendency` labels it early, on-beat or late.
  - `std_offset_ms`: consistency.
  - `pct_within_50ms`: share of steps landing within 50 ms of a beat.
  - `qqs_pattern_accuracy`: share of steps that sit in a clean quick-quick-slow group.

**`rhythm_analysis.png`**
1. **Timeline.** Foot-motion energy with the hysteresis thresholds. Green bands are `ACTIVE_MOVE` epochs and the purple lines are the sync claps.
2. **Zoom** on the first 6 s of dancing. Shows left/right foot horizontal speed (solid) and lift (dashed), the music beat grid (grey), and labelled footfalls (`L1, R2, L3-SLOW, R5, L6, R7-SLOW`).
3. **Timing offset** for every step, with a rolling mean per epoch. Drift towards early or late shows up immediately.

## 6. Troubleshooting
- **Sync `method` is `motion_xcorr`, or the offset looks wrong.** No clap was found in both cameras. Check `clap_frames_cam1/cam2` in `sync.json`, then set `overrides.frame_offset`. One way to find it: note the video frame of the clap in each camera, and use `cam2_frame - cam1_frame`.
- **Music `confidence` < 0.2, or every step is off by the same amount.** The music alignment is likely one beat out. Set `overrides.mp3_offset_sec`.
- **Too few or too many epochs.** Tune `segmenter.min_start_energy` / `min_stop_energy`, and see the energy curve in panel 1.
- **Beats 1-2-3 and 5-6-7 swapped.** The count is anchored on the first foot of the first complete quick-quick-slow group. If needed, force the beat-1 foot with `rhythm.count_one_foot` (`"L"` or `"R"`).
- **High reprojection error in `calibration`.** Usually a bad sync offset, or a camera that doesn't see the whole body.

## 7. Modules
| Module | Role |
|---|---|
| `landmarks.py` | MediaPipe 33-landmark index constants |
| `pose_2d.py` | PoseLandmarker (heavy, VIDEO mode) to `(N,33,3)` `[x_px, y_px, vis]`, cached `.npz` |
| `sync.py` | Visual clap detector, >2.5 kHz audio cross-check, hip/ankle motion fallback, manual override |
| `music_analyzer.py` | librosa HPSS beat grid; mp3 to cam1 alignment via a 60–250 Hz onset envelope cross-correlation |
| `calibrate_3d.py` | Approximate K. RANSAC F re-fitted on its inliers, E = KᵀFK, recoverPose. Triangulation, eye-to-heel height scaling, SVD floor fit to Y = 0 (+Y up) |
| `filters.py` | Vectorised 3D One Euro filter. NaN-gap safe; zero-lag forward/backward mode for offline use |
| `epoch_segmenter.py` | Foot-only motion energy, adaptive hysteresis, gap merging and short-burst removal. Epochs are trimmed to actual foot motion |
| `salsa_rhythm.py` | Footfalls at touchdown with sub-frame timing, L/R alternation and TAPs, QUICK/SLOW and 8-count labels, beat offsets, step lengths, summaries |
| `visualize.py` | `rhythm_analysis.png` |
| `main.py` | Config loading and merging, batch orchestration, exports, summary table |

Each stage also has its own CLI for debugging (`pose_2d.py`, `sync.py`, `music_analyzer.py`).

## 8. Tests
```bash
pytest -q
```
46 synthetic tests; no video, audio or model files are needed. They include an end-to-end run of `main.py` on two simulated subjects:
- a timeline of 10 s idle with a clap, 16 s of 180 BPM basic steps, 8 s idle, and 16 s more stepping;
- each subject projected into two virtual cameras;
- checks: sync, epoch boundaries, zero idle/clap false steps, 100% QUICK/SLOW and beat-count accuracy, beat offsets including a deliberate 60 ms late drift, and all exported files.
