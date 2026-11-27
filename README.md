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
- `overrides.mp3_offset_sec`: where the song starts on the cam1 timeline. Leave it `null` and the song is assumed to start with the video (`0.0 s`). Set a number only if the music started later (positive) or earlier (negative) than cam1. Convention: `cam1_time = mp3_time + mp3_offset_sec`. Automatic cross-correlation alignment is disabled because it was unreliable on noisy room audio; it can be re-enabled with `"music": {"auto_align": true}` in `defaults` or a session's `params`.

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
Any warnings follow the table: an audio/visual sync disagreement, few RANSAC inliers, or a tempo octave correction.

## 5. Outputs (`outputs/<subject_id>/`)
| File | Contents |
|---|---|
| `sync.json` | `frame_offset`, `method` (`manual` / `clap` / `clap+audio` / `motion_xcorr`), confidence, clap frames in each camera, audio cross-check |
| `epochs_and_steps.json` | Everything: sync, music (BPM, `mp3_offset_sec`, `method`: `default` = 0.0 s / `manual` = sessions.json, beat grid), calibration (K, R, t, scale, reprojection error), segmentation thresholds, epochs with per-epoch summaries, every step, overall summary, warnings, the merged parameters used |
| `steps.csv` | One row per footfall: `subject_id, epoch_id, cam1_frame, timestamp_sec, nearest_music_beat_sec, beat_offset_ms, foot, timing_type, beat_count, step_length_cm` |
| `rhythm_analysis.png` | Diagnostic plot (below) |
| `overlay_cam1.mp4` | cam1 with skeleton + live rhythm HUD (see section 6) |
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

## 6. Video overlay and rhythm dashboard

**Skeleton + rhythm overlay video.** `main.py` renders `outputs/<id>/overlay_cam1.mp4` after the analysis (skip it with `--no-overlay`). To re-render it on its own:
```bash
python overlay_video.py --config sessions.json --subject subject_01
```
For a quick preview of the first 20 s:
```bash
python overlay_video.py --config sessions.json --subject subject_01 --max-seconds 20
```
- **Skeleton.** The upper body is thin and semi-transparent. Legs, ankles, heels and toes are crisp and glowing: **electric cyan `#00E5FF` = left foot**, **magenta/coral `#FF4081` = right foot**.
- **Footfall callouts.** For about 0.6 s after each footfall, a ring and a floating tag appear beside that foot. The tag reads `LEFT FOOT • BEAT 3` over a badge:
  - `ON TIME (+12ms)` (green, ≤ ±45 ms);
  - `EARLY (-68ms)` / `LATE (+70ms)` (amber, ≤ ±90 ms);
  - `EARLY / LATE / OFF-BEAT (+115ms)` (red, > 90 ms).
- **Bottom 8-count strip.** `1 2 3 [4] 5 6 7 [8]` lights the active beat with a progress fill, and flashes red on a beat where a timing error happened.
- **Top chips.** `DANCING` / `IDLE`, the BPM, and the running on-time percentage.
- **Encoding.** With ffmpeg available (on PATH, or bundled with `imageio-ffmpeg`), the output is browser-friendly H.264 with the original camera audio.

**Rhythm dashboard (English, single screen).** It needs `epochs_and_steps.json` and ideally `overlay_cam1.mp4`:
```bash
python dashboard.py --config sessions.json --subject subject_01
```
It opens `http://127.0.0.1:8765/`. Options: `--port`, `--no-browser`, and `--read-only` (disables saving).
- **Layout.** It fits one widescreen viewport (`100vh`, no page scroll).
  - Left (≈55%): the video, transport, speed (`0.5x` / `0.75x` / `1.0x`), music / camera-audio / metronome toggles, the audio-sync calibration bar, and the **step error timeline**.
  - Right (≈45%): the tempo score and verdict, KPI badges (BPM, On-tempo %, Mean offset, QQS accuracy), the **coaching diagnosis**, and a scrollable **rhythm error & step log**.
- **Music sync.**
  - The song is served as a sample-accurate WAV, decoded once by ffmpeg to `outputs/<id>/song_sync.wav`. Browsers only seek `.mp3` approximately, which was the main source of drift.
  - **Music offset.** The page starts at `overrides.mp3_offset_sec` from `sessions.json`, or `0.000 s` if it is not set, so the song plays from its start together with the video. It never uses an old offset stored in `epochs_and_steps.json`; steps are re-scored at the offset in use. Audio starts directly inside the ▶ click, which browsers require. If a browser still blocks it, the calibration bar shows "audio blocked: click ▶".
  - The page keeps `audio.currentTime = video.currentTime − mp3_offset_sec` on play, pause, seek and rate change, and in a `requestAnimationFrame` loop that uses a frame-accurate video clock:
    - drift above 40 ms is corrected with a gentle playback-rate trim;
    - drift above 150 ms gets a hard re-seek;
    - for 1.5 s after each (re)start, anything above 10 ms is trimmed, removing play() start-up latency.
  - Music stays paused before the song starts, after it ends, and while the video is buffering. The live drift is shown in the calibration bar.
- **Calibration bar.**
  - `−50ms` / `−10` / `+10` / `+50ms` nudges, and a numeric `mp3_offset_sec` input. Every change re-scores all steps on the server.
  - **Tap Beat 1 Now** snaps the nearest song beat 1 to the current video moment.
  - **Save Offset to sessions.json** writes `overrides.mp3_offset_sec` through `POST /api/save-offset`. This endpoint takes JSON only, same-origin only, and writes the file atomically. Afterwards, re-run `python main.py --config sessions.json --session <id>` to re-score the steps and re-render the overlay with the new offset.
- **Step error timeline.** Each step's error in ms over the recording, with a green ±45 ms band.
  - Left foot = cyan ●, right foot = magenta ◆; amber = early/late; bright red = off-beat (> 90 ms).
  - Hovering shows the foot, time (s), beat and error.
  - Clicking a step replays from **1 s before** it.
- **Step log.** `All steps` / `Errors only` toggle; columns `Time (s)`, `Foot`, `Beat`, `Error (ms)`, `Status`. Clicking a row jumps the video to that step.
- **Keyboard.** Space = play/pause; ←/→ = ±1 s (Shift: ±0.1 s).
- **View switcher:** `Camera + Skeleton` | `Top-Down Footwork Map` | `Split View`. The map is a pure-vector top-down floor reconstruction from the stereo 3D feet (`points3d.npz`), synced with playback:
  - **Floor frame.** Per dance block, the origin is the centre stance before the first step, +y is the dancer's facing (FRONT, from heel→toe; falls back to PCA if toes aren't tracked), and +x is the dancer's right. The grid has 10 cm minor and 50 cm major lines.
  - **Ideal On-1 basic.** Dashed footprints and arrows scaled to `height_meters` (reach ≈ 0.16 × height, about 27 cm at 1.70 m):
    - 1 = left forward;
    - 2 = right in place;
    - 3 = left back to centre;
    - 5 = right backward;
    - 6 = left in place;
    - 7 = right back to centre;
    - 4 & 8 = feet together.
  - **Actual steps.** Solid footprints in cyan (left) / magenta (right) with a timing-coloured contour (green / amber / red), plus dotted error vectors with the distance to the ideal target.
  - **Window modes.** `Current 8-Count Cycle` follows playback, shows faded trails of the two previous cycles, and has live foot markers. `All Steps Overlay` shows every step as a footprint heatmap plus the yellow centre-drift path.
  - **Selection.** Clicking a table row, timeline marker or footprint highlights that footprint and shows its timing error (ms), step length (cm) and position offset (cm).
  - **Metrics:** main-step length L / R, forward (beat 1) vs backward (beat 5) symmetry %, mean placement error, and max centre drift. Placement error is measured against each 8-count cycle's own centre stance, so slow drift is reported separately and does not inflate every step's error. Up to two spatial bullets are added to the coaching panel.
  - **Export.** `main.py` also writes these as `footwork` (metrics, ideal pattern, per-step floor positions) and `floor_x_cm` / `floor_y_cm` / `placement_err_cm` per step in `epochs_and_steps.json`.
- **Off-beat warning.** If every step sits about half a beat from the pulse, the dashboard warns that the song's beat grid is probably on the off-beat. Check with the metronome, then fix it with *Tap Beat 1 Now* and save.

## 7. Live Webcam Coach (single laptop webcam, real time)

The offline two-camera pipeline is unchanged; this is an extra mode in the same dashboard.

Open the live page directly:
```bash
python dashboard.py --config sessions.json --live --camera 0
```
Or start the dashboard as usual and click **Live Webcam Coach** in the header:
```bash
python dashboard.py --config sessions.json
```
For the best frame rate on a CPU, add `--live-model lite`. This downloads `models/pose_landmarker_lite.task` (5.6 MB) on first use. Without the flag, `lite` or `full` is used if present in `models/`, otherwise `heavy`.
```bash
python dashboard.py --config sessions.json --live --live-model lite
```

1. Stand 2–3 m from the laptop, facing it, with your **whole body including your feet** in view.
2. Pick the camera and the music: the **built-in salsa clave / cowbell metronome** (120–200 BPM, son clave 3-2, cowbell accent on 1), or any song from `sessions.json` or `data/`.
3. Press **▶ Start Live Session** and dance the On-1 basic (1-2-3, hold, 5-6-7, hold).
4. Watch the skeleton, ankle callouts and the 8-count strip on the video, the **Live Top-Down Footwork Map**, and the score / coaching / step log on the right.
5. **Save Session** exports to `outputs/live_<timestamp>/`:
   - `cam1.mp4`, `overlay_cam1.mp4`;
   - `epochs_and_steps.json`, `steps.csv`, `sync.json`;
   - the song, or a `metronome.wav` click track matching the recording.

   It also adds the session to `sessions.json` (marked `"live": true`, which `main.py` skips). Its link opens it in **Recorded Sessions** with the full V2 player, error timeline and footwork map.

**How it works** (`realtime.py`)
- **The browser is the master clock.** It plays the song or the Web Audio metronome, and every 200 ms posts the audio position you hear (output latency subtracted) to `POST /api/live/clock`. The server maps that onto its own clock (median offset; pauses, seeks and BPM changes reset it). The header shows the sync jitter (typically ±2 ms).
- **Pose.** MediaPipe Pose Landmarker (VIDEO mode, 1 person) runs on the newest camera frame only, dropping older frames so it never lags behind real time:
  - 2D landmarks draw the skeleton;
  - 3D `pose_world_landmarks` (metres) drive step detection and the floor map.
- **Steps.**
  - **Signal.** A step is a change of the left-to-right foot vector (x, z), which hip motion doesn't affect, or one foot lifting (so in-place replace steps count). The foot that lifted is the stepping foot.
  - **Timing.** The plant time is refined on a rolling 2 s raw buffer: the moment the foot covered 97% of its travel, or its lift dropped. This avoids filter lag.
  - **Latency.** Detection lands about 60–150 ms after touchdown.
  - **Classification.** Each step is scored against the nearest beat: `ON TIME` (≤ ±45 ms), `EARLY` / `LATE` (≤ ±90 ms), `EARLY / LATE / OFF-BEAT` (> 90 ms). Stepping on 4 or 8 is flagged as "stepped on hold", and a same-foot double step as a tap.
- **Floor map.** Each new step is placed relative to the planted foot (chaining from step to step), in a frame aligned with the dancer's facing direction. Targets are the ideal On-1 basic scaled to the height you enter (mirrored automatically if you step on 1 with the right foot).
- **Coaching** updates every 2 bars, e.g. *"Rushing beat 3 on left foot (−91 ms) — hold your slow step."*

**Calibration & limits**
- **Cam latency** (default 60 ms) compensates the webcam's capture delay. If every step reads consistently late or early, adjust it. **Count +1** moves the 8-count if the "1" lands on the wrong beat (songs have no downbeat marker).
- **Facing direction.** A single camera cannot recover depth as well as the stereo pipeline. Face the camera roughly (within about 30°) for reliable forward/back steps.
- **Restarting the song.** If you restart the song during a recording, the saved replay aligns the music with its first playthrough.

### 7b. Ultra-light live coach (`live_coach.py`, standalone)

A separate single-file coach for slower laptops. It does not use `main.py`, `dashboard.py`, `overlay_video.py` or `realtime.py`, and needs only OpenCV, MediaPipe, NumPy and the standard library.
```bash
python live_coach.py --camera 0
```
It opens `http://127.0.0.1:8770/`. It downloads `models/pose_landmarker_lite.task` on first run, and falls back to `pose_landmarker_heavy.task` only when offline.

- **Zero-lag pipeline.**
  - The capture thread reads the camera at 640×360 (DirectShow on Windows, buffer 1) and *overwrites* a single latest-frame slot.
  - MediaPipe Pose Landmarker lite runs in `LIVE_STREAM` mode on the newest frame only. Frames arriving while it is busy are skipped, never queued.
  - A stream thread draws the newest camera frame with the latest skeleton at 30 fps and serves it as JPEG (quality 70) MJPEG on `GET /stream`. The video never waits for the model.
  - The page polls `GET /state` every 100 ms.
- **Music.** The browser plays an `.mp3` from `data/` or a Web Audio son-clave/cowbell metronome (120–200 BPM), and posts its beat clock to `POST /sync`. For songs, set the BPM and press **Tap 1** on a downbeat; no audio-analysis library is needed.
- **Feedback.**
  - Only the salsa-relevant landmarks are drawn: thin shoulder/hip lines, cyan left leg, magenta right leg.
  - A 0.5 s callout appears beside the ankle that stepped: `LEFT • Beat 1 • ON TIME` (green, ≤ ±45 ms), `EARLY` (amber) or `LATE` (red).
  - The bottom 8-count bar shows `1 2 3 [4] 5 6 7 [8]`.
  - The right panel shows the tempo score, on-tempo %, mean offset, a mini footwork pad (target vs. actual dots) and the last 8 steps.
- **Speed measured here:**
  - drawing + JPEG encoding take under 15 ms per 640×360 frame (asserted in the tests);
  - MediaPipe lite in `LIVE_STREAM` mode kept up with a 60 fps feed (~57 results/s, no person in frame) on this container's CPU;
  - with a deliberately slow 40 ms model, the video still streams at 30 fps.

## 8. Troubleshooting
- **Sync `method` is `motion_xcorr`, or the offset looks wrong.** No clap was found in both cameras. Check `clap_frames_cam1/cam2` in `sync.json`, then set `overrides.frame_offset`. One way to find it: note the video frame of the clap in each camera, and use `cam2_frame - cam1_frame`.
- **Every step is off by the same amount.** The song probably did not start exactly with the video. In the dashboard, turn on the metronome, use *Tap Beat 1 Now* or the ±50 ms nudges until the clicks match the music, then *Save Offset to sessions.json* and re-run `main.py` for that session.
- **Too few or too many epochs.** Tune `segmenter.min_start_energy` / `min_stop_energy`, and see the energy curve in panel 1.
- **Beats 1-2-3 and 5-6-7 swapped.** The count is anchored on the first foot of the first complete quick-quick-slow group. If needed, force the beat-1 foot with `rhythm.count_one_foot` (`"L"` or `"R"`).
- **High reprojection error in `calibration`.** Usually a bad sync offset, or a camera that doesn't see the whole body.

## 9. Modules
| Module | Role |
|---|---|
| `landmarks.py` | MediaPipe 33-landmark index constants |
| `pose_2d.py` | PoseLandmarker (heavy, VIDEO mode) to `(N,33,3)` `[x_px, y_px, vis]`, cached `.npz` |
| `sync.py` | Visual clap detector, >2.5 kHz audio cross-check, hip/ankle motion fallback, manual override |
| `music_analyzer.py` | librosa HPSS beat grid and BPM, anchored at `mp3_offset_sec` (0.0 s unless overridden); optional opt-in onset cross-correlation (`auto_align`) |
| `calibrate_3d.py` | Approximate K. RANSAC F re-fitted on its inliers, E = KᵀFK, recoverPose. Triangulation, eye-to-heel height scaling, SVD floor fit to Y = 0 (+Y up) |
| `filters.py` | Vectorised 3D One Euro filter. NaN-gap safe; zero-lag forward/backward mode for offline use |
| `epoch_segmenter.py` | Foot-only motion energy, adaptive hysteresis, gap merging and short-burst removal. Epochs are trimmed to actual foot motion |
| `salsa_rhythm.py` | Footfalls at touchdown with sub-frame timing, L/R alternation and TAPs, QUICK/SLOW and 8-count labels, beat offsets, step lengths, summaries |
| `visualize.py` | `rhythm_analysis.png` |
| `main.py` | Config loading and merging, batch orchestration, exports, summary table, overlay step |
| `overlay_video.py` | Skeleton (glowing legs and feet), foot-specific timing callouts and an 8-count strip rendered onto cam1; H.264 + audio via ffmpeg |
| `live_coach.py` | Standalone ultra-light webcam coach (OpenCV + MediaPipe lite + stdlib HTTP), latest-frame-only threads, 30 fps MJPEG |
| `realtime.py` | Live Webcam Coach: browser-clock sync, metronome/song beat grid, streaming step detector, live floor map, coaching, camera/MediaPipe threads, MJPEG, Save Session |
| `footwork.py` | Top-down floor reconstruction: facing-aligned floor frame, ideal On-1 basic scaled to height, placement error, step length, forward/back symmetry, drift |
| `dashboard.py` | Local single-screen rhythm dashboard (`http.server`): drift-corrected video + music player, offset calibration and saving, scorecard, coaching, top-down footwork map, error timeline and step log |

Each stage also has its own CLI for debugging (`pose_2d.py`, `sync.py`, `music_analyzer.py`).

## 10. Tests
```bash
pytest -q
```
140 synthetic tests (`tests/test_pipeline.py`, `tests/test_overlay_dashboard.py`, `tests/test_realtime.py`, `tests/test_live_coach.py`); no video, audio or model files are needed. They include an end-to-end run of `main.py` on two simulated subjects:
- a timeline of 10 s idle with a clap, 16 s of 180 BPM basic steps, 8 s idle, and 16 s more stepping;
- each subject projected into two virtual cameras;
- checks: sync, epoch boundaries, zero idle/clap false steps, 100% QUICK/SLOW and beat-count accuracy, beat offsets including a deliberate 60 ms late drift, and all exported files.

`tests/test_realtime.py` covers the live coach without a webcam:
- **Streaming step detection** on synthetic MediaPipe-like world-landmark streams: 30 / 24 fps, 200 BPM, a dancer turned 25°, and noise. Every step is detected with the right foot, the right 8-count and a timing error of ±30 ms or better, and nothing is detected while idle or clapping.
- **Live classification:** On Time / Early / Late / Off-beat, hold, tap.
- **Coaching** every 2 bars.
- **Clock and beat grid.**
- **Floor positions**, including the mirrored pattern for right-foot leads.
- **Model selection.**
- **End to end:** a threaded run of the live engine with a fake camera and pose estimator through every `/api/live/*` route, including the MJPEG stream, CSRF checks, Save Session, and replay in Recorded Sessions.

`tests/test_live_coach.py` covers the standalone coach without a webcam:
- the streaming step detector and ON TIME / EARLY / LATE classification on synthetic frames;
- the beat clock, the footwork pad, and the lite/heavy model fallback;
- overlay drawing speed;
- a threaded run with a fake 60 fps camera and a slow 40 ms fake model. It checks that frames are skipped rather than queued, the stream shows the newest frame at 30 fps, `/state` serialises, and `/sync`, `/start`, `/stop` work, with CSRF guards.
