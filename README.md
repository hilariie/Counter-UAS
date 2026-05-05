# Counter-UAS Perception-to-Intercept

https://github.com/user-attachments/assets/a017b7b8-32b0-46b8-a3c2-3db582819e3b

A simulated drone-defense system. Intruder drones approach a defended position; the system spots them with radar and a camera, decides which one is the biggest threat, locks a virtual gimbal camera onto it, and computes how to intercept it — all live, in a single operator dashboard.

Built solo on a single 6 GB GPU, in a Cosys-AirSim simulation. Portfolio project.

## What it is

A friendly drone (the "Ownship") hovers at a point in space and watches the sky. A synthetic radar sweeps the area. A wide-angle camera looks out, with a YOLO model behind it picking up anything that looks like a drone. When something flies into range, the system fuses the radar return with the camera detection into a single track, scores how dangerous it looks, and ranks it against everything else in the air.

The most dangerous track gets the camera's full attention: a second, zoomed-in gimbal camera slews onto it. A third model confirms it's really a drone. From that point the system tracks its 3D position and velocity, and continuously computes an intercept solution — where to send a defender if you wanted to take it down.

All of this is rendered live in a four-panel operator dashboard with a status strip and a top-down minimap.

## The headline moment

Mid-engagement the radar drops out. Within a third of a second the dashboard fades to red — a "RADAR DOWN — VISION ONLY" banner, blinking double border, hazard chevrons in the corners. The system flips to camera-only tracking. Uncertainty visibly grows on the minimap (the confidence circle around the target balloons). When the radar comes back the alert clears and the system snaps back to a tight, high-confidence track.

That sequence is the demo's punchline: the architecture is designed to degrade gracefully under sensor loss, and the UI shows the operator exactly when and why it happened.

## What you'd see in the operator UI

Single 1280×720 window, four panels:

- **Wide camera (left)** — live wide-angle view. Yellow boxes show where the radar is pointing the camera. Cyan boxes are camera detections.
- **Narrow gimbal (top right)** — zoomed-in view of the chosen target. Green box around it. A status pill in the corner shows what the controller is doing (acquiring, committed, lost). A big green banner appears when a valid intercept is computed.
- **Threat list (mid right)** — the top five tracks, ranked. Each row shows ID, range, bearing, and a score. The currently chosen target is highlighted. Small "R/B/S" chips show which sensors are contributing (Radar / Bearing camera / Stereo-from-motion). A coloured dot per row shows filter health.
- **Minimap (bottom left)** — top-down tactical view. The Ownship is a triangle that rotates with its heading. Range rings labeled at 100 m and 200 m. Tracks appear as coloured dots (red = dangerous, dim = low priority) with little arrows for their velocity. The chosen target gets a yellow ring and, on commit, a translucent red uncertainty ellipse. A dashed amber arc shows the intercept geometry.

Plus a status strip across the bottom right: mode, frame rate, elapsed time, lost-frame count, number of tracks, chosen ID.

## How it works (short version)

The pipeline runs once per frame:

1. **Sense** — synthetic radar returns range/bearing/range-rate; wide camera grabs a frame; a YOLO model runs on the camera frame.
2. **Cue** — radar drives the threat ranking by default; the wide camera corroborates by inspecting small regions around radar bearings. (If radar fails, the camera takes over the ranking job.)
3. **Slew** — the highest-ranked track gets the narrow gimbal pointed at it.
4. **Confirm** — a second YOLO inference on the narrow feed positively IDs the target.
5. **Estimate** — a Kalman filter fuses radar range and camera bearings into a 3D position+velocity estimate per track.
6. **Intercept** — once the target is positively confirmed, a lead-pursuit solver computes time-to-intercept and a heading toward the target.

The dashboard renders the whole picture every frame.

## Design choices that matter

The headline architectural decision: **radar leads by default, camera takes over when radar drops**. Real-world counter-drone systems work this way because radars are cheap, omnidirectional, and weather-tolerant, while cameras are high-resolution but narrow-field and compute-heavy. Mirroring that doctrine made the demo's sensor-failure sequence a natural fit instead of a contrived one.

The interception is gated on visual confirmation. A track that's only seen by radar gets a position estimate, but no intercept solution — the system refuses to compute a kill geometry against something it hasn't laid eyes on. That's the same positive-ID rule a real fielded system would enforce.

Under the hood: per-track extended Kalman filter in world NED, lead-pursuit proportional-navigation solver, commit-gated intercept release, hot-reloadable scoring weights. None of that is the point of the project — the point is that it ships, end to end, on one consumer GPU.

## Run it yourself

You'll need the Cosys-AirSim Blocks build (included), Python 3.10+, and a CUDA-capable GPU.

```powershell
# 1. Activate the venv
.\cuas-venv\Scripts\Activate.ps1

# 2. Launch the simulator (separate terminal)
.\Blocks_packaged_Windows_52_33\Windows\Blocks.exe -windowed -ResX=1280 -ResY=720

# 3. Headline demo: spawn intruders, run the full UI for 30 s
python cuas\scripts\05_operator_ui_demo.py --duration 30 --launch-intruders --state --intercept

# 4. Sensor-failure variant: kill the radar 12 s in, restore 8 s later
python cuas\scripts\05_operator_ui_demo.py --duration 30 --launch-intruders --state --intercept --fail-radar-at 12 --fail-radar-duration 8

# 5. Save the run as mp4
python cuas\scripts\05_operator_ui_demo.py --duration 30 --launch-intruders --state --intercept --save-video out.mp4
```

## Repo layout

- `cuas/cuas/` — the runtime package. One subfolder per subsystem: `perception/`, `cueing/`, `tracking/`, `state/`, `intercept/`, `viz/`, `sim/`.
- `cuas/scripts/` — three runnable demos (`03_cueing_demo.py`, `04_tracking_demo.py`, `05_operator_ui_demo.py`).
- `cuas/tests/` — pytest suite (~160 tests) covering geometry, scoring, association, EKF math, intercept geometry, and UI frame construction.
- `cuas/configs/` — YAML config (cueing scoring weights are hot-reloadable).
- `yolo/` — the training side: dataset converter, training script, fine-tuned weights for the wide and narrow detectors.

## Tests

```powershell
cuas-venv\Scripts\python.exe -m pytest cuas/tests/ -v
```

Tests marked `@pytest.mark.sim` or `@pytest.mark.gpu` are skipped by default; pass `-m sim` (with Blocks running) or `-m gpu` to include them.

## Stack

Python 3.10, PyTorch, Ultralytics YOLO + SAHI, OpenCV, Pygame, NumPy/SciPy, Cosys-AirSim 3.3.0 on Unreal Engine 5.2.1. Single Windows machine, RTX 2060 6 GB.

## License & contact

See `LICENSE`. Contact: holaryc@gmail.com.
