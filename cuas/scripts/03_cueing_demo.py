"""Module 2 end-to-end demo: perception + radar mock + cueing tracker.

Prints a live ranked-track table each frame. Edit configs/cueing_weights.yaml
mid-run to see the ranking shift.

Modes:
  --mode radar-led  (default): radar drives track creation; wide-FOV detector
      runs at --detect-hz on radar-cued ROIs only (on_cue cadence).
  --mode cv-led: detector runs full-frame (+ SAHI if not --no-sahi); radar
      enriches tracks with range/rate. For ablation and Sequence-3 sensor-fail.

Run from cuas/ (Blocks must already be running):
    python -m scripts.03_cueing_demo
    python -m scripts.03_cueing_demo --mode cv-led --no-sahi
    python -m scripts.03_cueing_demo --launch-intruders --save-frames
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import cosysairsim as airsim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuas.perception import WideFovDetector, WideFovFrameSource
from cuas.sim.radar_mock import RadarMock
from cuas.cueing import (
    CameraIntrinsics, CameraMount, CueingTracker, WeightsLoader,
)


DEFAULT_WEIGHTS_PT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yolo", "runs", "uav-2", "weights", "best.pt")
)
DEFAULT_WEIGHTS_YAML = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "cueing_weights.yaml")
)

OWNSHIP = "Ownship"
HOLD_NED = (0.0, 0.0, -10.0)
INTRUDERS = ["Z_Intruder1", "Z_Intruder2", "Z_Intruder3"]
WIDE_FOV_DEG = 90.0
WIDE_MOUNT_PITCH_DEG = -5.0  # matches settings - Copy.json wide camera


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["radar-led", "cv-led"], default="radar-led",
                   help="cueing mode (default: radar-led)")
    p.add_argument("--yolo-weights", default=DEFAULT_WEIGHTS_PT)
    p.add_argument("--cueing-weights", default=DEFAULT_WEIGHTS_YAML)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--slice", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--detect-hz", type=float, default=0.0,
                   help="on-cue detector cadence in radar-led mode (Hz); 0=disabled (default)")
    p.add_argument("--roi-size", type=int, default=256,
                   help="ROI box side length (px) for on-cue detection")
    p.add_argument("--launch-intruders", action="store_true",
                   help="command Z_Intruder1 onto a crossing path before the loop")
    p.add_argument("--no-clear", action="store_true",
                   help="don't ANSI-clear between frames; useful for piping output")
    p.add_argument("--profile", action="store_true",
                   help="print per-stage timing every 10 frames")
    p.add_argument("--ownship-pose-cache", type=int, default=5,
                   help="refresh Ownship pose every N frames; pass 1 to fetch every frame")
    p.add_argument("--track-timeout", type=float, default=3.0,
                   help="prune tracks unseen for this many seconds (default: 3.0)")
    p.add_argument("--track-gate", type=float, default=10.0,
                   help="track-maintenance association gate in degrees (default: 10.0)")
    p.add_argument("--min-hits-for-chosen", type=int, default=2)
    p.add_argument("--save-frames", action="store_true",
                   help="save annotated wide frames to --save-dir")
    p.add_argument("--save-dir", default="out/cueing_demo_frames",
                   help="directory for annotated frame output")
    return p.parse_args()


def quat_to_yaw_rad(q) -> float:
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


def bootstrap_ownship(c):
    c.enableApiControl(True, OWNSHIP)
    c.armDisarm(True, OWNSHIP)
    c.moveToPositionAsync(*HOLD_NED, velocity=30.0, vehicle_name=OWNSHIP).join()
    c.simSetCameraFov("wide", WIDE_FOV_DEG, vehicle_name=OWNSHIP)
    c.simSetCameraFov("narrow", 12.0, vehicle_name=OWNSHIP)


def launch_intruder1(c):
    c.enableApiControl(True, "Z_Intruder1")
    c.armDisarm(True, "Z_Intruder1")
    c.takeoffAsync(vehicle_name="Z_Intruder1").join()
    c.moveOnPathAsync(
        [airsim.Vector3r(150, -100, -25),
         airsim.Vector3r(100, 100, -30),
         airsim.Vector3r(50, -50, -20),
         airsim.Vector3r(200, 0, -25)],
        velocity=12.0,
        vehicle_name="Z_Intruder1",
    )


def annotate_frame(frame, dets, rois, ranked, chosen_id, mode, fps, frame_id):
    """Draw detections, cued ROIs, and chosen-target marker on a copy of frame."""
    out = frame.copy()
    # Cued ROI boxes (yellow) — radar-led only
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    # Detection bboxes (cyan)
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 255, 0), 2)
    # Chosen target: red cross at image centre (bearing unknown without Module 4)
    if chosen_id is not None:
        for rt in ranked:
            if rt.track.id == chosen_id:
                cx_img = out.shape[1] // 2
                cy_img = out.shape[0] // 2
                cv2.drawMarker(out, (cx_img, cy_img), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)
                break
    # Overlay text
    cv2.putText(out, f"mode={mode}  fps={fps:.1f}  f={frame_id}  trk={len(ranked)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render(ranked, t_elapsed, frame_id, fps, n_dets, n_radar, n_rois, chosen_id,
           weights_path, backend, ownship_yaw_deg, mode, clear: bool):
    if clear:
        sys.stdout.write("\033[2J\033[H")
    print(f"== Module 2 cueing demo ==  mode={mode}  t={t_elapsed:6.2f}s  "
          f"f={frame_id:5d}  fps={fps:5.2f}  backend={backend}")
    print(f"   inputs: cam_dets={n_dets}  radar_returns={n_radar}  rois={n_rois}  "
          f"ownship_yaw={ownship_yaw_deg:+.1f}°  weights={os.path.basename(weights_path)}")
    chosen_str = f"#{chosen_id}" if chosen_id is not None else "(none)"
    print(f"   tracks: {len(ranked):3d}   chosen target: {chosen_str}")
    print()
    print(f"  {'rank':>4} {'id':>4} {'age':>5} {'hits':>4} {'C':>1} {'R':>1} "
          f"{'range':>7} {'rate':>7} {'az':>7} {'el':>6} {'score':>6}  breakdown")
    for k, rt in enumerate(ranked[:10], start=1):
        t = rt.track
        age = t.age_s(t.last_t)
        cam = "Y" if (t.last_camera_t is not None and (t.last_t - t.last_camera_t) <= 0.5) else "."
        rad = "Y" if (t.last_radar_t is not None and (t.last_t - t.last_radar_t) <= 0.5) else "."
        rng = f"{t.range_m:7.0f}" if t.range_m is not None else "      ?"
        rate = f"{t.range_rate_mps:+7.1f}" if t.range_rate_mps is not None else "      ?"
        az = math.degrees(t.az_rad)
        el = math.degrees(t.el_rad)
        b = rt.breakdown
        bd = (f"r={b['range']:.2f} c={b['closing']:.2f} h={b['heading']:.2f} "
              f"p={b['persistence']:.2f} s={b['sensor_agree']:.2f} n={b['novelty']:.2f}")
        radar_id = t.last_radar_track_id or ""
        print(f"  {k:>4} #{t.id:<3} {age:5.1f} {t.hits:4d} {cam:>1} {rad:>1} "
              f"{rng} {rate} {az:+7.2f} {el:+6.2f} {rt.score:6.2f}  {bd}  {radar_id}")
    print()
    print("(edit configs/cueing_weights.yaml to live-tune the ranking)")


def main():
    args = parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()

    if args.launch_intruders:
        launch_intruder1(c)
    print(f"Ownship locked at {c.simGetVehiclePose(OWNSHIP).position}")

    cam_info = c.simGetCameraInfo("wide", vehicle_name=OWNSHIP)
    print(f"wide cam info: fov={getattr(cam_info, 'fov', 'n/a')}  "
          f"pose={cam_info.pose.position}")
    src = WideFovFrameSource(c, vehicle_name=OWNSHIP, camera_name="wide", fov_degrees=WIDE_FOV_DEG)
    grabbed = None
    for _ in range(20):
        grabbed = src.grab()
        if grabbed is not None:
            break
        time.sleep(0.1)
    if grabbed is None:
        print("ERROR: could not grab a frame from wide camera"); sys.exit(2)
    _, meta0 = grabbed
    print(f"wide frame size: {meta0.width}x{meta0.height}")

    intr = CameraIntrinsics.from_fov(meta0.width, meta0.height, WIDE_FOV_DEG)
    mount = CameraMount(yaw_deg=0.0, pitch_deg=WIDE_MOUNT_PITCH_DEG, roll_deg=0.0)

    # Only load YOLO if detection is actually needed.
    # radar-led + detect_hz=0 (default): pure radar, no camera, no model load.
    # radar-led + detect_hz>0: on-cue corroboration enabled by user.
    # cv-led: always load (detector is the primary sensor).
    need_detector = (args.mode == "cv-led") or (args.detect_hz > 0)
    detector = None
    if need_detector:
        cadence = "on_cue" if args.mode == "radar-led" else "live"
        use_sahi = False if args.mode == "radar-led" else (not args.no_sahi)
        detector = WideFovDetector(
            weights=args.yolo_weights,
            device=args.device,
            confidence_threshold=args.conf,
            cadence_mode=cadence,
            use_sahi=use_sahi,
            slice_height=args.slice,
            slice_width=args.slice,
        )
        print(f"detector backend: {detector.backend}  cadence: {cadence}  classes: {detector.class_names}")
    else:
        print("detector: disabled (radar-led, detect-hz=0) — pass --detect-hz N to enable on-cue corroboration")

    weights_loader = WeightsLoader(args.cueing_weights)
    tracker = CueingTracker(
        intrinsics=intr,
        mount=mount,
        weights=weights_loader.weights,
        track_gate_deg=args.track_gate,
        track_timeout_s=args.track_timeout,
        min_hits_for_chosen=args.min_hits_for_chosen,
        mode=args.mode,
    )
    radar = RadarMock(c, ownship_name=OWNSHIP, scan_hz=15.0)

    save_dir = None
    if args.save_frames:
        save_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        print(f"saving annotated frames to: {save_dir}")

    yaw_warned = False
    t0 = time.time()
    n_frames = 0
    fps_t0 = time.time()
    fps_n0 = 0
    fps = 0.0

    # Estimated camera FPS for detect cadence throttling (updated after warmup).
    fps_camera_est = 10.0

    pose_cache_period = max(1, int(args.ownship_pose_cache))
    cached_own_pose = None
    cached_yaw_rad = 0.0
    cached_pose_age = pose_cache_period

    stage_keys = ("grab", "pose", "detect", "radar", "tracker", "render")
    stage_acc = {k: 0.0 for k in stage_keys}
    stage_n = 0

    last_rois: list = []
    last_dets: list = []

    while time.time() - t0 < args.duration:
        t_grab = time.perf_counter()
        grabbed = src.grab()
        if grabbed is None:
            time.sleep(0.02); continue
        frame, meta = grabbed
        d_grab = time.perf_counter() - t_grab
        now_t = meta.timestamp

        t_pose = time.perf_counter()
        if cached_own_pose is None or cached_pose_age >= pose_cache_period:
            cached_own_pose = c.simGetObjectPose(OWNSHIP)
            cached_yaw_rad = quat_to_yaw_rad(cached_own_pose.orientation)
            cached_pose_age = 0
        else:
            cached_pose_age += 1
        ownship_yaw_rad = cached_yaw_rad
        own_pose = cached_own_pose
        d_pose = time.perf_counter() - t_pose

        if (not yaw_warned) and abs(math.degrees(ownship_yaw_rad)) > 2.0:
            print(f"WARN ownship yaw drifted to {math.degrees(ownship_yaw_rad):+.2f}° — bearings will be off")
            yaw_warned = True

        t_det = time.perf_counter()
        t_radar = time.perf_counter()

        if args.mode == "radar-led":
            radar_returns = radar.scan(INTRUDERS, t=now_t, ownship_pose=own_pose)
            d_radar = time.perf_counter() - t_radar

            if detector is not None and args.detect_hz > 0:
                detect_every_n = max(1, round(fps_camera_est / args.detect_hz))
                if n_frames % detect_every_n == 0:
                    last_rois = tracker.radar_to_rois(
                        radar_returns, ownship_yaw_rad,
                        meta.width, meta.height, args.roi_size,
                    )
                    last_dets = detector.detect(frame, meta.frame_id, meta.timestamp,
                                                rois=last_rois)
            dets = last_dets
        else:
            last_dets = detector.detect(frame, meta.frame_id, meta.timestamp)
            dets = last_dets
            d_radar_start = time.perf_counter()
            radar_returns = radar.scan(INTRUDERS, t=now_t, ownship_pose=own_pose)
            d_radar = time.perf_counter() - d_radar_start
            last_rois = []

        d_det = time.perf_counter() - t_det

        if weights_loader.maybe_reload():
            tracker.set_weights(weights_loader.weights)
            print(f"[hot-reload] {args.cueing_weights} -> w_range={weights_loader.weights.w_range} "
                  f"w_range_rate={weights_loader.weights.w_range_rate} "
                  f"w_sensor_agree={weights_loader.weights.w_sensor_agree}")

        t_trk = time.perf_counter()
        ranked = tracker.step(dets, radar_returns, ownship_yaw_rad, now_t)
        chosen = tracker.chosen_target_id(ranked)
        d_trk = time.perf_counter() - t_trk

        n_frames += 1
        if n_frames - fps_n0 >= 10:
            now_w = time.time()
            fps = (n_frames - fps_n0) / max(now_w - fps_t0, 1e-3)
            fps_camera_est = fps
            fps_t0 = now_w
            fps_n0 = n_frames

        t_render = time.perf_counter()
        backend_str = detector.backend if detector is not None else "disabled"
        render(
            ranked, t_elapsed=time.time() - t0, frame_id=meta.frame_id, fps=fps,
            n_dets=len(dets), n_radar=len(radar_returns), n_rois=len(last_rois),
            chosen_id=chosen, weights_path=args.cueing_weights,
            backend=backend_str, ownship_yaw_deg=math.degrees(ownship_yaw_rad),
            mode=args.mode, clear=not args.no_clear,
        )

        if save_dir is not None:
            annotated = annotate_frame(frame, dets, last_rois, ranked, chosen,
                                       args.mode, fps, meta.frame_id)
            cv2.imwrite(os.path.join(save_dir, f"f{meta.frame_id:06d}.jpg"), annotated)

        d_render = time.perf_counter() - t_render

        if args.profile:
            stage_acc["grab"] += d_grab
            stage_acc["pose"] += d_pose
            stage_acc["detect"] += d_det
            stage_acc["radar"] += d_radar
            stage_acc["tracker"] += d_trk
            stage_acc["render"] += d_render
            stage_n += 1
            if stage_n >= 10:
                avg_total = sum(stage_acc.values()) / stage_n
                stage_str = " ".join(
                    f"{k}={1000.0 * stage_acc[k] / stage_n:5.1f}ms" for k in stage_keys
                )
                print(f"[profile] {stage_str} total={1000.0*avg_total:5.1f}ms ({1.0/max(avg_total,1e-3):.2f} FPS)")
                stage_acc = {k: 0.0 for k in stage_keys}
                stage_n = 0

    elapsed = time.time() - t0
    final_fps = n_frames / max(elapsed, 1e-3)
    print()
    print("--- summary ---")
    print(f"mode:         {args.mode}")
    print(f"frames:       {n_frames}")
    print(f"FPS:          {final_fps:.2f}  (target >=10 for radar-led)")
    print(f"backend:      {detector.backend if detector is not None else 'disabled'}")
    print(f"final tracks: {len(tracker.tracks)}")
    print(f"pose cache:   refresh every {pose_cache_period} frames")
    if save_dir:
        print(f"frames saved: {save_dir}")


if __name__ == "__main__":
    main()
