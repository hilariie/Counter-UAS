"""Module 6 end-to-end demo: Pygame operator UI.

Run from d:\\Sim (Blocks must already be running):
    python cuas/scripts/05_operator_ui_demo.py --duration 20 --launch-intruders --state --intercept
    python cuas/scripts/05_operator_ui_demo.py --duration 30 --launch-intruders --state --intercept --fail-radar-at 12 --fail-radar-duration 8
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
from cuas.cueing import CameraIntrinsics, CameraMount, CueingTracker, WeightsLoader
from cuas.tracking import (
    NarrowFovFrameSource, GimbalController,
    NarrowDetTracker, NarrowTrackingController,
)
from cuas.state import StateEstimator, StateEstimate, SensorMask
from cuas.intercept import InterceptSolver
from cuas.viz import OperatorUI, build_ui_frame
from cuas.perception.frame_source import FrameMeta

OWNSHIP = "Ownship"
HOLD_NED = (0.0, 0.0, -10.0)
INTRUDERS = ["Z_Intruder1", "Z_Intruder2", "Z_Intruder3", "Z_Intruder4"]
WIDE_FOV_DEG = 90.0
NARROW_FOV_DEG = 20.0
WIDE_MOUNT_PITCH_DEG = -5.0

DEFAULT_WEIGHTS_PT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yolo", "new-runs", "uav-640", "uav-640", "weights", "best.pt")
)
DEFAULT_NARROW_WEIGHTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yolo", "new-runs", "uav-640", "uav-640", "weights", "best.engine")
)
DEFAULT_WEIGHTS_YAML = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "cueing_weights.yaml")
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["radar-led", "cv-led"], default="radar-led")
    p.add_argument("--yolo-weights", default=DEFAULT_NARROW_WEIGHTS)
    p.add_argument("--cueing-weights", default=DEFAULT_WEIGHTS_YAML)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--slice", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--detect-hz", type=float, default=2.5)
    p.add_argument("--roi-size", type=int, default=50)
    p.add_argument("--launch-intruders", action="store_true")
    p.add_argument("--ownship-pose-cache", type=int, default=5)
    p.add_argument("--track-timeout", type=float, default=3.0)
    p.add_argument("--track-gate", type=float, default=5.0)
    p.add_argument("--min-hits-for-chosen", type=int, default=2)
    p.add_argument("--gimbal-kp", type=float, default=0.6)
    p.add_argument("--drift-limit", type=float, default=8.0)
    p.add_argument("--narrow-fov", type=float, default=NARROW_FOV_DEG)
    p.add_argument("--lookahead-s", type=float, default=0.2)
    p.add_argument("--narrow-weights", default=DEFAULT_NARROW_WEIGHTS)
    p.add_argument("--narrow-imgsz", type=int, default=640)
    p.add_argument("--narrow-conf", type=float, default=0.1)
    p.add_argument("--warmup-s", type=float, default=0)
    p.add_argument("--commit-s", type=float, default=7.0)
    p.add_argument("--preempt-score", type=float, default=10.0)
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--loss-grace-s", type=float, default=1.5)
    p.add_argument("--intruder1-velocity", type=float, default=3.5)
    p.add_argument("--intruder2-velocity", type=float, default=3.5)
    p.add_argument("--intruder3-velocity", type=float, default=3.5)
    p.add_argument("--intruder4-velocity", type=float, default=3.5)
    p.add_argument("--fail-radar-at", type=float, default=-1.0)
    p.add_argument("--fail-radar-duration", type=float, default=-1.0)
    p.add_argument("--state", action="store_true")
    p.add_argument("--intercept", action="store_true")
    p.add_argument("--intercept-speed", type=float, default=25.0)
    p.add_argument("--save-video", default=None,
                   help="Path for output mp4 (e.g. demo_seq3.mp4)")
    p.add_argument("--video-fps", type=float, default=5)
    args = p.parse_args()
    if args.intercept and not args.state:
        args.state = True
        print("[intercept] auto-enabling --state", file=sys.stderr)
    return args


def quat_to_yaw_rad(q) -> float:
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


def launch_intruder(client, path, velocity):
    for intruder_name in INTRUDERS:
        client.enableApiControl(True, intruder_name)
        client.armDisarm(True, intruder_name)
        client.takeoffAsync(vehicle_name=intruder_name).join()
        client.moveOnPathAsync(path[intruder_name], velocity=velocity[intruder_name], vehicle_name=intruder_name)


def intruder_shut_down(client):
    for intruder_name in INTRUDERS:
        client.cancelLastTask(intruder_name)
        client.hoverAsync(vehicle_name=intruder_name)
        client.armDisarm(False, intruder_name)
        client.enableApiControl(False, intruder_name)


def main():
    args = parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()

    wide_src = WideFovFrameSource(c, vehicle_name=OWNSHIP, camera_name="wide",
                                  fov_degrees=WIDE_FOV_DEG)
    grabbed = None
    for _ in range(20):
        grabbed = wide_src.grab()
        if grabbed is not None:
            break
        time.sleep(0.1)
    if grabbed is None:
        print("ERROR: could not grab wide frame"); sys.exit(2)
    _, meta0 = grabbed
    print(f"wide frame: {meta0.width}x{meta0.height}")

    wide_intr = CameraIntrinsics.from_fov(meta0.width, meta0.height, WIDE_FOV_DEG)
    wide_mount = CameraMount(yaw_deg=0.0, pitch_deg=WIDE_MOUNT_PITCH_DEG, roll_deg=0.0)

    need_detector = (args.mode == "cv-led") or (args.detect_hz > 0) or (args.fail_radar_at > 0)
    detector = None
    if need_detector:
        cadence = "on_cue" if args.mode == "radar-led" else "live"
        use_sahi = False if args.mode == "radar-led" else (not args.no_sahi)
        if use_sahi:
            args.yolo_weights = DEFAULT_WEIGHTS_PT
        detector = WideFovDetector(
            weights=args.yolo_weights, device=args.device,
            confidence_threshold=args.conf, cadence_mode=cadence,
            use_sahi=use_sahi, slice_height=args.slice, slice_width=args.slice,
        )
        print(f"detector: {detector.backend} cadence={cadence}")

    weights_loader = WeightsLoader(args.cueing_weights)
    cueing = CueingTracker(
        intrinsics=wide_intr, mount=wide_mount, weights=weights_loader.weights,
        track_gate_deg=args.track_gate, track_timeout_s=args.track_timeout,
        min_hits_for_chosen=args.min_hits_for_chosen, mode=args.mode,
    )
    radar = RadarMock(c, ownship_name=OWNSHIP, scan_hz=15.0)

    narrow_src = NarrowFovFrameSource(c, vehicle_name=OWNSHIP, camera_name="narrow",
                                      fov_degrees=args.narrow_fov)
    narrow_intr = CameraIntrinsics.from_fov(640, 480, args.narrow_fov)
    gimbal = GimbalController(c, vehicle_name=OWNSHIP, camera_name="narrow",
                              kp_pixel=args.gimbal_kp)
    narrow_det = NarrowDetTracker(
        weights=args.narrow_weights, device=args.device,
        conf=args.narrow_conf, imgsz=args.narrow_imgsz,
    )
    ctrl = NarrowTrackingController(
        client=c, intrinsics=narrow_intr, ownship_name=OWNSHIP,
        warmup_s=args.warmup_s, commit_s=args.commit_s,
        confirm_frames=args.confirm_frames, loss_grace_s=args.loss_grace_s,
        bearing_drift_limit_rad=math.radians(args.drift_limit),
        lookahead_s=args.lookahead_s,
        preempt_score_threshold=args.preempt_score,
        _detector=narrow_det, _gimbal=gimbal,
    )

    state_est = StateEstimator(narrow_intr=narrow_intr) if args.state else None
    intercept_solver = (
        InterceptSolver(
            defended_asset_ned=np.array(list(HOLD_NED)),
            interceptor_speed_mps=args.intercept_speed,
        ) if args.intercept else None
    )

    pose_cache_period = max(1, int(args.ownship_pose_cache))
    cached_own_pose = None
    cached_yaw_rad = 0.0
    cached_pose_age = pose_cache_period

    current_mode = args.mode
    radar_failed = False
    last_rois: list = []
    last_dets: list = []
    state_estimates: list = []
    fps = 0.0
    fps_t0 = time.time()
    fps_n0 = 0
    fps_camera_est = 10.0
    n_lost = 0
    n_frames = 0
    t0 = time.time()

    if args.launch_intruders:
        intruder_paths = {
            INTRUDERS[0]: [airsim.Vector3r(-5, 0, -15),
                airsim.Vector3r(-10,  -5, -15),
                airsim.Vector3r(-10, 0, -15),
                airsim.Vector3r(-10,  -5, -15),
                airsim.Vector3r(-50,  -20, -20),
                airsim.Vector3r(-65,    0, -20),
                airsim.Vector3r(-75,    0, -20),
                airsim.Vector3r(-70,    0, -20),
                airsim.Vector3r(-65,    -4, -20),
                airsim.Vector3r(-65,    0, -20),
                airsim.Vector3r(-15, 5, -12),
                airsim.Vector3r(-5, 0, -10),
                airsim.Vector3r(-10,  5, -12),
                airsim.Vector3r(-50,  -20, -20),
                airsim.Vector3r(-65,    0, -25),
                airsim.Vector3r(-15, 5, -11),
                airsim.Vector3r(-5, 0, -10),
                airsim.Vector3r(-10,  5, -10),
                airsim.Vector3r(-50,  -20, -10),
                airsim.Vector3r(-65,    0, -25),
                airsim.Vector3r(-15, 5, -15),
            ],
            INTRUDERS[1]: [airsim.Vector3r(-5, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(20, 0, -12),
                airsim.Vector3r(20, -2, -12),
                airsim.Vector3r(20, -2, -12),
                airsim.Vector3r(20, -2, -12),
                airsim.Vector3r(20, -2, -12),
                airsim.Vector3r(20, -2, -12),
                airsim.Vector3r(20, -2, -12),
            ],
            INTRUDERS[2]: [airsim.Vector3r(-5, 0, -15),
                airsim.Vector3r(-10, 0, -15),
                airsim.Vector3r(-10, 0, -15),
                airsim.Vector3r(-10, -5, -15),
                airsim.Vector3r(50, 20, -20),
                airsim.Vector3r(50, 20, -20),
                airsim.Vector3r(50, 20, -20),
                airsim.Vector3r(65, 20, -20),
                airsim.Vector3r(65, -4, -20),
                airsim.Vector3r(50, 0, -20),
                airsim.Vector3r(15, 5, -12),
                airsim.Vector3r(-5, 0, -10),
                airsim.Vector3r(-10, 5, -12),
                airsim.Vector3r(50, -20, -20),
                airsim.Vector3r(50, 0, -25),
                airsim.Vector3r(15, 5, -11),
                airsim.Vector3r(-5, 0, -10),
            ],
            INTRUDERS[3]: [airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(0, 0, -15),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(50, -10, -12),
                airsim.Vector3r(50, -10, -12),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(50, -20, -12),
                airsim.Vector3r(25, -10, -12),
                airsim.Vector3r(25, -10, -12),
            ],
        }
        velocity = {
            INTRUDERS[0]: args.intruder1_velocity,
            INTRUDERS[1]: args.intruder2_velocity,
            INTRUDERS[2]: args.intruder3_velocity,
            INTRUDERS[3]: args.intruder4_velocity,
        }
        launch_intruder(c, intruder_paths, velocity)

    # --- Init UI ---
    ui = OperatorUI(save_video=args.save_video, video_fps=args.video_fps)
    ui.init()

    while time.time() - t0 < args.duration:
        if not ui.pump_events():
            break

        t_grab_wide = time.perf_counter()
        wide_grabbed = wide_src.grab()
        if wide_grabbed is None:
            time.sleep(0.02); continue
        wide_frame, wide_meta = wide_grabbed
        d_grab_wide = time.perf_counter() - t_grab_wide

        t_grab_narrow = time.perf_counter()
        narrow_grabbed = narrow_src.grab()
        d_grab_narrow = time.perf_counter() - t_grab_narrow

        now_t = wide_meta.timestamp

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

        t_elapsed = time.time() - t0

        if args.fail_radar_at > 0:
            is_in_failure_window = (t_elapsed >= args.fail_radar_at)
            if args.fail_radar_duration > 0:
                is_in_failure_window = is_in_failure_window and (t_elapsed < args.fail_radar_at + args.fail_radar_duration)

            if is_in_failure_window and not radar_failed:
                print(f"\n[MALFUNCTION] Radar failed at t={t_elapsed:.1f}s!")
                radar_failed = True
                current_mode = "cv-led"
                cueing.set_mode("cv-led")
                if detector:
                    detector.cadence_mode = "live"

            elif not is_in_failure_window and radar_failed and t_elapsed >= args.fail_radar_at:
                print(f"\n[RECOVERY] Radar back online at t={t_elapsed:.1f}s!")
                radar_failed = False
                current_mode = args.mode
                cueing.set_mode(args.mode)
                if detector:
                    detector.cadence_mode = "on_cue" if args.mode == "radar-led" else "live"

        t_det = time.perf_counter()
        t_radar = time.perf_counter()
        if current_mode == "radar-led":
            radar_returns = radar.scan(INTRUDERS, t=now_t, ownship_pose=own_pose) if not radar_failed else []
            d_radar = time.perf_counter() - t_radar
            if detector is not None and args.detect_hz > 0:
                detect_every_n = max(1, round(fps_camera_est / args.detect_hz))
                if n_frames % detect_every_n == 0:
                    last_rois = cueing.radar_to_rois(
                        radar_returns, ownship_yaw_rad,
                        wide_meta.width, wide_meta.height, args.roi_size,
                    )
                    last_dets = detector.detect(wide_frame, wide_meta.frame_id,
                                                wide_meta.timestamp, rois=last_rois)
        else:
            last_dets = detector.detect(wide_frame, wide_meta.frame_id, wide_meta.timestamp)
            d_radar_start = time.perf_counter()
            radar_returns = radar.scan(INTRUDERS, t=now_t, ownship_pose=own_pose) if not radar_failed else []
            d_radar = time.perf_counter() - d_radar_start
            last_rois = []
        d_det = time.perf_counter() - t_det

        if weights_loader.maybe_reload():
            cueing.set_weights(weights_loader.weights)

        _P0_TRACE = 27.0
        if state_estimates and radar_failed:
            for se in state_estimates:
                if se.track_id in cueing.tracks:
                    t = cueing.tracks[se.track_id]
                    est_range = float(np.linalg.norm(se.position_ned))
                    t.range_m = est_range
                    los_unit = se.position_ned / (est_range + 1e-6)
                    t.range_rate_mps = float(np.dot(se.velocity_ned, los_unit))
                    trace_pos = float(se.covariance[0, 0] + se.covariance[1, 1] + se.covariance[2, 2])
                    t.kinematic_confidence = _P0_TRACE / max(_P0_TRACE, trace_pos)
        elif not radar_failed:
            for t in cueing.tracks.values():
                t.kinematic_confidence = 1.0

        t_cueing = time.perf_counter()
        ranked = cueing.step(last_dets, radar_returns, ownship_yaw_rad, now_t)
        chosen_id = cueing.chosen_target_id(ranked)
        chosen_rt = next((rt for rt in ranked if rt.track.id == chosen_id), None)
        chosen_track = chosen_rt.track if chosen_rt else None
        chosen_score = chosen_rt.score if chosen_rt else 0.0
        d_cueing = time.perf_counter() - t_cueing

        t_tracking = time.perf_counter()
        if narrow_grabbed is not None:
            narrow_frame, narrow_meta = narrow_grabbed
        else:
            narrow_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            narrow_meta = FrameMeta(frame_id=n_frames, timestamp=now_t, width=640, height=480)
        ctrl_state = ctrl.step(chosen_track, ownship_yaw_rad, narrow_frame, narrow_meta,
                               target_score=chosen_score)
        if ctrl_state.lost:
            n_lost += 1
        d_tracking = time.perf_counter() - t_tracking

        filter_diags = {}
        intercept_solution = None
        if state_est is not None:
            ownship_ned = np.array([
                own_pose.position.x_val,
                own_pose.position.y_val,
                own_pose.position.z_val,
            ])
            state_estimates = state_est.step(
                ranked, radar_returns, ctrl_state,
                ownship_ned, ownship_yaw_rad, now_t,
            )
            filter_diags = {
                tid: (tf.consecutive_rejects, tf.last_nis, tf.reinit_count)
                for tid, tf in state_est._filters.items()
            }
        else:
            ownship_ned = np.array([
                own_pose.position.x_val,
                own_pose.position.y_val,
                own_pose.position.z_val,
            ])
        if intercept_solver is not None:
            intercept_solution = intercept_solver.step(state_estimates, ctrl_state, now_t)

        n_frames += 1
        if n_frames - fps_n0 >= 10:
            now_w = time.time()
            fps = (n_frames - fps_n0) / max(now_w - fps_t0, 1e-3)
            fps_camera_est = fps
            fps_t0 = now_w
            fps_n0 = n_frames

        ui_frame = build_ui_frame(
            t_elapsed=t_elapsed,
            frame_id=wide_meta.frame_id,
            fps=fps,
            mode=current_mode,
            radar_alive=not radar_failed,
            wide_frame_bgr=wide_frame,
            wide_meta=wide_meta,
            narrow_frame_bgr=narrow_frame,
            narrow_meta=narrow_meta,
            last_dets=last_dets,
            last_rois=last_rois,
            radar_returns=radar_returns,
            ranked=ranked,
            chosen_id=chosen_id,
            ctrl_state=ctrl_state,
            state_estimates=state_estimates if state_est is not None else [],
            filter_diags=filter_diags if state_est is not None else {},
            intercept_solution=intercept_solution,
            ownship_ned=ownship_ned,
            ownship_yaw_rad=ownship_yaw_rad,
            n_lost=n_lost,
        )
        ui.render(ui_frame)

    ui.close()

    elapsed = time.time() - t0
    final_fps = n_frames / max(elapsed, 1e-3)
    print()
    print("--- Module 6 summary ---")
    print(f"duration:    {elapsed:.1f}s")
    print(f"frames:      {n_frames}")
    print(f"FPS:         {final_fps:.2f}")
    print(f"total_lost:  {n_lost}")

    intruder_shut_down(c)


if __name__ == "__main__":
    main()
