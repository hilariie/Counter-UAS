"""Module 3 end-to-end demo: cueing + narrow-FOV SOT + gimbal.

Run from d:\\Sim (Blocks must already be running):
    python -m scripts.04_tracking_demo
    python -m scripts.04_tracking_demo --mode cv-led --duration 60 --save-frames
    python -m scripts.04_tracking_demo --sot-backend mil --gimbal-kp 0.6 --profile
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

OWNSHIP = "Ownship"
HOLD_NED = (0.0, 0.0, -10.0)
INTRUDERS = ["Z_Intruder1", "Z_Intruder2", "Z_Intruder3"]
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
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--slice", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--detect-hz", type=float, default=0.2)
    p.add_argument("--roi-size", type=int, default=256)
    p.add_argument("--launch-intruders", action="store_true")
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--ownship-pose-cache", type=int, default=5)
    p.add_argument("--track-timeout", type=float, default=3.0)
    p.add_argument("--track-gate", type=float, default=5.0)
    p.add_argument("--min-hits-for-chosen", type=int, default=2)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--save-dir", default="out/tracking_demo_frames")
    p.add_argument("--gimbal-kp", type=float, default=0.6)
    p.add_argument("--drift-limit", type=float, default=8.0,
                   help="gimbal bearing drift limit in degrees (default: 8)")
    p.add_argument("--narrow-fov", type=float, default=NARROW_FOV_DEG,
                   help="narrow camera FOV in degrees (default: 20)")
    p.add_argument("--lookahead-s", type=float, default=0.2,
                   help="bearing extrapolation lookahead in seconds (default: 0.2)")
    # narrow YOLO + ByteTrack (replaces OpenCV SOT)
    p.add_argument("--narrow-weights", default=DEFAULT_NARROW_WEIGHTS,
                   help="YOLO weights for narrow detector (TRT engine recommended)")
    p.add_argument("--narrow-imgsz", type=int, default=640)
    p.add_argument("--narrow-conf", type=float, default=0.1)
    # state machine timings
    p.add_argument("--warmup-s", type=float, default=2.5,
                   help="hold gimbal still after launch while radar matures (default: 2.5)")
    p.add_argument("--commit-s", type=float, default=7.0,
                   help="dwell on locked target ignoring cueing changes (default: 7)")
    p.add_argument("--preempt-score", type=float, default=3.0,
                   help="score threshold to break a COMMIT lock for a new target (default: 10.0)")
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--loss-grace-s", type=float, default=1.5)
    p.add_argument("--intruder1-velocity", type=float, default=5.0,
                   help="Z_Intruder1 path velocity (default: 5 m/s)")
    p.add_argument("--intruder2-velocity", type=float, default=8.0,
                   help="Z_Intruder2 path velocity (default: 8 m/s)")
    p.add_argument("--intruder3-velocity", type=float, default=5.0,
                   help="Z_Intruder3 path velocity (default: 5 m/s)")
    p.add_argument("--color-debug", action="store_true",
                   help="save raw frame at startup for color-grade verification")
    p.add_argument("--fail-radar-at", type=float, default=-1.0,
                   help="time in seconds to simulate radar failure and switch to cv-led (default: -1, disabled)")
    p.add_argument("--fail-radar-duration", type=float, default=-1.0,
                   help="duration in seconds for the radar failure. If -1, failure is permanent (default: -1)")
    p.add_argument("--state", action="store_true",
                   help="enable Module 4 EKF state estimation and show live error vs ground truth")
    p.add_argument("--intercept", action="store_true",
                   help="enable Module 5 PN intercept geometry (auto-enables --state)")
    p.add_argument("--intercept-speed", type=float, default=100.0,
                   help="interceptor speed in m/s (default: 100)")
    args = p.parse_args()
    if args.intercept and not args.state:
        args.state = True
        print("[intercept] auto-enabling --state (intercept requires EKF output)", file=sys.stderr)
    return args


def quat_to_yaw_rad(q) -> float:
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


def bootstrap_ownship(c):
    c.enableApiControl(True, OWNSHIP)
    c.armDisarm(True, OWNSHIP)
    c.moveToPositionAsync(*HOLD_NED, velocity=30.0, vehicle_name=OWNSHIP).join()
    c.simSetCameraFov("wide", WIDE_FOV_DEG, vehicle_name=OWNSHIP)
    c.simSetCameraFov("narrow", NARROW_FOV_DEG, vehicle_name=OWNSHIP)


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


def annotate_wide(frame, dets, rois, chosen_id, fps, frame_id):
    out = frame.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 255, 0), 2)
    if chosen_id is not None:
        cx = out.shape[1] // 2
        cy = out.shape[0] // 2
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(out, f"WIDE fps={fps:.1f} f={frame_id} chosen=#{chosen_id}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def annotate_narrow(frame, ctrl_state, fps):
    out = frame.copy()
    det = ctrl_state.det
    if det is not None:
        x, y, w, h = det.bbox_xywh
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
        tag = f"id={det.track_id} c={det.confidence:.2f}"
        cv2.putText(out, tag, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1)
    ih, iw = out.shape[:2]
    cv2.drawMarker(out, (iw // 2, ih // 2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    label = (f"NARROW {ctrl_state.state_name} tgt=#{ctrl_state.target_id} "
             f"rem={ctrl_state.state_remaining_s:.1f}s fps={fps:.1f}")
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if ctrl_state.lost:
        cv2.putText(out, "[LOST]", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def render(ranked, ctrl_state, t_elapsed, frame_id, fps, n_dets, n_radar,
           chosen_id, mode, ownship_yaw_deg, n_lost, clear: bool,
           state_estimates=None, filter_diags=None, intercept_solution=None):
    if clear:
        sys.stdout.write("\033[2J\033[H")
    print(f"== Module 3+4 tracking demo ==  mode={mode}  t={t_elapsed:6.2f}s  "
          f"f={frame_id:5d}  fps={fps:5.2f}")
    print(f"   cam_dets={n_dets}  radar={n_radar}  ownship_yaw={ownship_yaw_deg:+.1f}°  "
          f"tracks={len(ranked)}  chosen=#{chosen_id}  total_lost={n_lost}")
    gim = ctrl_state.gimbal_cmd
    print(f"   gimbal yaw={math.degrees(gim.yaw_rad):+.1f}°  "
          f"pitch={math.degrees(gim.pitch_rad):+.1f}°  "
          f"state={ctrl_state.state_name} rem={ctrl_state.state_remaining_s:.1f}s  "
          f"{'[LOST]' if ctrl_state.lost else ''}")
    if ctrl_state.det is not None:
        x, y, w, h = ctrl_state.det.bbox_xywh
        print(f"   det bbox=({x},{y},{w}×{h})  conf={ctrl_state.det.confidence:.2f}  "
              f"id={ctrl_state.det.track_id}")
    print()
    print(f"  {'rank':>4} {'id':>4} {'age':>5} {'hits':>4} {'range':>7} {'az':>7} {'el':>6} {'score':>6}")
    for k, rt in enumerate(ranked[:8], start=1):
        t = rt.track
        rng = f"{t.range_m:7.0f}" if t.range_m is not None else "      ?"
        print(f"  {k:>4} #{t.id:<3} {t.age_s(t.last_t):5.1f} {t.hits:4d} "
              f"{rng} {math.degrees(t.az_rad):+7.2f} {math.degrees(t.el_rad):+6.2f} {rt.score:6.2f}")

    if state_estimates:
        est_by_id = {se.track_id: se for se in state_estimates}
        print()
        print(f"  {'[EKF]':6} {'id':>4} {'px':>8} {'py':>8} {'pz':>8} "
              f"{'vx':>6} {'vy':>6} {'vz':>6} {'err_r':>7} {'cov_tr':>8} "
              f"{'sensors':>9} {'rej':>4} {'nis':>8} {'reinit':>6}")
        for rt in ranked[:8]:
            se = est_by_id.get(rt.track.id)
            if se is None:
                print(f"  {'[EKF]':6} #{rt.track.id:<3} {'(no estimate yet)':>60}")
                continue
            p = se.position_ned
            v = se.velocity_ned
            cov_tr = float(se.covariance[:3, :3].trace())
            sensors = []
            if SensorMask.RADAR in se.sensors_used:   sensors.append("R")
            if SensorMask.BEARING in se.sensors_used: sensors.append("B")
            if SensorMask.SFM in se.sensors_used:     sensors.append("S")
            sensor_str = "+".join(sensors) if sensors else "none"
            err_str = "      ?"
            if se.range_m_true is not None:
                import numpy as _np
                est_range = float(_np.linalg.norm(p))
                err_str = f"{abs(est_range - se.range_m_true):7.1f}"
            diag = (filter_diags or {}).get(se.track_id, (0, 0.0, 0))
            rej, nis, reinits = diag
            print(f"  {'[EKF]':6} #{se.track_id:<3} "
                  f"{p[0]:+8.1f} {p[1]:+8.1f} {p[2]:+8.1f} "
                  f"{v[0]:+6.2f} {v[1]:+6.2f} {v[2]:+6.2f} "
                  f"{err_str} {cov_tr:8.1f} "
                  f"{sensor_str:>9} {rej:>4} {nis:>8.1f} {reinits:>6}")

    if intercept_solution is None:
        if filter_diags is not None or state_estimates is not None:
            print()
            print("  INTERCEPT  —  (no committed target)")
    else:
        sol = intercept_solution
        print()
        if sol.feasible:
            hdg_deg = math.degrees(math.atan2(sol.launch_heading_unit[1], sol.launch_heading_unit[0]))
            cov_tr = float(sol.intercept_covariance.trace())
            print(f"  INTERCEPT  trk=#{sol.track_id}  "
                  f"TTI={sol.time_to_intercept_s:.1f}s  "
                  f"hdg={hdg_deg:+.1f}°  "
                  f"cov={cov_tr:.1f}m²")
        else:
            print(f"  INTERCEPT  trk=#{sol.track_id}  INFEASIBLE  ({sol.reason.value})")


def main():
    args = parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()
    # bootstrap_ownship(c)


    # Wide frame source + cueing
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

    # Narrow frame source + Module 3 controller
    narrow_src = NarrowFovFrameSource(c, vehicle_name=OWNSHIP, camera_name="narrow",
                                      fov_degrees=args.narrow_fov)
    narrow_intr = CameraIntrinsics.from_fov(640, 480, args.narrow_fov)
    gimbal = GimbalController(c, vehicle_name=OWNSHIP, camera_name="narrow",
                              kp_pixel=args.gimbal_kp)
    narrow_det = NarrowDetTracker(
        weights=args.narrow_weights, device=args.device,
        conf=args.narrow_conf, imgsz=args.narrow_imgsz,
    )
    print(f"narrow detector: {narrow_det.weights}")
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

    if args.color_debug:
        cd_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "out", "color_debug"))
        os.makedirs(cd_dir, exist_ok=True)
        cv2.imwrite(os.path.join(cd_dir, "wide_as_returned.png"), grabbed[0])
        print(f"[color-debug] wrote {cd_dir}/wide_as_returned.png")

    # Save dirs
    save_wide_dir = save_narrow_dir = None
    if args.save_frames:
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.save_dir))
        save_wide_dir = os.path.join(base, "wide")
        save_narrow_dir = os.path.join(base, "narrow")
        os.makedirs(save_wide_dir, exist_ok=True)
        os.makedirs(save_narrow_dir, exist_ok=True)
        print(f"saving frames to: {base}")

    pose_cache_period = max(1, int(args.ownship_pose_cache))
    cached_own_pose = None
    cached_yaw_rad = 0.0
    cached_pose_age = pose_cache_period

    stage_keys = ("grab_wide", "grab_narrow", "pose", "detect", "radar", "cueing", "tracking", "render")
    stage_acc = {k: 0.0 for k in stage_keys}
    stage_n = 0

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

    # --- 3. Launch intruders on a crossing trajectory ----------------------------
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
                airsim.Vector3r(-10,  0, -15),
                airsim.Vector3r(-10, 0, -15),
                airsim.Vector3r(-10,  -5, -15),
                airsim.Vector3r(50,  20, -20),
                airsim.Vector3r(50,    20, -20),
                airsim.Vector3r(50,    20, -20),
                airsim.Vector3r(65,    20, -20),
                airsim.Vector3r(65,    -4, -20),
                airsim.Vector3r(50,    0, -20),
                airsim.Vector3r(15, 5, -12),
                airsim.Vector3r(-5, 0, -10),
                airsim.Vector3r(-10,  5, -12),
                airsim.Vector3r(50,  -20, -20),
                airsim.Vector3r(50,    0, -25),
                airsim.Vector3r(15, 5, -11),
                airsim.Vector3r(-5, 0, -10),
                ],
        }
        velocity = {
            INTRUDERS[0]: args.intruder1_velocity,
            INTRUDERS[1]: args.intruder2_velocity,
            INTRUDERS[2]: args.intruder3_velocity
        }
        launch_intruder(c, intruder_paths, velocity)

    while time.time() - t0 < args.duration:
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

        # Malfunction check: handle radar failure, fallback, and recovery
        if args.fail_radar_at > 0:
            is_in_failure_window = (t_elapsed >= args.fail_radar_at)
            
            # If a duration is specified, bound the failure window
            if args.fail_radar_duration > 0:
                is_in_failure_window = is_in_failure_window and (t_elapsed < args.fail_radar_at + args.fail_radar_duration)

            # State Transition: Healthy -> Failed
            if is_in_failure_window and not radar_failed:
                print(f"\n[MALFUNCTION] Radar system failed at t={t_elapsed:.1f}s!")
                print(f"[FALLBACK] Switching CueingTracker to vision-only (cv-led) mode...")
                radar_failed = True
                current_mode = "cv-led"
                cueing.set_mode("cv-led")
                if detector:
                    detector.cadence_mode = "live"
                    
            # State Transition: Failed -> Recovered
            elif not is_in_failure_window and radar_failed and t_elapsed >= args.fail_radar_at:
                print(f"\n[RECOVERY] Radar system back online at t={t_elapsed:.1f}s!")
                print(f"[RESTORE] Reverting to original mode: {args.mode}...")
                radar_failed = False
                current_mode = args.mode
                cueing.set_mode(args.mode)
                if detector:
                    # Revert cadence based on the original startup mode
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

        # Feed EKF estimates back as discounted kinematic inputs during radar failure.
        # kinematic_confidence = P0_trace / trace(P_pos): 1.0 when covariance is at
        # init level, decays toward 0 as bearings-only EKF loses range observability.
        # Visual proxy terms in scoring activate as confidence drops (1 - confidence).
        _P0_TRACE = 27.0  # diag([9,9,9]) = 3m sigma per axis at init
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
            from cuas.perception.frame_source import FrameMeta
            narrow_meta = FrameMeta(frame_id=n_frames, timestamp=now_t, width=640, height=480)
        ctrl_state = ctrl.step(chosen_track, ownship_yaw_rad, narrow_frame, narrow_meta,
                               target_score=chosen_score)
        if ctrl_state.lost:
            n_lost += 1
            print(f"[lost] frame={wide_meta.frame_id} target=#{ctrl_state.target_id}")
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
        if intercept_solver is not None:
            intercept_solution = intercept_solver.step(state_estimates, ctrl_state, now_t)

        n_frames += 1
        if n_frames - fps_n0 >= 10:
            now_w = time.time()
            fps = (n_frames - fps_n0) / max(now_w - fps_t0, 1e-3)
            fps_camera_est = fps
            fps_t0 = now_w
            fps_n0 = n_frames

        t_render = time.perf_counter()
        render(
            ranked, ctrl_state,
            t_elapsed=time.time() - t0, frame_id=wide_meta.frame_id, fps=fps,
            n_dets=len(last_dets), n_radar=len(radar_returns),
            chosen_id=chosen_id, mode=current_mode,
            ownship_yaw_deg=math.degrees(ownship_yaw_rad),
            n_lost=n_lost, clear=not args.no_clear,
            state_estimates=state_estimates if state_est is not None else None,
            filter_diags=filter_diags if state_est is not None else None,
            intercept_solution=intercept_solution,
        )

        if save_wide_dir is not None:
            ann_wide = annotate_wide(wide_frame, last_dets, last_rois, chosen_id,
                                     fps, wide_meta.frame_id)
            cv2.imwrite(os.path.join(save_wide_dir, f"f{wide_meta.frame_id:06d}.jpg"), ann_wide)
            ann_narrow = annotate_narrow(narrow_frame, ctrl_state, fps)
            cv2.imwrite(os.path.join(save_narrow_dir, f"f{wide_meta.frame_id:06d}.jpg"), ann_narrow)

        d_render = time.perf_counter() - t_render

        if args.profile:
            stage_acc["grab_wide"] += d_grab_wide
            stage_acc["grab_narrow"] += d_grab_narrow
            stage_acc["pose"] += d_pose
            stage_acc["detect"] += d_det
            stage_acc["radar"] += d_radar
            stage_acc["cueing"] += d_cueing
            stage_acc["tracking"] += d_tracking
            stage_acc["render"] += d_render
            stage_n += 1
            if stage_n >= 10:
                avg_total = sum(stage_acc.values()) / stage_n
                parts = " ".join(
                    f"{k}={1000.0*stage_acc[k]/stage_n:5.1f}ms" for k in stage_keys
                )
                print(f"[profile] {parts} total={1000.0*avg_total:5.1f}ms "
                      f"({1.0/max(avg_total,1e-3):.2f} FPS)")
                stage_acc = {k: 0.0 for k in stage_keys}
                stage_n = 0

    elapsed = time.time() - t0
    final_fps = n_frames / max(elapsed, 1e-3)
    print()
    print("--- Module 3 summary ---")
    print(f"duration:    {elapsed:.1f}s")
    print(f"frames:      {n_frames}")
    print(f"FPS:         {final_fps:.2f}")
    print(f"tracks:      {len(cueing.tracks)}")
    print(f"total_lost:  {n_lost}")
    print(f"narrow_weights: {args.narrow_weights}")
    if save_wide_dir:
        print(f"wide saved:  {save_wide_dir}")
        print(f"narrow saved:{save_narrow_dir}")

    intruder_shut_down(c)

if __name__ == "__main__":
    main()
