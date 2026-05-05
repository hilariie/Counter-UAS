"""Unit tests for GimbalController math — no simulator required."""
import math
from unittest.mock import MagicMock, patch

import pytest

# Patch cosysairsim before importing gimbal so the module loads without the wheel.
import sys
airsim_mock = MagicMock()
airsim_mock.Vector3r = lambda x, y, z: MagicMock(x_val=x, y_val=y, z_val=z)
airsim_mock.Quaternionr = lambda x_val, y_val, z_val, w_val: MagicMock(
    x_val=x_val, y_val=y_val, z_val=z_val, w_val=w_val
)
airsim_mock.Pose = lambda pos, ori: MagicMock(position=pos, orientation=ori)
sys.modules.setdefault("cosysairsim", airsim_mock)

from cuas.tracking.gimbal import GimbalCommand, GimbalController, quat_from_yaw_pitch


def _make_gimbal(yaw_limit=math.radians(90), pitch_limit=math.radians(45),
                 max_slew=math.radians(999), kp=0.6):
    client = MagicMock()
    return GimbalController(
        client,
        yaw_limit_rad=yaw_limit,
        pitch_limit_rad=pitch_limit,
        max_slew_rate_rad_s=max_slew,
        kp_pixel=kp,
    ), client


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# 1. Body yaw = world az - ownship yaw (with wrap)
def test_slew_body_yaw_equals_world_az_minus_ownship_yaw():
    g, client = _make_gimbal()
    az = math.radians(30)
    ownship_yaw = math.radians(10)
    cmd = g.slew_to_world_bearing(az, 0.0, ownship_yaw)
    expected_yaw = _wrap_pi(az - ownship_yaw)
    assert abs(cmd.yaw_rad - expected_yaw) < 1e-9


# 2. Body pitch = -el
def test_slew_body_pitch_is_negative_elevation():
    g, _ = _make_gimbal()
    el = math.radians(15)
    cmd = g.slew_to_world_bearing(0.0, el, 0.0)
    assert abs(cmd.pitch_rad - (-el)) < 1e-9


# 3. Yaw clamped to ±yaw_limit
def test_slew_yaw_clamped():
    limit = math.radians(45)
    g, _ = _make_gimbal(yaw_limit=limit)
    cmd = g.slew_to_world_bearing(math.radians(120), 0.0, 0.0)
    assert abs(cmd.yaw_rad) <= limit + 1e-9


# 4. Pitch clamped to ±pitch_limit
def test_slew_pitch_clamped():
    limit = math.radians(30)
    g, _ = _make_gimbal(pitch_limit=limit)
    cmd = g.slew_to_world_bearing(0.0, -math.radians(60), 0.0)
    assert abs(cmd.pitch_rad) <= limit + 1e-9


# 5. nudge_pixel positive dx → positive yaw delta
def test_nudge_positive_dx_gives_positive_yaw_delta():
    g, _ = _make_gimbal()
    intr = MagicMock()
    intr.fx = 1500.0
    intr.fy = 1500.0
    # seed a zero command so we have a reference
    g._last_cmd = GimbalCommand(0.0, 0.0, __import__("time").time())
    cmd = g.nudge_pixel(dx_px=50.0, dy_px=0.0, intr=intr, ownship_yaw_rad=0.0)
    assert cmd.yaw_rad > 0.0


# 6. nudge_pixel positive dy (bbox below centre) → positive pitch delta (nose down)
def test_nudge_positive_dy_gives_positive_pitch_delta():
    g, _ = _make_gimbal()
    intr = MagicMock()
    intr.fx = 1500.0
    intr.fy = 1500.0
    g._last_cmd = GimbalCommand(0.0, 0.0, __import__("time").time())
    cmd = g.nudge_pixel(dx_px=0.0, dy_px=50.0, intr=intr, ownship_yaw_rad=0.0)
    assert cmd.pitch_rad > 0.0


# 7. Slew-rate limit: command delta capped per dt
def test_slew_rate_limit_caps_delta():
    import time as _time
    max_rate = math.radians(10)      # 10 deg/s
    g, _ = _make_gimbal(max_slew=max_rate)
    past = _time.time() - 1.0       # 1 second ago
    g._last_cmd = GimbalCommand(0.0, 0.0, past)
    # request a large yaw (60 deg), but only 1s * 10 deg/s = 10 deg allowed
    cmd = g.slew_to_world_bearing(math.radians(60), 0.0, 0.0)
    assert abs(cmd.yaw_rad) <= math.radians(10) + 1e-6


# 7b. Regression: _dispatch negates pitch before building the camera quaternion.
# AirSim's simSetCameraPose interprets the quaternion's pitch axis with the
# opposite sign of the NED math convention. The gimbal compensates inside
# _dispatch so callers can keep "positive pitch_rad = nose down" semantics.
def test_dispatch_negates_pitch_for_airsim_camera_pose():
    g, client = _make_gimbal()
    pitch_in = 0.5  # positive = nose down per gimbal API contract
    g._dispatch(0.0, pitch_in)
    # Inspect the Pose handed to simSetCameraPose
    args = client.simSetCameraPose.call_args.args
    pose = args[1]
    q = pose.orientation
    exp_w, exp_x, exp_y, exp_z = quat_from_yaw_pitch(0.0, -pitch_in)
    assert abs(q.w_val - exp_w) < 1e-9
    assert abs(q.x_val - exp_x) < 1e-9
    assert abs(q.y_val - exp_y) < 1e-9
    assert abs(q.z_val - exp_z) < 1e-9
    # The dispatched body pitch_rad in the returned cmd is unchanged
    # (still represents the requested "nose down by 0.5 rad" command).


# 8. Euler→Quaternion roundtrip matches reference
def test_quat_from_yaw_pitch_roundtrip():
    def _rot_matrix_from_quat(w, x, y, z):
        return [
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
            [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ]

    def _ref_rot(yaw, pitch):
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # R = Rz(yaw) @ Ry(pitch) (roll=0, body ZYX)
        return [
            [cy*cp,  -sy,  cy*sp],
            [sy*cp,   cy,  sy*sp],
            [  -sp,  0.0,     cp],
        ]

    for yaw, pitch in [(0.3, 0.1), (-1.1, 0.5), (0.0, 0.0), (math.pi/4, -math.pi/6)]:
        w, x, y, z = quat_from_yaw_pitch(yaw, pitch)
        got = _rot_matrix_from_quat(w, x, y, z)
        ref = _ref_rot(yaw, pitch)
        for i in range(3):
            for j in range(3):
                assert abs(got[i][j] - ref[i][j]) < 1e-6, \
                    f"yaw={yaw:.3f} pitch={pitch:.3f} R[{i}][{j}]: got {got[i][j]:.6f} ref {ref[i][j]:.6f}"
