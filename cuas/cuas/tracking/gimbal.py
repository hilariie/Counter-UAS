"""Narrow-FOV gimbal pose controller: pure math + AirSim dispatch."""
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cosysairsim as airsim


@dataclass(frozen=True)
class GimbalCommand:
    yaw_rad: float
    pitch_rad: float
    timestamp: float


def quat_from_yaw_pitch(yaw: float, pitch: float) -> Tuple[float, float, float, float]:
    """ZYX intrinsic Euler (yaw, pitch, roll=0) → quaternion (w, x, y, z)."""
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    w = cp * cy
    x = -sp * sy
    y = sp * cy
    z = cp * sy
    return w, x, y, z


class GimbalController:
    def __init__(self, client, vehicle_name: str = "Ownship",
                 camera_name: str = "narrow",
                 mount_position: Tuple[float, float, float] = (0.30, 0.0, -0.10),
                 yaw_limit_rad: float = math.radians(90),
                 pitch_limit_rad: float = math.radians(45),
                 max_slew_rate_rad_s: float = math.radians(120),
                 kp_pixel: float = 0.6):
        self._client = client
        self._vehicle = vehicle_name
        self._camera = camera_name
        self._mount_pos = airsim.Vector3r(*mount_position)
        self._yaw_limit = yaw_limit_rad
        self._pitch_limit = pitch_limit_rad
        self._max_slew = max_slew_rate_rad_s
        self._kp = kp_pixel
        self._last_cmd: Optional[GimbalCommand] = None

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _rate_limit(self, target_yaw: float, target_pitch: float) -> Tuple[float, float]:
        if self._last_cmd is None:
            return target_yaw, target_pitch
        dt = max(time.time() - self._last_cmd.timestamp, 1e-3)
        max_delta = self._max_slew * dt
        dy = self._wrap_pi(target_yaw - self._last_cmd.yaw_rad)
        dp = target_pitch - self._last_cmd.pitch_rad
        if abs(dy) > max_delta:
            dy = math.copysign(max_delta, dy)
        if abs(dp) > max_delta:
            dp = math.copysign(max_delta, dp)
        return self._wrap_pi(self._last_cmd.yaw_rad + dy), self._last_cmd.pitch_rad + dp

    def _dispatch(self, yaw_rad: float, pitch_rad: float) -> GimbalCommand:
        yaw_rad = max(-self._yaw_limit, min(self._yaw_limit, yaw_rad))
        pitch_rad = max(-self._pitch_limit, min(self._pitch_limit, pitch_rad))
        # AirSim's simSetCameraPose interprets the camera quaternion's pitch
        # axis with the opposite sign vs. NED math convention: a math-positive
        # pitch quaternion renders as nose-up, not nose-down. Negate pitch
        # only for the quaternion handed to the engine so the gimbal's
        # external contract (positive pitch_rad = nose down) stays intact.
        w, x, y, z = quat_from_yaw_pitch(yaw_rad, -pitch_rad)
        q = airsim.Quaternionr(x_val=x, y_val=y, z_val=z, w_val=w)
        self._client.simSetCameraPose(self._camera, airsim.Pose(self._mount_pos, q), self._vehicle)
        cmd = GimbalCommand(yaw_rad=yaw_rad, pitch_rad=pitch_rad, timestamp=time.time())
        self._last_cmd = cmd
        return cmd

    def slew_to_world_bearing(self, az_world_rad: float, el_world_rad: float,
                               ownship_yaw_rad: float) -> GimbalCommand:
        """Open-loop point at world-NED (az, el). Used for initial slew and post-loss reacquire."""
        body_yaw = self._wrap_pi(az_world_rad - ownship_yaw_rad)
        body_pitch = -el_world_rad
        body_yaw, body_pitch = self._rate_limit(body_yaw, body_pitch)
        return self._dispatch(body_yaw, body_pitch)

    def nudge_pixel(self, dx_px: float, dy_px: float, intr, ownship_yaw_rad: float) -> GimbalCommand:
        """Closed-loop P-controller on pixel error.

        Positive dx (bbox right of centre) → positive yaw delta (camera pans right).
        Positive dy (bbox below centre)    → positive pitch delta (camera tilts down,
                                              per the gimbal API: positive pitch_rad = nose down).
        """
        if self._last_cmd is None:
            cur_yaw, cur_pitch = 0.0, 0.0
        else:
            cur_yaw = self._last_cmd.yaw_rad
            cur_pitch = self._last_cmd.pitch_rad
        target_yaw = self._wrap_pi(cur_yaw + self._kp * dx_px / intr.fx)
        target_pitch = cur_pitch + self._kp * dy_px / intr.fy
        target_yaw, target_pitch = self._rate_limit(target_yaw, target_pitch)
        return self._dispatch(target_yaw, target_pitch)
