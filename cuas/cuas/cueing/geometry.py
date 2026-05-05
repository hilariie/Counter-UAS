"""Pixel -> world-NED bearing math.

Convention matches radar_mock:
    az = atan2(y_world_ned, x_world_ned)
    el = atan2(-z_world_ned, sqrt(x^2 + y^2))

NED is right-handed (x fwd, y right, z down). Camera image axes are (u right,
v down), with z_cam = 1 forward. AirSim's camera convention is NED-aligned, so
the camera-to-body remap is the identity rearrangement:
    body_x = z_cam (fwd) = 1
    body_y = x_cam (right) = (u - cx) / fx
    body_z = y_cam (down)  = (v - cy) / fy
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float) -> "CameraIntrinsics":
        fov = math.radians(fov_degrees)
        fx = (width / 2.0) / math.tan(fov / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)


@dataclass(frozen=True)
class CameraMount:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def pixel_to_body_ray(intr: CameraIntrinsics, u: float, v: float) -> np.ndarray:
    """Pixel (u, v) -> unit ray in body NED frame (x fwd, y right, z down)."""
    x_cam = (u - intr.cx) / intr.fx
    y_cam = (v - intr.cy) / intr.fy
    body = np.array([1.0, x_cam, y_cam])
    return body / np.linalg.norm(body)


def body_to_world_rotation(mount: CameraMount, ownship_yaw_rad: float) -> np.ndarray:
    """Compose mount + ownship yaw into a single body->world rotation.

    Mount is applied first (intrinsic): yaw, then pitch, then roll. Ownship
    yaw rotates body into world. Roll/pitch of ownship assumed 0 (mast-locked).
    """
    R_mount = (
        _rot_z(math.radians(mount.yaw_deg))
        @ _rot_y(math.radians(mount.pitch_deg))
        @ _rot_x(math.radians(mount.roll_deg))
    )
    R_ownship = _rot_z(ownship_yaw_rad)
    return R_ownship @ R_mount


def ray_to_az_el(ray_world: np.ndarray) -> Tuple[float, float]:
    """World-NED unit vector -> (az_rad, el_rad) matching radar convention."""
    x, y, z = float(ray_world[0]), float(ray_world[1]), float(ray_world[2])
    az = math.atan2(y, x)
    el = math.atan2(-z, math.sqrt(x * x + y * y))
    return az, el


def pixel_to_az_el(intr: CameraIntrinsics, mount: CameraMount,
                   ownship_yaw_rad: float, u: float, v: float) -> Tuple[float, float]:
    body = pixel_to_body_ray(intr, u, v)
    R = body_to_world_rotation(mount, ownship_yaw_rad)
    return ray_to_az_el(R @ body)


def angle_diff_rad(a: float, b: float) -> float:
    """Smallest signed difference a - b, wrapped to [-pi, pi]."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def az_el_to_pixel(intr: CameraIntrinsics, mount: CameraMount,
                   ownship_yaw_rad: float, az_rad: float,
                   el_rad: float) -> Optional[Tuple[float, float]]:
    """Inverse of pixel_to_az_el: bearing → (u, v) in wide-FOV pixel space.

    Returns None if the bearing is behind the camera or projects outside the
    image bounds.
    """
    world_ray = np.array([
        math.cos(el_rad) * math.cos(az_rad),
        math.cos(el_rad) * math.sin(az_rad),
        -math.sin(el_rad),
    ])
    R = body_to_world_rotation(mount, ownship_yaw_rad)
    body_ray = R.T @ world_ray
    if body_ray[0] <= 0:
        return None
    u = body_ray[1] / body_ray[0] * intr.fx + intr.cx
    v = body_ray[2] / body_ray[0] * intr.fy + intr.cy
    if 0.0 <= u < intr.width and 0.0 <= v < intr.height:
        return float(u), float(v)
    return None


def radar_roi(intr: CameraIntrinsics, mount: CameraMount,
              ownship_yaw_rad: float, az_rad: float, el_rad: float,
              roi_size: int = 256) -> Optional[Tuple[int, int, int, int]]:
    """Project radar bearing to a clamped pixel ROI (x1, y1, x2, y2)."""
    px = az_el_to_pixel(intr, mount, ownship_yaw_rad, az_rad, el_rad)
    if px is None:
        return None
    u, v = px
    half = roi_size // 2
    x1 = max(0, int(u) - half)
    y1 = max(0, int(v) - half)
    x2 = min(intr.width,  x1 + roi_size)
    y2 = min(intr.height, y1 + roi_size)
    return x1, y1, x2, y2
