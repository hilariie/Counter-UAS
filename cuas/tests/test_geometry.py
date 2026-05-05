import math

import numpy as np
import pytest

from cuas.cueing.geometry import (
    CameraIntrinsics, CameraMount, angle_diff_rad,
    pixel_to_az_el, pixel_to_body_ray, ray_to_az_el,
)


def test_intrinsics_from_fov_90deg():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    # tan(45°) = 1, so fx = (W/2)/1 = 640
    assert intr.fx == pytest.approx(640.0)
    assert intr.fy == pytest.approx(640.0)
    assert intr.cx == pytest.approx(640.0)
    assert intr.cy == pytest.approx(360.0)


def test_intrinsics_from_fov_60deg():
    intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
    expected_fx = (1280 / 2.0) / math.tan(math.radians(30.0))
    assert intr.fx == pytest.approx(expected_fx)


def test_pixel_center_with_zero_mount_zero_yaw():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount(yaw_deg=0, pitch_deg=0, roll_deg=0)
    az, el = pixel_to_az_el(intr, mount, 0.0, intr.cx, intr.cy)
    assert math.degrees(az) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(el) == pytest.approx(0.0, abs=1e-9)


def test_pixel_center_with_pitched_mount():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount(yaw_deg=0, pitch_deg=-5.0, roll_deg=0)
    az, el = pixel_to_az_el(intr, mount, 0.0, intr.cx, intr.cy)
    assert math.degrees(az) == pytest.approx(0.0, abs=1e-9)
    # Camera is pitched 5° down -> bearing elevation is -5°
    assert math.degrees(el) == pytest.approx(-5.0, abs=1e-9)


def test_pixel_right_of_center_gives_positive_azimuth():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount()
    # 100 px right at fx=640 -> tan-1(100/640) = 8.88°
    az, _ = pixel_to_az_el(intr, mount, 0.0, intr.cx + 100, intr.cy)
    assert math.degrees(az) == pytest.approx(math.degrees(math.atan(100.0 / 640.0)), abs=1e-6)


def test_pixel_left_of_center_gives_negative_azimuth():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount()
    az, _ = pixel_to_az_el(intr, mount, 0.0, intr.cx - 200, intr.cy)
    assert math.degrees(az) < 0


def test_pixel_below_center_gives_negative_elevation():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount()
    _, el = pixel_to_az_el(intr, mount, 0.0, intr.cx, intr.cy + 50)
    assert math.degrees(el) < 0


def test_ownship_yaw_rotates_bearing():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount()
    # Ownship rotated +90° (east). Center pixel should be due east (+y NED).
    az, el = pixel_to_az_el(intr, mount, math.radians(90.0), intr.cx, intr.cy)
    assert math.degrees(az) == pytest.approx(90.0, abs=1e-6)
    assert math.degrees(el) == pytest.approx(0.0, abs=1e-9)


def test_pixel_to_body_ray_is_unit_vector():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    for u, v in [(0, 0), (1280, 720), (640, 360), (10, 700)]:
        r = pixel_to_body_ray(intr, u, v)
        assert np.linalg.norm(r) == pytest.approx(1.0, abs=1e-9)


def test_ray_to_az_el_matches_radar_convention():
    # Forward in NED: (1, 0, 0) -> (az=0, el=0)
    az, el = ray_to_az_el(np.array([1.0, 0.0, 0.0]))
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(0.0)
    # Right (+y) -> az = +90°, el = 0
    az, el = ray_to_az_el(np.array([0.0, 1.0, 0.0]))
    assert math.degrees(az) == pytest.approx(90.0)
    # Up (-z) -> el = +90°
    az, el = ray_to_az_el(np.array([0.0, 0.0, -1.0]))
    assert math.degrees(el) == pytest.approx(90.0)


def test_angle_diff_wraps():
    assert angle_diff_rad(0.1, math.pi * 2 - 0.1) == pytest.approx(0.2, abs=1e-9)
    assert angle_diff_rad(-math.pi + 0.05, math.pi - 0.05) == pytest.approx(0.1, abs=1e-9)
    assert angle_diff_rad(0.5, 0.3) == pytest.approx(0.2)
    assert angle_diff_rad(0.3, 0.5) == pytest.approx(-0.2)
