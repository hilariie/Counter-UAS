"""Tests for az_el_to_pixel and radar_roi (inverse projection)."""
import math
import pytest

from cuas.cueing import (
    CameraIntrinsics, CameraMount,
    pixel_to_az_el, az_el_to_pixel, radar_roi,
)

W, H = 1280, 720
FOV = 90.0
INTR = CameraIntrinsics.from_fov(W, H, FOV)
MOUNT = CameraMount(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
YAW = 0.0


def test_roundtrip_center():
    u, v = INTR.cx, INTR.cy
    az, el = pixel_to_az_el(INTR, MOUNT, YAW, u, v)
    recovered = az_el_to_pixel(INTR, MOUNT, YAW, az, el)
    assert recovered is not None
    assert abs(recovered[0] - u) < 1.0
    assert abs(recovered[1] - v) < 1.0


def test_roundtrip_off_center():
    u, v = 300.0, 200.0
    az, el = pixel_to_az_el(INTR, MOUNT, YAW, u, v)
    recovered = az_el_to_pixel(INTR, MOUNT, YAW, az, el)
    assert recovered is not None
    assert abs(recovered[0] - u) < 1.0
    assert abs(recovered[1] - v) < 1.0


def test_roundtrip_with_mount_pitch():
    mount = CameraMount(yaw_deg=0.0, pitch_deg=-15.0, roll_deg=0.0)
    u, v = INTR.cx, INTR.cy + 50
    az, el = pixel_to_az_el(INTR, mount, YAW, u, v)
    recovered = az_el_to_pixel(INTR, mount, YAW, az, el)
    assert recovered is not None
    assert abs(recovered[0] - u) < 1.0
    assert abs(recovered[1] - v) < 1.0


def test_behind_camera_returns_none():
    # az = pi puts the target directly behind the camera (body_x < 0)
    result = az_el_to_pixel(INTR, MOUNT, YAW, az_rad=math.pi, el_rad=0.0)
    assert result is None


def test_outside_fov_returns_none():
    # Wide az angle far beyond 45° half-FOV of a 90° camera projects off-frame
    result = az_el_to_pixel(INTR, MOUNT, YAW, az_rad=math.radians(80.0), el_rad=0.0)
    assert result is None


def test_radar_roi_center():
    az, el = pixel_to_az_el(INTR, MOUNT, YAW, INTR.cx, INTR.cy)
    roi = radar_roi(INTR, MOUNT, YAW, az, el, roi_size=256)
    assert roi is not None
    x1, y1, x2, y2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    assert abs(cx - INTR.cx) < 2.0
    assert abs(cy - INTR.cy) < 2.0
    assert (x2 - x1) == 256
    assert (y2 - y1) == 256


def test_radar_roi_clamped_edge():
    # Target near the top-left corner; ROI should clamp so bounds stay valid
    az, el = pixel_to_az_el(INTR, MOUNT, YAW, 10.0, 10.0)
    roi = radar_roi(INTR, MOUNT, YAW, az, el, roi_size=256)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 >= 0 and y1 >= 0
    assert x2 <= W and y2 <= H


def test_radar_roi_returns_none_outside_fov():
    result = radar_roi(INTR, MOUNT, YAW, az_rad=math.pi, el_rad=0.0, roi_size=256)
    assert result is None


def test_radar_roi_custom_size():
    az, el = pixel_to_az_el(INTR, MOUNT, YAW, INTR.cx, INTR.cy)
    roi = radar_roi(INTR, MOUNT, YAW, az, el, roi_size=128)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert (x2 - x1) == 128
    assert (y2 - y1) == 128
