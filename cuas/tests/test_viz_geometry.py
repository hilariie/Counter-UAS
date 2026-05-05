import math
import numpy as np
import pytest
import pygame


@pytest.fixture(autouse=True, scope="module")
def _no_display():
    """Allow geometry imports without a display."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


from cuas.viz.geometry import (
    cov2d_to_ellipse_params,
    ned_to_minimap_px,
    minimap_px_to_ne,
)


def test_cov2d_to_ellipse_axis_aligned():
    P = np.diag([4.0, 1.0])
    major, minor, theta = cov2d_to_ellipse_params(P, n_sigma=2.0)
    # major axis = 2*sigma of larger eigenvalue = 2*sqrt(4) = 4
    # minor axis = 2*sigma of smaller eigenvalue = 2*sqrt(1) = 2
    assert abs(major - 4.0) < 1e-9
    assert abs(minor - 2.0) < 1e-9
    # theta ~ 0 (or pi) for axis-aligned
    assert abs(math.sin(theta)) < 1e-9


def test_cov2d_to_ellipse_45deg():
    angle = math.pi / 4
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    P = R @ np.diag([9.0, 1.0]) @ R.T
    major, minor, theta = cov2d_to_ellipse_params(P, n_sigma=1.0)
    assert abs(major - 3.0) < 1e-6
    assert abs(minor - 1.0) < 1e-6
    assert abs(abs(theta) - math.pi / 4) < 1e-6


def test_cov2d_non_psd_returns_zero():
    P = np.array([[1.0, 2.0], [2.0, 1.0]])   # indefinite (det < 0)
    major, minor, theta = cov2d_to_ellipse_params(P)
    assert major == 0.0
    assert minor == 0.0


def test_ned_to_minimap_roundtrip():
    rect = pygame.Rect(8, 448, 760, 264)
    origin = np.array([0.0, 0.0, 0.0])
    samples = [
        np.array([50.0, 30.0, 0.0]),
        np.array([-100.0, 80.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    for ned in samples:
        px, py = ned_to_minimap_px(ned, origin, rect)
        n, e = minimap_px_to_ne(px, py, origin, rect)
        assert abs(n - ned[0]) <= 1.0, f"N roundtrip failed: {ned[0]} -> {n}"
        assert abs(e - ned[1]) <= 1.0, f"E roundtrip failed: {ned[1]} -> {e}"
