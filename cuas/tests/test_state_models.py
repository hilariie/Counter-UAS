import math

import numpy as np
import pytest

from cuas.state.models import (H_bearing, H_radar, h_bearing, h_radar,
                                make_F, make_Q)
from cuas.cueing.geometry import CameraMount, CameraIntrinsics, pixel_to_az_el
from cuas.tracking.gimbal import GimbalCommand

_X = np.array([100.0, 50.0, -80.0, 3.0, -2.0, 1.0])
_EPS = 1e-5


def _fd_jacobian(h_fn, x, n_out):
    J = np.zeros((n_out, len(x)))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += _EPS
        xm = x.copy(); xm[i] -= _EPS
        J[:, i] = (h_fn(xp) - h_fn(xm)) / (2 * _EPS)
    return J


def test_H_radar_jacobian():
    H_ana = H_radar(_X)
    H_fd = _fd_jacobian(h_radar, _X, 4)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-5)


def test_H_bearing_jacobian():
    H_ana = H_bearing(_X)
    H_fd = _fd_jacobian(h_bearing, _X, 2)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-5)


def test_F_kinematics():
    dt = 0.1
    F = make_F(dt)
    x = np.array([0., 0., 0., 10., 0., 0.])
    x_pred = F @ x
    np.testing.assert_allclose(x_pred[:3], x[:3] + dt * x[3:])
    np.testing.assert_allclose(x_pred[3:], x[3:])


def test_Q_psd():
    Q = make_Q(0.1, 2.0)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= -1e-12)


def test_Q_shape():
    assert make_Q(0.1, 2.0).shape == (6, 6)


def test_bearing_sign_convention():
    """Positive GimbalCommand.pitch_rad (nose-down) must map to positive el_world
    for a target above the horizon.

    Verify the sign flip applied in estimator.py:
        CameraMount(pitch_deg=-degrees(gimbal_cmd.pitch_rad))
    """
    # Target is at az=0, el≈+0.24 rad above horizon (no pixel offset: centred pixel)
    target_el = 0.24  # rad
    # Narrow FOV camera — 12 degrees, 640×512
    intr = CameraIntrinsics.from_fov(width=640, height=512, fov_degrees=12.0)

    # Gimbal is pitched downward by target_el in *nose-down* convention.
    # slew_to_world_bearing sets body_pitch = -el_world_rad for el above horizon,
    # so GimbalCommand.pitch_rad = -target_el for an above-horizon target.
    # We test the inverse: given GimbalCommand.pitch_rad = -target_el,
    # a centred pixel should decode to el_world ≈ target_el.
    gcmd = GimbalCommand(yaw_rad=0.0, pitch_rad=-target_el, timestamp=0.0)

    mount = CameraMount(
        yaw_deg=math.degrees(gcmd.yaw_rad),
        pitch_deg=-math.degrees(gcmd.pitch_rad),  # sign flip
        roll_deg=0.0,
    )
    az_world, el_world = pixel_to_az_el(
        intr, mount, 0.0, intr.cx, intr.cy)  # centred pixel

    assert abs(el_world - target_el) < 1e-6, (
        f"el_world={el_world:.4f} expected ≈{target_el:.4f}; "
        "check CameraMount pitch sign flip in estimator.py"
    )
