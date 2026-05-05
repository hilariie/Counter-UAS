import math

import numpy as np
import pytest

from cuas.state.ekf import guard_divergence, predict, update
from cuas.state.models import (H_bearing, H_radar, P0, R_RADAR, R_bearing,
                                h_bearing, h_radar, make_F, make_Q)


def _radar_meas(x_true, rng):
    z_true = h_radar(x_true)
    noise = np.array([
        rng.normal(0, 3.0),
        rng.normal(0, math.radians(0.7)),
        rng.normal(0, math.radians(0.7)),
        rng.normal(0, 0.5),
    ])
    return z_true + noise


def test_predict_kinematics():
    x0 = np.array([0., 0., 0., 10., 0., 0.])
    P0_ = np.eye(6)
    F = make_F(1.0)
    Q = make_Q(1.0, 2.0)
    x_pred, P_pred = predict(x0, P0_, F, Q)
    assert abs(x_pred[0] - 10.0) < 1e-10
    assert P_pred[0, 0] > P0_[0, 0]


def test_predict_preserves_velocity():
    x0 = np.array([100., 50., -80., 3., -2., 1.])
    F = make_F(0.1)
    Q = make_Q(0.1, 2.0)
    x_pred, _ = predict(x0, np.eye(6), F, Q)
    np.testing.assert_allclose(x_pred[3:], x0[3:])


def test_update_converges_stationary():
    rng = np.random.default_rng(42)
    x_true = np.array([100., 50., -80., 0., 0., 0.])
    x = np.array([110., 60., -70., 5., -5., 5.])
    P = np.diag([100., 100., 100., 100., 100., 100.])
    F = make_F(0.0667)
    Q = make_Q(0.0667, 2.0)

    for _ in range(50):
        x, P = predict(x, P, F, Q)
        z = _radar_meas(x_true, rng)
        H = H_radar(x)
        x, P, nis, accepted = update(x, P, z, H, R_RADAR, h_radar,
                                     gate=9.49, angle_idx=1)

    pos_err = np.linalg.norm(x[:3] - x_true[:3])
    pos_std = math.sqrt(np.trace(P[:3, :3]) / 3)
    assert pos_err < 3 * pos_std + 5.0, f"pos_err={pos_err:.1f} too large"
    assert pos_err < 15.0


def test_gate_rejects_outlier():
    x_pred = np.array([100., 50., -80., 0., 0., 0.])
    P_pred = np.diag([9., 9., 9., 100., 100., 100.])
    # Inject a wildly wrong measurement (20-sigma range error)
    z_outlier = h_radar(x_pred) + np.array([60.0, 0, 0, 0])
    H = H_radar(x_pred)
    x_new, P_new, nis, accepted = update(
        x_pred, P_pred, z_outlier, H, R_RADAR, h_radar,
        gate=9.49, angle_idx=1)
    assert not accepted
    assert nis > 9.49
    np.testing.assert_array_equal(x_new, x_pred)


def test_bearing_reduces_uncertainty():
    x = np.array([200., 0., -50., -5., 0., 0.])
    P = P0.copy()
    P_before = P[0, 0]
    z_b = h_bearing(x)
    H = H_bearing(x)
    x_new, P_new, nis, accepted = update(
        x, P, z_b, H, R_bearing(), h_bearing, gate=5.99, angle_idx=0)
    assert accepted
    assert P_new[0, 0] < P_before


def test_nis_calibration():
    # Converge the filter first, then check NIS calibration at steady state.
    rng = np.random.default_rng(0)
    x_true = np.array([100., 50., -80., 0., 0., 0.])
    F = make_F(0.0667)
    Q = make_Q(0.0667, 2.0)
    x = x_true + np.array([10., 10., -10., 2., -2., 2.])
    P = P0.copy()
    for _ in range(100):
        x, P = predict(x, P, F, Q)
        z = _radar_meas(x_true, rng)
        H = H_radar(x)
        x, P, _, _ = update(x, P, z, H, R_RADAR, h_radar, angle_idx=1)

    # Now test NIS distribution at the converged (x, P)
    chi2_95 = 9.49
    inside = 0
    n = 500
    for _ in range(n):
        z = _radar_meas(x_true, rng)
        H = H_radar(x)
        _, _, nis, _ = update(x, P, z, H, R_RADAR, h_radar)
        if nis < chi2_95:
            inside += 1
    fraction = inside / n
    assert 0.88 <= fraction <= 1.00, f"NIS calibration fraction={fraction:.3f}"


def test_divergence_guard_resets():
    P_bad = np.diag([2e6, 1., 1., 1., 1., 1.])
    P_out = guard_divergence(P_bad)
    np.testing.assert_array_equal(P_out, P0)


def test_divergence_guard_passes_good():
    P_good = np.eye(6) * 10.0
    P_out = guard_divergence(P_good)
    np.testing.assert_array_equal(P_out, P_good)


# ---------------------------------------------------------------------------
# TrackFilter gate-rejection and re-init tests
# ---------------------------------------------------------------------------

from cuas.state.track_filter import TrackFilter


def _make_filter(max_rejects=5):
    return TrackFilter(
        track_id=1, accel_sigma=2.0,
        gate_radar=9.49, gate_bearing=5.99,
        sfm_baseline_m=5.0,
        max_consecutive_rejects=max_rejects,
    )


def test_consecutive_rejects_increments():
    tf = _make_filter()
    tf.init_from_radar(100.0, 0.0, 0.0, 0.0, t=0.0)
    # Feed a measurement 50 m off in range — will be gated
    for i in range(3):
        tf.update_radar(150.0, 0.0, 0.0, 0.0)
    assert tf.consecutive_rejects == 3


def test_acceptance_resets_reject_counter():
    tf = _make_filter()
    tf.init_from_radar(100.0, 0.0, 0.0, 0.0, t=0.0)
    for _ in range(3):
        tf.update_radar(150.0, 0.0, 0.0, 0.0)
    assert tf.consecutive_rejects == 3
    # Feed correct measurement — should be accepted
    tf.update_radar(100.0, 0.0, 0.0, 0.0)
    assert tf.consecutive_rejects == 0


def test_reinit_on_max_consecutive_rejects():
    tf = _make_filter(max_rejects=5)
    # Init at 200 m, true target actually at 30 m
    tf.init_from_radar(200.0, 0.0, 0.0, -3.0, t=0.0)
    assert tf.reinit_count == 0

    true_range, true_az, true_el, true_rr = 30.0, 0.1, -0.05, -2.0
    for _ in range(5):
        tf.update_radar(true_range, true_az, true_el, true_rr)

    # After 5 rejects the filter should have re-inited
    assert tf.reinit_count == 1
    assert tf.consecutive_rejects == 0
    # State should now be close to the true position
    est_range = float(np.linalg.norm(tf._x[:3]))
    assert abs(est_range - true_range) < 5.0, f"post-reinit range={est_range:.1f} expected ~{true_range}"


def test_reinit_then_converges():
    rng = np.random.default_rng(77)
    tf = _make_filter(max_rejects=5)
    tf.init_from_radar(200.0, 0.0, 0.0, 0.0, t=0.0)

    x_true = np.array([30.0, 5.0, -10.0, -3.0, 1.0, 0.0])
    F = make_F(0.0667)
    Q = make_Q(0.0667, 2.0)
    t = 0.0
    for i in range(40):
        t += 0.0667
        tf.step(t, np.zeros(3))
        z = _radar_meas(x_true, rng)
        tf.update_radar(z[0], z[1], z[2], z[3])
        x_true[:3] += x_true[3:] * 0.0667

    assert tf.reinit_count >= 1, "filter should have re-inited at least once"
    pos_err = np.linalg.norm(tf._x[:3] - x_true[:3])
    assert pos_err < 10.0, f"post-reinit pos_err={pos_err:.1f} m"


def test_last_nis_populated():
    tf = _make_filter()
    tf.init_from_radar(100.0, 0.0, 0.0, 0.0, t=0.0)
    tf.update_radar(100.0, 0.0, 0.0, 0.0)
    assert tf.last_nis >= 0.0
