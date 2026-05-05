import math

import numpy as np
import pytest

from cuas.state.sfm import SfmBuffer, SfmObs


def _obs(ownship, az, el, t=0.0):
    return SfmObs(timestamp=t, ownship_ned=np.array(ownship, dtype=float),
                  az_rad=az, el_rad=el)


def _bearing_to(target, origin):
    d = np.array(target, dtype=float) - np.array(origin, dtype=float)
    az = math.atan2(d[1], d[0])
    rxy = math.sqrt(d[0]**2 + d[1]**2)
    el = math.atan2(-d[2], rxy)
    return az, el


TARGET = [200., 0., -50.]
TRUE_RANGE = math.sqrt(200**2 + 50**2)


def _fill_buffer(buf: SfmBuffer, n=10, dx_per_step=2.0):
    for i in range(n):
        origin = [i * dx_per_step, 0., 0.]
        az, el = _bearing_to(TARGET, origin)
        buf.push(_obs(origin, az, el, t=float(i) * 0.1))


def test_two_ray_known_target():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5)
    _fill_buffer(buf, n=10, dx_per_step=2.0)
    result = buf.estimate(now_t=2.0)
    assert result is not None
    range_est, sigma = result
    assert abs(range_est - TRUE_RANGE) / TRUE_RANGE < 0.10


def test_insufficient_baseline_returns_none():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5)
    # Ownship barely moves — all same position
    for i in range(10):
        az, el = _bearing_to(TARGET, [0., 0., 0.])
        buf.push(_obs([0., 0., 0.], az, el, t=float(i)))
    result = buf.estimate(now_t=10.0)
    assert result is None


def test_too_few_observations():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5)
    for i in range(3):
        origin = [i * 5.0, 0., 0.]
        az, el = _bearing_to(TARGET, origin)
        buf.push(_obs(origin, az, el))
    assert buf.estimate(now_t=1.0) is None


def test_outlier_ray_rejected_by_median():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5)
    _fill_buffer(buf, n=9, dx_per_step=2.0)
    # Push one wildly wrong bearing (pointing 90° off)
    buf.push(_obs([20., 0., 0.], math.pi / 2, 0.0, t=10.0))
    result = buf.estimate(now_t=11.0)
    assert result is not None
    range_est, _ = result
    assert abs(range_est - TRUE_RANGE) / TRUE_RANGE < 0.20


def test_rate_limit():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5, rate_limit_s=1.0)
    _fill_buffer(buf, n=10, dx_per_step=2.0)
    r1 = buf.estimate(now_t=5.0)
    r2 = buf.estimate(now_t=5.5)  # within rate limit
    assert r1 is not None
    assert r2 is None


def test_clear_resets():
    buf = SfmBuffer(baseline_m=5.0, min_obs=5)
    _fill_buffer(buf, n=10, dx_per_step=2.0)
    buf.clear()
    assert buf.estimate(now_t=10.0) is None
