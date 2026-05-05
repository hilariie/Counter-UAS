import numpy as np
import pytest

from cuas.intercept.solver import solve_pn
from cuas.intercept.types import FeasibilityReason
from cuas.state.models import make_F

_P6 = np.diag([9.0, 9.0, 9.0, 4.0, 4.0, 4.0])
_ORIGIN = np.zeros(3)


def _sol(p_T, v_T, V_I=100.0, max_t=30.0, p_D=None):
    if p_D is None:
        p_D = _ORIGIN
    return solve_pn(p_T, v_T, p_D, _P6, V_I, max_t)


def test_stationary_target():
    p_T = np.array([50.0, 0.0, 0.0])
    v_T = np.zeros(3)
    sol = _sol(p_T, v_T)
    assert sol.feasible
    assert abs(sol.time_to_intercept_s - 0.5) < 1e-9
    np.testing.assert_allclose(sol.launch_heading_unit, np.array([1.0, 0.0, 0.0]), atol=1e-9)


def test_head_on_collision():
    # target at x=100, moving toward origin at 30 m/s; interceptor at V=100
    p_T = np.array([100.0, 0.0, 0.0])
    v_T = np.array([-30.0, 0.0, 0.0])
    sol = _sol(p_T, v_T)
    assert sol.feasible
    # analytic: dp=(100,0,0), a=900-10000=-9100, b=2*(-30*100)=-6000, c=10000
    # disc = 36e6 + 4*9100*10000 = 36e6 + 364e6 = 400e6; sqrt=20000
    # t1 = (6000-20000) / (-18200) = 14000/18200 ≈ 0.7692
    t_expected = 14000.0 / 18200.0
    assert abs(sol.time_to_intercept_s - t_expected) < 1e-9


def test_crossing_target():
    # target to the side, moving laterally; interceptor should lead ahead
    p_T = np.array([0.0, 100.0, 0.0])
    v_T = np.array([50.0, 0.0, 0.0])    # moving perpendicular
    sol = _sol(p_T, v_T)
    assert sol.feasible
    # intercept_point should be ahead of p_T in the x direction
    assert sol.intercept_point_ned[0] > 0.0
    # heading should not simply point at p_T (it should be aimed at the lead point)
    # p_T unit from origin is (0,1,0); if heading had no x-component that would be non-leading
    assert abs(sol.launch_heading_unit[0]) > 1e-3


def test_intercept_point_consistent():
    p_T = np.array([80.0, 40.0, -20.0])
    v_T = np.array([-5.0, 3.0, 1.0])
    sol = _sol(p_T, v_T)
    assert sol.feasible
    t = sol.time_to_intercept_s
    dist = float(np.linalg.norm(sol.intercept_point_ned - _ORIGIN))
    assert abs(dist - 100.0 * t) < 1e-6


def test_target_faster_opening():
    # target faster than interceptor and moving away
    p_T = np.array([100.0, 0.0, 0.0])
    v_T = np.array([200.0, 0.0, 0.0])   # opening at 200 m/s; V_I=100
    sol = _sol(p_T, v_T)
    assert not sol.feasible
    assert sol.reason == FeasibilityReason.TARGET_OUTRUNNING


def test_target_faster_inbound_still_feasible():
    # target faster than interceptor but flying toward asset
    p_T = np.array([200.0, 0.0, 0.0])
    v_T = np.array([-150.0, 0.0, 0.0])  # inbound; V_I=100
    sol = _sol(p_T, v_T)
    assert sol.feasible
    # analytic: dp=(200,0,0), a=22500-10000=12500, b=2*(-150*200)=-60000, c=40000
    # disc=3.6e9-4*12500*40000=3.6e9-2e9=1.6e9; sqrt=40000
    # t1=(60000-40000)/25000=0.8
    assert abs(sol.time_to_intercept_s - 0.8) < 1e-9


def test_max_flight_time_filter():
    # far target; valid root but beyond max_flight_time_s
    p_T = np.array([500.0, 0.0, 0.0])
    v_T = np.zeros(3)
    sol = _sol(p_T, v_T, V_I=100.0, max_t=4.0)  # TTI=5s, limit=4s
    assert not sol.feasible
    assert sol.reason == FeasibilityReason.EXCEEDS_MAX_FLIGHT_TIME


def test_degenerate_speed_match():
    # |v_T| == V_I exactly → leading coefficient a ≈ 0, linear fallback
    V_I = 100.0
    p_T = np.array([100.0, 0.0, 0.0])
    v_T = np.array([-100.0, 0.0, 0.0])  # |v_T|=100 = V_I, inbound
    sol = _sol(p_T, v_T, V_I=V_I)
    assert sol.feasible
    # linear: b*t + c = 0 → t = -c/b
    # dp=(100,0,0), b=2*(-100*100)=-20000, c=10000 → t=0.5
    assert abs(sol.time_to_intercept_s - 0.5) < 1e-9
    dist = float(np.linalg.norm(sol.intercept_point_ned - _ORIGIN))
    assert abs(dist - V_I * 0.5) < 1e-9


def test_covariance_grows_with_tti():
    # same input cov, larger TTI → larger covariance trace
    P6 = np.eye(6) * 4.0

    # fast intercept: nearby target
    p_near = np.array([10.0, 0.0, 0.0])
    sol_near = solve_pn(p_near, np.zeros(3), _ORIGIN, P6, 100.0, 30.0)

    # slower intercept: far target
    p_far = np.array([200.0, 0.0, 0.0])
    sol_far = solve_pn(p_far, np.zeros(3), _ORIGIN, P6, 100.0, 30.0)

    assert sol_near.feasible and sol_far.feasible
    assert sol_far.time_to_intercept_s > sol_near.time_to_intercept_s
    tr_near = float(np.trace(sol_near.intercept_covariance))
    tr_far = float(np.trace(sol_far.intercept_covariance))
    assert tr_far > tr_near


def test_covariance_uses_module4_F():
    p_T = np.array([60.0, 0.0, 0.0])
    v_T = np.zeros(3)
    P6 = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
    sol = solve_pn(p_T, v_T, _ORIGIN, P6, 100.0, 30.0)
    assert sol.feasible
    t = sol.time_to_intercept_s
    F = make_F(t)
    expected_cov = (F @ P6 @ F.T)[:3, :3]
    np.testing.assert_allclose(sol.intercept_covariance, expected_cov, atol=1e-12)
