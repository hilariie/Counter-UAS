import logging
from typing import List, Optional

import numpy as np

from cuas.intercept.types import FeasibilityReason, InterceptSolution
from cuas.state.models import make_F
from cuas.state.types import StateEstimate
from cuas.tracking.controller import ControllerState

_log = logging.getLogger(__name__)

_EPS = 1e-6


def solve_pn(
    p_T: np.ndarray,
    v_T: np.ndarray,
    p_D: np.ndarray,
    P6: np.ndarray,
    V_I: float,
    max_flight_time_s: float,
) -> InterceptSolution:
    """Closed-form lead-pursuit intercept. track_id and timestamp are placeholders (0)."""
    dp = p_T - p_D
    vT_sq = float(np.dot(v_T, v_T))
    VI_sq = V_I * V_I

    a = vT_sq - VI_sq
    b = 2.0 * float(np.dot(v_T, dp))
    c = float(np.dot(dp, dp))

    t_intercept = None
    reason = None

    if abs(a) < _EPS:
        # Linear fallback when |v_T| ≈ V_I
        if abs(b) < _EPS:
            reason = FeasibilityReason.DEGENERATE_GEOMETRY
        else:
            t_lin = -c / b
            if t_lin > 0.0:
                t_intercept = t_lin
            else:
                reason = FeasibilityReason.DEGENERATE_GEOMETRY
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            reason = FeasibilityReason.TARGET_OUTRUNNING
        else:
            sqrt_disc = disc ** 0.5
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            candidates = [t for t in (t1, t2) if t > 1e-9]
            if not candidates:
                reason = FeasibilityReason.TARGET_OUTRUNNING
            else:
                t_intercept = min(candidates)

    if reason is not None:
        return InterceptSolution(
            track_id=0, timestamp=0.0, feasible=False,
            time_to_intercept_s=None, intercept_point_ned=None,
            launch_heading_unit=None, intercept_covariance=None,
            reason=reason,
        )

    if t_intercept > max_flight_time_s:
        return InterceptSolution(
            track_id=0, timestamp=0.0, feasible=False,
            time_to_intercept_s=None, intercept_point_ned=None,
            launch_heading_unit=None, intercept_covariance=None,
            reason=FeasibilityReason.EXCEEDS_MAX_FLIGHT_TIME,
        )

    intercept_pt = p_T + v_T * t_intercept
    heading_vec = intercept_pt - p_D
    heading_unit = heading_vec / np.linalg.norm(heading_vec)

    F = make_F(t_intercept)
    P_t = F @ P6 @ F.T
    cov3 = P_t[:3, :3].copy()

    return InterceptSolution(
        track_id=0, timestamp=0.0, feasible=True,
        time_to_intercept_s=t_intercept,
        intercept_point_ned=intercept_pt,
        launch_heading_unit=heading_unit,
        intercept_covariance=cov3,
        reason=None,
    )


class InterceptSolver:
    def __init__(
        self,
        defended_asset_ned: np.ndarray,
        interceptor_speed_mps: float = 100.0,
        max_flight_time_s: float = 30.0,
    ):
        self._p_D = defended_asset_ned.copy()
        self._V_I = interceptor_speed_mps
        self._max_t = max_flight_time_s

    def step(
        self,
        state_estimates: List[StateEstimate],
        ctrl_state: ControllerState,
        now_t: float,
    ) -> Optional[InterceptSolution]:
        if ctrl_state.state_name != "COMMIT":
            return None

        target_id = ctrl_state.target_id
        se = next((s for s in state_estimates if s.track_id == target_id), None)
        if se is None:
            _log.debug("COMMIT target_id=%s not yet in state_estimates", target_id)
            return None

        sol = solve_pn(
            p_T=se.position_ned,
            v_T=se.velocity_ned,
            p_D=self._p_D,
            P6=se.covariance,
            V_I=self._V_I,
            max_flight_time_s=self._max_t,
        )
        sol.track_id = target_id
        sol.timestamp = now_t
        return sol
