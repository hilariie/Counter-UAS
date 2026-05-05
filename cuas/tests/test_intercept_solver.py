import numpy as np
import pytest

from cuas.intercept.solver import InterceptSolver
from cuas.intercept.types import InterceptSolution
from cuas.state.types import SensorMask, StateEstimate
from cuas.tracking.controller import ControllerState
from cuas.tracking.gimbal import GimbalCommand

_P6 = np.diag([9.0, 9.0, 9.0, 4.0, 4.0, 4.0])
_ASSET = np.array([0.0, 0.0, -10.0])
_CMD = GimbalCommand(yaw_rad=0.0, pitch_rad=0.0, timestamp=0.0)


def _state(tid, pos, vel):
    x = np.concatenate([pos, vel])
    return StateEstimate(track_id=tid, timestamp=1.0, state=x, covariance=_P6,
                         sensors_used=SensorMask.RADAR)


def _ctrl(state_name, target_id=None):
    return ControllerState(det=None, gimbal_cmd=_CMD, target_id=target_id,
                           lost=False, state_name=state_name)


def _solver():
    return InterceptSolver(defended_asset_ned=_ASSET, interceptor_speed_mps=100.0,
                           max_flight_time_s=30.0)


def test_returns_none_when_not_commit():
    solver = _solver()
    se = _state(1, np.array([50.0, 0.0, 0.0]), np.zeros(3))
    for name in ("IDLE", "WARMUP", "ACQUIRING"):
        assert solver.step([se], _ctrl(name, target_id=1), now_t=1.0) is None


def test_returns_solution_when_commit():
    solver = _solver()
    se = _state(1, np.array([50.0, 0.0, 0.0]), np.zeros(3))
    sol = solver.step([se], _ctrl("COMMIT", target_id=1), now_t=5.0)
    assert sol is not None
    assert isinstance(sol, InterceptSolution)
    assert sol.track_id == 1
    assert sol.timestamp == 5.0


def test_returns_none_when_target_id_missing_from_states():
    solver = _solver()
    se = _state(2, np.array([50.0, 0.0, 0.0]), np.zeros(3))
    # COMMIT for track 99, but only track 2 in estimates
    assert solver.step([se], _ctrl("COMMIT", target_id=99), now_t=1.0) is None


def test_solution_track_id_matches_controller_target():
    solver = _solver()
    se1 = _state(1, np.array([30.0, 0.0, 0.0]), np.zeros(3))
    se2 = _state(2, np.array([80.0, 0.0, 0.0]), np.zeros(3))
    sol = solver.step([se1, se2], _ctrl("COMMIT", target_id=2), now_t=1.0)
    assert sol is not None
    assert sol.track_id == 2


def test_empty_state_estimates_during_commit():
    solver = _solver()
    assert solver.step([], _ctrl("COMMIT", target_id=1), now_t=1.0) is None
