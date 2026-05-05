"""Per-track EKF instance: initialisation, predict, update, and output."""
import math
from typing import Optional

import numpy as np

from .ekf import guard_divergence, predict, update
from .models import (H_bearing, H_radar, P0, R_RADAR, R_bearing, h_bearing,
                     h_radar, make_F, make_Q)
from .sfm import SfmBuffer, SfmObs
from .types import SensorMask, StateEstimate

# Re-initialise filter after this many consecutive gate rejections.
# A stuck filter (bad init or diverged velocity) permanently gates out every
# measurement; this prevents it from being irrecoverable.
_MAX_CONSECUTIVE_REJECTS = 5


class TrackFilter:
    def __init__(self, track_id: int, accel_sigma: float,
                 gate_radar: float, gate_bearing: float,
                 sfm_baseline_m: float,
                 max_consecutive_rejects: int = _MAX_CONSECUTIVE_REJECTS):
        self._id = track_id
        self._accel_sigma = accel_sigma
        self._gate_radar = gate_radar
        self._gate_bearing = gate_bearing
        self._max_rejects = max_consecutive_rejects
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self._sfm = SfmBuffer(baseline_m=sfm_baseline_m)
        self._sensors = SensorMask.NONE
        # Diagnostics — surfaced in demo display and unit tests
        self._consecutive_rejects: int = 0
        self._reinit_count: int = 0
        self._last_nis: float = 0.0

    @property
    def is_initialised(self) -> bool:
        return self._x is not None

    @property
    def consecutive_rejects(self) -> int:
        return self._consecutive_rejects

    @property
    def reinit_count(self) -> int:
        return self._reinit_count

    @property
    def last_nis(self) -> float:
        return self._last_nis

    def init_from_radar(self, range_m: float, az_rad: float, el_rad: float,
                        range_rate_mps: float, t: float) -> None:
        ce = math.cos(el_rad)
        px = range_m * ce * math.cos(az_rad)
        py = range_m * ce * math.sin(az_rad)
        pz = -range_m * math.sin(el_rad)
        # Only the LOS component of velocity is observable from range-rate.
        # Transverse components are zero-initialised; P0 gives σ=10 m/s uncertainty.
        los = np.array([px, py, pz]) / range_m
        vx, vy, vz = los * range_rate_mps
        self._x = np.array([px, py, pz, vx, vy, vz])
        self._P = P0.copy()
        self._last_t = t

    def step(self, t: float, ownship_ned: np.ndarray) -> None:
        """Predict forward to time t. Call before any updates."""
        if self._x is None or self._last_t is None:
            self._last_t = t
            return
        dt = max(t - self._last_t, 0.0)
        self._last_t = t
        self._sensors = SensorMask.NONE
        if dt > 0:
            F = make_F(dt)
            Q = make_Q(dt, self._accel_sigma)
            self._x, self._P = predict(self._x, self._P, F, Q)
            self._P = guard_divergence(self._P)

    def update_radar(self, range_m: float, az_rad: float,
                     el_rad: float, range_rate_mps: float) -> None:
        if self._x is None:
            return
        z = np.array([range_m, az_rad, el_rad, range_rate_mps])
        H = H_radar(self._x)
        self._x, self._P, nis, accepted = update(
            self._x, self._P, z, H, R_RADAR, h_radar,
            gate=self._gate_radar, angle_idx=1)
        self._last_nis = nis
        if accepted:
            self._sensors |= SensorMask.RADAR
            self._consecutive_rejects = 0
        else:
            self._consecutive_rejects += 1
            if self._consecutive_rejects >= self._max_rejects:
                # Filter is stuck — re-init from this measurement.
                self.init_from_radar(range_m, az_rad, el_rad, range_rate_mps,
                                     self._last_t or 0.0)
                self._consecutive_rejects = 0
                self._reinit_count += 1

    def update_bearing(self, az_rad: float, el_rad: float,
                       confidence: float = 1.0) -> None:
        if self._x is None:
            return
        z = np.array([az_rad, el_rad])
        H = H_bearing(self._x)
        R = R_bearing(confidence)
        self._x, self._P, nis, accepted = update(
            self._x, self._P, z, H, R, h_bearing,
            gate=self._gate_bearing, angle_idx=0)
        if accepted:
            self._sensors |= SensorMask.BEARING

    def push_sfm(self, obs: SfmObs) -> None:
        self._sfm.push(obs)

    def try_sfm_update(self, now_t: float) -> None:
        if self._x is None:
            return
        result = self._sfm.estimate(now_t)
        if result is None:
            return
        range_est, sigma = result
        p = self._x[:3]
        r = float(np.linalg.norm(p))
        if r < 1e-3:
            return
        H_sfm = np.zeros((1, 6))
        H_sfm[0, :3] = p / r
        R_sfm = np.array([[sigma ** 2]])
        z_sfm = np.array([range_est])
        h_sfm = lambda x: np.array([float(np.linalg.norm(x[:3]))])
        self._x, self._P, _, accepted = update(
            self._x, self._P, z_sfm, H_sfm, R_sfm, h_sfm)
        if accepted:
            self._sensors |= SensorMask.SFM

    def init_from_sfm_if_possible(self, now_t: float) -> None:
        """Attempt to initialise from SfM when no radar range is available."""
        if self._x is not None:
            return
        result = self._sfm.estimate(now_t)
        if result is None:
            return
        buf = list(self._sfm._buf)
        if not buf:
            return
        latest = buf[-1]
        range_est, _ = result
        self.init_from_radar(range_est, latest.az_rad, latest.el_rad, 0.0, now_t)
        # Inflate covariance heavily — no radar, uncertain velocity
        self._P = np.diag([400.0, 400.0, 400.0, 400.0, 400.0, 400.0])

    def to_estimate(self, range_m_true: Optional[float] = None) -> StateEstimate:
        return StateEstimate(
            track_id=self._id,
            timestamp=self._last_t or 0.0,
            state=self._x.copy(),
            covariance=self._P.copy(),
            sensors_used=self._sensors,
            range_m_true=range_m_true,
        )
