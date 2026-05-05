"""Stereo-from-motion ring buffer + lstsq triangulation.

Provides a vision-only range seed when radar is denied.
"""
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SfmObs:
    timestamp:   float
    ownship_ned: np.ndarray  # (3,) world NED
    az_rad:      float
    el_rad:      float


def _ray(az: float, el: float) -> np.ndarray:
    ce = math.cos(el)
    return np.array([ce * math.cos(az), ce * math.sin(az), -math.sin(el)])


class SfmBuffer:
    _BEARING_SIGMA = 0.7e-3  # rad, matches R_BEARING_BASE

    def __init__(self, capacity: int = 20, baseline_m: float = 5.0,
                 min_sweep_rad: float = math.radians(2.0),
                 min_obs: int = 5, rate_limit_s: float = 1.0):
        self._buf: deque[SfmObs] = deque(maxlen=capacity)
        self._baseline_m = baseline_m
        self._min_sweep = min_sweep_rad
        self._min_obs = min_obs
        self._rate_limit = rate_limit_s
        self._last_estimate_t: float = -1e9

    def push(self, obs: SfmObs) -> None:
        self._buf.append(obs)

    def estimate(self, now_t: float) -> Optional[Tuple[float, float]]:
        """Return (range_m, sigma_range) or None if conditions not met."""
        if len(self._buf) < self._min_obs:
            return None
        if now_t - self._last_estimate_t < self._rate_limit:
            return None

        obs = list(self._buf)
        positions = np.array([o.ownship_ned for o in obs])
        baseline = float(np.max(
            np.linalg.norm(positions - positions[0:1], axis=1)
        ))
        az_vals = np.array([o.az_rad for o in obs])
        el_vals = np.array([o.el_rad for o in obs])
        sweep = max(float(az_vals.max() - az_vals.min()),
                    float(el_vals.max() - el_vals.min()))

        if baseline < self._baseline_m and sweep < self._min_sweep:
            return None

        d0 = _ray(obs[0].az_rad, obs[0].el_rad)
        t0_vals = []
        for k in range(1, len(obs)):
            dk = _ray(obs[k].az_rad, obs[k].el_rad)
            A = np.column_stack([d0, -dk])
            b = obs[k].ownship_ned - obs[0].ownship_ned
            res, *_ = np.linalg.lstsq(A, b, rcond=None)
            t0_vals.append(float(res[0]))

        t0 = float(np.median(t0_vals))
        if t0 < 5.0 or t0 > 3000.0:
            return None

        sigma = float(np.clip(
            self._BEARING_SIGMA * t0 ** 2 / max(baseline, 1.0),
            5.0, 80.0
        ))
        self._last_estimate_t = now_t
        return t0, sigma

    def clear(self) -> None:
        self._buf.clear()
        self._last_estimate_t = -1e9
