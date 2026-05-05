"""EKF process and measurement models for Module 4.

State vector: x = [px, py, pz, vx, vy, vz] in world NED (metres, m/s).
Convention matches radar_mock and geometry.py:
    az = atan2(py, px)
    el = atan2(-pz, sqrt(px²+py²))
"""
import math

import numpy as np


# ---------------------------------------------------------------------------
# Process model — constant velocity
# ---------------------------------------------------------------------------

def make_F(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def make_Q(dt: float, accel_sigma: float) -> np.ndarray:
    s2 = accel_sigma ** 2
    q = s2 * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                        [dt ** 3 / 2.0, dt ** 2]])
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[np.ix_([i, i + 3], [i, i + 3])] = q
    return Q


# ---------------------------------------------------------------------------
# Radar measurement model — 4D: [range, az, el, range_rate]
# ---------------------------------------------------------------------------

def h_radar(x: np.ndarray) -> np.ndarray:
    px, py, pz, vx, vy, vz = x
    r = math.sqrt(px*px + py*py + pz*pz)
    rxy = math.sqrt(px*px + py*py)
    az = math.atan2(py, px)
    el = math.atan2(-pz, rxy)
    rr = (px*vx + py*vy + pz*vz) / r
    return np.array([r, az, el, rr])


def H_radar(x: np.ndarray) -> np.ndarray:
    px, py, pz, vx, vy, vz = x
    r = math.sqrt(px*px + py*py + pz*pz)
    rxy = math.sqrt(px*px + py*py)
    r2 = r * r
    rxy2 = rxy * rxy

    H = np.zeros((4, 6))

    # row 0 — range
    H[0, 0] = px / r
    H[0, 1] = py / r
    H[0, 2] = pz / r

    # row 1 — azimuth
    H[1, 0] = -py / rxy2
    H[1, 1] =  px / rxy2

    # row 2 — elevation
    H[2, 0] =  px * pz / (r2 * rxy)
    H[2, 1] =  py * pz / (r2 * rxy)
    H[2, 2] = -rxy / r2

    # row 3 — range-rate
    dot = px*vx + py*vy + pz*vz
    H[3, 0] = (vx*r2 - px*dot) / (r2 * r)
    H[3, 1] = (vy*r2 - py*dot) / (r2 * r)
    H[3, 2] = (vz*r2 - pz*dot) / (r2 * r)
    H[3, 3] = px / r
    H[3, 4] = py / r
    H[3, 5] = pz / r

    return H


# Default radar measurement noise — matches radar_mock noise parameters
R_RADAR = np.diag([9.0,      # range_sigma=3m → 9 m²
                   1.49e-4,  # angle_sigma=0.7° in rad
                   1.49e-4,
                   0.25])    # rate_sigma=0.5 m/s


# ---------------------------------------------------------------------------
# Bearing measurement model — 2D: [az, el]
# ---------------------------------------------------------------------------

def h_bearing(x: np.ndarray) -> np.ndarray:
    px, py, pz = x[0], x[1], x[2]
    rxy = math.sqrt(px*px + py*py)
    az = math.atan2(py, px)
    el = math.atan2(-pz, rxy)
    return np.array([az, el])


def H_bearing(x: np.ndarray) -> np.ndarray:
    return H_radar(x)[1:3, :]


_PIXEL_SIGMA_RAD = 0.7e-3  # 1 px narrow FOV + servo error budget

R_BEARING_BASE = np.diag([_PIXEL_SIGMA_RAD**2, _PIXEL_SIGMA_RAD**2])


def R_bearing(confidence: float = 1.0) -> np.ndarray:
    """Scale bearing noise by detection confidence."""
    if confidence >= 0.5:
        return R_BEARING_BASE.copy()
    scale = min(9.0, (0.5 / confidence) ** 2)
    return R_BEARING_BASE * scale


# ---------------------------------------------------------------------------
# Default initial covariance
# ---------------------------------------------------------------------------

P0 = np.diag([9.0, 9.0, 9.0, 100.0, 100.0, 100.0])
