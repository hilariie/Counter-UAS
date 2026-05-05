"""EKF predict / update — pure numpy, no I/O."""
import math
from typing import Callable, Optional, Tuple

import numpy as np

from .models import P0

_WRAP = lambda a: (a + math.pi) % (2.0 * math.pi) - math.pi

_POS_DIVERGE = 1e6
_VEL_DIVERGE = 1e4


def predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    gate: Optional[float] = None,
    angle_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """EKF measurement update.

    Returns (x_new, P_new, nis, accepted).
    accepted=False when the measurement is outside the chi-squared gate.
    """
    innov = z - h_fn(x_pred)
    if angle_idx is not None:
        innov[angle_idx] = _WRAP(innov[angle_idx])

    S = H @ P_pred @ H.T + R
    nis = float(innov @ np.linalg.solve(S, innov))

    if gate is not None and nis > gate:
        return x_pred, P_pred, nis, False

    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    x_new = x_pred + K @ innov
    P_new = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_new, P_new, nis, True


def guard_divergence(P: np.ndarray) -> np.ndarray:
    """Reset covariance to P0 if any diagonal element has blown up."""
    diag = P.diagonal()
    if np.any(diag[:3] > _POS_DIVERGE) or np.any(diag[3:] > _VEL_DIVERGE):
        return P0.copy()
    return P
