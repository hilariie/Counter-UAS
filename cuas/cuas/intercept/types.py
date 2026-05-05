from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class FeasibilityReason(str, Enum):
    TARGET_OUTRUNNING = "target_outrunning_interceptor"
    EXCEEDS_MAX_FLIGHT_TIME = "exceeds_max_flight_time"
    DEGENERATE_GEOMETRY = "degenerate_geometry"


@dataclass
class InterceptSolution:
    track_id: int
    timestamp: float
    feasible: bool
    time_to_intercept_s: Optional[float]
    intercept_point_ned: Optional[np.ndarray]      # (3,) world NED
    launch_heading_unit: Optional[np.ndarray]      # (3,) unit vector, world NED
    intercept_covariance: Optional[np.ndarray]     # (3,3) at intercept point, world NED
    reason: Optional[FeasibilityReason] = None
