from dataclasses import dataclass
from enum import Flag, auto
from typing import Optional

import numpy as np


class SensorMask(Flag):
    NONE    = 0
    RADAR   = auto()
    BEARING = auto()
    SFM     = auto()


@dataclass
class StateEstimate:
    track_id:     int
    timestamp:    float
    state:        np.ndarray   # (6,) [px, py, pz, vx, vy, vz] world NED m / m/s
    covariance:   np.ndarray   # (6,6) world NED
    sensors_used: SensorMask = SensorMask.NONE
    range_m_true: Optional[float] = None  # ground truth, eval only — never fed to filter

    @property
    def position_ned(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity_ned(self) -> np.ndarray:
        return self.state[3:]
