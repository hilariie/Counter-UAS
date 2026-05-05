"""Track-level synthetic radar driven by AirSim ground truth.
Models a notional X-band C-UAS radar (Echodyne-class).

Uses simGetObjectPose for both ownship and intruders. simGetObjectPose returns
world-NED coordinates; simGetVehiclePose returns each vehicle's pose in its
own local frame (origin at its spawn) which is wrong for cross-vehicle range
calculations.
"""
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class RadarReturn:
    track_id: str
    range_m: float
    az_rad: float          # +x forward, +y left, +z up (radar frame)
    el_rad: float
    range_rate_mps: float
    rcs_dbsm: float
    timestamp: float
    range_m_true: float = 0.0
    az_rad_true: float = 0.0
    el_rad_true: float = 0.0

class RadarMock:
    def __init__(self, client, ownship_name="Ownship", scan_hz=15.0,
                 max_range_m=2000.0, range_sigma=3.0, angle_sigma_deg=0.7,
                 rate_sigma=0.5, rcs_dbsm_default=-15.0, health=1.0):
        self.client = client
        self.ownship = ownship_name
        self.dt = 1.0 / scan_hz
        self.max_range = max_range_m
        self.range_sigma = range_sigma
        self.angle_sigma = np.deg2rad(angle_sigma_deg)
        self.rate_sigma = rate_sigma
        self.rcs = rcs_dbsm_default
        self.health = health
        self._prev_ranges = {}
        self._last_t = None

    def _pd(self, range_m, rcs_dbsm):
        snr_db = rcs_dbsm + 4*10*np.log10(1500.0/max(range_m, 1.0))
        pd = 1.0 / (1.0 + np.exp(-(snr_db - (-5.0))))
        return float(np.clip(self.health * pd, 0.0, 1.0))

    def scan(self, intruder_names: List[str], t: float, ownship_pose=None) -> List[RadarReturn]:
        # Accept a pre-fetched ownship pose so the caller can avoid a duplicate
        # simGetObjectPose RPC each frame. Falls back to fetching internally.
        if ownship_pose is None:
            own = self.client.simGetObjectPose(self.ownship).position
        else:
            own = ownship_pose.position if hasattr(ownship_pose, "position") else ownship_pose
        out = []
        for name in intruder_names:
            p = self.client.simGetObjectPose(name).position
            dx, dy, dz = p.x_val - own.x_val, p.y_val - own.y_val, p.z_val - own.z_val
            r_true = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            if r_true > self.max_range:
                continue
            az_true = float(np.arctan2(dy, dx))
            el_true = float(np.arctan2(-dz, np.sqrt(dx*dx + dy*dy)))
            rr = 0.0
            if name in self._prev_ranges and self._last_t is not None:
                rr = (r_true - self._prev_ranges[name]) / max(t - self._last_t, 1e-3)
            self._prev_ranges[name] = r_true
            if np.random.rand() > self._pd(r_true, self.rcs):
                continue
            out.append(RadarReturn(
                track_id=name,
                range_m=r_true + np.random.randn()*self.range_sigma,
                az_rad=az_true + np.random.randn()*self.angle_sigma,
                el_rad=el_true + np.random.randn()*self.angle_sigma,
                range_rate_mps=rr + np.random.randn()*self.rate_sigma,
                rcs_dbsm=self.rcs + np.random.randn()*1.5,
                timestamp=t,
                range_m_true=r_true,
                az_rad_true=az_true,
                el_rad_true=el_true,
            ))
        self._last_t = t
        return out