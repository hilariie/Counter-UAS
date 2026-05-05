"""Persistent track maintained by CueingTracker."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    id: int
    created_t: float
    last_t: float
    az_rad: float
    el_rad: float
    hits: int = 1
    range_m: Optional[float] = None
    range_rate_mps: Optional[float] = None
    last_camera_t: Optional[float] = None
    last_radar_t: Optional[float] = None
    last_radar_track_id: Optional[str] = None
    last_camera_conf: Optional[float] = None
    # EKF-derived confidence: 1.0 = real radar, <1.0 = EKF substitute (degrades with covariance)
    kinematic_confidence: float = 1.0
    # Bbox area history for visual proxies (normalized to image area)
    bbox_area_frac: Optional[float] = None
    bbox_area_frac_prev: Optional[float] = None
    bbox_area_prev_t: Optional[float] = None

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.created_t)

    def has_camera(self, now: float, window: float) -> bool:
        return self.last_camera_t is not None and (now - self.last_camera_t) <= window

    def has_radar(self, now: float, window: float) -> bool:
        return self.last_radar_t is not None and (now - self.last_radar_t) <= window
