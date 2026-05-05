"""StateEstimator — top-level Module 4 class consumed by the main loop."""
import math
from typing import Dict, List, Optional

import numpy as np

from ..cueing.geometry import CameraIntrinsics, CameraMount, pixel_to_az_el
from .sfm import SfmObs
from .track_filter import TrackFilter
from .types import SensorMask, StateEstimate


class StateEstimator:
    def __init__(
        self,
        narrow_intr: CameraIntrinsics,
        q_accel_sigma: float = 2.0,
        gate_chi2_4dof: float = 9.49,
        gate_chi2_2dof: float = 5.99,
        track_timeout_s: float = 5.0,
        sfm_baseline_m: float = 5.0,
    ):
        self._intr = narrow_intr
        self._accel_sigma = q_accel_sigma
        self._gate_radar = gate_chi2_4dof
        self._gate_bearing = gate_chi2_2dof
        self._timeout = track_timeout_s
        self._sfm_baseline = sfm_baseline_m
        self._filters: Dict[int, TrackFilter] = {}
        self._last_seen: Dict[int, float] = {}

    def _get_or_create(self, track_id: int) -> TrackFilter:
        if track_id not in self._filters:
            self._filters[track_id] = TrackFilter(
                track_id=track_id,
                accel_sigma=self._accel_sigma,
                gate_radar=self._gate_radar,
                gate_bearing=self._gate_bearing,
                sfm_baseline_m=self._sfm_baseline,
            )
        return self._filters[track_id]

    def step(
        self,
        ranked_tracks,          # List[RankedTrack]
        radar_returns,          # List[RadarReturn]
        ctrl_state,             # ControllerState | None
        ownship_ned: np.ndarray,
        ownship_yaw_rad: float,
        now_t: float,
    ) -> List[StateEstimate]:
        radar_by_id: Dict[str, object] = {r.track_id: r for r in radar_returns}

        active_ids = set()
        estimates: List[StateEstimate] = []

        for rt in ranked_tracks:
            track = rt.track
            tid = track.id
            active_ids.add(tid)
            self._last_seen[tid] = now_t

            tf = self._get_or_create(tid)
            tf.step(now_t, ownship_ned)

            # --- Radar update ---
            radar_key = track.last_radar_track_id
            range_m_true: Optional[float] = None
            if radar_key and radar_key in radar_by_id:
                r = radar_by_id[radar_key]
                range_m_true = getattr(r, "range_m_true", None)
                if not tf.is_initialised:
                    tf.init_from_radar(r.range_m, r.az_rad, r.el_rad,
                                       r.range_rate_mps, now_t)
                else:
                    tf.update_radar(r.range_m, r.az_rad, r.el_rad, r.range_rate_mps)

            # --- Bearing update from narrow tracker ---
            bearing_az: Optional[float] = None
            bearing_el: Optional[float] = None
            if (ctrl_state is not None
                    and ctrl_state.state_name in ("LOCKED", "COMMIT")
                    and ctrl_state.target_id == tid
                    and ctrl_state.det is not None
                    and ctrl_state.gimbal_cmd is not None):
                det = ctrl_state.det
                gcmd = ctrl_state.gimbal_cmd
                cx = det.bbox_xywh[0] + det.bbox_xywh[2] / 2.0
                cy_px = det.bbox_xywh[1] + det.bbox_xywh[3] / 2.0
                # CameraMount.pitch_deg > 0 = nose-UP (_rot_y convention).
                # GimbalCommand.pitch_rad > 0 = nose-DOWN (external convention).
                # Negate pitch when building effective mount.
                mount = CameraMount(
                    yaw_deg=math.degrees(gcmd.yaw_rad),
                    pitch_deg=-math.degrees(gcmd.pitch_rad),
                    roll_deg=0.0,
                )
                bearing_az, bearing_el = pixel_to_az_el(
                    self._intr, mount, ownship_yaw_rad, cx, cy_px)
                if tf.is_initialised:
                    tf.update_bearing(bearing_az, bearing_el, det.confidence)

            # --- SfM: push every step using best available bearing ---
            if bearing_az is not None:
                az_sfm, el_sfm = bearing_az, bearing_el
            elif track.az_rad is not None:
                az_sfm, el_sfm = track.az_rad, track.el_rad
            else:
                az_sfm = el_sfm = None

            if az_sfm is not None:
                tf.push_sfm(SfmObs(
                    timestamp=now_t,
                    ownship_ned=ownship_ned.copy(),
                    az_rad=az_sfm,
                    el_rad=el_sfm,
                ))

            if not tf.is_initialised:
                tf.init_from_sfm_if_possible(now_t)
            elif SensorMask.RADAR not in tf._sensors:
                tf.try_sfm_update(now_t)

            if tf.is_initialised:
                estimates.append(tf.to_estimate(range_m_true))

        # --- Drop stale filters ---
        stale = [tid for tid, last_t in self._last_seen.items()
                 if now_t - last_t > self._timeout]
        for tid in stale:
            self._filters.pop(tid, None)
            self._last_seen.pop(tid, None)

        return estimates
