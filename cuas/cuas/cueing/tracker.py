"""CueingTracker: fuses camera detections + radar returns into ranked tracks.

Two modes:
  radar-led (default): radar returns are primary observations; camera detections
    are corroboration only (no camera-only track spawning).
  cv-led: camera detections are primary; radar enriches tracks with range/rate.

Two-stage greedy NN association per frame:
  Stage A (cross-sensor): cam bearings <-> radar bearings, gate=fusion_gate_deg.
  Stage B (track maintenance): observations <-> existing tracks, gate=track_gate_deg.

Bearings throughout are in world NED radians (matches radar_mock convention).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import greedy_assign
from .geometry import (
    CameraIntrinsics, CameraMount, angle_diff_rad, pixel_to_az_el,
    radar_roi,
)
from .scoring import score_track
from .track import Track
from .weights_config import CueingWeights


@dataclass(frozen=True)
class FusedObservation:
    az_rad: float
    el_rad: float
    timestamp: float
    range_m: Optional[float] = None
    range_rate_mps: Optional[float] = None
    camera_conf: Optional[float] = None
    radar_track_id: Optional[str] = None
    bbox_area_frac: Optional[float] = None


@dataclass(frozen=True)
class RankedTrack:
    track: Track
    score: float
    breakdown: Dict[str, float]


def _bearing_cost_rad(a_az: float, a_el: float, b_az: float, b_el: float) -> float:
    d_az = angle_diff_rad(a_az, b_az)
    d_el = angle_diff_rad(a_el, b_el)
    return math.sqrt(d_az * d_az + d_el * d_el)


class CueingTracker:
    def __init__(self,
                 intrinsics: CameraIntrinsics,
                 mount: CameraMount,
                 weights: CueingWeights,
                 fusion_gate_deg: float = 1.5,
                 track_gate_deg: float = 3.0,
                 track_timeout_s: float = 6.0,
                 min_hits_for_chosen: int = 2,
                 defended_asset_ned: Tuple[float, float, float] = (0.0, 0.0, -10.0),
                 mode: str = "radar-led"):
        self.intrinsics = intrinsics
        self.mount = mount
        self.weights = weights
        self.fusion_gate_rad = math.radians(fusion_gate_deg)
        self.track_gate_rad = math.radians(track_gate_deg)
        self.track_timeout_s = track_timeout_s
        self.min_hits_for_chosen = max(1, int(min_hits_for_chosen))
        self.defended_asset_ned = defended_asset_ned
        self.mode = mode

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def set_weights(self, weights: CueingWeights) -> None:
        self.weights = weights

    def set_mode(self, new_mode: str) -> None:
        """Switch cueing mode at runtime; existing tracks are preserved."""
        self.mode = new_mode

    def radar_to_rois(self, radar_returns, ownship_yaw_rad: float,
                      frame_width: int, frame_height: int,
                      roi_size: int = 256) -> List[Tuple[int, int, int, int]]:
        """Project radar bearings to pixel ROI boxes for on-cue detector."""
        rois = []
        for r in radar_returns:
            box = radar_roi(self.intrinsics, self.mount, ownship_yaw_rad,
                            float(r.az_rad), float(r.el_rad), roi_size)
            if box is not None:
                rois.append(box)
        return rois

    def _camera_detections_to_bearings(self, detections, ownship_yaw_rad: float
                                       ) -> List[Tuple[float, float, float, Optional[float]]]:
        """Returns (az, el, conf, bbox_area_frac) per detection."""
        img_area = self.intrinsics.width * self.intrinsics.height
        bearings = []
        for d in detections:
            az, el = pixel_to_az_el(self.intrinsics, self.mount,
                                    ownship_yaw_rad, d.cx, d.cy)
            bbox_area_frac = (d.width * d.height) / img_area if img_area > 0 else None
            bearings.append((az, el, float(d.confidence), bbox_area_frac))
        return bearings

    def _stage_a_fuse(self, cam_bearings, radar_returns, now_t: float
                      ) -> List[FusedObservation]:
        """CV-led cross-sensor fusion: both camera and radar can spawn tracks."""
        m, n = len(cam_bearings), len(radar_returns)
        cost = np.full((m, n), np.inf, dtype=float)
        for i, (az_c, el_c, _, _bbox) in enumerate(cam_bearings):
            for j, r in enumerate(radar_returns):
                cost[i, j] = _bearing_cost_rad(az_c, el_c, r.az_rad, r.el_rad)
        pairs = greedy_assign(cost, self.fusion_gate_rad)
        matched_c, matched_r = set(), set()
        obs: List[FusedObservation] = []
        for i, j in pairs:
            az_c, el_c, conf, bbox_frac = cam_bearings[i]
            r = radar_returns[j]
            obs.append(FusedObservation(
                az_rad=az_c, el_rad=el_c,
                timestamp=now_t,
                range_m=float(r.range_m),
                range_rate_mps=float(r.range_rate_mps),
                camera_conf=conf,
                radar_track_id=str(r.track_id),
                bbox_area_frac=bbox_frac,
            ))
            matched_c.add(i)
            matched_r.add(j)
        for i, (az_c, el_c, conf, bbox_frac) in enumerate(cam_bearings):
            if i in matched_c:
                continue
            obs.append(FusedObservation(
                az_rad=az_c, el_rad=el_c,
                timestamp=now_t,
                camera_conf=conf,
                bbox_area_frac=bbox_frac,
            ))
        for j, r in enumerate(radar_returns):
            if j in matched_r:
                continue
            obs.append(FusedObservation(
                az_rad=float(r.az_rad), el_rad=float(r.el_rad),
                timestamp=now_t,
                range_m=float(r.range_m),
                range_rate_mps=float(r.range_rate_mps),
                radar_track_id=str(r.track_id),
            ))
        return obs

    def _stage_b_track_match(self, obs: List[FusedObservation], now_t: float) -> None:
        track_ids = list(self.tracks.keys())
        m, n = len(obs), len(track_ids)
        if m and n:
            cost = np.full((m, n), np.inf, dtype=float)
            for i, o in enumerate(obs):
                for j, tid in enumerate(track_ids):
                    t = self.tracks[tid]
                    cost[i, j] = _bearing_cost_rad(o.az_rad, o.el_rad, t.az_rad, t.el_rad)
            pairs = greedy_assign(cost, self.track_gate_rad)
        else:
            pairs = []

        matched_obs = set()
        for i, j in pairs:
            o = obs[i]
            t = self.tracks[track_ids[j]]
            self._update_track(t, o)
            matched_obs.add(i)

        for i, o in enumerate(obs):
            if i in matched_obs:
                continue
            self._spawn_track(o)

        stale = [tid for tid, t in self.tracks.items() if (now_t - t.last_t) > self.track_timeout_s]
        for tid in stale:
            del self.tracks[tid]

    def _update_track(self, t: Track, o: FusedObservation) -> None:
        prev_t = t.last_t
        t.az_rad = o.az_rad
        t.el_rad = o.el_rad
        t.last_t = o.timestamp
        t.hits += 1
        if o.camera_conf is not None:
            t.last_camera_t = o.timestamp
            t.last_camera_conf = o.camera_conf
        if o.range_m is not None:
            t.range_m = o.range_m
            t.range_rate_mps = o.range_rate_mps
            t.last_radar_t = o.timestamp
            t.last_radar_track_id = o.radar_track_id
        if o.bbox_area_frac is not None:
            t.bbox_area_frac_prev = t.bbox_area_frac
            t.bbox_area_prev_t = prev_t
            t.bbox_area_frac = o.bbox_area_frac

    def _spawn_track(self, o: FusedObservation) -> None:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(
            id=tid,
            created_t=o.timestamp,
            last_t=o.timestamp,
            az_rad=o.az_rad,
            el_rad=o.el_rad,
            hits=1,
            range_m=o.range_m,
            range_rate_mps=o.range_rate_mps,
            last_camera_t=o.timestamp if o.camera_conf is not None else None,
            last_radar_t=o.timestamp if o.range_m is not None else None,
            last_radar_track_id=o.radar_track_id,
            last_camera_conf=o.camera_conf,
            bbox_area_frac=o.bbox_area_frac,
        )

    def _update_cv_led(self, detections, radar_returns, ownship_yaw_rad: float,
                       now_t: float) -> None:
        cam_bearings = self._camera_detections_to_bearings(detections, ownship_yaw_rad)
        obs = self._stage_a_fuse(cam_bearings, radar_returns, now_t)
        self._stage_b_track_match(obs, now_t)

    def _update_radar_led(self, detections, radar_returns, ownship_yaw_rad: float,
                          now_t: float) -> None:
        # Radar returns are the primary observation stream.
        radar_obs: List[FusedObservation] = [
            FusedObservation(
                az_rad=float(r.az_rad), el_rad=float(r.el_rad),
                timestamp=now_t,
                range_m=float(r.range_m),
                range_rate_mps=float(r.range_rate_mps),
                radar_track_id=str(r.track_id),
            )
            for r in radar_returns
        ]
        # Camera detections are corroboration: match against radar obs and
        # enrich those that match; discard unmatched camera (no camera-only
        # track spawning in radar-led mode).
        cam_bearings = self._camera_detections_to_bearings(detections, ownship_yaw_rad)
        if radar_obs and cam_bearings:
            m, n = len(cam_bearings), len(radar_obs)
            cost = np.full((m, n), np.inf, dtype=float)
            for i, (az_c, el_c, _, _bbox) in enumerate(cam_bearings):
                for j, o in enumerate(radar_obs):
                    cost[i, j] = _bearing_cost_rad(az_c, el_c, o.az_rad, o.el_rad)
            for ci, ri in greedy_assign(cost, self.fusion_gate_rad):
                _, _, conf, bbox_frac = cam_bearings[ci]
                o = radar_obs[ri]
                radar_obs[ri] = FusedObservation(
                    az_rad=o.az_rad, el_rad=o.el_rad,
                    timestamp=o.timestamp,
                    range_m=o.range_m, range_rate_mps=o.range_rate_mps,
                    camera_conf=conf,
                    radar_track_id=o.radar_track_id,
                    bbox_area_frac=bbox_frac,
                )
        self._stage_b_track_match(radar_obs, now_t)

    def step(self, detections, radar_returns, ownship_yaw_rad: float, now_t: float
             ) -> List[RankedTrack]:
        if self.mode == "radar-led":
            self._update_radar_led(detections, list(radar_returns), ownship_yaw_rad, now_t)
        else:
            self._update_cv_led(detections, list(radar_returns), ownship_yaw_rad, now_t)
        ranked = []
        for t in self.tracks.values():
            s, br = score_track(t, self.weights, now_t)
            ranked.append(RankedTrack(track=t, score=s, breakdown=br))
        ranked.sort(key=lambda rt: rt.score, reverse=True)
        return ranked

    def chosen_target_id(self, ranked: List[RankedTrack]) -> Optional[int]:
        """Top-ranked track that has been hit at least min_hits_for_chosen
        times. Falls back to top-1 if no track qualifies (e.g. cold start).
        """
        if not ranked:
            return None
        for rt in ranked:
            if rt.track.hits >= self.min_hits_for_chosen:
                return rt.track.id
        return ranked[0].track.id
