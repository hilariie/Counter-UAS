"""Threat scoring for cueing tracks.

Each component is in [0, 1] before its weight; the total is a weighted sum.

When kinematic_confidence < 1.0 (EKF substitute during radar failure):
  - Kinematic terms (range, closing, heading) are multiplied by kinematic_confidence
  - Visual proxy terms (bbox size, bbox growth rate) are weighted by (1 - kinematic_confidence)
This gives continuous blending: radar-healthy = full 3D kinematics, EKF-diverged = visual proxies.
"""
from typing import Dict, Tuple

from .track import Track
from .weights_config import CueingWeights


def _range_term(track: Track, w: CueingWeights) -> float:
    if track.range_m is None:
        return 0.0
    return max(0.0, 1.0 - track.range_m / w.range_scale_m)


def _closing_term(track: Track, w: CueingWeights) -> float:
    if track.range_rate_mps is None:
        return 0.0
    # Negative range_rate = closing (matches radar_mock).
    return min(1.0, max(0.0, -track.range_rate_mps / w.rate_scale_mps))


def _persistence_term(track: Track, now: float, w: CueingWeights) -> float:
    return min(1.0, track.age_s(now) / max(w.persistence_tau_s, 1e-3))


def _sensor_agree_term(track: Track, now: float, w: CueingWeights) -> float:
    if track.has_camera(now, w.sensor_agree_window_s) and track.has_radar(now, w.sensor_agree_window_s):
        return 1.0
    return 0.0


def _novelty_term(track: Track, now: float, w: CueingWeights) -> float:
    return max(0.0, 1.0 - track.age_s(now) / max(w.novelty_tau_s, 1e-3))


def _bbox_size_term(track: Track, w: CueingWeights) -> float:
    """Bbox area as a range proxy: larger bbox = closer target."""
    if track.bbox_area_frac is None:
        return 0.0
    return min(1.0, track.bbox_area_frac / max(w.bbox_size_scale_frac, 1e-6))


def _bbox_growth_term(track: Track, w: CueingWeights) -> float:
    """Fractional bbox area growth rate as a closing-rate proxy.

    d(ln A)/dt = (A_curr - A_prev) / (A_prev * dt) is scale-invariant:
    a target whose angular footprint doubles per second scores 1.0 at
    bbox_growth_scale_rps = 1.0, regardless of absolute size.
    """
    if (track.bbox_area_frac is None
            or track.bbox_area_frac_prev is None
            or track.bbox_area_prev_t is None):
        return 0.0
    dt = track.last_t - track.bbox_area_prev_t
    if dt < 1e-3 or track.bbox_area_frac_prev < 1e-9:
        return 0.0
    frac_growth_rate = (track.bbox_area_frac - track.bbox_area_frac_prev) / (track.bbox_area_frac_prev * dt)
    return min(1.0, max(0.0, frac_growth_rate / max(w.bbox_growth_scale_rps, 1e-6)))


def score_track(track: Track, w: CueingWeights, now: float) -> Tuple[float, Dict[str, float]]:
    c = track.kinematic_confidence          # 1.0 = real radar, <1.0 = EKF substitute
    proxy_w = 1.0 - c                       # visual proxies activate as kinematics fade

    r = _range_term(track, w)
    rr = _closing_term(track, w)
    # Heading-toward-asset proxied by closing rate until Module 4 ships a
    # velocity vector. Same input, separate weight, so the YAML can de-emphasize
    # one without the other.
    h = rr
    p = _persistence_term(track, now, w)
    s = _sensor_agree_term(track, now, w)
    n = _novelty_term(track, now, w)
    bs = _bbox_size_term(track, w)
    bg = _bbox_growth_term(track, w)

    total = (
        c * (w.w_range * r + w.w_range_rate * rr + w.w_heading * h)
        + w.w_persistence * p
        + w.w_sensor_agree * s
        + w.w_novelty * n
        + proxy_w * (w.w_bbox_size * bs + w.w_bbox_growth * bg)
    )
    breakdown = {
        "range": r, "closing": rr, "heading": h,
        "persistence": p, "sensor_agree": s, "novelty": n,
        "bbox_size": bs, "bbox_growth": bg,
        "kinematic_conf": c,
        "total": total,
    }
    return total, breakdown
