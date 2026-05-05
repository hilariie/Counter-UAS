"""Tests for CueingTracker mode dispatch (radar-led vs cv-led).

Covers MODULE_2_PLAN exit criteria #4 (mode switch preserves tracks) and #5
(unit tests pass) without any simulator or GPU dependency.
"""
import math
import pytest

from cuas.cueing import (
    CameraIntrinsics, CameraMount, CueingTracker, CueingWeights,
)
from cuas.perception.detection import Detection
from cuas.sim.radar_mock import RadarReturn


def make_tracker(mode="radar-led", **overrides):
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount(yaw_deg=0, pitch_deg=0, roll_deg=0)
    args = dict(intrinsics=intr, mount=mount, weights=CueingWeights(),
                fusion_gate_deg=1.5, track_gate_deg=3.0, track_timeout_s=6.0,
                min_hits_for_chosen=1, mode=mode)
    args.update(overrides)
    return CueingTracker(**args), intr, mount


def make_det(u, v, *, timestamp=0.0, conf=0.9):
    return Detection(x1=u - 10, y1=v - 10, x2=u + 10, y2=v + 10,
                     confidence=conf, class_id=0, class_name="uav",
                     frame_id=0, timestamp=timestamp)


def make_radar(az_deg, el_deg=0.0, range_m=100.0, rate=-10.0, t=0.0, name="R1"):
    return RadarReturn(
        track_id=name,
        range_m=range_m,
        az_rad=math.radians(az_deg),
        el_rad=math.radians(el_deg),
        range_rate_mps=rate,
        rcs_dbsm=-15.0,
        timestamp=t,
    )


# ── mode attribute ────────────────────────────────────────────────────────────

def test_default_mode_is_radar_led():
    tr, _, _ = make_tracker()
    assert tr.mode == "radar-led"


def test_set_mode_cv_led_preserves_tracks():
    tr, intr, _ = make_tracker(mode="radar-led")
    # Seed a track via a radar return in radar-led mode.
    r = make_radar(0.0)
    tr.step([], [r], ownship_yaw_rad=0.0, now_t=1.0)
    ids_before = set(tr.tracks.keys())
    assert ids_before, "no track spawned — test premise failed"

    tr.set_mode("cv-led")
    ids_after = set(tr.tracks.keys())
    assert ids_after == ids_before


def test_set_mode_round_trip_preserves_tracks():
    tr, intr, _ = make_tracker(mode="cv-led")
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    tr.step([det], [], ownship_yaw_rad=0.0, now_t=1.0)
    ids_before = set(tr.tracks.keys())

    tr.set_mode("radar-led")
    tr.set_mode("cv-led")
    assert set(tr.tracks.keys()) == ids_before


# ── radar-led behaviour ───────────────────────────────────────────────────────

def test_radar_led_radar_return_spawns_track():
    tr, _, _ = make_tracker(mode="radar-led")
    r = make_radar(5.0)
    ranked = tr.step([], [r], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    assert ranked[0].track.range_m is not None


def test_radar_led_camera_only_does_not_spawn_track():
    tr, intr, _ = make_tracker(mode="radar-led")
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    ranked = tr.step([det], [], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 0, "camera-only detection must not spawn a track in radar-led mode"


def test_radar_led_camera_matching_radar_sets_cam_seen():
    tr, intr, _ = make_tracker(mode="radar-led")
    # Radar at boresight (az=0, el=0) — same direction as image centre.
    r = make_radar(0.0, el_deg=0.0, t=1.0)
    # Camera detection at image centre projects to (az≈0, el≈0).
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    ranked = tr.step([det], [r], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    t = ranked[0].track
    assert t.last_camera_t is not None, "camera should have confirmed radar track"
    assert t.last_radar_t is not None


# ── cv-led behaviour ──────────────────────────────────────────────────────────

def test_cv_led_camera_only_spawns_track():
    tr, intr, _ = make_tracker(mode="cv-led")
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    ranked = tr.step([det], [], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1


def test_cv_led_fuses_radar_range_into_camera_track():
    tr, intr, _ = make_tracker(mode="cv-led")
    r = make_radar(0.0, el_deg=0.0, range_m=120.0, t=1.0)
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    ranked = tr.step([det], [r], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    assert ranked[0].track.range_m == pytest.approx(120.0, abs=5.0)


# ── mid-run mode switch ───────────────────────────────────────────────────────

def test_mode_switch_mid_run_track_ids_unchanged():
    tr, intr, _ = make_tracker(mode="radar-led")
    # Run 3 frames in radar-led with two intruders.
    r1 = make_radar(10.0, name="A")
    r2 = make_radar(-10.0, name="B")
    for t in [1.0, 2.0, 3.0]:
        tr.step([], [r1, r2], ownship_yaw_rad=0.0, now_t=t)
    ids_before = set(tr.tracks.keys())
    assert len(ids_before) == 2

    tr.set_mode("cv-led")
    # One more frame in cv-led with same radar returns.
    tr.step([], [r1, r2], ownship_yaw_rad=0.0, now_t=4.0)
    assert set(tr.tracks.keys()) == ids_before, "track IDs must not change on mode switch"
