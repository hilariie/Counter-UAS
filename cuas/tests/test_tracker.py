"""End-to-end tests for CueingTracker — covers Module 2 exit-criteria #2 and #3
without the simulator. Uses synthetic Detection and RadarReturn instances.
"""
import math

import pytest

from cuas.cueing import (
    CameraIntrinsics, CameraMount, CueingTracker, CueingWeights,
)
from cuas.perception.detection import Detection
from cuas.sim.radar_mock import RadarReturn


def make_tracker(**overrides):
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    mount = CameraMount(yaw_deg=0, pitch_deg=0, roll_deg=0)
    # Tests use the v1.0 gate/timeout (tighter) so association edge cases
    # remain meaningful; the v1.1 production defaults are looser.
    # mode="cv-led" matches the original mode-agnostic behaviour these tests
    # were written for; radar-led / mode-dispatch tests live in
    # test_cueing_tracker_mode_dispatch.py.
    args = dict(intrinsics=intr, mount=mount, weights=CueingWeights(),
                fusion_gate_deg=1.5, track_gate_deg=2.0, track_timeout_s=3.0,
                min_hits_for_chosen=1, mode="cv-led")
    args.update(overrides)
    return CueingTracker(**args), intr, mount


def make_det(u, v, *, frame_id=0, timestamp=0.0, conf=0.9):
    # 20x20 box around (u, v).
    return Detection(x1=u - 10, y1=v - 10, x2=u + 10, y2=v + 10,
                     confidence=conf, class_id=0, class_name="uav",
                     frame_id=frame_id, timestamp=timestamp)


def make_radar(name, az_deg, el_deg, range_m=400.0, rate=-10.0, t=0.0):
    return RadarReturn(
        track_id=name,
        range_m=range_m,
        az_rad=math.radians(az_deg),
        el_rad=math.radians(el_deg),
        range_rate_mps=rate,
        rcs_dbsm=-15.0,
        timestamp=t,
    )


def test_camera_only_detection_spawns_one_track():
    tr, intr, _ = make_tracker()
    det = make_det(intr.cx, intr.cy, timestamp=1.0)
    ranked = tr.step([det], [], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    t = ranked[0].track
    assert t.range_m is None
    assert t.last_camera_t == 1.0
    assert t.last_radar_t is None


def test_radar_only_return_spawns_one_track():
    tr, _, _ = make_tracker()
    rad = make_radar("Z_Intruder1", 10.0, 0.0, t=1.0)
    ranked = tr.step([], [rad], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    t = ranked[0].track
    assert t.range_m == pytest.approx(400.0)
    assert t.last_radar_t == 1.0
    assert t.last_camera_t is None
    assert t.last_radar_track_id == "Z_Intruder1"


def test_camera_and_radar_at_same_bearing_fuse_into_one_track():
    """Exit criterion #2: when both sensors see the same intruder, exactly one
    fused track exists (not one cam-only + one radar-only)."""
    tr, intr, _ = make_tracker()
    # 100 px right of center @ fx=640 -> az = atan(100/640) = 8.88°
    az_deg = math.degrees(math.atan(100.0 / 640.0))
    det = make_det(intr.cx + 100, intr.cy, timestamp=1.0)
    rad = make_radar("Z_Intruder1", az_deg, 0.0, t=1.0)
    ranked = tr.step([det], [rad], ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 1
    t = ranked[0].track
    assert t.last_camera_t == 1.0
    assert t.last_radar_t == 1.0
    assert t.range_m == pytest.approx(400.0)


def test_two_widely_separated_intruders_form_two_tracks():
    tr, intr, _ = make_tracker()
    az_left = math.degrees(math.atan(-200.0 / 640.0))
    az_right = math.degrees(math.atan(+200.0 / 640.0))
    dets = [
        make_det(intr.cx - 200, intr.cy, timestamp=1.0),
        make_det(intr.cx + 200, intr.cy, timestamp=1.0),
    ]
    rads = [
        make_radar("A", az_left, 0.0, range_m=300.0, t=1.0),
        make_radar("B", az_right, 0.0, range_m=600.0, t=1.0),
    ]
    ranked = tr.step(dets, rads, ownship_yaw_rad=0.0, now_t=1.0)
    assert len(ranked) == 2
    radar_ids = {rt.track.last_radar_track_id for rt in ranked}
    assert radar_ids == {"A", "B"}


def test_track_id_is_stable_across_frames():
    """Exit criterion #3: track IDs persist across frames."""
    tr, intr, _ = make_tracker()
    az_deg = math.degrees(math.atan(100.0 / 640.0))
    for k in range(5):
        det = make_det(intr.cx + 100, intr.cy, frame_id=k, timestamp=float(k) * 0.1)
        rad = make_radar("Z_Intruder1", az_deg, 0.0, t=float(k) * 0.1)
        ranked = tr.step([det], [rad], ownship_yaw_rad=0.0, now_t=float(k) * 0.1)
    assert len(tr.tracks) == 1
    t = next(iter(tr.tracks.values()))
    assert t.id == 1
    assert t.hits == 5


def test_stale_track_is_pruned():
    tr, _, _ = make_tracker(track_timeout_s=0.5)
    rad = make_radar("X", 0.0, 0.0, t=0.0)
    tr.step([], [rad], ownship_yaw_rad=0.0, now_t=0.0)
    assert len(tr.tracks) == 1
    # Step well past the timeout with no observations.
    tr.step([], [], ownship_yaw_rad=0.0, now_t=2.0)
    assert len(tr.tracks) == 0


def test_reacquisition_after_prune_yields_new_id():
    tr, _, _ = make_tracker(track_timeout_s=0.5)
    rad = make_radar("X", 0.0, 0.0, t=0.0)
    ranked1 = tr.step([], [rad], ownship_yaw_rad=0.0, now_t=0.0)
    first_id = ranked1[0].track.id
    tr.step([], [], ownship_yaw_rad=0.0, now_t=2.0)  # prunes
    rad2 = make_radar("X", 0.0, 0.0, t=3.0)
    ranked2 = tr.step([], [rad2], ownship_yaw_rad=0.0, now_t=3.0)
    new_id = ranked2[0].track.id
    assert new_id != first_id


def test_chosen_target_id_is_top_ranked():
    tr, intr, _ = make_tracker()
    # Close, fast-closing target is more threatening than a far, receding one.
    az_deg = math.degrees(math.atan(100.0 / 640.0))
    det = make_det(intr.cx + 100, intr.cy, timestamp=1.0)
    rad_close = make_radar("close", az_deg, 0.0, range_m=100.0, rate=-25.0, t=1.0)
    rad_far = make_radar("far", -az_deg, 0.0, range_m=1500.0, rate=+10.0, t=1.0)
    det_far = make_det(intr.cx - 100, intr.cy, timestamp=1.0)
    ranked = tr.step([det, det_far], [rad_close, rad_far], ownship_yaw_rad=0.0, now_t=1.0)
    chosen = tr.chosen_target_id(ranked)
    assert chosen == ranked[0].track.id
    # Verify chosen is the close target by its preserved radar id.
    assert tr.tracks[chosen].last_radar_track_id == "close"


def test_set_weights_takes_effect_next_step():
    tr, intr, _ = make_tracker()
    az_deg = math.degrees(math.atan(100.0 / 640.0))
    det = make_det(intr.cx + 100, intr.cy, timestamp=1.0)
    rad = make_radar("X", az_deg, 0.0, range_m=100.0, rate=-25.0, t=1.0)
    ranked_a = tr.step([det], [rad], ownship_yaw_rad=0.0, now_t=1.0)
    score_a = ranked_a[0].score

    # Zero out every weight -> total score must be 0 next step.
    tr.set_weights(CueingWeights(
        w_range=0.0, w_range_rate=0.0, w_heading=0.0,
        w_persistence=0.0, w_sensor_agree=0.0, w_novelty=0.0,
    ))
    ranked_b = tr.step([det], [rad], ownship_yaw_rad=0.0, now_t=1.1)
    assert score_a > 0.0
    assert ranked_b[0].score == 0.0


def test_chosen_target_id_none_when_no_tracks():
    tr, _, _ = make_tracker()
    ranked = tr.step([], [], ownship_yaw_rad=0.0, now_t=0.0)
    assert tr.chosen_target_id(ranked) is None


def test_min_hits_for_chosen_skips_one_hit_track():
    """A multi-hit track must be chosen over a higher-scoring 1-hit track."""
    tr, intr, _ = make_tracker(min_hits_for_chosen=2)
    az_lo = math.degrees(math.atan(-200.0 / 640.0))
    az_hi = math.degrees(math.atan(+200.0 / 640.0))
    # Build up a 2-hit track at az_lo over two frames (lower score: far range).
    rad_persistent = make_radar("persistent", az_lo, 0.0, range_m=400.0, rate=0.0, t=0.0)
    tr.step([], [rad_persistent], ownship_yaw_rad=0.0, now_t=0.0)
    rad_persistent_2 = make_radar("persistent", az_lo, 0.0, range_m=400.0, rate=0.0, t=0.5)
    tr.step([], [rad_persistent_2], ownship_yaw_rad=0.0, now_t=0.5)

    # Now introduce a single-frame hot track at az_hi (close + closing fast).
    rad_hot = make_radar("hot", az_hi, 0.0, range_m=50.0, rate=-30.0, t=1.0)
    rad_persistent_3 = make_radar("persistent", az_lo, 0.0, range_m=400.0, rate=0.0, t=1.0)
    ranked = tr.step([], [rad_hot, rad_persistent_3], ownship_yaw_rad=0.0, now_t=1.0)

    # Top-1 by score is the new "hot" track (1 hit). chosen must skip it.
    assert ranked[0].track.last_radar_track_id == "hot"
    chosen = tr.chosen_target_id(ranked)
    assert tr.tracks[chosen].last_radar_track_id == "persistent"


def test_min_hits_for_chosen_falls_back_to_top1_when_nothing_qualifies():
    """Cold start: no track has enough hits yet -> still pick top-1."""
    tr, intr, _ = make_tracker(min_hits_for_chosen=5)
    az_deg = math.degrees(math.atan(100.0 / 640.0))
    rad = make_radar("X", az_deg, 0.0, range_m=100.0, rate=-25.0, t=1.0)
    ranked = tr.step([], [rad], ownship_yaw_rad=0.0, now_t=1.0)
    chosen = tr.chosen_target_id(ranked)
    assert chosen == ranked[0].track.id  # fallback path
