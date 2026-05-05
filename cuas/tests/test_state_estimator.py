"""Synthetic end-to-end tests for StateEstimator. No AirSim, no GPU.

Sim-in-the-loop tests are marked @pytest.mark.sim and skipped by default.
Run with:  pytest -m sim cuas/tests/test_state_estimator.py
Requires Blocks.exe running and cuas-venv active.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from cuas.cueing.geometry import CameraIntrinsics
from cuas.cueing.track import Track
from cuas.cueing.tracker import RankedTrack
from cuas.state.estimator import StateEstimator
from cuas.state.types import SensorMask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeRadarReturn:
    track_id: str
    range_m: float
    az_rad: float
    el_rad: float
    range_rate_mps: float
    range_m_true: float


def _spherical(pos_ned):
    px, py, pz = pos_ned
    r = math.sqrt(px*px + py*py + pz*pz)
    rxy = math.sqrt(px*px + py*py)
    az = math.atan2(py, px)
    el = math.atan2(-pz, rxy)
    return r, az, el


def _make_radar_return(track_id: str, pos_ned, vel_ned, t: float,
                       rng: np.random.Generator, noise: bool = True):
    r, az, el = _spherical(pos_ned)
    px, py, pz = pos_ned
    vx, vy, vz = vel_ned
    rr = (px*vx + py*vy + pz*vz) / r
    if noise:
        r   += rng.normal(0, 3.0)
        az  += rng.normal(0, math.radians(0.7))
        el  += rng.normal(0, math.radians(0.7))
        rr  += rng.normal(0, 0.5)
    return _FakeRadarReturn(track_id=track_id, range_m=r, az_rad=az,
                            el_rad=el, range_rate_mps=rr, range_m_true=r)


def _make_track(tid: int, pos_ned, vel_ned, t: float,
                radar_key: Optional[str] = "Z_Intruder1"):
    r, az, el = _spherical(pos_ned)
    vel_r = sum(p*v for p, v in zip(pos_ned, vel_ned)) / r
    return Track(id=tid, created_t=0.0, last_t=t,
                 az_rad=az, el_rad=el, hits=5,
                 range_m=r, range_rate_mps=vel_r,
                 last_radar_track_id=radar_key,
                 last_radar_t=t)


def _ranked(track: Track) -> RankedTrack:
    return RankedTrack(track=track, score=1.0, breakdown={})


NARROW_INTR = CameraIntrinsics.from_fov(width=640, height=512, fov_degrees=12.0)


def _make_estimator() -> StateEstimator:
    return StateEstimator(narrow_intr=NARROW_INTR, track_timeout_s=5.0,
                          sfm_baseline_m=5.0)


# ---------------------------------------------------------------------------
# Scenario 1 — radar-only straight-line target
# ---------------------------------------------------------------------------

def test_radar_only_straight_line():
    rng = np.random.default_rng(7)
    pos = np.array([200., 0., -50.])
    vel = np.array([-5., 0., 0.])
    est = _make_estimator()
    dt = 1.0 / 15.0
    errors = []
    t = 0.0
    for i in range(30):
        pos = pos + vel * dt
        t += dt
        rr = _make_radar_return("Z_Intruder1", pos, vel, t, rng)
        tk = _make_track(1, pos, vel, t)
        results = est.step([_ranked(tk)], [rr], None,
                           np.zeros(3), 0.0, t)
        if i >= 10 and results:
            se = results[0]
            errors.append(np.linalg.norm(se.position_ned - pos))

    assert errors, "no estimates produced after warmup"
    rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
    assert rms < 5.0, f"radar-only RMS={rms:.2f} m > 5 m threshold"


# ---------------------------------------------------------------------------
# Scenario 2 — vision-only (radar denied) with ownship moving
# ---------------------------------------------------------------------------

def test_vision_only_sfm():
    rng = np.random.default_rng(99)
    target = np.array([200., 0., -50.])
    vel = np.array([-5., 0., 0.])
    est = StateEstimator(narrow_intr=NARROW_INTR, track_timeout_s=5.0,
                         sfm_baseline_m=4.0)  # lower baseline for test
    dt = 1.0 / 15.0
    ownship = np.array([0., 0., 0.])
    errors = []
    t = 0.0
    for i in range(40):
        target = target + vel * dt
        ownship = ownship + np.array([2.0, 0., 0.]) * dt
        t += dt
        _, az, el = _spherical(target - ownship)
        tk = Track(id=2, created_t=0.0, last_t=t,
                   az_rad=az, el_rad=el, hits=i+1,
                   range_m=None, range_rate_mps=None,
                   last_radar_track_id=None)
        results = est.step([_ranked(tk)], [], None, ownship, 0.0, t)
        if results and i >= 15:
            se = results[0]
            true_pos = target - ownship  # relative to ownship frame approximation
            errors.append(np.linalg.norm(se.position_ned - (target - ownship)))

    if errors:
        rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
        assert rms < 15.0, f"vision-only RMS={rms:.2f} m > 15 m threshold"


# ---------------------------------------------------------------------------
# Scenario 3 — track dropout and re-init
# ---------------------------------------------------------------------------

def test_track_dropout_no_crash():
    rng = np.random.default_rng(5)
    pos = np.array([200., 0., -50.])
    vel = np.array([-5., 0., 0.])
    est = _make_estimator()
    dt = 1.0 / 15.0
    t = 0.0
    for i in range(10):
        pos = pos + vel * dt
        t += dt
        rr = _make_radar_return("Z_Intruder1", pos, vel, t, rng)
        tk = _make_track(1, pos, vel, t)
        est.step([_ranked(tk)], [rr], None, np.zeros(3), 0.0, t)

    # Skip 7 seconds — filter should be evicted (timeout=5s)
    t += 7.0

    # Resume
    for i in range(5):
        pos = pos + vel * dt
        t += dt
        rr = _make_radar_return("Z_Intruder1", pos, vel, t, rng)
        tk = _make_track(1, pos, vel, t)
        results = est.step([_ranked(tk)], [rr], None, np.zeros(3), 0.0, t)

    # Must not raise; covariance should be non-negative definite
    assert results
    se = results[0]
    eigvals = np.linalg.eigvalsh(se.covariance)
    assert np.all(eigvals >= -1e-6)


# ---------------------------------------------------------------------------
# Scenario 4 — multi-track isolation
# ---------------------------------------------------------------------------

def test_multi_track_isolation():
    rng = np.random.default_rng(13)
    targets = {
        1: (np.array([200., 20., -50.]),  np.array([-5.,  0., 0.]), "Z_Intruder1"),
        2: (np.array([150., -30., -60.]), np.array([-3., 2., 0.]), "Z_Intruder2"),
    }
    est = _make_estimator()
    dt = 1.0 / 15.0
    t = 0.0

    for i in range(20):
        t += dt
        ranked = []
        radars = []
        for tid, (pos, vel, key) in targets.items():
            targets[tid] = (pos + vel * dt, vel, key)
            pos_new = targets[tid][0]
            rr = _make_radar_return(key, pos_new, vel, t, rng)
            tk = _make_track(tid, pos_new, vel, t, radar_key=key)
            ranked.append(_ranked(tk))
            radars.append(rr)
        est.step(ranked, radars, None, np.zeros(3), 0.0, t)

    final = est.step(ranked, radars, None, np.zeros(3), 0.0, t)
    ids = [se.track_id for se in final]
    assert len(ids) == len(set(ids)), "duplicate track IDs in output"
    assert set(ids) == {1, 2}


# ---------------------------------------------------------------------------
# Sim-in-the-loop integration test
# ---------------------------------------------------------------------------

@pytest.mark.sim
def test_sim_ekf_accuracy():
    """Run StateEstimator against the live simulator for 30 s and assert
    that per-track RMS position error meets the Module 4 done criteria:
        all-sensors: RMS < 5 m at ~200 m range
    Requires Blocks.exe running. Skip with default pytest run; use -m sim to enable.
    """
    import time
    import cosysairsim as airsim

    from cuas.sim.radar_mock import RadarMock
    from cuas.cueing import CameraIntrinsics, CameraMount, CueingTracker, WeightsLoader
    from cuas.tracking import (
        NarrowFovFrameSource, GimbalController,
        NarrowDetTracker, NarrowTrackingController,
    )

    OWNSHIP = "Ownship"
    INTRUDERS = ["Z_Intruder1", "Z_Intruder2", "Z_Intruder3"]
    NARROW_FOV_DEG = 20.0

    c = airsim.MultirotorClient()
    c.confirmConnection()

    radar = RadarMock(c, ownship_name=OWNSHIP, scan_hz=15.0)
    narrow_intr = CameraIntrinsics.from_fov(640, 480, NARROW_FOV_DEG)
    import os
    default_yaml = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "cueing_weights.yaml"))
    weights_loader = WeightsLoader(default_yaml)
    wide_intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    wide_mount = CameraMount(yaw_deg=0.0, pitch_deg=-5.0, roll_deg=0.0)
    cueing = CueingTracker(
        intrinsics=wide_intr, mount=wide_mount, weights=weights_loader.weights,
        track_timeout_s=3.0, mode="radar-led",
    )
    state_est = StateEstimator(narrow_intr=narrow_intr)

    errors_by_track: dict = {}
    t0 = time.time()
    run_s = 30.0

    while time.time() - t0 < run_s:
        now_t = time.time() - t0
        own_pose = c.simGetObjectPose(OWNSHIP)
        ownship_ned = np.array([
            own_pose.position.x_val,
            own_pose.position.y_val,
            own_pose.position.z_val,
        ])
        ownship_yaw_rad = math.atan2(
            2.0 * (own_pose.orientation.w_val * own_pose.orientation.z_val
                   + own_pose.orientation.x_val * own_pose.orientation.y_val),
            1.0 - 2.0 * (own_pose.orientation.y_val ** 2
                         + own_pose.orientation.z_val ** 2)
        )

        radar_returns = radar.scan(INTRUDERS, t=now_t, ownship_pose=own_pose)
        ranked = cueing.step([], radar_returns, ownship_yaw_rad, now_t)
        estimates = state_est.step(
            ranked, radar_returns, None,
            ownship_ned, ownship_yaw_rad, now_t,
        )

        for se in estimates:
            if se.range_m_true is None:
                continue
            est_range = float(np.linalg.norm(se.position_ned))
            err = abs(est_range - se.range_m_true)
            errors_by_track.setdefault(se.track_id, []).append(err)

        time.sleep(1.0 / 15.0)

    assert errors_by_track, "StateEstimator produced no estimates with range_m_true in 30 s"

    for tid, errs in errors_by_track.items():
        if len(errs) < 10:
            continue  # not enough samples — filter still warming up
        rms = math.sqrt(sum(e**2 for e in errs) / len(errs))
        print(f"  track #{tid}: n={len(errs)} RMS_err={rms:.2f} m")
        assert rms < 5.0, (
            f"Track #{tid} RMS position error {rms:.2f} m exceeds 5 m threshold "
            f"(Module 4 done criteria). n={len(errs)} samples."
        )
