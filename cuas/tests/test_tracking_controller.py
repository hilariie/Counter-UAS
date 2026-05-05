"""Unit tests for the v2 NarrowTrackingController state machine.

Mocks: cosysairsim wheel, NarrowDetTracker, GimbalController, time.time().
"""
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

airsim_mock = MagicMock()
airsim_mock.Vector3r = lambda x, y, z: MagicMock()
airsim_mock.Quaternionr = MagicMock(return_value=MagicMock())
airsim_mock.Pose = MagicMock(return_value=MagicMock())
sys.modules.setdefault("cosysairsim", airsim_mock)

from cuas.cueing.geometry import CameraIntrinsics
from cuas.cueing.track import Track
from cuas.perception.frame_source import FrameMeta
from cuas.tracking import controller as ctrl_mod
from cuas.tracking.controller import NarrowTrackingController
from cuas.tracking.gimbal import GimbalCommand, GimbalController
from cuas.tracking.narrow_detector import NarrowDet, NarrowDetTracker


_INTR = CameraIntrinsics.from_fov(640, 480, 12.0)
_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_META = FrameMeta(frame_id=0, timestamp=0.0, width=640, height=480)


@pytest.fixture
def fake_clock(monkeypatch):
    state = {"t": 1000.0}
    monkeypatch.setattr(ctrl_mod, "_now", lambda: state["t"])
    return state


def _track(tid=1, az=0.1, el=0.05, range_m=100.0, t=0.0):
    return Track(id=tid, created_t=t, last_t=t, az_rad=az, el_rad=el, range_m=range_m)


def _make_ctrl(detector_returns=None, warmup_s=5.0, commit_s=10.0,
               confirm_frames=3, loss_grace_s=1.5):
    client = MagicMock()
    det = MagicMock(spec=NarrowDetTracker)
    det.step.return_value = detector_returns
    det.reset = MagicMock()

    gimbal = MagicMock(spec=GimbalController)
    cmd = GimbalCommand(0.0, 0.0, 0.0)
    gimbal.slew_to_world_bearing.return_value = cmd
    gimbal.nudge_pixel.return_value = cmd

    ctrl = NarrowTrackingController(
        client=client, intrinsics=_INTR,
        warmup_s=warmup_s, commit_s=commit_s,
        confirm_frames=confirm_frames, loss_grace_s=loss_grace_s,
        _detector=det, _gimbal=gimbal,
    )
    return ctrl, det, gimbal


# ---------------------------------------------------------------- WARMUP

def test_warmup_holds_gimbal_and_ignores_target(fake_clock):
    ctrl, det, gimbal = _make_ctrl()
    state = ctrl.step(_track(), 0.0, _FRAME, _META)
    assert state.state_name == "WARMUP"
    gimbal.slew_to_world_bearing.assert_not_called()
    gimbal.nudge_pixel.assert_not_called()
    det.step.assert_not_called()
    assert state.state_remaining_s > 4.9


def test_warmup_exits_after_warmup_s(fake_clock):
    ctrl, det, gimbal = _make_ctrl(warmup_s=5.0)
    ctrl.step(_track(), 0.0, _FRAME, _META)        # enters WARMUP
    fake_clock["t"] += 5.01                         # past warmup
    state = ctrl.step(_track(), 0.0, _FRAME, _META) # should leave WARMUP
    assert state.state_name in ("ACQUIRING", "COMMIT")


# ---------------------------------------------------------------- ACQUIRING → COMMIT

def _detection(cx=320, cy=240, conf=0.9, tid=42):
    return NarrowDet(bbox_xywh=(cx - 20, cy - 20, 40, 40), confidence=conf,
                     track_id=tid, frame_id=0)


def test_confirm_frames_promotes_to_commit(fake_clock):
    ctrl, det, gimbal = _make_ctrl(detector_returns=_detection(), confirm_frames=3)
    fake_clock["t"] += 0.0
    # Skip warmup
    ctrl.step(_track(), 0.0, _FRAME, _META)
    fake_clock["t"] += 6.0
    s1 = ctrl.step(_track(), 0.0, _FRAME, _META)
    assert s1.state_name == "ACQUIRING"
    s2 = ctrl.step(_track(), 0.0, _FRAME, _META)
    s3 = ctrl.step(_track(), 0.0, _FRAME, _META)
    # 3 consecutive detections → COMMIT
    assert s3.state_name == "COMMIT"


# ---------------------------------------------------------------- COMMIT dwell

def test_commit_ignores_target_id_changes(fake_clock):
    ctrl, det, gimbal = _make_ctrl(detector_returns=_detection(), commit_s=10.0)
    ctrl.step(_track(), 0.0, _FRAME, _META)
    fake_clock["t"] += 6.0
    for _ in range(3):
        ctrl.step(_track(tid=1), 0.0, _FRAME, _META)
    assert ctrl._state.value == "COMMIT"

    # cueing flips to a totally different target — controller must ignore it
    fake_clock["t"] += 1.0
    state = ctrl.step(_track(tid=99, az=1.5), 0.0, _FRAME, _META)
    assert state.state_name == "COMMIT"
    assert state.target_id == 1


def test_commit_returns_to_acquiring_after_dwell(fake_clock):
    ctrl, det, gimbal = _make_ctrl(detector_returns=_detection(), commit_s=10.0)
    ctrl.step(_track(), 0.0, _FRAME, _META)
    fake_clock["t"] += 6.0
    for _ in range(3):
        ctrl.step(_track(), 0.0, _FRAME, _META)
    assert ctrl._state.value == "COMMIT"

    fake_clock["t"] += 10.5  # past commit_s
    state = ctrl.step(_track(tid=99, az=1.5), 0.0, _FRAME, _META)
    assert state.state_name in ("ACQUIRING", "COMMIT")  # re-enters via new target
    # After commit window, controller should be willing to slew to a new target
    assert gimbal.slew_to_world_bearing.call_count >= 2


# ---------------------------------------------------------------- loss-of-detection

def test_loss_grace_returns_to_acquiring(fake_clock):
    ctrl, det, gimbal = _make_ctrl(detector_returns=_detection(),
                                    commit_s=10.0, loss_grace_s=1.5)
    ctrl.step(_track(), 0.0, _FRAME, _META)
    fake_clock["t"] += 6.0
    for _ in range(3):
        ctrl.step(_track(), 0.0, _FRAME, _META)
    assert ctrl._state.value == "COMMIT"

    # detector starts returning None
    det.step.return_value = None
    fake_clock["t"] += 0.5
    s1 = ctrl.step(_track(), 0.0, _FRAME, _META)
    assert s1.state_name == "COMMIT"  # within grace
    fake_clock["t"] += 2.0
    s2 = ctrl.step(_track(), 0.0, _FRAME, _META)
    assert s2.state_name == "ACQUIRING"


# ---------------------------------------------------------------- IDLE on no target

def test_idle_when_no_chosen_target_after_warmup(fake_clock):
    ctrl, det, gimbal = _make_ctrl()
    ctrl.step(None, 0.0, _FRAME, _META)
    fake_clock["t"] += 6.0
    state = ctrl.step(None, 0.0, _FRAME, _META)
    assert state.state_name == "IDLE"
    gimbal.slew_to_world_bearing.assert_not_called()
