import numpy as np
import pytest

from cuas.cueing.track import Track
from cuas.cueing.tracker import RankedTrack
from cuas.state.types import StateEstimate, SensorMask
from cuas.tracking.controller import ControllerState
from cuas.tracking.gimbal import GimbalCommand
from cuas.perception.frame_source import FrameMeta
from cuas.viz.ui_frame import build_ui_frame


def _make_ctrl_state():
    return ControllerState(
        det=None,
        gimbal_cmd=GimbalCommand(yaw_rad=0.0, pitch_rad=0.0, timestamp=0.0),
        target_id=None,
        lost=False,
        state_name="IDLE",
        state_remaining_s=0.0,
    )


def test_build_ui_frame_from_synthetic_snapshot():
    track = Track(id=42, created_t=0.0, last_t=1.0, az_rad=0.1, el_rad=-0.05, hits=5)
    ranked = [RankedTrack(track=track, score=3.7, breakdown={})]

    wide_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    narrow_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wide_meta = FrameMeta(frame_id=1, timestamp=1.0, width=640, height=480)
    narrow_meta = FrameMeta(frame_id=1, timestamp=1.0, width=640, height=480)

    se = StateEstimate(
        track_id=42,
        timestamp=1.0,
        state=np.zeros(6),
        covariance=np.eye(6),
        sensors_used=SensorMask.RADAR,
    )

    frame = build_ui_frame(
        t_elapsed=5.0,
        frame_id=1,
        fps=20.0,
        mode="radar-led",
        radar_alive=True,
        wide_frame_bgr=wide_frame,
        wide_meta=wide_meta,
        narrow_frame_bgr=narrow_frame,
        narrow_meta=narrow_meta,
        last_dets=[],
        last_rois=[],
        radar_returns=[],
        ranked=ranked,
        chosen_id=42,
        ctrl_state=_make_ctrl_state(),
        state_estimates=[se],
        filter_diags={42: (0, 1.2, 0)},
        intercept_solution=None,
        ownship_ned=np.zeros(3),
        ownship_yaw_rad=0.0,
        n_lost=0,
    )

    assert frame.radar_alive is True
    assert frame.mode == "radar-led"
    assert frame.ranked[0].track.id == 42
    assert frame.chosen_id == 42
    assert frame.state_estimates[0].track_id == 42
    assert frame.filter_diags[42] == (0, 1.2, 0)
