"""Module 3 coordinator: narrow-FOV track maintenance state machine.

States: IDLE → WARMUP → ACQUIRING → LOCKED → COMMIT → ACQUIRING (cycle).

WARMUP holds the gimbal still on script start so radar can mature.
COMMIT keeps the gimbal on the YOLO-confirmed target for a fixed dwell
even if cueing's chosen_target.id changes — simulating Modules 4 & 5
working an intercept solution against the locked threat.
"""
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from cuas.cueing.geometry import CameraIntrinsics
from cuas.cueing.track import Track
from cuas.perception.frame_source import FrameMeta
from cuas.tracking.gimbal import GimbalCommand, GimbalController
from cuas.tracking.narrow_detector import NarrowDet, NarrowDetTracker


class _State(Enum):
    IDLE = "IDLE"
    WARMUP = "WARMUP"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    COMMIT = "COMMIT"


@dataclass
class ControllerState:
    det: Optional[NarrowDet]
    gimbal_cmd: GimbalCommand
    target_id: Optional[int]
    lost: bool
    state_name: str
    state_remaining_s: float = 0.0


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _now() -> float:
    return time.time()


class NarrowTrackingController:
    def __init__(self, client, intrinsics: CameraIntrinsics,
                 ownship_name: str = "Ownship",
                 narrow_weights: Optional[str] = None,
                 narrow_device: str = "cuda:0",
                 narrow_conf: float = 0.25,
                 narrow_imgsz: int = 640,
                 warmup_s: float = 5.0,
                 commit_s: float = 10.0,
                 confirm_frames: int = 3,
                 loss_grace_s: float = 1.5,
                 bearing_drift_limit_rad: float = math.radians(8),
                 lookahead_s: float = 0.2,
                 preempt_score_threshold: float = 10.0,
                 _detector: Optional[NarrowDetTracker] = None,
                 _gimbal: Optional[GimbalController] = None):
        self._intr = intrinsics
        self._warmup_s = warmup_s
        self._commit_s = commit_s
        self._confirm_frames = confirm_frames
        self._loss_grace_s = loss_grace_s
        self._drift_limit = bearing_drift_limit_rad
        self._lookahead_s = lookahead_s
        self._preempt_threshold = preempt_score_threshold

        if _detector is not None:
            self._det = _detector
        else:
            if narrow_weights is None:
                raise ValueError("narrow_weights must be provided when _detector is not injected")
            self._det = NarrowDetTracker(
                weights=narrow_weights, device=narrow_device,
                conf=narrow_conf, imgsz=narrow_imgsz,
            )
        self._gimbal = _gimbal or GimbalController(client, vehicle_name=ownship_name)

        self._state = _State.IDLE
        self._target_id: Optional[int] = None
        self._consecutive_dets = 0
        self._last_det_t: Optional[float] = None
        self._commit_started_t: Optional[float] = None
        self._warmup_started_t: Optional[float] = None
        self._first_step = True
        self._last_cmd = GimbalCommand(0.0, 0.0, _now())

        self._prev_bearing: Optional[Tuple[float, float]] = None
        self._prev_bearing_t: Optional[float] = None
        self._bearing_rate: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ helpers

    def _predicted_bearing(self, track: Track) -> Tuple[float, float]:
        az, el = track.az_rad, track.el_rad
        if self._bearing_rate is not None and self._lookahead_s > 0.0:
            az = _wrap_pi(az + self._bearing_rate[0] * self._lookahead_s)
            el = el + self._bearing_rate[1] * self._lookahead_s
        return az, el

    def _update_bearing_rate(self, track: Track) -> None:
        if self._prev_bearing is not None and self._prev_bearing_t is not None:
            dt = track.last_t - self._prev_bearing_t
            if 0.0 < dt < 1.0:
                d_az = _wrap_pi(track.az_rad - self._prev_bearing[0])
                d_el = track.el_rad - self._prev_bearing[1]
                az_dot, el_dot = d_az / dt, d_el / dt
                if self._bearing_rate is None:
                    self._bearing_rate = (az_dot, el_dot)
                else:
                    a = 0.4
                    self._bearing_rate = (
                        a * az_dot + (1.0 - a) * self._bearing_rate[0],
                        a * el_dot + (1.0 - a) * self._bearing_rate[1],
                    )
        self._prev_bearing = (track.az_rad, track.el_rad)
        self._prev_bearing_t = track.last_t

    def _is_new_physical_target(self, track: Track) -> bool:
        if track.id == self._target_id:
            return False
        if self._target_id is None or self._prev_bearing is None or self._state == _State.IDLE:
            return True
        last_az, last_el = self._prev_bearing
        return (abs(_wrap_pi(track.az_rad - last_az)) > self._drift_limit
                or abs(track.el_rad - last_el) > self._drift_limit)

    def _slew_predicted(self, track: Track, ownship_yaw_rad: float) -> GimbalCommand:
        pred_az, pred_el = self._predicted_bearing(track)
        return self._gimbal.slew_to_world_bearing(pred_az, pred_el, ownship_yaw_rad)

    def _reset_target_state(self) -> None:
        self._consecutive_dets = 0
        self._last_det_t = None
        self._commit_started_t = None
        self._prev_bearing = None
        self._prev_bearing_t = None
        self._bearing_rate = None
        self._det.reset()

    def _state_remaining_s(self, now: float) -> float:
        if self._state == _State.WARMUP and self._warmup_started_t is not None:
            return max(0.0, self._warmup_s - (now - self._warmup_started_t))
        if self._state == _State.COMMIT and self._commit_started_t is not None:
            return max(0.0, self._commit_s - (now - self._commit_started_t))
        return 0.0

    # ------------------------------------------------------------------ step

    def step(self, chosen_target: Optional[Track], ownship_yaw_rad: float,
             frame: np.ndarray, frame_meta: FrameMeta,
             target_score: float = 0.0) -> ControllerState:
        now = _now()

        # First call ever → enter WARMUP regardless of cueing state.
        if self._first_step:
            self._first_step = False
            self._state = _State.WARMUP
            self._warmup_started_t = now

        # WARMUP: hold gimbal pose, ignore everything until elapsed.
        if self._state == _State.WARMUP:
            if now - self._warmup_started_t >= self._warmup_s:
                self._state = _State.IDLE
                self._warmup_started_t = None
            else:
                return ControllerState(
                    det=None, gimbal_cmd=self._last_cmd,
                    target_id=None, lost=False,
                    state_name=_State.WARMUP.value,
                    state_remaining_s=self._state_remaining_s(now),
                )

        # COMMIT: servo on YOLO output; ignore minor cueing switches unless they exceed threshold.
        if self._state == _State.COMMIT:
            # Check for high-score preemption
            should_preempt = (
                chosen_target is not None
                and chosen_target.id != self._target_id
                and target_score >= self._preempt_threshold
            )

            if should_preempt or (now - self._commit_started_t >= self._commit_s):
                self._state = _State.ACQUIRING
                self._commit_started_t = None
                # fall through to ACQUIRING logic below
            else:
                det = self._det.step(frame)
                lost = False
                if det is not None:
                    self._last_det_t = now
                    cx = det.bbox_xywh[0] + det.bbox_xywh[2] / 2.0
                    cy = det.bbox_xywh[1] + det.bbox_xywh[3] / 2.0
                    dx = cx - self._intr.cx
                    dy = cy - self._intr.cy
                    self._last_cmd = self._gimbal.nudge_pixel(dx, dy, self._intr, ownship_yaw_rad)
                else:
                    if (self._last_det_t is not None
                            and now - self._last_det_t > self._loss_grace_s):
                        # detector lost the drone for too long; bounce back to ACQUIRING
                        self._state = _State.ACQUIRING
                        self._commit_started_t = None
                        self._consecutive_dets = 0
                        lost = True
                if self._state == _State.COMMIT:
                    return ControllerState(
                        det=det, gimbal_cmd=self._last_cmd,
                        target_id=self._target_id, lost=lost,
                        state_name=_State.COMMIT.value,
                        state_remaining_s=self._state_remaining_s(now),
                    )

        # Outside COMMIT, cueing decides the target.
        if chosen_target is None:
            self._state = _State.IDLE
            self._target_id = None
            self._reset_target_state()
            return ControllerState(
                det=None, gimbal_cmd=self._last_cmd,
                target_id=None, lost=False,
                state_name=_State.IDLE.value,
            )

        new_phys = self._is_new_physical_target(chosen_target)
        if new_phys:
            self._reset_target_state()
        self._update_bearing_rate(chosen_target)
        self._target_id = chosen_target.id

        if self._state in (_State.IDLE, _State.LOCKED):
            self._state = _State.ACQUIRING
            self._consecutive_dets = 0
        if new_phys and self._state != _State.ACQUIRING:
            self._state = _State.ACQUIRING
            self._consecutive_dets = 0

        # ACQUIRING: slew to predicted bearing, run detector, count confirmations.
        if self._state == _State.ACQUIRING:
            self._last_cmd = self._slew_predicted(chosen_target, ownship_yaw_rad)
            det = self._det.step(frame)
            lost = False
            if det is not None:
                self._consecutive_dets += 1
                self._last_det_t = now
                if self._consecutive_dets >= self._confirm_frames:
                    self._state = _State.COMMIT
                    self._commit_started_t = now
                    return ControllerState(
                        det=det, gimbal_cmd=self._last_cmd,
                        target_id=self._target_id, lost=False,
                        state_name=_State.COMMIT.value,
                        state_remaining_s=self._state_remaining_s(now),
                    )
            else:
                self._consecutive_dets = 0
            return ControllerState(
                det=det, gimbal_cmd=self._last_cmd,
                target_id=self._target_id, lost=lost,
                state_name=_State.ACQUIRING.value,
            )

        # Should not be reached.
        return ControllerState(
            det=None, gimbal_cmd=self._last_cmd,
            target_id=self._target_id, lost=False,
            state_name=self._state.value,
        )
