from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cuas.cueing.tracker import RankedTrack
from cuas.tracking.controller import ControllerState
from cuas.state.types import StateEstimate
from cuas.intercept.types import InterceptSolution
from cuas.perception.frame_source import FrameMeta


@dataclass(frozen=True)
class UIFrame:
    t_elapsed: float
    frame_id: int
    fps: float
    mode: str
    radar_alive: bool
    wide_frame_bgr: np.ndarray
    wide_meta: FrameMeta
    narrow_frame_bgr: np.ndarray
    narrow_meta: FrameMeta
    last_dets: list
    last_rois: list
    radar_returns: list
    ranked: List[RankedTrack]
    chosen_id: Optional[int]
    ctrl_state: ControllerState
    state_estimates: List[StateEstimate]
    filter_diags: Dict[int, Tuple[int, float, int]]
    intercept_solution: Optional[InterceptSolution]
    ownship_ned: np.ndarray
    ownship_yaw_rad: float
    n_lost: int

    class Config:
        arbitrary_types_allowed = True


def build_ui_frame(
    *,
    t_elapsed: float,
    frame_id: int,
    fps: float,
    mode: str,
    radar_alive: bool,
    wide_frame_bgr: np.ndarray,
    wide_meta: FrameMeta,
    narrow_frame_bgr: np.ndarray,
    narrow_meta: FrameMeta,
    last_dets: list,
    last_rois: list,
    radar_returns: list,
    ranked: List[RankedTrack],
    chosen_id: Optional[int],
    ctrl_state: ControllerState,
    state_estimates: Optional[List[StateEstimate]] = None,
    filter_diags: Optional[Dict[int, Tuple[int, float, int]]] = None,
    intercept_solution: Optional[InterceptSolution] = None,
    ownship_ned: Optional[np.ndarray] = None,
    ownship_yaw_rad: float = 0.0,
    n_lost: int = 0,
) -> UIFrame:
    return UIFrame(
        t_elapsed=t_elapsed,
        frame_id=frame_id,
        fps=fps,
        mode=mode,
        radar_alive=radar_alive,
        wide_frame_bgr=wide_frame_bgr,
        wide_meta=wide_meta,
        narrow_frame_bgr=narrow_frame_bgr,
        narrow_meta=narrow_meta,
        last_dets=last_dets,
        last_rois=last_rois,
        radar_returns=radar_returns,
        ranked=ranked,
        chosen_id=chosen_id,
        ctrl_state=ctrl_state,
        state_estimates=state_estimates or [],
        filter_diags=filter_diags or {},
        intercept_solution=intercept_solution,
        ownship_ned=ownship_ned if ownship_ned is not None else np.zeros(3),
        ownship_yaw_rad=ownship_yaw_rad,
        n_lost=n_lost,
    )
