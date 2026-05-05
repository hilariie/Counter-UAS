"""YOLO + ByteTrack narrow-FOV detection-tracker.

Replaces the previous OpenCV SOT. Runs Ultralytics YOLO every step and
relies on the built-in ByteTrack association for stable IDs.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class NarrowDet:
    bbox_xywh: Tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int]
    frame_id: int


class NarrowDetTracker:
    def __init__(self,
                 weights: str,
                 device: str = "cuda:0",
                 conf: float = 0.25,
                 imgsz: int = 640,
                 tracker_yaml: str = "bytetrack.yaml"):
        if not Path(weights).exists():
            raise FileNotFoundError(f"Narrow YOLO weights not found: {weights}")
        self.weights = weights
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.tracker_yaml = tracker_yaml
        self._frame_id = -1

        from ultralytics import YOLO
        try:
            self._model = YOLO(weights, task="detect")
        except Exception:
            # TRT engine load can fail (cuda/TRT mismatch); fall back to .pt sibling.
            pt = str(Path(weights).with_suffix(".pt"))
            if not Path(pt).exists():
                raise
            self._model = YOLO(pt, task="detect")
            self.weights = pt

    def reset(self) -> None:
        # Ultralytics persists tracker state across track() calls for the same
        # model instance; reinstantiate to clear it.
        from ultralytics import YOLO
        self._model = YOLO(self.weights, task="detect")

    def step(self, frame_bgr: np.ndarray) -> Optional[NarrowDet]:
        self._frame_id += 1
        results = self._model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None

        h, w = frame_bgr.shape[:2]
        cx_img, cy_img = w / 2.0, h / 2.0

        # Pick highest-conf box; tie-break by proximity to centre.
        best_i = self._select(xyxy, confs, cx_img, cy_img)
        x1, y1, x2, y2 = xyxy[best_i]
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        tid = int(ids[best_i]) if ids is not None else None
        return NarrowDet(
            bbox_xywh=bbox,
            confidence=float(confs[best_i]),
            track_id=tid,
            frame_id=self._frame_id,
        )

    @staticmethod
    def _select(xyxy: np.ndarray, confs: np.ndarray, cx: float, cy: float) -> int:
        # Prefer top confidence; if multiple within 90% of max, pick closest to centre.
        thresh = 0.9 * confs.max()
        candidates = np.where(confs >= thresh)[0]
        if len(candidates) == 1:
            return int(candidates[0])
        ctrs_x = 0.5 * (xyxy[candidates, 0] + xyxy[candidates, 2])
        ctrs_y = 0.5 * (xyxy[candidates, 1] + xyxy[candidates, 3])
        d2 = (ctrs_x - cx) ** 2 + (ctrs_y - cy) ** 2
        return int(candidates[int(np.argmin(d2))])
