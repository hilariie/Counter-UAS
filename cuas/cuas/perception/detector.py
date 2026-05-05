"""YOLO + SAHI wide-FOV detector wrapper.

Tries SAHI's ultralytics backend first, falls back to yolov8 backend on
older sahi versions (same loader for ultralytics-saved weights).
"""
from pathlib import Path
from typing import List, Optional

import numpy as np

from .detection import Detection


def _load_sahi_model(weights: str, device: str, conf: float):
    from sahi import AutoDetectionModel
    last_err: Optional[Exception] = None
    for model_type in ("ultralytics", "yolov8"):
        try:
            m = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=weights,
                confidence_threshold=conf,
                device=device,
            )
            return m, model_type
        except Exception as e:  # SAHI raises ValueError or ImportError on unknown types
            last_err = e
    raise RuntimeError(f"Could not load SAHI model from {weights}: {last_err}")


class WideFovDetector:
    def __init__(self,
                 weights: str,
                 device: str = "cuda:0",
                 confidence_threshold: float = 0.25,
                 cadence_mode: str = "live",
                 use_sahi: bool = True,
                 slice_height: int = 384,
                 slice_width: int = 384,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2,
                 class_names: Optional[List[str]] = None):
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.weights = weights
        self.device = device
        self.conf = float(confidence_threshold)
        self.cadence_mode = cadence_mode
        self.use_sahi = bool(use_sahi)
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        self._sahi_model = None
        self._sahi_backend: Optional[str] = None
        self._yolo = None
        self.class_names: List[str] = class_names or []

        if self.use_sahi:
            self._sahi_model, self._sahi_backend = _load_sahi_model(weights, device, self.conf)
            inner = getattr(self._sahi_model, "model", None)
            if inner is not None and hasattr(inner, "names") and not self.class_names:
                names = inner.names
                self.class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        else:
            from ultralytics import YOLO
            self._yolo = YOLO(weights)
            if not self.class_names:
                names = self._yolo.names
                self.class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    @property
    def backend(self) -> str:
        if self.use_sahi:
            return f"sahi:{self._sahi_backend}"
        return "ultralytics"

    def _name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float,
               rois: Optional[List[tuple]] = None) -> List[Detection]:
        if rois is not None:
            return self._detect_on_cue(frame_bgr, rois, frame_id, timestamp)
        if self.use_sahi:
            return self._detect_sahi(frame_bgr, frame_id, timestamp)
        return self._detect_whole(frame_bgr, frame_id, timestamp)

    def _detect_on_cue(self, frame_bgr: np.ndarray, rois: List[tuple],
                       frame_id: int, timestamp: float) -> List[Detection]:
        out: List[Detection] = []
        for (x1, y1, x2, y2) in rois:
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            for d in self._detect_whole(crop, frame_id, timestamp):
                out.append(Detection(
                    x1=d.x1 + x1, y1=d.y1 + y1,
                    x2=d.x2 + x1, y2=d.y2 + y1,
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    frame_id=d.frame_id,
                    timestamp=d.timestamp,
                ))
        return out

    def _detect_sahi(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        from sahi.predict import get_sliced_prediction
        # SAHI expects RGB.
        rgb = frame_bgr[:, :, ::-1]
        result = get_sliced_prediction(
            rgb,
            self._sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0,
        )
        out: List[Detection] = []
        for p in result.object_prediction_list:
            bb = p.bbox
            cid = int(p.category.id)
            out.append(Detection(
                x1=float(bb.minx), y1=float(bb.miny),
                x2=float(bb.maxx), y2=float(bb.maxy),
                confidence=float(p.score.value),
                class_id=cid,
                class_name=self._name(cid),
                frame_id=frame_id,
                timestamp=timestamp,
            ))
        return out

    def _detect_whole(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        results = self._yolo.predict(
            frame_bgr, conf=self.conf, device=self.device, verbose=False
        )
        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                out.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=float(c),
                    class_id=int(k),
                    class_name=self._name(int(k)),
                    frame_id=frame_id,
                    timestamp=timestamp,
                ))
        return out
