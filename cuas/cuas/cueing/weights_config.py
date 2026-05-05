"""Cueing weights + scales, with hot-reloadable YAML loader."""
import os
from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass(frozen=True)
class CueingWeights:
    w_range: float = 1.0
    w_range_rate: float = 1.5
    w_heading: float = 1.0
    w_persistence: float = 0.5
    w_sensor_agree: float = 1.5
    w_novelty: float = 0.3
    # Visual proxy weights (active when kinematic_confidence < 1.0)
    w_bbox_size: float = 0.8
    w_bbox_growth: float = 2.0

    range_scale_m: float = 200.0
    rate_scale_mps: float = 30.0
    persistence_tau_s: float = 8.0
    novelty_tau_s: float = 5.0
    sensor_agree_window_s: float = 0.5
    # Visual proxy scales
    bbox_size_scale_frac: float = 0.04   # area fraction that maps to score 1.0
    bbox_growth_scale_rps: float = 0.5   # fractional growth rate (per second) at score 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "CueingWeights":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        kwargs = {k: float(v) for k, v in data.items() if k in valid}
        unknown = set(data.keys()) - valid
        if unknown:
            print(f"[CueingWeights] ignoring unknown keys in {path}: {sorted(unknown)}")
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


class WeightsLoader:
    """Watches a YAML file's mtime and reloads on change."""
    def __init__(self, path: str):
        self.path = path
        self._mtime: Optional[float] = None
        self.weights = CueingWeights()
        self.maybe_reload()

    def maybe_reload(self) -> bool:
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return False
        if self._mtime is not None and m == self._mtime:
            return False
        try:
            self.weights = CueingWeights.from_yaml(self.path)
        except Exception as e:
            print(f"[WeightsLoader] failed to reload {self.path}: {e}")
            return False
        self._mtime = m
        return True
