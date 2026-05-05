"""Smoke tests for NarrowDetTracker. Mocks ultralytics.YOLO; no GPU required."""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Avoid importing the real ultralytics at module-import time.
sys.modules.setdefault("cosysairsim", MagicMock())


class _FakeBoxes:
    def __init__(self, xyxy, conf, ids=None):
        self.xyxy = MagicMock()
        self.xyxy.cpu.return_value.numpy.return_value = np.asarray(xyxy, dtype=float)
        self.conf = MagicMock()
        self.conf.cpu.return_value.numpy.return_value = np.asarray(conf, dtype=float)
        if ids is not None:
            self.id = MagicMock()
            self.id.cpu.return_value.numpy.return_value = np.asarray(ids)
        else:
            self.id = None

    def __len__(self):
        return len(self.conf.cpu.return_value.numpy.return_value)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _patched_yolo_class(track_results):
    instance = MagicMock()
    instance.track.return_value = track_results
    klass = MagicMock(return_value=instance)
    return klass, instance


def test_step_returns_highest_conf_box(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    boxes = _FakeBoxes(
        xyxy=[[100, 100, 140, 140], [300, 220, 340, 260]],
        conf=[0.55, 0.92],
        ids=[7, 11],
    )
    klass, instance = _patched_yolo_class([_FakeResult(boxes)])
    with patch("ultralytics.YOLO", klass):
        from cuas.tracking.narrow_detector import NarrowDetTracker
        det = NarrowDetTracker(weights=str(weights), device="cpu", conf=0.25)
        out = det.step(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out is not None
    assert out.confidence == pytest.approx(0.92)
    assert out.track_id == 11
    assert out.bbox_xywh == (300, 220, 40, 40)


def test_step_returns_none_on_empty(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    boxes = _FakeBoxes(xyxy=np.zeros((0, 4)), conf=np.zeros((0,)))
    klass, _ = _patched_yolo_class([_FakeResult(boxes)])
    with patch("ultralytics.YOLO", klass):
        from cuas.tracking.narrow_detector import NarrowDetTracker
        det = NarrowDetTracker(weights=str(weights), device="cpu")
        out = det.step(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out is None


def test_missing_weights_raises(tmp_path):
    from cuas.tracking.narrow_detector import NarrowDetTracker
    with pytest.raises(FileNotFoundError):
        NarrowDetTracker(weights=str(tmp_path / "nope.pt"), device="cpu")
