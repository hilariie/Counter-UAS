import pytest

from cuas.perception.detection import Detection


def _det(**kwargs):
    base = dict(x1=10.0, y1=20.0, x2=30.0, y2=60.0,
                confidence=0.9, class_id=0, class_name="uav",
                frame_id=5, timestamp=1.5)
    base.update(kwargs)
    return Detection(**base)


def test_centroid_and_size():
    d = _det()
    assert d.cx == pytest.approx(20.0)
    assert d.cy == pytest.approx(40.0)
    assert d.width == pytest.approx(20.0)
    assert d.height == pytest.approx(40.0)


def test_detection_is_frozen():
    d = _det()
    with pytest.raises(Exception):
        d.confidence = 0.5  # frozen dataclass -> FrozenInstanceError
