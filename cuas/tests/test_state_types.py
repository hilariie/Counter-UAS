import numpy as np
import pytest

from cuas.state.types import SensorMask, StateEstimate


def _make() -> StateEstimate:
    return StateEstimate(
        track_id=1,
        timestamp=0.0,
        state=np.arange(6, dtype=float),
        covariance=np.eye(6),
    )


def test_state_shape():
    se = _make()
    assert se.state.shape == (6,)
    assert se.covariance.shape == (6, 6)


def test_position_velocity_slices():
    se = _make()
    np.testing.assert_array_equal(se.position_ned, se.state[:3])
    np.testing.assert_array_equal(se.velocity_ned, se.state[3:])


def test_sensor_mask_none_falsy():
    assert not SensorMask.NONE


def test_sensor_mask_or_truthy():
    mask = SensorMask.RADAR | SensorMask.BEARING
    assert mask
    assert SensorMask.RADAR in mask
    assert SensorMask.BEARING in mask
    assert SensorMask.SFM not in mask


def test_range_m_true_defaults_none():
    assert _make().range_m_true is None


def test_sensors_used_defaults_none():
    assert _make().sensors_used == SensorMask.NONE
