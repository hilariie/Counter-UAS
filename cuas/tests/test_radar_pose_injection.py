"""Verify RadarMock.scan accepts a pre-fetched ownship pose so the demo loop
can avoid a duplicate simGetObjectPose RPC every frame.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from cuas.sim.radar_mock import RadarMock


def _pose(x, y, z):
    return SimpleNamespace(position=SimpleNamespace(x_val=x, y_val=y, z_val=z))


def test_scan_with_ownship_pose_skips_internal_rpc():
    np.random.seed(0)
    client = MagicMock()
    intruder_pose = _pose(100.0, 0.0, -10.0)
    client.simGetObjectPose.return_value = intruder_pose

    radar = RadarMock(client, ownship_name="Ownship")
    own_pose = _pose(0.0, 0.0, -10.0)

    returns = radar.scan(["I1"], t=0.0, ownship_pose=own_pose)
    assert len(returns) == 1

    names_called = [call.args[0] for call in client.simGetObjectPose.call_args_list]
    assert "Ownship" not in names_called
    assert names_called == ["I1"]


def test_scan_without_pose_falls_back_to_internal_rpc():
    np.random.seed(0)
    client = MagicMock()
    own_pose = _pose(0.0, 0.0, -10.0)
    intruder_pose = _pose(100.0, 0.0, -10.0)

    def fake_pose(name):
        return own_pose if name == "Ownship" else intruder_pose
    client.simGetObjectPose.side_effect = fake_pose

    radar = RadarMock(client, ownship_name="Ownship")
    returns = radar.scan(["I1"], t=0.0)
    assert len(returns) == 1

    names_called = [call.args[0] for call in client.simGetObjectPose.call_args_list]
    assert names_called == ["Ownship", "I1"]


def test_scan_accepts_bare_position_object():
    """If caller passes the .position directly (no wrapping pose), still works."""
    np.random.seed(0)
    client = MagicMock()
    intruder_pose = _pose(100.0, 0.0, -10.0)
    client.simGetObjectPose.return_value = intruder_pose

    radar = RadarMock(client, ownship_name="Ownship")
    bare_position = SimpleNamespace(x_val=0.0, y_val=0.0, z_val=-10.0)

    returns = radar.scan(["I1"], t=0.0, ownship_pose=bare_position)
    assert len(returns) == 1
    names_called = [call.args[0] for call in client.simGetObjectPose.call_args_list]
    assert "Ownship" not in names_called
