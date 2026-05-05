"""Make cuas/ importable when pytest is launched from anywhere."""
import os
import sys
from unittest.mock import MagicMock

# Stub the AirSim wheel so tests that import tracking/perception modules
# don't require the real cosysairsim package to be installed.
# The factory lambdas mirror what gimbal.py passes to each constructor so
# that tests inspecting the resulting Pose/Quaternionr see real float attrs.
_airsim_stub = MagicMock()
_airsim_stub.Vector3r = lambda x, y, z: MagicMock(x_val=x, y_val=y, z_val=z)
_airsim_stub.Quaternionr = lambda x_val, y_val, z_val, w_val: MagicMock(
    x_val=x_val, y_val=y_val, z_val=z_val, w_val=w_val
)
_airsim_stub.Pose = lambda pos, ori: MagicMock(position=pos, orientation=ori)
sys.modules.setdefault("cosysairsim", _airsim_stub)

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUAS_ROOT = os.path.dirname(_HERE)
if _CUAS_ROOT not in sys.path:
    sys.path.insert(0, _CUAS_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: requires Blocks simulator running (use -m sim)")
    config.addinivalue_line("markers", "gpu: requires GPU / CUDA (use -m gpu)")


def pytest_collection_modifyitems(config, items):
    # Skip sim/gpu tests unless explicitly selected via -m
    expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    for item in items:
        if "sim" in item.keywords and "sim" not in expr:
            item.add_marker(pytest.mark.skip(reason="requires Blocks simulator (use -m sim)"))
        elif "gpu" in item.keywords and "gpu" not in expr:
            item.add_marker(pytest.mark.skip(reason="requires GPU/CUDA (use -m gpu)"))
