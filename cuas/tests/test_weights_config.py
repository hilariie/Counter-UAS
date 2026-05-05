import os
import time

import pytest
import yaml

from cuas.cueing.weights_config import CueingWeights, WeightsLoader


def test_defaults_match_design():
    w = CueingWeights()
    # Defaults match Module 2 v1.1 retune (scene-scaled: 14-220 m intruders,
    # 60+ s-old tracks). Scales reduced from v1.0 so terms discriminate
    # across the actual range distribution instead of saturating.
    assert w.w_range == 1.0
    assert w.w_range_rate == 1.5
    assert w.w_heading == 1.0
    assert w.w_persistence == 0.5
    assert w.w_sensor_agree == 1.5
    assert w.w_novelty == 0.3
    assert w.range_scale_m == 200.0
    assert w.rate_scale_mps == 30.0
    assert w.persistence_tau_s == 8.0


def test_from_yaml_loads_overrides(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump({"w_range": 2.5, "range_scale_m": 1000.0}))
    w = CueingWeights.from_yaml(str(p))
    assert w.w_range == 2.5
    assert w.range_scale_m == 1000.0
    # Untouched fields keep defaults.
    assert w.w_range_rate == 1.5


def test_from_yaml_ignores_unknown_keys(tmp_path, capsys):
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump({"w_range": 2.0, "bogus_key": 99}))
    w = CueingWeights.from_yaml(str(p))
    assert w.w_range == 2.0
    captured = capsys.readouterr()
    assert "bogus_key" in captured.out  # warning printed


def test_to_yaml_roundtrip(tmp_path):
    p = tmp_path / "w.yaml"
    original = CueingWeights(w_range=3.3, persistence_tau_s=7.0)
    original.to_yaml(str(p))
    reloaded = CueingWeights.from_yaml(str(p))
    assert reloaded == original


def test_weights_loader_initial_load(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump({"w_range": 4.2}))
    loader = WeightsLoader(str(p))
    assert loader.weights.w_range == 4.2


def test_weights_loader_detects_mtime_change(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump({"w_range": 1.0}))
    loader = WeightsLoader(str(p))
    assert loader.weights.w_range == 1.0
    assert loader.maybe_reload() is False
    # Bump mtime explicitly (file write may land in the same OS tick).
    new_t = os.path.getmtime(p) + 5.0
    p.write_text(yaml.safe_dump({"w_range": 9.9}))
    os.utime(p, (new_t, new_t))
    assert loader.maybe_reload() is True
    assert loader.weights.w_range == 9.9


def test_weights_loader_handles_missing_file(tmp_path):
    p = tmp_path / "missing.yaml"
    loader = WeightsLoader(str(p))
    # Falls back to defaults; reload returns False on missing file.
    assert loader.weights == CueingWeights()
    assert loader.maybe_reload() is False
