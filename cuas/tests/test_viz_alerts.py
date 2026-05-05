import math
import pytest

from cuas.viz.alerts import RadarFailAlert


def test_radar_fail_alert_edges():
    alert = RadarFailAlert(screen_size=(1280, 720))

    # Initially all quiet
    alert.update(radar_alive=True, t_now=0.0)
    assert alert._active is False
    assert alert._fade_t == 0.0

    # Failure edge: radar goes down
    alert.update(radar_alive=False, t_now=0.1)
    assert alert._active is True
    assert alert._failure_t == pytest.approx(0.1)

    # Fade ramps up over FADE_IN_S
    alert.update(radar_alive=False, t_now=0.1 + RadarFailAlert.FADE_IN_S)
    assert alert._fade_t == pytest.approx(1.0)

    # Recovery edge
    alert.update(radar_alive=True, t_now=1.0)
    assert alert._active is False
    assert alert._failure_t is None

    # Fade ramps down
    alert.update(radar_alive=True, t_now=1.0 + RadarFailAlert.FADE_OUT_S)
    assert alert._fade_t == pytest.approx(0.0)


def test_pulse_strictly_positive():
    alert = RadarFailAlert()
    alert._active = True
    alert._fade_t = 1.0

    for i in range(200):
        alert._t_now = i * 0.05
        pulse = alert._pulse()
        assert pulse >= 0.55, f"pulse={pulse} at t={alert._t_now:.2f}"
        assert pulse <= 1.0,  f"pulse={pulse} at t={alert._t_now:.2f}"
