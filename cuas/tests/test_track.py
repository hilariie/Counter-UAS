from cuas.cueing.track import Track


def _t(now=10.0, **kwargs):
    base = dict(id=1, created_t=now - 2.0, last_t=now,
                az_rad=0.0, el_rad=0.0, hits=5,
                last_camera_t=now - 0.1, last_radar_t=now - 0.2)
    base.update(kwargs)
    return Track(**base)


def test_age_is_now_minus_created():
    t = _t(now=10.0, created_t=7.5)
    assert t.age_s(10.0) == 2.5


def test_age_clamped_at_zero():
    t = _t(created_t=10.0)
    assert t.age_s(5.0) == 0.0


def test_has_camera_within_window():
    t = _t(last_camera_t=9.5)
    assert t.has_camera(10.0, 0.5) is True
    assert t.has_camera(10.0, 0.4) is False


def test_has_camera_none_means_false():
    t = _t(last_camera_t=None)
    assert t.has_camera(10.0, 1.0) is False


def test_has_radar_within_window():
    t = _t(last_radar_t=9.6)
    assert t.has_radar(10.0, 0.5) is True
    assert t.has_radar(10.0, 0.3) is False


def test_has_radar_none_means_false():
    t = _t(last_radar_t=None)
    assert t.has_radar(10.0, 5.0) is False
