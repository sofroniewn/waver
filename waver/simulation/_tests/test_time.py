from waver.simulation._time import Time

def test_time():
    """Test instantiating a time object."""
    time = Time(step=1e-3, duration=1e-1)

    assert time.step == 1e-3
    assert time.duration == 1e-1
    assert time.nsteps == 100
    assert time.nsteps_detected == 100
    assert len(time.values) == time.nsteps


def test_time_temporal_downsample():
    """Test instantiating a time object with a temporal_downsample."""
    time = Time(step=1e-3, duration=1e-1, temporal_downsample=2)

    assert time.step == 1e-3
    assert time.duration == 1e-1
    assert time.nsteps == 100
    assert time.nsteps_detected == 50
    assert len(time.values) == time.nsteps