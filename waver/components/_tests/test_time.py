from waver.components._time import Time

def test_time():
    """Test instantiating a time object."""
    time = Time(1e-3, 1e-1)

    assert time.step == 1e-3
    assert time.duration == 1e-1
    assert time.nsteps == 100
    assert len(time.values) == time.nsteps
