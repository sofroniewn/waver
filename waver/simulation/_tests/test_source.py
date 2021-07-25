import numpy as np

from waver.simulation._source import Source


def test_source():
    """Test instantiating a source object."""
    location = (None, None)
    shape = (2, 2)
    spacing = 0.1
    weight = np.ones((2, 2))
    source = Source(location=location,
                    shape=shape,
                    spacing=spacing,
                    period=0.1,
                    phase=0,
                    ncycles=None)

    assert np.all(source.weight == weight)
    assert source.period == 0.1
    assert source.phase == 0
    assert source.ncycles is None

    np.testing.assert_array_equal(source.weight, weight)
    # Test keypoints from first wave is on sine curve
    np.testing.assert_almost_equal(source.profile(0), 0)
    np.testing.assert_almost_equal(source.profile(0.025), 1)
    np.testing.assert_almost_equal(source.profile(0.05), 0)
    np.testing.assert_almost_equal(source.profile(0.1), 0)

    # Test keypoints from 11th wave is on sine curve
    np.testing.assert_almost_equal(source.profile(1), 0)
    np.testing.assert_almost_equal(source.profile(1.025), 1)
    np.testing.assert_almost_equal(source.profile(1.05), 0)
    np.testing.assert_almost_equal(source.profile(1.1), 0)

    # Test full value is correct
    np.testing.assert_almost_equal(source.value(0), 0 * weight)
    np.testing.assert_almost_equal(source.value(0.025), weight)


def test_pulsed_source():
    """Test instantiating a source object."""
    location = (None, None)
    shape = (2, 2)
    spacing = 0.1
    weight = np.ones((2, 2))
    source = Source(location=location,
                    shape=shape,
                    spacing=spacing,
                    period=0.1,
                    phase=0,
                    ncycles=5)

    assert np.all(source.weight == weight)
    assert source.period == 0.1
    assert source.phase == 0
    assert source.ncycles == 5

    # Test keypoints from first wave is on sine curve
    np.testing.assert_almost_equal(source.profile(0), 0)
    np.testing.assert_almost_equal(source.profile(0.025), 1)
    np.testing.assert_almost_equal(source.profile(0.05), 0)
    np.testing.assert_almost_equal(source.profile(0.1), 0)

    # Test keypoints from 11th wave are now zero
    np.testing.assert_almost_equal(source.profile(1), 0)
    np.testing.assert_almost_equal(source.profile(1.025), 0)
    np.testing.assert_almost_equal(source.profile(1.05), 0)
    np.testing.assert_almost_equal(source.profile(1.1), 0)


def test_spatial_source():
    """Test instantiating a source object."""
    location = (0, None)
    shape = (2, 2)
    spacing = 0.1
    weight = np.zeros((2, 2))
    weight[0] = 1
    source = Source(location=location,
                    shape=shape,
                    spacing=spacing,
                    period=0.1,
                    phase=0,
                    ncycles=None)

    assert np.all(source.weight == weight)
    assert source.period == 0.1
    assert source.phase == 0
    assert source.ncycles is None

    # Test keypoints from first wave is on sine curve
    np.testing.assert_almost_equal(source.profile(0), 0)
    np.testing.assert_almost_equal(source.profile(0.025), 1)

    # Test full value is correct
    np.testing.assert_almost_equal(source.value(0), 0 * weight)
    np.testing.assert_almost_equal(source.value(0.025), weight)