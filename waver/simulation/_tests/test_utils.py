import pytest

from waver.simulation._utils import location_to_index, fourier_sample, ifft_sample_1D

def test_location_to_index():
    """Test instantiating a time object."""
    index = location_to_index((10, None, 20), 0.1, (100,))

    assert index == (99,)


@pytest.mark.parametrize("shape", [(128,), (32, 32), ((16, 16, 16))])
def test_fourier_sample(shape):
    """Test sampling an nD array with a fourier method."""
    values = fourier_sample(shape)

    assert values.shape == shape


@pytest.mark.parametrize("length", [32, 16])
def test_1D_ifft_sample(length):
    """Test sampling a 1D array with an ifft method."""
    values = ifft_sample_1D(length)

    assert values.shape == (length,)
