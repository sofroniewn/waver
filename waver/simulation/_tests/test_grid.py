import pytest

from waver.simulation._grid import Grid

@pytest.mark.parametrize('size', [(.1, .1, .1), (.1, .1), (.1,)])
def test_grid(size):
    """Test instantiating the grid."""
    grid = Grid(size=size, spacing=100e-6)

    assert grid.size == size
    assert grid.ndim == len(size)
    assert len(grid.shape) == grid.ndim
    assert grid.shape == (1000, ) * grid.ndim
