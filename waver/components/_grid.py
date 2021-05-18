from functools import lru_cache
from typing import NamedTuple, Tuple

class Grid(NamedTuple):
    """Grid that the simulation is defined on.
    
    Parameters
    ----------
    size : tuple of float
        Size of the grid in meters. Length of size determines the
        dimensionality of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    speed : float or array
        Speed of the wave in meters per second. If a float then
        speed is assumed constant across the whole grid. If an
        array then must be the same shape as the shape of the grid.
    """
    size: Tuple[float, ...]
    spacing: float
    speed: float

    @property
    @lru_cache(1)
    def ndim(self):
        """int: Dimensionality of the grid."""
        return len(self.size)

    @property
    @lru_cache(1)
    def shape(self):
        """tuple of int: Shape of the grid."""
        return tuple(int(_size//self.spacing) for _size in self.size)
