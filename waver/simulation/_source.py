from functools import lru_cache
from typing import NamedTuple

import numpy as np

from ._utils import location_to_index


class Source(NamedTuple):
    """Source for the grid.
    
    Note that the source has a fixed spatial weight that
    varies with a certain temporal profile that is sinusoidal
    either contiously or for a fixed number of cycles.

    Parameters
    ----------
    location : tuple of float or None
        Location of source in m. If None is passed at a certain location
        of the tuple then the source is broadcast along the full extent
        of that axis. For example a source of `(0.1, 0.2, 0.1)` is a
        point source in 3D at the point x=10cm, y=20cm, z=10cm. A source of
        `(0.1, None, 0.1)` is a line source in 3D at x=10cm, z=10cm extending
        the full length of y.
    shape : tuple of int
        Shape of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    period : float
        Period of the source in seconds.
    ncycles : int or None
        If None, source is considered to be continous, otherwise
        it will only run for ncycles.
    phase : float
        Phase offset of the source in radians.
    """
    location: tuple
    shape: tuple
    spacing: tuple
    period: float
    ncycles: int
    phase: float

    @property
    @lru_cache(1)
    def index(self):
        """tuple of int: Location of source in grid."""
        return location_to_index(self.location, self.spacing, self.shape)

    @property
    @lru_cache(1)
    def weight(self):
        """array: Spatial weights of the source on the grid."""
        weight = np.zeros(self.shape)
        weight[self.index] = 1
        return weight

    def profile(self, time):
        """Get temporal profile of a certain time.
        
        Parameters
        ----------
        time : float
            Time in seconds through simulation,

        Returns
        -------
        float
            Value of the source at that moment in time.
        """
        if self.ncycles is None or time / self.period <= self.ncycles:
            return np.sin(2 * np.pi * time / self.period + self.phase)
        else:
            return 0

    def value(self, time):
        """Get value of the source on grid at a certain time.
        
        Parameters
        ----------
        time : float
            Time in seconds through simulation.

        Returns
        -------
        array
            Value of the source at that moment in time over the
            whole grid.
        """
        return self.weight * self.profile(time)