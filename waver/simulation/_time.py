from functools import lru_cache
from typing import NamedTuple

class Time(NamedTuple):
    """Time that the simulation is defined over.

    Parameters
    ----------
    step : float
        Timestep for the simulation in seconds.
    duration : float
        Length of the simulation in seconds.
    """
    step: float
    duration: float

    @property
    @lru_cache(1)
    def nsteps(self):
        """int: Number of timesteps in the simulation."""
        return int(self.duration//self.step)

    @property
    @lru_cache(1)
    def values(self):
        """tuple of float: Values of timesteps in the simulation."""
        return tuple((t * self.step) for t in range(self.nsteps))
