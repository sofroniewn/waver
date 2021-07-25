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
    temporal_downsample : int, optional
        Temporal downsample factor.
    """
    step: float
    duration: float
    temporal_downsample: int=1

    @property
    @lru_cache(1)
    def nsteps(self):
        """int: Number of timesteps in the simulation."""
        return int(self.duration / self.step)

    @property
    @lru_cache(1)
    def nsteps_detected(self):
        """int: Number of detected timesteps."""
        return int((self.nsteps - 1) // self.temporal_downsample + 1)

    @property
    @lru_cache(1)
    def values(self):
        """tuple of float: Values of timesteps in the simulation."""
        return tuple((t * self.step) for t in range(self.nsteps))
