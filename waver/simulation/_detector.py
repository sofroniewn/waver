from functools import lru_cache
from typing import NamedTuple


class Detector(NamedTuple):
    """Detector for the grid.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    spatial_downsample : int
        Spatial downsample factor.
    temporal_downsample : int
        Temporal downsample factor.
    """
    shape: tuple
    spacing: tuple
    spatial_downsample: int
    temporal_downsample: int

    @property
    @lru_cache(1)
    def downsample_index(self):
        """tuple of int: Location of detector in simulation grid."""
        return (slice(None, None, self.spatial_downsample),) * len(self.shape)

    @property
    @lru_cache(1)
    def downsample_shape(self):
        """tuple of int: Shape of detector grid."""
        return tuple(int((s-1)//self.spatial_downsample) + 1 for s in self.shape)

    @property
    @lru_cache(1)
    def downsample_spacing(self):
        """tuple of int: Shape of detector grid."""
        return tuple(s * self.spatial_downsample for s in self.spacing)
