import numpy as np
from functools import lru_cache
from typing import NamedTuple

from ._utils import sample_boundary


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
    boundary : int, optional
        If greater than zero, then number of pixels on the boundary
        to detect at, in downsampled coordinates. If zero then detection
        is done over the full grid.
    edge : int, optional
        If provided detect only at that particular "edge", which in 1D is
        a point, 2D a line, 3D a plane etc. The particular edge is determined
        by indexing around the grid. It None is provided then all edges are
        used.
    """
    shape: tuple
    spacing: tuple
    spatial_downsample: int=1
    boundary: int=0
    edge: int=None

    @property
    @lru_cache(1)
    def grid_index(self):
        """tuple of int: Location of detector grid in simulation grid."""
        return (slice(None, None, self.spatial_downsample),) * len(self.shape)

    @property
    @lru_cache(1)
    def grid_shape(self):
        """tuple of int: Shape of detector grid."""
        return tuple(int((s-1)//self.spatial_downsample) + 1 for s in self.shape)

    @property
    @lru_cache(1)
    def grid_spacing(self):
        """tuple of float: Spacing of detector grid."""
        return tuple(s * self.spatial_downsample for s in self.spacing)

    @property
    @lru_cache(1)
    def downsample_shape(self):
        """tuple of int: Shape of detector."""
        if self.boundary == 0:
            return self.grid_shape
        else:
            if self.edge is None:
                # Record number of pixels on each boundary
                boundary_shape = list(self.grid_shape)
                boundary_shape.pop(0)
                for dim in range(len(self.grid_shape)):
                    # Move through each dimension, considering all dims aside from that one
                    # which form an n-1 dimensional face
                    tmp_shape = list(self.grid_shape)
                    tmp_shape.pop(dim)
                    if boundary_shape != tmp_shape:
                        raise ValueError(f'This grid shape {self.grid_shape} does not allow for full'
                                          ' boundary detection, try detecting at a single "edge" instead.')
                return (int(2 * len(self.grid_shape) * self.boundary),) + tuple(boundary_shape)         
            else:
                dim = self.edge % len(self.grid_shape)
                # Consider all dims aside from that one which form an n-1 dimensional face
                boundary_shape = list(self.grid_shape)
                boundary_shape.pop(dim)
                # Add number of pixels on this face just once
                return (int(self.boundary),) + tuple(boundary_shape)         

    def sample(self, wave):
        """Sample wave only at boundary.

        Parameters
        ----------
        wave : array
            Wave that should be sampled

        Returns
        -------
        array
            Wave sampled at the boundary.
        """
        return sample_boundary(wave, self.boundary, self.edge)