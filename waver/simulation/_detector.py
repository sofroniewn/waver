import numpy as np
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
    spatial_downsample: int
    temporal_downsample: int
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
                n_boundary_pixels = 0
                for dim in range(len(self.grid_shape)):
                    # Move through each dimension, considering all dims aside from that one
                    # which form an n-1 dimensional face
                    tmp_shape = list(self.grid_shape)
                    tmp_shape.pop(dim)
                    # Add number of pixels on this face twice, once for each side.
                    n_boundary_pixels += 2 * np.product(tmp_shape)
                return (int(self.boundary * n_boundary_pixels),)
            else:
                dim = self.edge % len(self.grid_shape)
                # Consider all dims aside from that one which form an n-1 dimensional face
                tmp_shape = list(self.grid_shape)
                tmp_shape.pop(dim)
                # Add number of pixels on this face just once
                n_boundary_pixels = np.product(tmp_shape)
                return (int(self.boundary * n_boundary_pixels),)            
