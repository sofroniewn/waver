import numpy as np
from scipy.fft import ifft


def location_to_index(location, spacing, shape):
    """Convert a location to an index.

    Parameters
    ----------
    location: tuple of float or None
        Location of source in m. If None is passed at a certain location
        of the tuple then the source is broadcast along the full extent
        of that axis. For example a source of `(0.1, 0.2, 0.1)` is a
        point source in 3D at the point x=10cm, y=20cm, z=10cm. A source of
        `(0.1, None, 0.1)` is a line source in 3D at x=10cm, z=10cm extending
        the full length of y.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    shape : tuple of int
        Shape of the grid. This constrains the valid indices allowed.

    Returns
    -------
    tuple of int or slice
        Location of source in grid. Where source is broadcast along
        the whole axis a slice is used.
    """
    index = tuple(int(loc // spacing) if loc is not None else None for loc in location)

    # Ensure index is positive
    index = tuple(max(0, ind) if ind is not None else None for ind in index)

    # Ensure index is less than shape
    index = tuple(min(s-1, ind) if ind is not None else None for s, ind in zip(shape, index))

    index = tuple(ind if ind is not None else slice(None) for ind in index)

    return index


def sample_boundary(wave, boundary, edge=None):
    """Sample wave only at boundary.

    Parameters
    ----------
    wave : array
        Wave that should be sampled
    boundary : int
        Number of pixels at boundary that should be sampled. If zero
        then full wave is returned
    edge : int, optional
        If provided detect only at that particular "edge", which in 1D is
        a point, 2D a line, 3D a plane etc. The particular edge is determined
        by indexing around the grid. It None is provided then all edges are
        used.        

    Returns
    -------
    array
        Wave sampled at the boundary.
    """
    if boundary == 0:
        return wave

    if edge is None:
        wave_detected = []
        # Move through boundaries and try and extract each "recorded" signal
        for dim in range(wave.ndim):
            index = [slice(None)] * wave.ndim
            # Take lower and upper edges
            for edge_slice in [slice(0, boundary), slice(-boundary, wave.shape[dim])]:
                index[dim] = edge_slice
                # Extract edge, move boundary axis to beginning
                wave_at_boundary = np.moveaxis(wave[tuple(index)], dim, 0)
                wave_detected.append(wave_at_boundary)
        # Concatenate boundaries along "zero" axis
        return np.concatenate(wave_detected, axis=0)
    else:
        index = [slice(None)] * wave.ndim
        dim = edge % wave.ndim
        if edge >= wave.ndim:
            edge_slice = slice(-boundary, wave.shape[dim])
        else:
            edge_slice = slice(0, boundary)
        index[dim] = edge_slice
        # Extract edge, move boundary axis to beginning
        wave_at_boundary = np.moveaxis(wave[tuple(index)], dim, 0)
        return wave_at_boundary


subs = ['', 'i', 'i,j', 'i,j,k', 'i,j,k,l']


def generate_grid_speed(method, shape, speed_range):
    """Generate a speed distribution according to sampling method.

    Parameters
    ----------
    method : str
        Method for generating the speed distribution.
    shape : tuple
        Shape of grid that the speed distribution should be defined on.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.

    Returns
    -------
    speed : np.ndarray
        Speed values matched to the shape of the grid, and in the
        allowed range, sampled according to input method.
    """
    if method == 'flat':
        speed = speed_range[0] * np.ones(shape)
    elif method == 'random':
        speed = speed_range[0] + np.random.random(shape) * (speed_range[1] - speed_range[0])
    elif method == 'ifft':
        values = []
        for length in shape:
            values.append(ifft_sample_1D(length))
        output = np.einsum(subs[len(values)], *values)
        speed = speed_range[0] + output * (speed_range[1] - speed_range[0])
    elif method == 'mixed_random_ifft':
        if np.random.rand() > 0.5:
            speed = generate_grid_speed('random', shape, speed_range)
        else:
            speed = generate_grid_speed('ifft', shape, speed_range)
    else:
        raise ValueError(f'Speed sampling method {method} not recognized for this grid shape')

    return speed


def ifft_sample_1D(length):
    """Sample in 1D based on an ifft method.

    Parameters
    ----------
    length : int
        Length of array to be generated.

    Returns
    -------
    np.ndarray
        1D array randomly sampled with ifft method.
    """
    freq_cutoff = np.random.randint(length)
    weights = np.random.random((freq_cutoff,))
    weights = weights / np.sum(weights)
    values = np.zeros((length,))
    values[:freq_cutoff] = length * weights

    shift = np.random.randint(length)
    output = np.roll(ifft(values), shift)
    return np.clip(np.abs(output), 0, 1)
