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


def sample_boundary(wave, boundary):
    """Sample wave only at boundary.

    Parameters
    ----------
    wave: array
        Wave that should be sampled
    boundary: int
        Number of pixels at boundary that should be sampled. If zero
        then full wave is returned

    Returns
    -------
    array
        Wave sampled at the boundary.
    """
    if boundary == 0:
        return wave

    wave_detected = []
    # Move through boundaries and try and extract each "recorded" signal
    for dim in range(wave.ndim):
        index = [slice(None)] * wave.ndim
        # Take lower and upper edges
        for edge in [slice(0, boundary), slice(-boundary, wave.shape[dim])]:
            index[dim] = edge
            # Extract edge, move boundary axis to end and flatten
            wave_at_boundary = np.moveaxis(wave[tuple(index)], dim, -1).flatten()
            wave_detected.append(wave_at_boundary)

    return np.concatenate(wave_detected)


def generate_speed_array(method, grid, speed_range):
    """Generate a speed distribution according to sampling method.

    Parameters
    ----------
    method : str
        Method for generating the speed distribution.
    grid : waver.components.Grid
        Grid that the speed distribution should be defined on.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.

    Returns
    -------
    speed : np.ndarray
        Speed values matched to the shape of the grid, and in the
        allowed range, sampled according to input method.
    """
    shape = grid.shape
    if method == 'flat':
        speed = speed_range[0] * np.ones(shape)
    elif method == 'random':
        speed = speed_range[0] + np.random.random(shape) * (speed_range[1] - speed_range[0])
    elif method == 'ifft' and len(shape) == 1:
        shape = shape[0]
        freq_cutoff = np.random.randint(shape)
        weights = np.random.random((freq_cutoff,))
        weights = weights / np.sum(weights)
        values = np.zeros((shape,))
        values[:freq_cutoff] = shape * weights

        shift = np.random.randint(shape)
        output = np.roll(ifft(values), shift)
        output = np.clip(np.abs(output), 0, 1)
        speed = speed_range[0] + output * (speed_range[1] - speed_range[0])
    elif method == 'mixed_random_ifft':
        if np.random.rand() > 0.5:
            speed = generate_speed_array('random', grid, speed_range)
        else:
            speed = generate_speed_array('ifft', grid, speed_range)
    else:
        raise ValueError(f'Speed sampling method {method} not recognized for this grid shape')

    return speed