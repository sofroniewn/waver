import numpy as np
from scipy.fft import ifft


def sample_speed(method, grid, speed_range):
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
        Random speed distribution matched to the shape of
        the grid, sampled according to input method. 
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
    else:
        raise ValueError(f'Speed sampling method {method} not recognized')

    return speed