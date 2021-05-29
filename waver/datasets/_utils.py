import numpy as np


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
    return speed_range[0] * np.ones(grid.shape)