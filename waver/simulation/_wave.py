import numpy as np
from scipy.ndimage import laplace


def wave_equantion_update(U_1, U_0, c, Q_1, dt, dx, boundary='open'):
    """Calculate the updated wave using a FDTD method.

    We use a second order method that takes the current wave,
    the wave at the previous time point, the spatially varying
    speed, the forcing function at the current time point, and
    the spatial and temporal grid steps and calculates the value
    of the wave at the next time step.

    Parameters
    ----------
    U_1 : array
        Value of the wave at the current time point.
    U_0 : array
        Value of the wave at the previous time point.
    c : array
        Speed of the wave in meters per second. Must be the
        same shape as the grid.
    Q_1 : array
        Value of the forcing function on the wave equation at the
        current time point. Must be same shape as the grid.
    dt : float
        Timestep for the simulation in seconds.
    dx : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    boundary : str, optional
        Behavior of the simulation at the boundary. If `open` then
        the wave continues to propogate unimpeeded. If None, the
        wave is reflected.

    Returns
    -------
    U_2 : array
        Value of the wave at the next time point.
    """
    U_2 = 2 * U_1 - U_0 + (c * dt / dx)**2 * laplace(U_1, mode='constant')

    if boundary == 'open':
        # Enforce open boundary condition
        grad = np.gradient(U_1)
        if U_1.ndim > 1:
            grad = np.sum(grad, axis=0)

        for boundary in range(U_1.ndim):
            for edge, sign in zip((0, -1), (1, -1)):
                index = [slice(None)] * U_1.ndim
                index[boundary] = edge
                index = tuple(index)
                U_2[index] = U_1[index] + sign * (c[index] * dt / dx) * grad[index]

    # Add forcing function, after boundary condition met
    U_2 = U_2 + Q_1

    return U_2
