import numpy as np
from scipy.ndimage import laplace


def wave_equantion_update(U_1, U_0, c, Q_1, dt, dx, boundary):
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
    boundary : int
        Thickness of a perfectly matched layer at the boundary.

    Returns
    -------
    U_2 : array
        Value of the wave at the next time point.
    """

    U_2 = 2 * U_1 - U_0 + (c * dt / dx)**2 * laplace(U_1, mode='constant')


    # Attempt a perfectly matched layer correction
    grad = np.gradient(U_1)
    # If 1D then an array is returned, convert into list
    if U_1.ndim == 1:
        grad = [grad]

    for dim in range(U_1.ndim):
        slice_indices = (slice(0, boundary), slice(-boundary, U_1.shape[dim]))
        shape_boundary = list(U_1.shape)
        shape_boundary[dim] = boundary
        ones_shape = [1] * U_1.ndim
        ones_shape[dim] = boundary
        for edge, sign in zip(slice_indices, (1, -1)):
            index = [slice(None)] * U_1.ndim
            index[dim] = edge
            index = tuple(index)
            if sign == -1:
                attenuation_ramp = np.linspace(0, 1, boundary)
            else:
                attenuation_ramp = np.linspace(1, 0, boundary)
            attenuation = np.reshape(attenuation_ramp, ones_shape)
            attenuation = np.broadcast_to(attenuation, shape_boundary)
            U_2[index] = U_2[index] - sign * (c[index] * dt / dx) * grad[dim][index] * attenuation

    # Add forcing function, after boundary condition met
    U_2 = U_2 + Q_1

    return U_2
