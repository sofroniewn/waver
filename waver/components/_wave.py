from scipy.ndimage import laplace


def wave_equantion_update(U_1, U_0, c, Q_1, dt, dx):
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
    c : float or array
        Speed of the wave in meters per second. If a float then
        speed is assumed constant across the whole grid. If an
        array then must be the same shape as the grid.
    Q_1 : array
        Value of the forcing function on the wave equation at the
        current time point. Must be same shape as the grid.
    dt : float
        Timestep for the simulation in seconds.
    dx : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.

    Returns
    -------
    U_2 : array
        Value of the wave at the next time point.
    """
    return 2 * U_1 - U_0 + (c * dt / dx)**2 * laplace(U_1) + dt**2 * Q_1
