import numpy as np
from tqdm import tqdm
# from napari.qt import progress as tqdm

from ._utils import generate_grid_speed
from .simulation import Simulation


def run_single_source(size, spacing, location, period, duration, max_speed, time_step=None, pml_thickness=20,
                   speed=None, min_speed=0, spatial_downsample=1, temporal_downsample=1,
                   boundary=0, edge=None, ncycles=1, phase=0, progress=True, leave=False):
    """Convenience method to run a single simulation with a single source.

    Parameters
    ----------
    size : tuple of float
        Size of the grid in meters. Length of size determines the
        dimensionality of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    location : tuple of float or None
        Location of source in m. If None is passed at a certain location
        of the tuple then the source is broadcast along the full extent
        of that axis. For example a source of `(0.1, 0.2, 0.1)` is a
        point source in 3D at the point x=10cm, y=20cm, z=10cm. A source of
        `(0.1, None, 0.1)` is a line source in 3D at x=10cm, z=10cm extending
        the full length of y.
    period : float
        Period of the source in seconds.    
    duration : float
        Length of the simulation in seconds.
    max_speed : float, optional
        Maximum speed of the wave in meters per second. If passed then
        this speed will be used to derive the time step.
    time_step : float, optional
        Time step to use if stable.
    pml_thickness : int
        Thickness of any perfectly matched layer in pixels.
    speed : float, array, or str, optional
        Speed of the wave in meters per second. If a float then
        speed is assumed constant across the whole grid. If an
        array then must be the same shape as the grid. Note that
        the speed is assumed contant in time. Or string with a method for 
            generating a random speed distribution. 
    min_speed : float, optional
        Minimum allowed speed value.
    spatial_downsample : int, optional
        Spatial downsample factor.
    temporal_downsample : int, optional
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
    ncycles : int or None
        If None, source is considered to be continous, otherwise
        it will only run for ncycles.
    phase : float
        Phase offset of the source in radians.
    progress : bool, optional
        Show progress bar or not.
    leave : bool, optional
        Leave progress bar or not.

    Returns
    -------
    wave : np.ndarray
        Array of wave sampled on detector.
    speed : np.ndarray
        Array of speed values sampled on grid.
    """

    # Create a simulation
    sim = Simulation(size=size, spacing=spacing, max_speed=max_speed, time_step=time_step, pml_thickness=pml_thickness)

    if isinstance(speed, str):
        # Generate speed according to method.
        speed = generate_grid_speed(speed, sim.grid.shape, (min_speed, max_speed))

    # Set speed array
    if speed is not None:
        sim.set_speed(speed=speed, min_speed=min_speed, max_speed=max_speed)

    # Add source
    sim.add_source(location=location, period=period, ncycles=ncycles, phase=phase)

    # Add detector grid
    sim.add_detector(spatial_downsample=spatial_downsample,
                     boundary=boundary, edge=edge)

    # Run simulation
    sim.run(duration=duration, temporal_downsample=temporal_downsample, progress=progress, leave=leave)

    # Return simulation wave and speed data
    return sim.detected_wave, np.expand_dims(sim.grid_speed, axis=0)


def run_multiple_sources(size, spacing, sources, duration, max_speed, time_step=None, pml_thickness=20,
                   speed=None, min_speed=0, spatial_downsample=1, temporal_downsample=1,
                   boundary=0, edge=None, progress=True, leave=False):
    """Convenience method to run a single simulation with multiple sources.

    Parameters
    ----------
    size : tuple of float
        Size of the grid in meters. Length of size determines the
        dimensionality of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    sources : list of dict
        List of sources to use with the same grid. Each source is a
        dict of Simulation.add_source kwargs.
    duration : float
        Length of the simulation in seconds.
    max_speed : float, optional
        Maximum speed of the wave in meters per second. If passed then
        this speed will be used to derive the time step.
    time_step : float, optional
        Time step to use if stable.
    pml_thickness : int
        Thickness of any perfectly matched layer in pixels.
    speed : float, array, or str, optional
        Speed of the wave in meters per second. If a float then
        speed is assumed constant across the whole grid. If an
        array then must be the same shape as the grid. Note that
        the speed is assumed contant in time. Or string with a method for 
            generating a random speed distribution. 
    min_speed : float, optional
        Minimum allowed speed value.
    spatial_downsample : int, optional
        Spatial downsample factor.
    temporal_downsample : int, optional
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
    progress : bool, optional
        Show progress bar or not.
    leave : bool, optional
        Leave progress bar or not.

    Returns
    -------
    wave : np.ndarray
        Array of wave sampled on detector.
    speed : np.ndarray
        Array of speed values sampled on grid.
    """
    if isinstance(speed, str):
        # Create a simulation
        sim = Simulation(size=size, spacing=spacing, max_speed=max_speed, time_step=time_step, pml_thickness=pml_thickness)

        # Generate speed according to method
        speed = generate_grid_speed(speed, sim.grid.shape, (min_speed, max_speed))


    detected_waves = []

    # Move through sources
    for j, source in enumerate(tqdm(sources, leave=False)):
        wave, grid_speed = run_single_source(size=size, spacing=spacing, **source, pml_thickness=pml_thickness,
                duration=duration, max_speed=max_speed, time_step=time_step, speed=speed, min_speed=min_speed,
                spatial_downsample=spatial_downsample, temporal_downsample=temporal_downsample,
                boundary=boundary, edge=edge, progress=progress, leave=leave)
        detected_waves.append(wave)

    # Return simulation wave and speed data
    return np.stack(detected_waves, axis=0), np.expand_dims(grid_speed, axis=0)