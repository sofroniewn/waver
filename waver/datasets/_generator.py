import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm

from ..simulation import Simulation
from ._utils import sample_speed


def generate_simulation_datasets(*, path, splits, runs, seed=0, size, spacing,
                                 duration, speed, speed_range, sources, reduced=False, boundary=2):
    """Generate simulation datasets.

    Parameters
    ----------
    path : str
        Root path where simulation data will be stored.
    splits : list of str
        List of splits to bucket simulations into. Most commonly
        `('train', 'test')`.
    runs : list of int
        Number of different speed distributions. Must be same length
        as number of splits.
    seed : int, optional
        Seed to initialize random number generator with.
    size : tuple of float
        Size of the grid in meters. Length of size determines the
        dimensionality of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    duration : float
        Length of the simulation in seconds.
    speed : str, array, list
        String describing sampling method for generating
        speed distributions. If array then must have dims one more 
        than spatial dims of grid, and each element along the first
        dimension much have same shape as grid. If list then must be
        same length as splits.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.
    sources : list of dict
        List of source parameters. The simulation will be rerun
        with each set of source parameters.
    reduced : bool
        If saving data in reduced format where detection has
        already occured along the boundary.
    boundary : int
        Thickness of boundary where detection is happening in
        pixels.

    Returns
    -------
    dataset : list of zarr.hierarchy.Group
        List of simulation datasets.
    """
    # Initialize seed
    np.random.seed(seed)

    # Create datasets folder
    path = Path(path)
    path.mkdir(exist_ok=True)

    # Move through splits
    datasets = []
    for i, split in enumerate(tqdm(splits)):
        # Create dataset
        split_path = path / f'{split}.zarr'
        
        # Split speed.
        if isinstance(speed, list):
            split_speed = speed[i]
        else:
            split_speed = speed

        dataset = generate_simulation_dataset(
                                              path=split_path,
                                              runs=runs[i],
                                              size=size,
                                              spacing=spacing,
                                              duration=duration,
                                              speed=split_speed,
                                              speed_range=speed_range,
                                              sources=sources,
                                              reduced=reduced,
                                              boundary=boundary,
                                             )
        datasets.append(dataset)
    
    return datasets


def generate_simulation_dataset(*, path, runs, size, spacing, duration, speed, speed_range, sources, reduced=False, boundary=2):
    """Generate simulation datasets.

    Parameters
    ----------
    path : str or Pathlib.Path
        Root path where simulation data will be stored.
    runs : int
        Number of different speed distributions.
    size : tuple of float
        Size of the grid in meters. Length of size determines the
        dimensionality of the grid.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    duration : float
        Length of the simulation in seconds.
    speed : str, array
        String describing sampling method for generating
        speed distributions. If array then must have dims one more 
        than spatial dims of grid, and each element along the first
        dimension much have same shape as grid.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.
    sources : list of dict
        List of source parameters. The simulation will be rerun
        with each set of source parameters.
    reduced : bool
        If saving data in reduced format where detection has
        already occured along the boundary.
    boundary : int
        Thickness of boundary where detection is happening in
        pixels.

    Returns
    -------
    dataset : zarr.hierarchy.Group
        Simulation dataset.
    """
    # Convert path to pathlib path
    path = Path(path)

    # Create base simulation
    base_simulation = Simulation(size=size, spacing=spacing, duration=duration, speed=speed_range[1], max_speed=speed_range[1])
    grid_shape = base_simulation.grid.shape
    time_nsteps = base_simulation.time.nsteps
    time_step = base_simulation.time.step

    # Handle speed processing
    if isinstance(speed, str):
        speed_method = speed
    else:
        speed_method = None
        runs = len(speed)


    # Create dataset
    dataset = zarr.open(path.as_posix(), mode='w')

    # Add dataset attributes
    dataset.attrs['waver'] = True
    dataset.attrs['dataset'] = True
    dataset.attrs['runs'] = runs

    # Simulation attributes
    dataset.attrs['size'] = size
    dataset.attrs['spacing'] = spacing
    dataset.attrs['grid_shape'] = grid_shape
    dataset.attrs['duration'] = duration
    dataset.attrs['time_step'] = time_step
    dataset.attrs['time_nsteps'] = time_nsteps
    dataset.attrs['speed'] = speed_method
    dataset.attrs['speed_range'] = speed_range
    dataset.attrs['sources'] = sources
    dataset.attrs['reduced'] = reduced
    dataset.attrs['boundary'] = boundary

 
    # Generate array containers
    if reduced:
        speed_shape = (runs, ) + tuple(grid_shape)
        n_boundary = 0
        if len(grid_shape) == 1:
            n_boundary = 2
        else:
            for dim in range(len(grid_shape)):
                tmp_shape = list(grid_shape)
                tmp_shape.pop(dim)
                n_boundary += 2 * np.product(tmp_shape)
        n_detected = len(sources) * boundary * n_boundary
        wave_shape = (runs, n_detected, time_nsteps)
        speed_array = dataset.zeros('speed', shape=speed_shape, chunks=(1,) + (None,) * len(grid_shape))
        wave_array = dataset.zeros('wave', shape=wave_shape, chunks=(1, None, None))

    else:
        full_simulation_shape = (runs, len(sources), time_nsteps) + tuple(grid_shape)
        full_simulation_chunks = (1,) + (64,) * (len(grid_shape) + 2)

        speed_array = dataset.zeros('speed', shape=full_simulation_shape, chunks=full_simulation_chunks)
        wave_array = dataset.zeros('wave', shape=full_simulation_shape, chunks=full_simulation_chunks)
        source_array = dataset.zeros('source', shape=full_simulation_shape, chunks=full_simulation_chunks)

    # Move through runs
    for run in tqdm(range(runs), leave=False):

        # Generate speed distribution on the grid
        if speed_method is None:
            speed_on_grid = speed[run]
        else:
            speed_on_grid = sample_speed(speed, base_simulation.grid, speed_range)

        # If saving reduced representation only save detected part
        if reduced:
            wave_detected = []

        # Move through sources
        for j, source in enumerate(tqdm(sources, leave=False)):
            # Create a new simulation
            sim = Simulation(size=size, spacing=spacing, speed=speed_on_grid, duration=duration, max_speed=speed_range[1])

            # Add the source
            sim.add_source(**source)

            # Run simulation
            sim.run(progress=False)

            if reduced:
                # Move through boundaries and try and extract each "recorded" signal
                for dim in range(len(grid_shape)):
                    index = [slice(None)] * (len(grid_shape) + 1)
                    # Take lower edge
                    index[dim + 1] = slice(0, boundary)
                    wave_detected.append(sim.wave[tuple(index)])
                    # Take upper edge
                    index[dim + 1] = slice(-boundary, len(sim.wave[dim+1]))
                    wave_detected.append(sim.wave[tuple(index)])
            else:
                 # Save source and wave from run
                speed_array[run, j] = sim.full_speed
                source_array[run, j] = sim.source
                wave_array[run, j] = sim.wave
        
        if reduced:
            # Save source and wave from run
            speed_array[run] = speed_on_grid
            # Concatenate waves across source/ space and transpose so
            # first axis is time
            wave_array[run] = np.concatenate(wave_detected, axis=1).T

    return dataset