import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm

from ..simulation import Simulation
from ._utils import sample_speed


def generate_simulation_datasets(*, path, splits, runs, seed=0, size, spacing, duration, speed, speed_range, sources):
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
    speed : str
        String describing sampling method for generating
        speed distributions.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.
    sources : list of dict
        List of source parameters. The simulation will be rerun
        with each set of source parameters.

    Returns
    -------
    dataset : list of zarr.hierarchy.Group
        List of simulation datasets.
    """
    # Initialize seed
    np.random.seed(seed)

    # Create datasets folder
    path = Path(path) / f'wave_simulation_{seed}'
    path.mkdir(exist_ok=True)

    # Move through splits
    datasets = []
    for i, split in enumerate(tqdm(splits)):
        # Create dataset
        split_path = path / f'{split}.zarr'
        dataset = generate_simulation_dataset(
                                              path=split_path,
                                              runs=runs[i],
                                              size=size,
                                              spacing=spacing,
                                              duration=duration,
                                              speed=speed,
                                              speed_range=speed_range,
                                              sources=sources,
                                             )
        datasets.append(dataset)
    
    return datasets


def generate_simulation_dataset(*, path, runs, size, spacing, duration, speed, speed_range, sources):
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
    speed : str
        String describing sampling method for generating
        speed distributions.
    speed_range : tuple of float
        Minimum and maximum allowed speeds.
    sources : list of dict
        List of source parameters. The simulation will be rerun
        with each set of source parameters.

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
    dataset.attrs['time_nsteps'] = time_nsteps
    dataset.attrs['speed'] = speed
    dataset.attrs['speed_range'] = speed_range
    dataset.attrs['sources'] = sources

 
    # Generate array containers
    full_simulation_shape = (runs, len(sources), time_nsteps) + tuple(grid_shape)
    full_simulation_chunks = (1, 1) + (64,) * (len(grid_shape) + 1)
    print('aaa', full_simulation_shape, full_simulation_chunks)

    speed_array = dataset.zeros('speed', shape=full_simulation_shape, chunks=full_simulation_chunks)
    wave_array = dataset.zeros('wave', shape=full_simulation_shape, chunks=full_simulation_chunks)
    source_array = dataset.zeros('source', shape=full_simulation_shape, chunks=full_simulation_chunks)

    # Move through runs
    for run in tqdm(range(runs), leave=False):

        # Generate speed distribution on the grid
        speed_on_grid = sample_speed(speed, base_simulation.grid, speed_range)

        # Move through sources
        for j, source in enumerate(tqdm(sources, leave=False)):

            # Create a new simulation
            sim = Simulation(size=size, spacing=spacing, speed=speed_on_grid, duration=duration, max_speed=speed_range[1])

            # Add the source
            sim.add_source(**source)

            # Run simulation
            sim.run(progress=False)

            # Save source and wave from run
            speed_array[run, j] = sim.full_speed
            source_array[run, j] = sim.source
            wave_array[run, j] = sim.wave

    return dataset