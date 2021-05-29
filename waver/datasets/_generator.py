import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm

from ..simulation import Grid, Simulation
from ._utils import sample_speed


def generate_simulation_dataset(*, path, size, spacing, duration, speed, speed_range, sources, splits, runs, seed=0):
    """Generate a simulation dataset.

    Parameters
    ----------
    path : str
        Root path where simulation data will be stored.
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
    splits : list of str
        List of splits to bucket simulations into. Most commonly
        `('train', 'test')`.
    runs : list of int
        Number of different speed distributions. Must be same length
        as number of splits.
    seed : int, optional
        Seed to initialize random number generator with.

    Returns
    -------
    dataset : zarr.hierarchy.Group
        Full simulation dataset.
    """
    # Initialize seed
    np.random.seed(seed)

    # Create grid for simulation
    grid = Grid(size=size, spacing=spacing)

    # Create dataset
    path = Path(path) / 'wave_simulation.zarr'
    dataset = zarr.open(path.as_posix(), mode='w')

    # Add dataset attributes
    dataset.attrs['size'] = size
    dataset.attrs['spacing'] = spacing
    dataset.attrs['duration'] = duration
    dataset.attrs['speed'] = speed
    dataset.attrs['speed_range'] = speed_range
    dataset.attrs['sources'] = sources
    dataset.attrs['splits'] = splits
    dataset.attrs['runs'] = runs
    dataset.attrs['seed'] = seed

    # Move through splits
    for i, split in enumerate(tqdm(splits)):
        # Create group for this data split
        split_group = dataset.create_group(split)

        # Move through runs
        for run in tqdm(range(runs[i]), leave=False):
            # Create group for this data run
            run_group = split_group.create_group(f'{run:08}')

            # Add run attributes
            run_group.attrs['size'] = size
            run_group.attrs['spacing'] = spacing
            run_group.attrs['duration'] = duration
            run_group.attrs['sources'] = sources

            # Generate speed distribution on the grid
            speed_on_grid = sample_speed(speed, grid, speed_range)

            # Save speed on run
            speed_array = run_group.zeros('speed', shape=speed_on_grid.shape) # ToDo work on chunks
            speed_array[:] = speed_on_grid

            # Move through sources
            for j, source in enumerate(tqdm(sources, leave=False)):
                # Create group for this data source
                source_group = run_group.create_group(f'{j:08}')

                # Create a new simulation
                sim = Simulation(size=size, spacing=spacing, speed=speed_on_grid, duration=duration)

                # Add the source
                sim.add_source(**source)

                # Run simulation
                sim.run(progress=False)

                # Save source from run
                source_array = source_group.zeros('source', shape=sim.source.shape) # ToDo work on chunks
                source_array[:] = sim.source

                # Save wave from run
                wave_array = source_group.zeros('wave', shape=sim.wave.shape) # ToDo work on chunks
                wave_array[:] = sim.wave

    return dataset