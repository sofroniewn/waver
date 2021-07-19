import inspect
import zarr
from pathlib import Path
from tqdm import tqdm

from ..simulation import run_multiple_sources


def generate_simulation_dataset(path, runs, **kawrgs):
    """Generate and save a simulation dataset.

    Parameters
    ----------
    path : str
        Root path where simulation data will be stored.
    runs : int
        Number of runs to use.
    kawrgs :
        run_simulation kwargs.

    Returns
    -------
    dataset : zarr.hierarchy.Group
        Simulation dataset.
    """
    # Convert path to pathlib path
    path = Path(path)

    # Create dataset
    dataset = zarr.open(path.as_posix(), mode='w')

    # Add dataset attributes
    dataset.attrs['waver'] = True
    dataset.attrs['dataset'] = True
    dataset.attrs['runs'] = runs

    # Add simulation attributes based on kwargs and defaults
    parameters = inspect.signature(run_multiple_sources).parameters
    for param, value in parameters.items():
        if param in kawrgs:
            dataset.attrs[param] = kawrgs[param]
        else:
            dataset.attrs[param] = value.default

    # Initialize speed and wave arrays
    speed_array = None
    wave_array = None
    
    # Move through runs
    for run in tqdm(range(runs), leave=False):
        wave, speed = run_multiple_sources(**kawrgs)
        if speed_array is None:
            speed_array = dataset.zeros('speed', shape=(runs, ) + speed.shape, chunks=(1,) + (None,) * speed.ndim)
        if wave_array is None:
            wave_array = dataset.zeros('wave', shape=(runs, ) + wave.shape, chunks=(1,) + (None,) * wave.ndim)

        speed_array[run] = speed
        wave_array[run] = wave

    return dataset