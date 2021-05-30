import zarr
from pathlib import Path


def load_simulation_dataset(path):
    """Load a simulation dataset.

    Parameters
    ----------
    path : str
        Path to simulation data to load.

    Returns
    -------
    dataset : zarr.hierarchy.Group
        Loaded simulation dataset.
    """

    # Load dataset
    path = Path(path)
    dataset = zarr.open(path.as_posix(), mode='r')

    # If dataset is a full dataset return it
    if dataset.attrs['waver'] and dataset.attrs['dataset']:
        return [
                (dataset['wave'], {'name':'wave'}),
                (dataset['speed'], {'name':'speed'}),
                (dataset['source'], {'name':'source'}),
        ]
    else:
        raise ValueError(f'Dataset at {path} not valid waver simulation')
