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
                (dataset['source'], {'name':'source', 'visible':False, 'colormap':'PiYG', 'contrast_limits':(-1, 1)}),
                (dataset['speed'], {'name':'speed', 'visible':False}),
                (dataset['wave'], {'name':'wave', 'colormap':'PiYG', 'contrast_limits':(-2.5, 2.5)}),
        ]
    else:
        raise ValueError(f'Dataset at {path} not valid waver simulation')
