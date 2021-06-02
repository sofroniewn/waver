from waver import datasets
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

    metadata = dataset.attrs.asdict()

    # If dataset is a full dataset return it
    if dataset.attrs['waver'] and dataset.attrs['dataset']:
        if 'reduced' in metadata and metadata['reduced']:
            return [
                    (dataset['speed'], {'name':'speed', 'visible':False, 'metadata':metadata}),
                    (dataset['wave'], {'name':'wave', 'colormap':'PiYG', 'contrast_limits':(-2.5, 2.5), 'metadata':metadata}),
            ]
        else:
            return [
                    (dataset['source'], {'name':'source', 'visible':False, 'colormap':'PiYG', 'contrast_limits':(-1, 1), 'metadata':metadata}),
                    (dataset['speed'], {'name':'speed', 'visible':False, 'metadata':metadata}),
                    (dataset['wave'], {'name':'wave', 'colormap':'PiYG', 'contrast_limits':(-2.5, 2.5), 'metadata':metadata}),
            ]
    else:
        raise ValueError(f'Dataset at {path} not valid waver simulation')
