from napari_plugin_engine import napari_hook_implementation
from pathlib import Path
import zarr

from .datasets import load_simulation_dataset


@napari_hook_implementation
def napari_get_reader(path):
    """Implementation of the napari_get_reader hook specification.
    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
    
    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # Inspect dataset
    path = Path(path)
    dataset = zarr.open(path.as_posix(), mode='r')

    # If dataset is a full dataset return reader
    if dataset.attrs['waver'] and dataset.attrs['dataset']:
        return load_simulation_dataset
    else:
        return None
