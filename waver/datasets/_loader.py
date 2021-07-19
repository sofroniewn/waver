from napari.utils import Colormap
from pathlib import Path
import zarr


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
        # Return simulation wave data
        clim = max(dataset['wave'][0].max(), abs(dataset['wave'][0].min())) / 3**dataset['wave'][0].ndim
        wave_cmap = Colormap([[0.55, 0, .32, 1], [0, 0, 0, 0], [0.15, 0.4, 0.1, 1]], name='PBlG')
        wave_dict = {'colormap': wave_cmap, 'contrast_limits':[-clim, clim], 'name': 'wave', 'metadata':metadata}
        speed_cmap = Colormap([[0, 0, 0, 0], [0.7, 0.5, 0, 1]], name='Orange')
        speed_dict = {'colormap': speed_cmap, 'visible': False, 'contrast_limits':(metadata['min_speed'], metadata['max_speed']),
                      'name': 'speed', 'metadata':metadata}
        return [(dataset['wave'], wave_dict, 'image'), (dataset['speed'], speed_dict, 'image')]
    else:
        raise ValueError(f'Dataset at {path} not valid waver simulation')
