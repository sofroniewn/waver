import napari
import numpy as np
from napari.utils import Colormap

from ..simulation import run_multiple_sources


def run_and_visualize(**kawrgs):
    """Visualize a simulation.

    Parameters
    ----------
    kawrgs :
        run_multiple_sources kwargs.
    """
    wave, speed = run_multiple_sources(**kawrgs)

    clim = max(wave[0].max(), abs(wave[0].min())) / 3**wave[0].ndim
    wave_cmap = Colormap([[0.55, 0, .32, 1], [0, 0, 0, 0], [0.15, 0.4, 0.1, 1]], name='PBlG')
    wave_dict = {'colormap': wave_cmap, 'contrast_limits':[-clim, clim], 'name': 'wave',
        'metadata':kawrgs, 'interpolation': 'bilinear'}

    speed_cmap = Colormap([[0, 0, 0, 0], [0.7, 0.5, 0, 1]], name='Orange')
    speed_dict = {'colormap': speed_cmap, 'opacity': 0.5,
                'name': 'speed', 'metadata':kawrgs, 'interpolation': 'bilinear'}
    
    viewer = napari.Viewer()
    viewer.add_image(np.atleast_2d(np.squeeze(wave)), **wave_dict)
    viewer.add_image(np.atleast_2d(speed[0, 0]), **speed_dict)

    napari.run()