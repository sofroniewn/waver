from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from napari.utils import Colormap
from .simulation import run_single_source
from .simulation._utils import fourier_sample


@magic_factory(call_button="run",
              ndim={'min': 1, 'max':3, 'step': 1},
              length={'min': .1, 'max':1e3, 'step': .1, 'label': 'length (mm)'},
              spacing={'min': 10, 'max':10_000, 'step': 10, 'label': 'spacing (um)'},
              min_speed={'min': 10, 'max':1e5, 'step': 1e1, 'label': 'min speed (m/s)'},
              max_speed={'min': 10, 'max':1e5, 'step': 1e1, 'label': 'max speed (m/s)'},
              time_step={'min': 10, 'max':1e6, 'step': 1e2, 'label': 'time step (ns)'},
              duration={'min': 10, 'max':1e6, 'step': 10, 'label': 'duration (us)'},
              frequency={'min': 1, 'max':1e4, 'step': 10, 'label': 'frequency (kHz)'},
              method={'choices': ['flat', 'random', 'ifft', 'custom']},
              spatial_downsample={'min': 1, 'max':100, 'step': 1, 'label': 'spatial ds'},
              temporal_downsample={'min': 1, 'max':100, 'step': 1, 'label': 'temporal ds'},
              boundary={'min': 0, 'max':100, 'step': 1},
)
def simulation(
               ndim: int=2,
               length: float=12.8,
               spacing: float=100,
               min_speed: float=343, 
               max_speed: float=686, 
               time_step: float=50,
               duration: float=60,
               frequency: float=500,
               method: str='flat',
               custom: 'napari.layers.Layer'=None,
               spatial_downsample: int=1,
               temporal_downsample: int=1,
               boundary: int=0,
               ) -> 'napari.types.LayerDataTuple':
    """Run a single simulation.
    """
    size = ndim * (length / 1e3,) # length always same in all dimensions
    spacing = spacing / 1e6
    time_ste = time_step / 1e9
    period = 1 / frequency / 1e3
    location = (s/2 for s in size)
    duration = duration / 1e6

    # Add randomly sampled speed array
    if method == 'custom':
        raw_speed = custom.data
        if hasattr(custom, 'num_colors'):
            raw_speed = raw_speed / raw_speed.max()
        else:
            import numpy as np

            raw_speed = np.clip(raw_speed, custom.contrast_limits[0], custom.contrast_limits[1])
            raw_speed = (raw_speed - custom.contrast_limits[0]) / (custom.contrast_limits[1] - custom.contrast_limits[0])
        speed = raw_speed * (max_speed - min_speed) + min_speed
    else:
        speed = method


    detected_wave, detected_speed = run_single_source(size=size, spacing=spacing, location=location,
        duration=duration, max_speed=max_speed, min_speed=min_speed, speed=speed, period=period,
        spatial_downsample=spatial_downsample, temporal_downsample=temporal_downsample,
        boundary=boundary
    )

    # Return simulation wave data
    clim = max(detected_wave.max(), abs(detected_wave.min())) / 3**ndim
    wave_cmap = Colormap([[0.55, 0, .32, 1], [0, 0, 0, 0], [0.15, 0.4, 0.1, 1]], name='PBlG')
    wave_dict = {'colormap': wave_cmap, 'contrast_limits':[-clim, clim], 'name': 'wave'}
    speed_cmap = Colormap([[0, 0, 0, 0], [0.7, 0.5, 0, 1]], name='Orange')
    speed_dict = {'colormap': speed_cmap, 'opacity': 0.5, 'contrast_limits':(min_speed, max_speed), 'name': 'speed'}
    return [(detected_wave, wave_dict, 'Image'), (detected_speed, speed_dict, 'Image')]


@magic_factory(call_button="run",
              ndim={'min': 1, 'max':3, 'step': 1},
              length={'min': 1, 'max':1e5, 'step': 1},
              )
def sample_fourier(
               ndim: int=2,
               length: int=32,
               ) -> 'napari.types.LayerDataTuple':
    """Run a single simulation.
    """
    shape = ndim * (length,) # length always same in all dimensions
    values = fourier_sample(shape)
    values_dict = {'colormap': 'viridis', 'contrast_limits':[0, 1], 'name': 'fourier sample', 'interpolation': 'bilinear'}
    return [(values, values_dict, 'Image')]


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [simulation, sample_fourier]