from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from napari.utils import Colormap
from .simulation import Simulation
from .datasets import sample_speed


# ToDo, fix startup values for comboboxes
# ToDo, expose spatial/ temporal downsample factors

@magic_factory(call_button="run",
              ndim={'min': 1, 'max':3, 'step': 1},
              length={'min': .1, 'max':1e3, 'step': .1, 'label': 'length (mm)'},
              spacing={'min': 10, 'max':10_000, 'step': 10, 'label': 'spacing (um)'},
              min_speed={'min': 10, 'max':1e5, 'step': 1e1, 'label': 'speed (m/s)'},
              max_speed={'min': 10, 'max':1e5, 'step': 1e1, 'label': 'speed (m/s)'},
              time_step={'min': 10, 'max':1e6, 'step': 1e2, 'label': 'time step (ns)'},
              duration={'min': 10, 'max':1e6, 'step': 10, 'label': 'duration (us)'},
              frequency={'min': 1, 'max':1e4, 'step': 10, 'label': 'frequency (kHz)'},
              method={'choices': ['flat', 'random', 'ifft']}
)
def simulation(
               ndim: int=2,
               length: float=12.8,
               spacing: float=100,
               min_speed: float=50, 
               max_speed: float=100, 
               time_step: float=200,
               duration: float=60,
               frequency: float=500,
               method: str='flat',
               ) -> 'napari.types.LayerDataTuple':
    """Run a single simulation.
    """
    size = ndim * (length / 1e3,) # length always same in all dimensions

    # Create simulation
    sim = Simulation(size=size, spacing=spacing / 1e6, speed=max_speed, time_step=time_step / 1e9)
    
    # Add randomly sampled speed array
    speed_array = sample_speed(method, sim.grid, (min_speed, max_speed))
    sim._speed = speed_array

    # Add source
    period = 1 / frequency / 1e3
    sim.add_source(location=(s/2 for s in size), period=period, ncycles=1)

    # Add detector grid
    sim.add_detector()

    # Run simulation
    sim.run(duration=duration / 1e6)

    # Return simulation wave data
    clim = max(sim.wave.max(), abs(sim.wave.min())) / 3**ndim
    wave_cmap = Colormap([[1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]], name='MBlG')
    wave_dict = {'colormap': wave_cmap, 'contrast_limits':[-clim, clim], 'name': 'wave'}
    speed_cmap = Colormap([[0, 0, 0, 0], [0.7, 0.5, 0, 1]], name='Orange')
    speed_dict = {'colormap': speed_cmap, 'opacity': 0.5, 'contrast_limits':(min_speed, max_speed), 'name': 'speed'}
    return [(speed_array, speed_dict, 'Image'), (sim.wave, wave_dict, 'Image')]

# (speed_array, speed_dict, 'Image'), 
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return simulation