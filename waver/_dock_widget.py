from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from .simulation import Simulation


# ToDo, fix startup values for comboboxes
# ToDo, expose spatial/ temporal downsample factors
# ToDo, allow for "speed" to be inputed as an array somehow ....

@magic_factory(call_button="run",
              ndim={'min': 1, 'max':3, 'step': 1},
              length={'min': .1, 'max':1e3, 'step': .1, 'label': 'length (mm)'},
              spacing={'min': 10, 'max':10_000, 'step': 10, 'label': 'spacing (um)'},
              speed={'min': 10, 'max':1e5, 'step': 1e1, 'label': 'speed (m/s)'},
              time_step={'min': 10, 'max':1e6, 'step': 1e2, 'label': 'time step (ns)'},
              duration={'min': 10, 'max':1e6, 'step': 10, 'label': 'duration (us)'},
              frequency={'min': 1, 'max':1e4, 'step': 10, 'label': 'frequency (kHz)'},
)
def simulation(
               ndim: int=2,
               length: float=12.8,
               spacing: float=100,
               speed: float=343, 
               time_step: float=200,
               duration: float=60,
               frequency: float=500,               
               ) -> 'napari.types.LayerDataTuple':
    """Run a single simulation.
    """
    size = ndim * (length / 1e3,) # length always same in all dimensions

    # Create simulation
    sim = Simulation(size=size, spacing=spacing / 1e6, speed=speed, time_step=time_step / 1e9)

    # Add source
    period = 1 / frequency / 1e3
    sim.add_source(location=(s/2 for s in size), period=period, ncycles=1)

    # Add detector grid
    sim.add_detector()

    # Run simulation
    sim.run(duration=duration / 1e6)

    # Return simulation wave data
    clim = max(sim.wave.max(), abs(sim.wave.min()))
    layer_dict = {'colormap': 'PiYG', 'contrast_limits':[-clim, clim], 'name': 'wave'}
    return (sim.wave, layer_dict, 'Image')


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return simulation