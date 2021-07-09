from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from .simulation import Simulation


@magic_factory(call_button="run",
              spacing={'min': 1e-6, 'max':1},
              speed={'min': 1, 'max':1e5},
              time_step={'min': 1e-10, 'max':1},
)
def simulation(spacing: float=1e-3,
               speed: float=343, 
               time_step: float=200e-9) -> 'napari.types.LayerDataTuple':
    """Run a single simulation.
    """
    # Should be provided by magicgui, right now hard coded
    size = (12.8e-3, 12.8e-3) # how to do tuple - could be 1D, 2D, 3D?
    spacing=1e-5
    speed=343
    time_step=200e-9
    period=5e-6
    duration=60e-6

    # Create simulation
    sim = Simulation(size=size, spacing=spacing, speed=speed, time_step=time_step)

    # Add source
    sim.add_source(location=(s/2 for s in size), period=period, ncycles=1)

    # Add detector grid
    sim.add_detector()

    # Run simulation
    sim.run(duration=duration)

    # Return simulation wave data
    layer_dict = {'colormap': 'PiYG', 'contrast_limits':[-1.5, 1.5], 'name': 'wave'}
    return (sim.wave, layer_dict, 'Image')


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return simulation