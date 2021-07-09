from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
import numpy as np


@magic_factory(call_button="run",
              spacing={'min': 1e-6, 'max':1},
              speed={'min': 1, 'max':1e5},
              time_step={'min': 1e-10, 'max':1},
)
def simulation(spacing: float=1e-3,
               speed: float=343, 
               time_step: float=200e-9) -> 'napari.types.ImageData':
    print(f"you have selected {spacing} {speed} {time_step}")
    return np.random.random((50, 50))


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return simulation



#     # Create simulation, 12.8mm at 100um spacing, at 200ns resolution
# sim_dict = {'size': (384e-4,),
#             'spacing': 1e-4,
#             'speed': 343,
#             'time_step': 200e-9,
#             }

# sim = Simulation(**sim_dict)

# # Add a point source in the center, 200kHz pulse for one cycle
# sim.add_source(location=(0,), period=5e-6, ncycles=1)

# # Add detector grid at full spatial and temporal resolution
# sim.add_detector(spatial_downsample=3)

# # Run simulation for 60.8us
# sim.run(duration=120e-6)