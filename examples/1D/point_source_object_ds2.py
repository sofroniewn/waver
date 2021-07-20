import numpy as np
import napari
from waver.simulation import Simulation

# Define variable speed matrix
# speed = 343 * np.ones((1600, 1600))
# speed[500:550, 720:770] = 2*343

# Create simulation, 12.8mm at 100um spacing, at 200ns resolution
sim_dict = {'size': (384e-4,),
            'spacing': 1e-4,
            'max_speed': 343,
            'time_step': 200e-9,
            }

sim = Simulation(**sim_dict)

# Add a point source in the center, 200kHz pulse for one cycle
sim.add_source(location=(0,), period=5e-6, ncycles=1)

# Add detector grid at full spatial and temporal resolution
sim.add_detector(spatial_downsample=3, boundary=20, edge=0)

# Run simulation for 60.8us
sim.run(duration=120e-6)

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave, speed and source
viewer.add_image(sim.source, visible=False, name='source')
# viewer.add_image(sim.detector_speed, colormap='bop orange', name='speed')
viewer.add_image(sim.wave, colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5), name='wave')
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()