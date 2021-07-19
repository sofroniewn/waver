import numpy as np
import napari
from waver.simulation import Simulation

# Define variable speed matrix
speed = 343 * np.ones((399, 399))
speed[100:150, 220:270] = 2*343
# Create simulation, 4mm x 4mm, 10um spacing, for 10us
sim = Simulation(size=(4e-3, 4e-3), spacing=1e-5, max_speed=speed.max())
sim.set_speed(speed=speed)

# Add a point source in the center, 1MHz pulse for 1us
sim.add_source(location=(2e-3, 2e-3), period=1e-6, ncycles=1)

# Add default detector
sim.add_detector(spatial_downsample=4, temporal_downsample=5, boundary=4)

# Run simulation
sim.run(duration=1e-5)

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave, speed and source
viewer.add_image(sim.source, visible=False, name='source')
viewer.add_image(sim.speed, colormap='bop orange', name='speed')
viewer.add_image(sim.wave, colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5), name='wave')
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()