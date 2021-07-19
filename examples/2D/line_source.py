import numpy as np
import napari
from waver.simulation import Simulation

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (with 100ns timesteps)
size = (12.8e-3, 12.8e-3)
spacing = 1e-4
duration = 6.08e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)
speed = np.random.random((128, 128)) * 343 + 343

# Define sources, a single 40KHz pulse at the left and right edges
sources = {'location':(None, 0), 'period':5e-6, 'ncycles':1}

sim = Simulation(size=size, spacing=spacing, max_speed=speed_range[1])
sim.set_speed(speed=speed)

# Add a point source in the center, 1MHz pulse for 1us
sim.add_source(**sources)

# Add default detector
sim.add_detector()

# Run simulation
sim.run(duration=duration, progress=True, leave=True)

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave, speed and source
viewer.add_image(sim.source, visible=False)
# viewer.add_image(sim.speed, colormap='bop orange')
cmap = napari.utils.Colormap(name='RBB', colors=[[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1]])
viewer.add_image(sim.wave, colormap=cmap, opacity=0.9, contrast_limits=(-2.5, 2.5))
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()