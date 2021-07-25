import numpy as np
import napari
from waver.simulation import Simulation

# Create simulation, 4mm x 4mm, 10um spacing
max_speed = 686
sim = Simulation(size=(4e-3,), spacing=1e-5, max_speed=max_speed)

# Define variable speed matrix
speed = max_speed * np.ones((399,)) / 2
speed[100:150] = max_speed
sim.set_speed(speed=speed)

# Add a point source in the center, 1MHz pulse for 1us
sim.add_source(location=(2e-3,), period=1e-6, ncycles=1)
# Add default detector
sim.add_detector()
# Run simulation for 15us
sim.run(duration=15e-6)


# Create a napari viewer
viewer = napari.Viewer()

# Add detected wave and speed
viewer.add_image(sim.wave, name='wave', colormap='PiYG', contrast_limits=(-1.5, 1.5))
viewer.add_image(np.tile(sim.detector_speed, (sim.wave.shape[0], 1)), colormap='bop orange',
    name='speed', contrast_limits=(343, 686), opacity=0.4)

# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()