import numpy as np
import napari
from waver.components import Simulation

# Define variable speed matrix
speed = 343 * np.ones((399, 399))
speed[100:150, 220:270] = 2*343
# Create simulation, 4mm x 4mm, 10um spacing, for 10us
sim = Simulation(size=(4e-3, 4e-3), spacing=1e-5, speed=speed, duration=1e-5)
# Report the simulation timestep (10ns in this case)
print('Time step in (s) is ', sim.time.step)
# Add a point source in the center, 1MHz pulse for 2us
sim.add_source(location=(2e-3, 2e-3), period=1e-6, ncycles=2)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave, speed and source
viewer.add_image(sim.source.weight)
viewer.add_image(sim.speed, colormap='bop orange')
viewer.add_image(sim.wave[::5], colormap='PiYG', opacity=0.9)
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()