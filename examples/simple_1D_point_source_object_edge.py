import numpy as np
import napari
from waver.simulation import Simulation

# Define variable speed matrix
speed = 343 * np.ones((399,))
speed[100:150] = 2*343
speed[200:220] = 0.5*343
speed[250:280] = 1.5*343
# Create simulation, 4mm x 4mm, 10um spacing, for 10us
sim = Simulation(size=(4e-3,), spacing=1e-5, speed=speed, duration=15e-6)
# Report the simulation timestep (10ns in this case)
print('Time step in (s) is ', sim.time.step)
# Add a point source in at the left edge, 1MHz pulse for 1us
sim.add_source(location=(1e-5,), period=1e-6, ncycles=1)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()

# Add simulated wave, speed and source
viewer.add_image(-(np.expand_dims(sim.speed, axis=(0, )) - 343) / 4, colormap='bop orange')
viewer.add_image(np.expand_dims(-sim.wave[::5], axis=1), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))

# # Add simulated endpoints timeseries
viewer.add_image(500 + 10*np.expand_dims(-sim.wave[:, 0], axis=0), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))
viewer.add_image(500 + 100 + 10*np.expand_dims(-sim.wave[:, -1], axis=0), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))

# Set to 1D plotting
viewer.dims.ndisplay = 1

# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()