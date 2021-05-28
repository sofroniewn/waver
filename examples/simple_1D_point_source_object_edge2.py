import numpy as np
import napari
from waver.components import Simulation

# Define variable speed matrix
speed = 343 * np.ones((500,))
speed[100:150] = 2*343
speed[200:220] = 0.5*343
speed[250:280] = 1.5*343
# Create simulation, 10cm, 100um spacing, for 100us
sim = Simulation(size=(5e-2,), spacing=1e-4, speed=speed, duration=25e-5)
# Report the simulation timestep (10ns in this case)
print('Time step in (s) is ', sim.time.step)
# Add a point source in at the left edge, 1MHz pulse for 1us
sim.add_source(location=(1e-4,), period=1e-5, ncycles=1)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()
viewer.dims.axis_labels = 'tx'

viewer.add_image(-sim.wave[::5], colormap='PiYG', contrast_limits=(-15, 15), name='wave')

full_speed = (np.repeat([sim.speed], 500, axis=0) - 343) / 343
cmap = napari.utils.Colormap([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]], name='custom')
viewer.add_image(-30*full_speed, colormap=cmap, contrast_limits=(-2, 2), name='speed')

mask = np.zeros((500, 500))
mask[:, 50:-50] = 1
cmap2 = napari.utils.Colormap([[0, 0, 0, 0], [0, 0, 0, 1]], name='mask')
viewer.add_image(mask, colormap=cmap2, contrast_limits=(0, 1), visible=False)

# # Add simulated wave, speed and source
# viewer.add_image(-(np.expand_dims(sim.speed, axis=(0, )) - 343) / 4, colormap='bop orange')
# viewer.add_image(np.expand_dims(-sim.wave[::5], axis=1), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))

# # # Add simulated endpoints timeseries
# viewer.add_image(500 + 10*np.expand_dims(-sim.wave[:, 0], axis=0), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))
# viewer.add_image(500 + 100 + 10*np.expand_dims(-sim.wave[:, -1], axis=0), colormap='PiYG', opacity=0.9, contrast_limits=(-1.5, 1.5))

# Set to 1D plotting
viewer.dims.ndisplay = 2

# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()