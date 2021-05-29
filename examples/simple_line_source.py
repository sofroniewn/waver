import napari
from waver.simulation import Simulation


# Create simulation
sim = Simulation(size=(1e-4, 1e-2), spacing=1e-5, speed=343, duration=1e-5)
# Report the simulation timestep
print('Time step in (s) is ', sim.time.step)
# Add a line source at the edge
sim.add_source(location=(None, 0), period=1e-6)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave and source
viewer.add_image(sim.source.weight)
viewer.add_image(sim.wave)
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()