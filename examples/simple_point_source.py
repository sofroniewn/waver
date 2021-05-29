import napari
from waver.simulation import Simulation


# Create simulation
sim = Simulation(size=(4e-3, 4e-3), spacing=1e-5, speed=343, duration=5e-6)
# Report the simulation timestep
print('Time step in (s) is ', sim.time.step)
# Add a point source in the center
sim.add_source(location=(2e-3, 2e-3), period=1e-6, ncycles=2)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()
# Add simulated wave and source
viewer.add_image(sim.source.weight)
viewer.add_image(sim.wave[::5], colormap='PiYG')
# Add simulation to the console
viewer.update_console({'sim': sim})

# Run napari
napari.run()