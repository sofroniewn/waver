from waver.components import Simulation
import napari

# Create simulation
sim = Simulation(size=(1e-4, 1e-2), spacing=1e-5, speed=343, duration=1e-5)

# Add the source
sim.add_source(location=(None, 0), frequency=1e6)
# Run simulation
sim.run()

# Create a napari viewer
viewer = napari.Viewer()

# Add simulated wave
viewer.add_image(sim.source.weight)
viewer.add_image(sim.wave)

viewer.update_console({'sim': sim})
# Run napari
napari.run()