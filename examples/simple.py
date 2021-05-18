from waver.components import Simulation
import napari

# Create simulation
sim = Simulation(size=(0.1, 0.1), spacing=1e-3, speed=343, duration=1e-2)

# Add the source
sim.add_source(location=(0, None), frequency=10)
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