
import napari
from waver.datasets import load_simulation_dataset

# Define root path for simulation
path = './simulation_dataset.zarr'

dataset = load_simulation_dataset(path)

# Load data into napari
viewer = napari.Viewer()
for data in dataset:
    viewer._add_layer_from_data(*data)

napari.run()