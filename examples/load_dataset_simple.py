
import napari
from waver import datasets
from waver.datasets import load_simulation_dataset

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_001/' + 'wave_simulation_0/train.zarr'

dataset = load_simulation_dataset(path)

# Load data into napari
viewer = napari.Viewer()
for data in dataset:
    viewer._add_layer_from_data(*data)

napari.run()