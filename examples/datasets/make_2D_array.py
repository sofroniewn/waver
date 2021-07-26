import numpy as np
from waver.datasets import generate_simulation_dataset

# Define root path for simulation
path = './simulation_dataset.zarr'

# Set normed_image to be runs
rescaled_image = np.eye(128, 128)
normed_image = 343 + 343 * rescaled_image
runs = [normed_image]

# Define a simulation, 12.8mm, 100um spacing
sim_params = {
    'size': (12.8e-3, 12.8e-3),
    'spacing': 100e-6,
    'duration': 80e-6,
    'min_speed': 343,
    'max_speed': 686,
    'speed': 'mixed_random_ifft',
    'time_step': 50e-9,
    'sources': [{
        'location': (None, 0),
        'period': 5e-6,
        'ncycles':1,
    }],
    'temporal_downsample': 2,
    'boundary': 1,
    'edge': 1,
}

# Run and save simulation
generate_simulation_dataset(path, runs, **sim_params)