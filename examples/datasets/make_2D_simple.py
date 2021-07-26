from waver.datasets import generate_simulation_dataset

# Define root path for simulation
path = './simulation_dataset.zarr'
runs = 5

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (with 100ns timesteps)
sim_params = {
    'size': (12.8e-3, 12.8e-3),
    'spacing': 100e-6,
    'duration': 80e-6,
    'min_speed': 343,
    'max_speed': 686,
    'speed': 343,
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