import numpy as np
from waver.datasets import run_and_visualize

# Define a simulation, 6.4mm, 100um spacing
sim_params = {
    'size': (6.4e-3,) * 3,
    'spacing': 100e-6,
    'duration': 20e-6,
    'min_speed': 343,
    'max_speed': 686,
    'speed': 686,
    'time_step': 50e-9,
    'temporal_downsample': 2,
    'sources': [{
        'location': (3.2e-3,) * 3,
        'period': 5e-6,
        'ncycles':1,
    }]
}

# Run and visualize simulation
run_and_visualize(**sim_params)