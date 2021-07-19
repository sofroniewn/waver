from numpy.lib.utils import source
from waver.datasets import generate_simulation_dataset

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/2D_simulations/test'
runs = 3

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3, 12.8e-3)
spacing = 1e-4
time_step = 50e-9
min_speed = 343
max_speed = 686
duration = 30e-6

# Define a speed sampling method
speed = 'random' # 'ifft'

# Define sources, a single 40KHz pulse at the left and right edges
sources = [
    {'location':(0, None), 'period':5e-6, 'ncycles':1},
    {'location':(size[0], None), 'period':5e-6, 'ncycles':1},
    {'location':(None, 0), 'period':5e-5, 'ncycles':1},
    {'location':(None, size[0]), 'period':5e-6, 'ncycles':1},
]

# Generate simulation dataset according to the above configuration
dataset = generate_simulation_dataset(
                                       path=path,
                                       runs=runs,
                                       size=size,
                                       spacing=spacing,
                                       duration=duration,
                                       min_speed=min_speed,
                                       max_speed=max_speed,
                                       time_step=time_step,
                                       speed=speed,
                                       sources=sources,
                                       temporal_downsample=2,
                                     )

print(dataset)

########### Other Sources
# # Define sources, a 2MHz, 5MHz, and 100KHz pulse at the left and right edges
# sources = [
#     {'location':(0,) * len(size), 'period':2e-6, 'ncycles':1},
#     {'location':(0,) * len(size), 'period':5e-6, 'ncycles':1},
#     {'location':(0,) * len(size), 'period':1e-5, 'ncycles':1},
#     {'location': size, 'period':2e-6, 'ncycles':1},
#     {'location': size, 'period':5e-6, 'ncycles':1},
#     {'location': size, 'period':1e-5, 'ncycles':1},
# ]