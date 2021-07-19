from numpy.lib.utils import source
from waver.datasets import generate_simulation_dataset

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/1D_simulations/12_8mm_at_100um.zarr'
runs = 1000

# Define a simulation, 12.8mm, 100um spacing, for 60.8us
size = (12.8e-3,)
spacing = 1e-4
time_step = 50e-9
temporal_downsample = 2
min_speed = 343
max_speed = 686
duration = 60.8e-6

# Define a speed sampling method
speed = 'mixed_random_ifft' # 'ifft'

# Define sources, a single 200KHz pulse at the left and right edges
sources = [
    {'location':(0,), 'period':5e-6, 'ncycles':1},
    {'location':(size[0],), 'period':5e-6, 'ncycles':1},
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
                                       temporal_downsample=temporal_downsample,
                                       boundary=1
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