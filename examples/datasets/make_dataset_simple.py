from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/2D_simulations/12_8mm_at_100um/200kHz_f'

reduced = False

# Consider train and test splits
splits = ['train']
runs = [1]

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3, 12.8e-3)
spacing = 1e-4
duration = 6.08e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)

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
dataset = generate_simulation_datasets(
                                       path=path,
                                       size=size,
                                       spacing=spacing,
                                       duration=duration,
                                       speed_range=speed_range,
                                       speed=speed,
                                       sources=sources,
                                       splits=splits,
                                       runs=runs,
                                       reduced=reduced
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