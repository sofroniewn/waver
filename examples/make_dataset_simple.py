from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_017r/'

reduced = True

# Consider train and test splits
splits = ['train', 'test']
runs = [7000, 3000]

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3,)
spacing = 1e-4
duration = 6.08e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)

# Define a speed sampling method
speed = 'ifft'

# Define sources, a single 100KHz pulse at the left and right edges
sources = [
    {'location':(0,) * len(size), 'period':2e-6, 'ncycles':1},
    {'location':(0,) * len(size), 'period':5e-6, 'ncycles':1},
    {'location':(0,) * len(size), 'period':1e-5, 'ncycles':1},
    {'location': size, 'period':2e-6, 'ncycles':1},
    {'location': size, 'period':5e-6, 'ncycles':1},
    {'location': size, 'period':1e-5, 'ncycles':1},
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
