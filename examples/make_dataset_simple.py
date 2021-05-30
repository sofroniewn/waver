from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_003/'

# Consider train and test splits
splits = ['train', 'test']
runs = [70, 30]

# Define a simulation, 2.56mm, 100um spacing, for 120us
size = (2.56e-2,)
spacing = 1e-4
duration = 12e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)

# Define a speed sampling method
speed = 'random'

# Define sources, a single 100KHz pulse at the left and right edges
sources = [
    {'location':(0,) * len(size), 'period':1e-5, 'ncycles':1},
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
                                     )

print(dataset)
