from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_001/'

# Consider train and test splits
splits = ['train', 'test']
runs = [70, 30]

# Define a simulation, 5mm, 100um spacing, for 15us
size = (5e-2,)
spacing = 1e-4
duration = 15e-6

# Define a speed range, min speed of sound in air
speed_range = (343, 1000)

# Define a speed sampling method
speed = ''

# Define sources, a single 100KHz pulse at the left and right edges
sources = [
    {'location':(1e-4,), 'period':1e-5, 'ncycles':1},
    {'location':(5e-2 - 1e-4,), 'period':1e-5, 'ncycles':1},
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
