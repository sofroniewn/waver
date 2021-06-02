from skimage import data
from scipy import ndimage as ndi
from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_007_reduced/'

reduced = True

# Consider train and test splits
splits = ['astro']
runs = [None]

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3,)
spacing = 1e-4
duration = 6.08e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)

# Define a custom speed based on an image
full_image = data.astronaut().mean(axis=2)
# full_image = data.camera()
full_image = full_image / full_image.max()
rescaled_image = ndi.zoom(full_image, 128/512)
normed_image = speed_range[0] + (speed_range[1] - speed_range[0]) * rescaled_image
speed = [normed_image]

# Define sources, a single 100KHz pulse at the left and right edges
sources = [
    {'location':(0,) * len(size), 'period':1e-5, 'ncycles':1},
    {'location': size, 'period':1e-5, 'ncycles':1},
]

# import napari
# napari.view_image(speed[0])
# napari.run()

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
                                       reduced=reduced,
                                     )
