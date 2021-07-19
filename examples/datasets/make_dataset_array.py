from numpy.lib.utils import source
from waver.datasets import generate_simulation_dataset

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/2D_simulations/test_astro.zarr'

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3,)
spacing = 1e-4
time_step = 50e-9
min_speed = 343
max_speed = 686
duration = 60e-6

# Define a custom speed based on an image
from skimage import data
import scipy.ndimage as ndi

full_image = data.astronaut().mean(axis=2)
# full_image = data.camera()
full_image = full_image / full_image.max()
rescaled_image = ndi.zoom(full_image, 128/full_image.shape[0])
normed_image = min_speed + (max_speed - min_speed) * rescaled_image

# Set normed_image to be runs
runs = normed_image



# Define sources, a single 40KHz pulse at the left and right edges
sources = [
    {'location':(0,), 'period':5e-6, 'ncycles':1},
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
                                       sources=sources,
                                       temporal_downsample=2,
                                       boundary=1,
                                     )

print(dataset)