from scipy import ndimage as ndi
from waver.datasets import generate_simulation_datasets

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_018r/'

reduced = True

# Consider train and test splits
splits = ['astro', 'astroT']
runs = [None, None]

# Define a simulation, 12.8mm, 100um spacing, for 60.8us (leads to 100ns timesteps)
size = (12.8e-3,)
spacing = 1e-4
duration = 6.08e-5

# Define a speed range, min speed of sound in air
speed_range = (343, 686)

# Define a custom speed based on an image
from skimage import data
full_image = data.astronaut().mean(axis=2)
# full_image = data.camera()

# from torchvision.datasets import MNIST
# from torchvision import transforms
# mnist_train = MNIST('/Users/nsofroniew/Documents/code/data/MNIST', download = False,
#                     transform = transforms.Compose([
#                         transforms.ToTensor(),
#                     ]), train = True)
# full_image = mnist_train[0][0][0].detach().numpy()

full_image = full_image / full_image.max()
rescaled_image = ndi.zoom(full_image, 128/full_image.shape[0])
normed_image = speed_range[0] + (speed_range[1] - speed_range[0]) * rescaled_image
speed = [normed_image, normed_image.T]

# Define sources, a single 40KHz pulse at the left and right edges
sources = [
    {'location':(0,) * len(size), 'period':2.5e-5, 'ncycles':1},
    {'location': size, 'period':2.5e-5, 'ncycles':1},
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
