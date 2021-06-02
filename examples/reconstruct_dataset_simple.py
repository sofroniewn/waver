from glob import glob
import napari

from waver.inversion._data_module import WaverDataModule
from waver.inversion._lightning_module import WaverInversion

# Define path for simulation
# path = '/Users/nsofroniew/Documents/inverting_physics/tests_006'
# ckpt_path = path + '/lightning_logs/version_3/checkpoints/epoch=12-step=2560.ckpt'
path = '/Users/nsofroniew/Documents/inverting_physics/tests_008r'
ckpt_path = path + '/lightning_logs/version_1/checkpoints/*.ckpt'
ckpt_path = glob(ckpt_path)[0]

print(ckpt_path)

# Create WaverDataModule
dm = WaverDataModule(path, predict='astro')
dm.setup()
n_channels = int(dm.dims_input[0])
n_spatial_points = int(dm.dims_output[0])

# Load model from checkpoint
model = WaverInversion.load_from_checkpoint(ckpt_path, n_channels=n_channels, n_spatial_points=n_spatial_points)
model.eval();

# Pass prediction set through model
i, batch = next(enumerate(dm.predict_dataloader()))
detected_wave, speed = batch

# Pass through network
predicted_speed = model(detected_wave)
speed = speed.detach().numpy()
detected_wave = detected_wave.detach().numpy()
predicted_speed = predicted_speed.detach().numpy()


### Run again!
predicted_speed_N = predicted_speed
# Create WaverDataModule
dm = WaverDataModule(path, predict='astroT')
dm.setup()
n_channels = int(dm.dims_input[0])
n_spatial_points = int(dm.dims_output[0])

# Load model from checkpoint
model = WaverInversion.load_from_checkpoint(ckpt_path, n_channels=n_channels, n_spatial_points=n_spatial_points)
model.eval();

# Pass prediction set through model
i, batch = next(enumerate(dm.predict_dataloader()))
detected_wave, speed = batch

# Pass through network
predicted_speed = model(detected_wave)
speed = speed.detach().numpy()
detected_wave = detected_wave.detach().numpy()
predicted_speed_T = predicted_speed.detach().numpy()

predicted_speed = (predicted_speed_T.T + predicted_speed_N) / 2
speed = speed.T

# View in napari
viewer = napari.Viewer()
viewer.add_image(predicted_speed, colormap='viridis', contrast_limits=[0, 1], interpolation='bilinear')
viewer.add_image(speed, colormap='viridis', contrast_limits=[0, 1], interpolation='bilinear')
viewer.grid.enabled = True

# Launch napari
napari.run()