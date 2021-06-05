
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from waver.inversion._data_module import WaverDataModule
from waver.inversion._lightning_module import WaverInversion


# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_017r/' 

# Create WaverDataModule
dm = WaverDataModule(path)
dm.setup()
n_channels = int(dm.dims_input[0])
n_spatial_points = int(dm.dims_output[0])

# Create WaverInversion
inverter = WaverInversion(n_channels=n_channels, n_spatial_points=n_spatial_points)

# Create Trainer
trainer = Trainer(
                  default_root_dir=path,
                 )
trainer.fit(inverter, dm)

