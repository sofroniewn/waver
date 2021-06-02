from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from ._dataset import WaverSimulationDataset


class WaverDataModule(LightningDataModule):
    """Data module for waver simulation data.

    Parameters
    ----------
    data_dir : str
        Location of wave simulation data.
    batch_size : int, optional
        Batch size.
    """
    def __init__(self, data_dir: str = './', batch_size=32):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def prepare_data(self):
        """Skip preparing data."""
        pass

    def setup(self, stage=None):
        """Setup data."""
        # Assign train/ val datasets for use in dataloaders
        if stage in (None, 'fit'):
            train_dataset = WaverSimulationDataset(self.data_dir, mode='train')
            split = [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))]
            self._waver_train, self._waver_val = random_split(train_dataset, split)
            self.dims_input = self._waver_train[0][0].shape
            self.dims_output = self._waver_train[0][1].shape
            self.dims = self.dims_input

        if stage in (None, 'test'):
            self._waver_test = WaverSimulationDataset(self.data_dir, mode='test')
            self.dims_input = getattr(self, 'dims_input', self._waver_test[0][0].shape)
            self.dims_output = getattr(self, 'dims_output', self._waver_test[0][1].shape)
            self.dims = getattr(self, 'dims', self.dims_input)

    def train_dataloader(self):
        """Return data loader for training."""
        return DataLoader(self._waver_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return data loader for validation."""
        return DataLoader(self._waver_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return data loader for testing."""
        return DataLoader(self._waver_test, batch_size=self.batch_size, shuffle=False)
