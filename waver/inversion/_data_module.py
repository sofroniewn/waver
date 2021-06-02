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
    def __init__(self, data_dir: str = './', batch_size=32, predict='predict'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self._predict = predict

    def prepare_data(self):
        """Skip preparing data."""
        pass

    def setup(self, stage=None):
        """Setup data."""
        # Assign train/ val datasets for use in dataloaders
        if stage in (None, 'fit'):
            train_dataset = WaverSimulationDataset(self.data_dir, mode='train')
            train_split = int(0.9 * len(train_dataset))
            split = [train_split, len(train_dataset) - train_split]
            self._waver_train, self._waver_val = random_split(train_dataset, split)
            self.dims_input = self._waver_train[0][0].shape
            self.dims_output = self._waver_train[0][1].shape
            self.dims = self.dims_input

        if stage in (None, 'test'):
            self._waver_test = WaverSimulationDataset(self.data_dir, mode='test')
            self.dims_input = getattr(self, 'dims_input', self._waver_test[0][0].shape)
            self.dims_output = getattr(self, 'dims_output', self._waver_test[0][1].shape)
            self.dims = getattr(self, 'dims', self.dims_input)

        if stage in (None, 'predict'):
            try:
                self._waver_predict = WaverSimulationDataset(self.data_dir, mode=self._predict)
                self.dims_input = getattr(self, 'dims_input', self._waver_predict[0][0].shape)
                self.dims_output = getattr(self, 'dims_output', self._waver_predict[0][1].shape)
                self.dims = getattr(self, 'dims', self.dims_input)
            except:
                self._waver_predict = None

    def train_dataloader(self):
        """Return data loader for training."""
        return DataLoader(self._waver_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return data loader for validation."""
        return DataLoader(self._waver_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return data loader for testing."""
        return DataLoader(self._waver_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        """Return data loader for prediction."""
        return DataLoader(self._waver_predict, batch_size=128, shuffle=False)