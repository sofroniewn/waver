
from monai.networks.nets import BasicUNet
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.nn.functional import interpolate, mse_loss


class WaverInversion(LightningModule):
    """Data module for waver simulation data.

    Parameters
    ----------
    n_channels : int
        Number of channels of input wave data.
    n_spatial_points : int
        Number of spatial points in speed.
    """
    def __init__(self, n_channels, n_spatial_points):
        super().__init__()

        # Store additional parameters
        self._n_channels = n_channels
        self._n_spatial_points = n_spatial_points

        # Create model
        self._model = BasicUNet(1, n_channels, 1)

    def forward(self, x):
        # In lightning, forward defines the prediction/inference actions

        # Pass data through network
        output = self._model(x)

        # Resize and squeeze output to match speed shape
        predicted_speed = interpolate(output, size=self._n_spatial_points).squeeze(1)

        return predicted_speed

    def training_step(self, batch, _):
        # training_step defined the train loop.
        # It is independent of forward
        detected_wave, speed = batch
        predicted_speed = self.forward(detected_wave)
        loss = mse_loss(speed, predicted_speed)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        # training_step defined the train loop.
        # It is independent of forward
        detected_wave, speed = batch
        predicted_speed = self.forward(detected_wave)
        loss = mse_loss(speed, predicted_speed)

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer