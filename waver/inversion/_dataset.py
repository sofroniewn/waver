from pathlib import Path

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from ..datasets import load_simulation_dataset


class WaverSimulationDataset(Dataset):
    def __init__(self, data_dir, border=2, mode='train'):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.border = border

        # Load simulation dataset
        dataset = load_simulation_dataset(self.data_dir / (mode + '.zarr'))
        self._speed_data = dataset[0][0]
        self._wave_data = dataset[1][0]
        self.reduced = len(dataset) == 2

    def __len__(self):
        return len(self._wave_data)
    
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            Index of sample from dataset.

        Returns
        -------
        2-tuple of torch.Torch
            Tuple of detected wave and speed.
        """
        if self.reduced:
            speed = np.asarray(self._speed_data[index])
            wave = np.asarray(self._wave_data[index])
        else:
            # Assume speed is constant in time and for every source
            # Speed is now in spatial steps
            speed = np.asarray(self._speed_data[index, 0, 0])

            # Wave is sources x time steps x spatial steps
            wave = self._wave_data[index]
        
            # # Filter speed!!!!!!!!!
            # from scipy.signal import sosfiltfilt, butter
            # sos = butter(4, .3, output='sos')
            # speed = sosfiltfilt(sos, speed).copy()

            # Shrink wave to only contain the border
            # ToDo generalize code from more than one spatial dimension
            b = self.border
            if b is not None:
                detected_wave = np.concatenate([np.asarray(wave[:, :, :b]), np.asarray(wave[:, :, -b:])], axis=2)
            else:
                # If no border assume whole wave is detected
                detected_wave = np.asarray(wave)

            # Reshape wave to combine detected spatial points and sources into one channels axis
            # reshaped_wave is not (sources * b) x time
            # ToDo generalize code from more than one spatial dimension
            wave = detected_wave.transpose((2, 0, 1)).reshape(-1, detected_wave.shape[1])

        # Put speed in 0, 1 range
        speed = (speed - 343) / 343

        # Convert to Tensors
        detected_wave = Tensor(wave).float()
        speed = Tensor(speed).float()
        
        return detected_wave, speed