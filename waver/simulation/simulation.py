import numpy as np
# from tqdm import tqdm
from napari.qt import progress as tqdm

from ._detector import Detector
from ._grid import Grid
from ._source import Source
from ._time import Time
from ._utils import sample_boundary
from ._wave import wave_equantion_update


class Simulation:
    """Simulation of wave equation for a certain time on a defined grid.

    The simulation is always assumed to start from zero wave initial
    conditions, but is driven by a source. Therefore the `add_source`
    method must be called before the simulation can be `run`.

    In order to record anything a dector must be added. Therefore the
    `add_detector` method must be called before the simulation can be
    `run`.

    Right now only one source and one detector can be used per simulation.
    """
    def __init__(self, *, size, spacing, speed, max_speed=None, time_step=None):
        """
        Parameters
        ----------
        size : tuple of float
            Size of the grid in meters. Length of size determines the
            dimensionality of the grid.
        spacing : float
            Spacing of the grid in meters. The grid is assumed to be
            isotropic, all dimensions use the same spacing.
        speed : float or array
            Speed of the wave in meters per second. If a float then
            speed is assumed constant across the whole grid. If an
            array then must be the same shape as the grid. Note that
            the speed is assumed contant in time.
        max_speed : float, optional
            Maximum speed of the wave in meters per second. If passed then
            this speed will be used to derive the time step.
        time_step : float, optional
            Time step to use if stable.
        """

        # Create grid
        self._grid = Grid(size=size, spacing=spacing)

        # Creat speed array
        self._speed = np.full(self._grid.shape, speed)

        # Calculate the theoretically optical courant number
        # given the dimensionality of the grid
        courant_number = 0.9 / float(self.grid.ndim) ** (0.5)

        # Based on the counrant number and the maximum speed
        # calculate the largest stable time step
        if max_speed is None:
            max_speed = np.max(self.speed)
        max_step = courant_number * self.grid.spacing / max_speed 

        # Round step, i.e. 5.047e-7 => 5e-7
        power =  np.power(10, np.floor(np.log10(max_step)))
        coef = int(np.floor(max_step / power))
        step = coef * power

        # If time step is provided and it would be stable use it
        if time_step is not None:
            step = min(step, time_step)

        # Set the time step informatioan for        
        self._time_step = step

        # Initialize some unset attributes
        self._time = None
        self._source = None
        self._detector = None
        self._wave_current = None
        self._wave_previous = None
        self._wave_array = None
        self._source_array = None
        self._run = False

    @property
    def grid(self):
        """Grid: Grid that simulation is defined on."""
        return self._grid

    @property
    def detector(self):
        """Decector: detector that simulation is recorded over."""
        return self._detector

    @property
    def time(self):
        """Time: Time that simulation is defined over."""
        return self._time
    
    @property
    def speed(self):
        """Array: Speed of the wave in meters per second."""
        return self._speed

    @property
    def detector_speed(self):
        """Array: Speed of the wave in meters per second on detector."""
        return self._speed[self.detector.grid_index]

    @property
    def source(self):
        """array: Array for the source."""
        if self._run:
            return self._source_array
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    @property
    def wave(self):
        """array: Array for the wave."""
        if self._run:
            return self._wave_array
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    def _setup_run(self, duration):
        """Setup run of the simulation.

        Parameters
        ----------
        duration : float
            Length of the simulation in seconds.
        """
        # Create time object based on duration of run
        self._time = Time(step=self._time_step, duration=duration)

        self._wave_current = np.zeros(self.grid.shape)
        self._wave_previous = np.zeros(self.grid.shape)

        # Create detector arrays for wave and source
        full_shape = (int((self.time.nsteps - 1) // self.detector.temporal_downsample + 1),) + self.detector.downsample_shape
        self._wave_array = np.zeros(full_shape)
        self._source_array = np.zeros(full_shape)

    def run(self, duration, *, progress=True, leave=False):
        """Run the simulation.
        
        Note a source and a detector must be added before the simulation
        can be run.

        Parameters
        ----------
        duration : float
            Length of the simulation in seconds.
        progress : bool, optional
            Show progress bar or not.
        leave : bool, optional
            Leave progress bar or not.
        """
        # Setup the simulation for the requested duration
        self._setup_run(duration=duration)

        if self._source is None:
            raise ValueError('Please add a source before running, use Simulation.add_source')

        if self._detector is None:
            raise ValueError('Please add a detector before running, use Simulation.add_detector')

        for current_step in tqdm(range(self.time.nsteps), disable=not progress, leave=leave):
            current_time = self.time.step * current_step

            # Get current source values
            source_current = self._source.value(current_time)

            # Compute the next wave values
            wave_tmp =  wave_equantion_update(U_1=self._wave_current, 
                                              U_0=self._wave_previous,
                                              c=self.speed,
                                              Q_1=source_current,
                                              dt=self.time.step,
                                              dx=self.grid.spacing
                                             )

            self._wave_previous = self._wave_current
            self._wave_current = wave_tmp

            # Record wave using detector
            if current_step % self.detector.temporal_downsample == 0:
                index = int(current_step // self.detector.temporal_downsample)
                wave_current_ds = self._wave_current[self.detector.grid_index]
                source_current_ds = source_current[self.detector.grid_index]
                if self.detector.boundary > 0:
                    self._wave_array[index] = sample_boundary(wave_current_ds, self.detector.boundary)
                    self._source_array[index] = sample_boundary(source_current_ds, self.detector.boundary)
                else:
                    self._wave_array[index] = wave_current_ds
                    self._source_array[index] = source_current_ds

        self._run = True

    def add_detector(self, *, spatial_downsample=1, temporal_downsample=1, boundary=0):
        """Add a detector to the simulaiton.
        
        Note this must be done before the simulation can be run.

        Parameters
        ----------
        spatial_downsample : int, optional
            Spatial downsample factor.
        temporal_downsample : int, optional
            Temporal downsample factor.
        boundary : int, optional
            If greater than zero, then number of pixels on the boundary
            to detect at, in downsampled coordinates. If zero then detection
            is done over the full grid.
        """
        self._run = False
        self._detector = Detector(shape=self.grid.shape,
                                  spacing=self.grid.spacing,
                                  spatial_downsample=spatial_downsample,
                                  temporal_downsample=temporal_downsample,
                                  boundary=boundary,
                                 )

    def add_source(self, *, location, period, ncycles=None, phase=0,):
        """Add a source to the simulaiton.
        
        Note this must be done before the simulation can be run.

        The added source will be a sinusoid with a fixed spatial weight
        and vary either contiously or for a fixed number of cycles.

        Parameters
        ----------
        location : tuple of float or None
            Location of source in m. If None is passed at a certain location
            of the tuple then the source is broadcast along the full extent
            of that axis. For example a source of `(0.1, 0.2, 0.1)` is a
            point source in 3D at the point x=10cm, y=20cm, z=10cm. A source of
            `(0.1, None, 0.1)` is a line source in 3D at x=10cm, z=10cm extending
            the full length of y.
        period : float
            Period of the source in seconds.
        ncycles : int or None
            If None, source is considered to be continous, otherwise
            it will only run for ncycles.
        phase : float
            Phase offset of the source in radians.
        """
        self._run = False
        self._source = Source(location=location,
                              shape=self.grid.shape,
                              spacing=self.grid.spacing,
                              period=period,
                              ncycles=ncycles,
                              phase=phase)

    def set_boundaries(boundaries):
        """Set boundary conditions
        
        Parameters
        ----------
        boundaries : list of 2-tuple of str
            For each axis, a 2-tuple of the boundary conditions where the 
            first and second values correspond to low and high boundaries
            of the axis. The acceptable boundary conditions are `PML` and
            `periodic` for Perfectly Matched Layer, and periodic conditions
            respectively.
        """
        # Not yet implemented
        pass