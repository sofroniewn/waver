import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm
# from napari.qt import progress as tqdm

from ._detector import Detector
from ._grid import Grid
from ._source import Source
from ._time import Time
from ._wave import WaveEquation


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
    def __init__(self, *, size, spacing, max_speed, time_step=None, pml_thickness=20):
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
        max_speed : float
            Maximum speed of the wave in meters per second. This
            this speed will be used to derive largest allowed 
            time step.
        time_step : float, optional
            Time step to use if simulation will be stable. It must be
            smaller than the largest allowed time step.
        pml_thickness : int
            Thickness of any perfectly matched layer in pixels.
        """

        # Create grid
        self._grid = Grid(size=size, spacing=spacing, pml_thickness=pml_thickness)
        
        # Set default speed array
        self._max_speed = max_speed
        self._grid_speed = np.full(self.grid.shape, max_speed)

        # Calculate the theoretically optical courant number
        # given the dimensionality of the grid
        courant_number = 0.9 / float(self.grid.ndim) ** (0.5)

        # Based on the counrant number and the maximum speed
        # calculate the largest stable time step
        max_step = courant_number * self.grid.spacing / max_speed 

        # If time step is provided and it would be stable use it
        if time_step is not None:
            if time_step <= max_step:
                self._time_step = time_step
            else:
                raise ValueError(f'Provided time step {time_step} is larger than maximum allowed time step of {max_step}')
        else:
            # Round step, i.e. 5.047e-7 => 5e-7
            power =  np.power(10, np.floor(np.log10(max_step)))
            coef = int(np.floor(max_step / power))
            self._time_step = coef * power

        # Initialize some unset attributes
        self._record_with_pml = None
        self._time = None
        self._source = None
        self._detector = None
        self._wave_equation = None
        self._detected_wave = None
        self._run = False

    @property
    def grid(self):
        """Grid: Grid that simulation is defined on."""
        return self._grid

    @property
    def grid_speed(self):
        """Array: Speed of the wave in meters per second on the grid."""
        return self._grid_speed

    @property
    def time(self):
        """Time: Time that simulation is defined over."""
        return self._time

    @property
    def detector(self):
        """Decector: detector that simulation is recorded over."""
        return self._detector

    @property
    def detected_source(self):
        """array: Source for the wave on the detector."""
        if self._run:
            return self._detected_source
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    @property
    def detected_wave(self):
        """array: Array for the wave."""
        if self._run:
            return self._detected_wave
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    def set_speed(self, speed, min_speed=0, max_speed=None):
        """Set speed values defined on the simulation grid.
        
        Parameters
        ----------
        speed : np.ndarray, str
            Speed values defined on simulation grid. 
        min_speed : float
            Minimum allowed speed value.
        min_speed : float
            Maximum allowed speed value. Note cannot be larger
            than the maximum speed value allowed by the sample grid
            spaceing and time step.
        """
        if max_speed is None:
            max_speed = self._max_speed
        else:
            max_speed = min(max_speed, self._max_speed)

        speed = np.clip(speed, min_speed, max_speed)
        if getattr(speed, 'ndim', None) == self.grid.ndim:
            self._grid_speed = ndi.zoom(speed, np.divide(self.grid.shape, speed.shape))
        else:
            self._grid_speed = np.full(self.grid.shape, speed)

    def _setup_run(self, duration, temporal_downsample=1):
        """Setup run of the simulation for a given duration.

        Parameters
        ----------
        duration : float
            Length of the simulation in seconds.
        temporal_downsample : int, optional
            Temporal downsample factor.
        """
        # Create time object based on duration of run
        self._time = Time(step=self._time_step, duration=duration, temporal_downsample=temporal_downsample)

        # Pad grid speed if a pml is being used
        grid_speed = np.pad(self.grid_speed, self.grid.pml_thickness, 'edge')

        # Initialize new wave equation
        wave = np.zeros(self.grid.full_shape)
        self._wave_equation = WaveEquation(wave,
                                           c=grid_speed,
                                           dt=self.time.step,
                                           dx=self.grid.spacing,
                                           pml=self.grid.pml_thickness
                                           )

        # Create detector arrays for wave and source
        full_shape = (self.time.nsteps_detected,) + self.detector.downsample_shape
        self._detected_wave = np.zeros(full_shape)
        self._detected_source = np.zeros(full_shape)

    def run(self, duration, *, temporal_downsample=1, progress=True, leave=False):
        """Run the simulation for a given duration.
        
        Note a source and a detector must be added before the simulation
        can be run.

        Parameters
        ----------
        duration : float
            Length of the simulation in seconds.
        temporal_downsample : int, optional
            Temporal downsample factor.
        progress : bool, optional
            Show progress bar or not.
        leave : bool, optional
            Leave progress bar or not.
        """
        # Setup the simulation for the requested duration
        self._setup_run(duration=duration, temporal_downsample=temporal_downsample)

        if self._source is None:
            raise ValueError('Please add a source before running, use Simulation.add_source')

        if self._detector is None:
            raise ValueError('Please add a detector before running, use Simulation.add_detector')

        if self.grid.pml_thickness > 0 and not self._record_with_pml:
            recorded_slice = (slice(self.grid.pml_thickness, -self.grid.pml_thickness),) * self.grid_speed.ndim
        else:
            recorded_slice = (slice(None), ) * self.grid_speed.ndim

        for current_step in tqdm(range(self.time.nsteps), disable=not progress, leave=leave):
            current_time = self.time.step * current_step

            # Get current source values
            source_current = self._source.value(current_time)
            source_current = np.pad(source_current, self.grid.pml_thickness, 'constant')

            # Compute the next wave values
            self._wave_equation.update(Q=source_current)
            wave_current = self._wave_equation.wave

            # If recored timestep then use detector
            if current_step % self._time.temporal_downsample == 0:
                index = int(current_step // self._time.temporal_downsample)

                # Record wave on detector
                wave_current = wave_current[recorded_slice]                
                wave_current_ds = wave_current[self.detector.grid_index]
                self._detected_wave[index] = self.detector.sample(wave_current_ds)

                # Record source on detector
                source_current = source_current[recorded_slice]
                source_current_ds = source_current[self.detector.grid_index]
                self._detected_source[index] = self.detector.sample(source_current_ds)

        # Simulation has finished running
        self._run = True

    def add_detector(self, *, spatial_downsample=1, boundary=0, edge=None, with_pml=False):
        """Add a detector to the simulaiton.
        
        Note this must be done before the simulation can be run.

        Parameters
        ----------
        spatial_downsample : int, optional
            Spatial downsample factor.
        boundary : int, optional
            If greater than zero, then number of pixels on the boundary
            to detect at, in downsampled coordinates. If zero then detection
            is done over the full grid.
        edge : int, optional
            If provided detect only at that particular "edge", which in 1D is
            a point, 2D a line, 3D a plane etc. The particular edge is determined
            by indexing around the grid. It None is provided then all edges are
            used.
        with_pml : bool, optional
            If detector should also record values at the perfectly matched layer.
            The boundary should always be set to zero if this option is used.
        """
        self._run = False
        self._record_with_pml = with_pml
        if self._record_with_pml:
            grid_shape = self.grid.full_shape
        else:
            grid_shape = self.grid.shape
        self._detector = Detector(shape=grid_shape,
                                  spacing=self.grid.spacing,
                                  spatial_downsample=spatial_downsample,
                                  boundary=boundary,
                                  edge=edge,
                                 )

    def add_source(self, *, location, period, ncycles=None, phase=0):
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
