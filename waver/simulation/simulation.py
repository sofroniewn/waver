import numpy as np
from tqdm import tqdm

from ._grid import Grid
from ._source import Source
from ._time import Time
from ._wave import wave_equantion_update


class Simulation:
    """Simulation of wave equation for a certain time on a defined grid.

    The simulation is always assumed to start from zero wave initial
    conditions, but is driven by a source. Therefore the `add_source`
    method must be called before the simulation can be `run`.
    """
    def __init__(self, *, size, spacing, speed, duration, max_speed=None):
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
            array then must be the same shape as the grid.
        max_speed : float, optional
            Maximum speed of the wave in meters per second. If passed then
            this speed will be used to derive timestep.
        duration : float
            Length of the simulation in seconds.
        """
        self._grid = Grid(size=size, spacing=spacing)
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
        self._time = Time(step=step, duration=duration)

        self._wave = np.zeros((self.time.nsteps,) + self.grid.shape)
        self._source_array = np.zeros((self.time.nsteps,) + self.grid.shape)
        self._speed_array = np.zeros((self.time.nsteps,) + self.grid.shape)
        self._source = None
        self._run = False

    @property
    def grid(self):
        """Grid: Grid that simulation is defined on."""
        return self._grid

    @property
    def time(self):
        """Time: Time that simulation is defined over."""
        return self._time
    
    @property
    def speed(self):
        """Array or float: Speed of the wave in meters per second."""
        return self._speed

    @property
    def source(self):
        """array: Array for the source."""
        if self._run:
            return self._source_array
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    @property
    def full_speed(self):
        """array: Array for the speed."""
        if self._run:
            return self._speed_array
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    @property
    def wave(self):
        """array: Array for the wave."""
        if self._run:
            return self._wave
        else:
            raise ValueError('Simulation must be run first, use Simulation.run()')

    def run(self, *, progress=True, leave=False):
        """Run the simulation.
        
        Note a source must be added before the simulation can be run.

        Parameters
        ----------
        progress : bool, optional
            Show progress bar or not.
        leave : bool, optional
            Leave progress bar or not.
        """
        # Reset wave to 0
        self._wave *= 0

        if self._source is None:
            raise ValueError('Please add a source before running, use Simulation.add_source()')

        for current_step in tqdm(range(self.time.nsteps), disable=not progress, leave=leave):
            # Take wave to be zero for first two time steps 
            current_time = self.time.step * current_step

            # Save source values
            self._speed_array[current_step] = self.speed

            # Save source values
            self._source_array[current_step] = self._source.value(current_time)

            # Save wave values
            self._wave[current_step] = wave_equantion_update(U_1=self._wave[current_step - 1], 
                                                                U_0=self._wave[current_step - 2],
                                                                c=self.speed,
                                                                Q_1=self._source_array[current_step],
                                                                dt=self.time.step,
                                                                dx=self.grid.spacing
                                                                )
        self._run = True

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