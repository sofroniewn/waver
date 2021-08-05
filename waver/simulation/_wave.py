import numpy as np
from ._utils import gradient, divergence, make_pml_sigma


class WaveEquation:
    """Class that does the wave equation update
    
    Attributes
    ---------- 
    """
    def __init__(self, wave, *, c, dt, dx, pml=0):

        # Store update parameters
        self._dt = dt
        self._D = dt / dx
        self._c = c
        self._c2 = c ** 2
        self._pml_thickness = pml
        self._sigma_max = pml
        self._ndim = wave.ndim

        # Initialize Pressure and Velocity
        self._P = wave
        self._v = np.zeros((self._ndim,) + wave.shape)

        # Initialize with an empty auxillary factors
        self._psi = np.zeros(self._v.shape)

        # Create sigma factor for pml
        self._sigma = make_pml_sigma(wave.shape, self._sigma_max, self._pml_thickness)
        self._sigma_factors = [np.product([s for i, s in enumerate(self._sigma) if i != dim], axis=0)
                                for dim in range(len(self._sigma))]

    def update(self, Q=0):
        """Update the wave equation"""

        # Update velocity vector array
        grad_P = gradient(self._P)
        pml_correction = self._dt * self._c * self._sigma * self._v
        self._v -= self._D * grad_P + pml_correction

        # Update pressure scalar array
        div_v = divergence(self._v)
        pml_correction = self._dt * self._c * np.sum(self._sigma, axis=0) * self._P
        # Additional factor using auxilary variables don't seem to be needed ......
        # for dim, psi in enumerate(self._psi):
        #     pml_correction += self._D * self._c2 * self._sigma_factors[dim] * gradient(self._psi[dim], axis=dim)
        self._P -=  self._D * self._c2 * div_v + pml_correction - Q

        # Note auxilary variables don't seem to be needed ......
        # Update auxilary equations for perfectly matched layer correction
        # self._psi += self._dt * self._c * self._v

    @property
    def wave(self):
        """np.ndarray: Wave."""
        return self._P


# class WaveEquation:
#     """Class that does the wave equation update
    
#     Attributes
#     ---------- 
#     """
#     def __init__(self, wave, *, c, dt, dx, pml=0):

#         # Store update parameters
#         self._D = dt / dx
#         self._c2 = c ** 2
        
#         # Initialize Pressure and Velocity
#         self._wave = wave
#         self._wave_1 = wave

#     def update(self, Q=0):
#         """Update the wave equation"""

#         wave = 2 * self._wave - self._wave_1 + self._c2 * self._D**2 * laplace(self._wave, mode='constant') + Q
#         self._wave_1 = self._wave
#         self._wave = wave

#     @property
#     def wave(self):
#         """np.ndarray: Wave."""
#         return self._wave
