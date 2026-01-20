import numpy as np
from scipy.integrate import quad


class PowerSpectrum:
    def __init__(self, power_spec_func, ndim=3):
        self.func = power_spec_func
        self.norm = 1.0
        if ndim not in [1, 2, 3]:
            raise ValueError("Invalid number of dimensions! Must be 1, 2, or 3.")
        self.ndim = ndim

    def __call__(self, k):
        return self.norm * self.func(k)

    def E(self, k):
        if self.ndim == 1:
            return self.func(k)
        elif self.ndim == 2:
            return 2.0 * np.pi * self.func(k) * k
        elif self.ndim == 3:
            return 4.0 * np.pi * self.func(k) * k * k

    def renormalize(self, f_rms):
        self.norm = (
            f_rms**2 / quad(self.E, 0.0, 100.0, points=(0.1, 1.0, 10.0, 30.0))[0]
        )


def plaw_with_cutoffs(l_min, l_max, alpha, ndim=3):
    """
    Power-law power spectrum with exponential cutoffs at small and large scales.

    Parameters
    ----------
    l_min : float
        Minimum scale (smallest wavelength) cutoff.
    l_max : float
        Maximum scale (largest wavelength) cutoff.
    alpha : float
        Power-law index.
    ndim : int, optional
        Number of dimensions (1, 2, or 3). Default is 3.
    """

    def _pspec(k):
        return (1.0 + (k * l_max) ** 2) ** (0.5 * alpha) * np.exp(-((k * l_min) ** 2))

    return PowerSpectrum(_pspec, ndim=ndim)
