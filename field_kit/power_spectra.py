import numpy as np
from scipy.integrate import quad


two_pi = 2.0*np.pi


class PowerSpectrum:
    def __init__(self, power_spec_func, ndim=3):
        self.func = power_spec_func
        self.norm = 1.0
        if ndim not in [1, 2, 3]:
            raise ValueError("Invalid number of dimensions! Must be 1, 2, or 3.")
        self.ndim = ndim
        self.prefactor = two_pi**ndim

    def __call__(self, k):
        return self.norm * self.func(k)

    def E(self, k):
        if self.ndim == 1:
            e = self(k)
        elif self.ndim == 2:
            e = 2.0 * np.pi * self(k) * k
        elif self.ndim == 3:
            e = 4.0 * np.pi * self(k) * k * k
        return self.prefactor * e

    def A(self, k):
        return np.sqrt(self.E(k)*k)

    def integrate_E(self, kmin, kmax):
        points = np.array([0.001, 0.01, 0.1, 0.3]) * (kmax - kmin) + kmin
        return float(quad(self.E, kmin, kmax, points=points)[0])

    def renormalize(self, f_rms, kmin=0.0, kmax=100.0):
        self.norm = (
            f_rms**2 / self.integrate_E(kmin, kmax)
        )


class PowerLawBetaModel(PowerSpectrum):
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
    def __init__(self, l_min, l_max, alpha, ndim=3):

        self.l_min = l_min
        self.l_max = l_max
        self.k_min_i = l_min/two_pi
        self.k_max_i = l_max/two_pi
        self.alpha = alpha

        def _pspec(k):
            return (1.0 + (k * k_max_inv) ** 2) ** (0.5 * alpha) * np.exp(-((k * k_min_i) ** 2))

	super().__init__(_pspec, length_scale=l_min, ndim=ndim)
