"""
3D Gaussian Random Field generation with a specified power spectrum.
"""

import numpy as np
from numba import njit, prange
from scipy.fft import fftfreq


sqrt2 = 2.0**0.5


def make_jit_power_spec(power_spec, **njit_kwargs):
    user_jit = njit(**njit_kwargs)(power_spec.func)
    norm = power_spec.norm
    @njit
    def _power_spec(k):
        return norm*user_jit(k)  # user_func must be jittable!
    return _power_spec


@njit(parallel=True, fastmath=True)
def compute_sigma_1d(kx, nx, power_spec):
    sigma = np.zeros(nx)

    for i in prange(nx):
        if kx[i] == 0.0:
            sigma[i] = 0.0
        else:
            sigma[i] = np.sqrt(power_spec(kx[i]))

    return sigma

@njit(parallel=True, fastmath=True)
def compute_sigma_2d(kx, ky, nx, ny, power_spec):
    sigma = np.zeros((nx, ny))

    for i in prange(nx):
        for j in prange(ny):
            kk = np.sqrt(kx[i] * kx[i] + ky[j] * ky[j])
            if kk == 0.0:
                sigma[i, j] = 0.0
            else:
                sigma[i, j] = np.sqrt(power_spec(kk))

    return sigma


@njit(parallel=True, fastmath=True)
def compute_sigma_3d(kx, ky, kz, nx, ny, nz, power_spec):
    sigma = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                kk = np.sqrt(kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k])
                if kk == 0.0:
                    sigma[i, j, k] = 0.0
                else:
                    sigma[i, j, k] = np.sqrt(power_spec(kk))

    return sigma


@njit(fastmath=True)
def enforce_hermitian_symmetry_1d(f_hat):
    nx = f_hat.shape[0]
    for i in range(nx // 2 + 1): # only loop over half
        i_ = (-i) % nx
        if i != i_:
            f_hat[i_] = np.conj(f_hat[i])
        else:
            f_hat[i] = complex(f_hat[i].real, 0.0)  # ensure real on Nyquist
    return f_hat


@njit(fastmath=True)
def enforce_hermitian_symmetry_2d(f_hat):
    nx, ny = f_hat.shape
    for i in range(nx):
        for j in range(ny // 2 + 1):  # only loop over half
            i_, j_ = (-i) % nx, (-j) % ny
            if (i, j) != (i_, j_):
                f_hat[i_, j_] = np.conj(f_hat[i, j])
            else:
                f_hat[i, j] = complex(f_hat[i, j].real, 0.0)  # ensure real on Nyquist
    return f_hat


@njit(fastmath=True)
def enforce_hermitian_symmetry_3d(f_hat):
    nx, ny, nz = f_hat.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz // 2 + 1):  # only loop over half
                i_, j_, k_ = (-i) % nx, (-j) % ny, (-k) % nz
                if (i, j, k) != (i_, j_, k_):
                    f_hat[i_, j_, k_] = np.conj(f_hat[i, j, k])
                else:
                    f_hat[i, j, k] = complex(f_hat[i, j, k].real, 0.0)  # ensure real on Nyquist
    return f_hat


class GaussianRandomField:
    """
    Parameters
    ----------
    left_edge : array-like
        The lower edge of the box [kpc] for each of the dimensions.
    right_edge : array-like
        The upper edge of the box [kpc] for each of the dimensions.
    ddims : array-like
        The number of grids in each of the axes.
    power_spec : PowerSpectrum object
        The power spectrum function for the field.
    seed : int, optional
        Random seed for reproducibility.
    """
    def __init__(self, left_edge, right_edge, ddims, power_spec, seed=None):
        self.left_edge = np.atleast_1d(left_edge).astype("float64")
        self.right_edge = np.atleast_1d(right_edge).astype("float64")
        self.ddims = np.atleast_1d(ddims).astype("int")
        self.width = self.right_edge - self.left_edge
        self.delta = self.width / self.ddims
        self.ndim = self.left_edge.size
        self.x = np.linspace(self.left_edge[0]+0.5*self.delta[0], self.right_edge[0]-0.5*self.delta[0], self.ddims[0])
        self.y = np.linspace(self.left_edge[1]+0.5*self.delta[1], self.right_edge[1]-0.5*self.delta[1], self.ddims[1])
        self.z = np.linspace(self.left_edge[2]+0.5*self.delta[2], self.right_edge[2]-0.5*self.delta[2], self.ddims[2])

        self.kx = fftfreq(self.ddims[0], d=self.delta[0])
        if self.ndim > 1:
            self.ky = fftfreq(self.ddims[1], d=self.delta[1])
        if self.ndim == 3:
            self.kz = fftfreq(self.ddims[2], d=self.delta[2])

        self.pspec = make_jit_power_spec(power_spec)
        self.sigma = None
        self.prng = np.random.default_rng(seed=seed)

    def _compute_sigma(self):
        if self.ndim == 1:
            sigma = compute_sigma_1d(self.kx, self.ddims[0], self.pspec)
        elif self.ndim == 2:
            sigma = compute_sigma_2d(self.kx, self.ky, self.ddims[0], self.ddims[1], self.pspec)
        else:
            sigma = compute_sigma_3d(self.kx, self.ky, self.kz, self.ddims[0], self.ddims[1], self.ddims[2], self.pspec)
        sigma /= np.sqrt(np.prod(self.width))
        self.sigma = sigma

    def generate_scalar_field_realization(self):
        """
        Generate a random realization of the field as a scalar field.
        """

        if self.sigma is None:
            self._compute_sigma()

        v_real = self.prng.normal(size=self.ddims)*self.sigma
        v_imag = self.prng.normal(size=self.ddims)*self.sigma

        v = v_real + 1j * v_imag
        v /= sqrt2

        if self.ndim == 1:
            v = enforce_hermitian_symmetry_1d(v)
        elif self.ndim == 2:
            v = enforce_hermitian_symmetry_2d(v)
        else:
            v = enforce_hermitian_symmetry_3d(v)

        v = np.fft.ifftn(v, norm="forward")

        return v.real

    def generate_vector_field_realization(self):
        """
        Generate a random realization of the field as a vector field.
        """

        field = []
        for i in range(self.ndim):
            field.append(self.generate_scalar_field_realization())
        return np.asarray(field)
