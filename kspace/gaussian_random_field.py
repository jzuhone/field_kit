"""
3D Gaussian Random Field generation with a specified power spectrum.
"""

import numpy as np
from numba import njit, prange

from .base_field import RandomField
from .constants import sqrt2
from .utils import (
    enforce_hermitian_symmetry_1d,
    enforce_hermitian_symmetry_2d,
    enforce_hermitian_symmetry_3d,
)


def _make_jit_power_spec(power_spec, **njit_kwargs):
    """
    Wrap a PowerSpectrum object's function in a numba-jitted callable
    that also applies its current normalization, for use inside the
    other njit-compiled functions in this module.

    Parameters
    ----------
    power_spec : PowerSpectrum
        The power spectrum to wrap. Its `func` attribute must be
        numba-jittable.
    **njit_kwargs
        Additional keyword arguments passed to numba.njit when compiling
        `power_spec.func`.

    Returns
    -------
    callable
        A jitted function of the wavenumber k returning the normalized
        power spectrum.
    """
    user_jit = njit(**njit_kwargs)(power_spec.func)
    norm = power_spec.norm

    @njit
    def _power_spec(k):
        return norm * user_jit(k)  # user_func must be jittable!

    return _power_spec


@njit(parallel=True, fastmath=True)
def compute_sigma_1d(kx, nx, power_spec):
    """
    Compute the per-mode standard deviation sqrt(P(k)) on a 1D
    wavenumber grid, with the k=0 mode set to zero.

    Parameters
    ----------
    kx : ndarray
        The wavenumbers along the axis.
    nx : int
        The number of grid points.
    power_spec : callable
        A jitted function of the wavenumber k returning the power
        spectrum (see make_jit_power_spec).

    Returns
    -------
    ndarray
        The standard deviation array, of shape (nx,).
    """
    sigma = np.zeros(nx)

    for i in prange(nx):
        if kx[i] == 0.0:
            sigma[i] = 0.0
        else:
            sigma[i] = np.sqrt(power_spec(kx[i]))

    return sigma


@njit(parallel=True, fastmath=True)
def compute_sigma_2d(kx, ky, nx, ny, power_spec):
    """
    Compute the per-mode standard deviation sqrt(P(|k|)) on a 2D
    wavenumber grid, with the k=0 mode set to zero.

    Parameters
    ----------
    kx : ndarray
        The wavenumbers along the first axis.
    ky : ndarray
        The wavenumbers along the second axis.
    nx : int
        The number of grid points along the first axis.
    ny : int
        The number of grid points along the second axis.
    power_spec : callable
        A jitted function of the wavenumber k returning the power
        spectrum (see make_jit_power_spec).

    Returns
    -------
    ndarray
        The standard deviation array, of shape (nx, ny).
    """
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
    """
    Compute the per-mode standard deviation sqrt(P(|k|)) on a 3D
    wavenumber grid, with the k=0 mode set to zero.

    Parameters
    ----------
    kx : ndarray
        The wavenumbers along the first axis.
    ky : ndarray
        The wavenumbers along the second axis.
    kz : ndarray
        The wavenumbers along the third axis.
    nx : int
        The number of grid points along the first axis.
    ny : int
        The number of grid points along the second axis.
    nz : int
        The number of grid points along the third axis.
    power_spec : callable
        A jitted function of the wavenumber k returning the power
        spectrum (see make_jit_power_spec).

    Returns
    -------
    ndarray
        The standard deviation array, of shape (nx, ny, nz).
    """
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


class GaussianRandomField(RandomField):
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
        super().__init__(left_edge, right_edge, ddims, seed)
        self.pspec = _make_jit_power_spec(power_spec)
        self.sigma = None

    def _compute_sigma(self):
        if self.ndim == 1:
            sigma = compute_sigma_1d(self.kx, self.ddims[0], self.pspec)
        elif self.ndim == 2:
            sigma = compute_sigma_2d(
                self.kx, self.ky, self.ddims[0], self.ddims[1], self.pspec
            )
        else:
            sigma = compute_sigma_3d(
                self.kx,
                self.ky,
                self.kz,
                self.ddims[0],
                self.ddims[1],
                self.ddims[2],
                self.pspec,
            )
        sigma /= np.sqrt(np.prod(self.width))
        self.sigma = sigma

    def _generate_field_realization(self):
        if self.sigma is None:
            self._compute_sigma()

        v_real = self.prng.normal(size=self.ddims) * self.sigma
        v_imag = self.prng.normal(size=self.ddims) * self.sigma

        v = v_real + 1j * v_imag
        v /= sqrt2

        if self.ndim == 1:
            v = enforce_hermitian_symmetry_1d(v)
        elif self.ndim == 2:
            v = enforce_hermitian_symmetry_2d(v)
        else:
            v = enforce_hermitian_symmetry_3d(v)

        return v
