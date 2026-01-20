"""
3D Gaussian Random Field generation with a specified power spectrum.
"""

import numpy as np
from numba import njit, prange


sqrt2 = 2.0**0.5


def make_jit_power_spec(power_spec, **njit_kwargs):
    user_jit = njit(**njit_kwargs)(power_spec.func)
    norm = power_spec.norm
    @njit
    def _power_spec(k):
        return norm*user_jit(k)  # user_func must be jittable!
    return _power_spec


@njit(parallel=True, fastmath=True)
def compute_pspec_1d(kx, nx, power_spec):
    sigma = np.zeros(nx)

    for i in prange(nx):
        if kx[i] == 0.0:
            sigma[i] = 0.0
        else:
            sigma[i] = np.sqrt(power_spec(kx[i]))

    return sigma

@njit(parallel=True, fastmath=True)
def compute_pspec_2d(kx, ky, nx, ny, power_spec):
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
def compute_pspec_3d(kx, ky, kz, nx, ny, nz, power_spec):
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


def make_gaussian_random_field(
    left_edge, right_edge, ddims, power_spec, seed=None
):
    """
    Parameters
    ----------
    left_edge : array-like
        The lower edge of the box [kpc] for each of the dimensions.
    right_edge : array-like
        The upper edge of the box [kpc] for each of the dimensions.
    ddims : array-like
        The number of grids in each of the axes.
    power_spec : callable
        The power spectrum function for the field.
    seed : int, optional
        Random seed for reproducibility.
    """

    prng = np.random.default_rng(seed=seed)

    left_edge = np.atleast_1d(left_edge).astype("float64")
    right_edge = np.atleast_1d(right_edge).astype("float64")
    ddims = np.atleast_1d(ddims).astype("int")
    width = right_edge - left_edge
    deltas = width / ddims
    ndim = left_edge.size
    if ndim == 1:
        dx = deltas[0]
        nx = ddims[0]
    elif ndim == 2:
        dx, dy = deltas
        nx, ny = ddims
    else:
        dx, dy, dz = deltas
        nx, ny, nz = ddims
    kx = np.arange(nx, dtype="float64")
    kx[kx > nx // 2] = kx[kx > nx // 2] - nx
    kx /= nx * dx    
    if ndim > 1:
        ky = np.arange(ny, dtype="float64")
        ky[ky > ny // 2] = ky[ky > ny // 2] - ny
        ky /= ny * dy
    if ndim == 3:
        kz = np.arange(nz, dtype="float64")
        kz[kz > nz // 2] = kz[kz > nz // 2] - nz
        kz /= nz * dz

    pspec = make_jit_power_spec(power_spec)

    v_real = prng.normal(size=ddims)
    v_imag = prng.normal(size=ddims)
        
    if ndim == 1:
        sigma = compute_pspec_1d(kx, nx, pspec)
    elif ndim == 2:
        sigma = compute_pspec_2d(kx, ky, nx, ny, pspec)
    else:
        sigma = compute_pspec_3d(kx, ky, kz, nx, ny, nz, pspec) 
    sigma /= np.sqrt(np.prod(width))
    
    v_real *= sigma
    v_imag *= sigma

    v = v_real + 1j * v_imag
    v /= sqrt2

    if ndim == 1:
        v = enforce_hermitian_symmetry_1d(v)
    elif ndim == 2:
        v = enforce_hermitian_symmetry_2d(v)
    else:
        v = enforce_hermitian_symmetry_3d(v)

    v = np.fft.ifftn(v, norm="forward")

    return v.real
