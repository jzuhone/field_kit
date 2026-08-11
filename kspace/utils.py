import numpy as np
from numba import njit


@njit(fastmath=True)
def enforce_hermitian_symmetry_1d(f_hat):
    """
    Enforce Hermitian symmetry (f_hat[-k] = conj(f_hat[k])) on a 1D array
    of complex Fourier coefficients in place, so its inverse FFT is
    real-valued. The Nyquist mode (which has no distinct -k mirror) is
    forced to be purely real.

    Parameters
    ----------
    f_hat : ndarray
        The complex Fourier coefficients, modified in place.

    Returns
    -------
    ndarray
        The same array, with Hermitian symmetry enforced.
    """
    nx = f_hat.shape[0]
    for i in range(nx // 2 + 1):  # only loop over half
        i_ = (-i) % nx
        if i != i_:
            f_hat[i_] = np.conj(f_hat[i])
        else:
            f_hat[i] = complex(f_hat[i].real, 0.0)  # ensure real on Nyquist
    return f_hat


@njit(fastmath=True)
def enforce_hermitian_symmetry_2d(f_hat):
    """
    Enforce Hermitian symmetry (f_hat[-k] = conj(f_hat[k])) on a 2D array
    of complex Fourier coefficients in place, so its inverse FFT is
    real-valued. The Nyquist modes (which have no distinct -k mirror) are
    forced to be purely real.

    Parameters
    ----------
    f_hat : ndarray
        The complex Fourier coefficients, modified in place.

    Returns
    -------
    ndarray
        The same array, with Hermitian symmetry enforced.
    """
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
    """
    Enforce Hermitian symmetry (f_hat[-k] = conj(f_hat[k])) on a 3D array
    of complex Fourier coefficients in place, so its inverse FFT is
    real-valued. The Nyquist modes (which have no distinct -k mirror) are
    forced to be purely real.

    Parameters
    ----------
    f_hat : ndarray
        The complex Fourier coefficients, modified in place.

    Returns
    -------
    ndarray
        The same array, with Hermitian symmetry enforced.
    """
    nx, ny, nz = f_hat.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz // 2 + 1):  # only loop over half
                i_, j_, k_ = (-i) % nx, (-j) % ny, (-k) % nz
                if (i, j, k) != (i_, j_, k_):
                    f_hat[i_, j_, k_] = np.conj(f_hat[i, j, k])
                else:
                    f_hat[i, j, k] = complex(
                        f_hat[i, j, k].real, 0.0
                    )  # ensure real on Nyquist
    return f_hat
