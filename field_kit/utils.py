import numpy as np
from numba import njit


@njit(fastmath=True)
def enforce_hermitian_symmetry_1d(f_hat):
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
                    f_hat[i, j, k] = complex(
                        f_hat[i, j, k].real, 0.0
                    )  # ensure real on Nyquist
    return f_hat
