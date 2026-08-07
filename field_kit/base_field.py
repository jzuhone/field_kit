import numpy as np
from scipy.fft import fftfreq

from .constants import two_pi
from .fourier_analysis import FFTArray, FourierAnalysis


class BaseField:
    def __init__(self, left_edge, right_edge, ddims):
        self.left_edge = np.atleast_1d(left_edge).astype("float64")
        self.right_edge = np.atleast_1d(right_edge).astype("float64")
        self.ddims = np.atleast_1d(ddims).astype("int")
        self.width = self.right_edge - self.left_edge
        self.delta = self.width / self.ddims
        self.dV = np.prod(self.delta)
        self.ndim = self.left_edge.size
        self.x = np.linspace(
            self.left_edge[0] + 0.5 * self.delta[0],
            self.right_edge[0] - 0.5 * self.delta[0],
            self.ddims[0],
        )
        if self.ndim > 1:
            self.y = np.linspace(
                self.left_edge[1] + 0.5 * self.delta[1],
                self.right_edge[1] - 0.5 * self.delta[1],
                self.ddims[1],
            )
        if self.ndim == 3:
            self.z = np.linspace(
                self.left_edge[2] + 0.5 * self.delta[2],
                self.right_edge[2] - 0.5 * self.delta[2],
                self.ddims[2],
            )

        self.kx = two_pi * fftfreq(self.ddims[0], d=self.delta[0])
        if self.ndim > 1:
            self.ky = two_pi * fftfreq(self.ddims[1], d=self.delta[1])
        if self.ndim == 3:
            self.kz = two_pi * fftfreq(self.ddims[2], d=self.delta[2])


class RandomField(BaseField):
    def __init__(self, left_edge, right_edge, ddims, seed):
        super().__init__(left_edge, right_edge, ddims)
        self.prng = np.random.default_rng(seed=seed)

    def _generate_field_realization(self):
        raise NotImplementedError

    def generate_scalar_field_realization(self, fourier_space=False):
        """
        Generate a random realization of the field as a scalar field.
        The shape of the array will be the size of the grid.

        Parameters
        ----------
        fourier_space : boolean, optional
            If True, return the field in Fourier space (i.e., the
            complex Fourier coefficients). If False, return the
            field in real space. Default is False.
        """
        v = self._generate_field_realization()

        if fourier_space:
            return FFTArray(v * self.dV, delta=self.delta)
        else:
            v = np.fft.ifftn(v, norm="forward")
            return v.real

    def generate_vector_field_realization(
        self, fourier_space=False, divergence_free=False
    ):
        """
        Generate a random realization of the field as a vector field.
        The first index of the NumPy array will have the size of the
        dimension of the field (e.g., 3 for a 3D vector field), and
        the remaining indices will have the size of the grid.

        Parameters
        ----------
        fourier_space : boolean, optional
            If True, return the field in Fourier space (i.e., the
            complex Fourier coefficients). If False, return the
            field in real space. Default is False.
        divergence_free : boolean, optional
            If True, generate a divergence-free vector field. Default is False.
        """
        if self.ndim == 1:
            raise NotImplementedError("Vector fields are not implemented for 1D.")

        field = []
        for _ in range(self.ndim):
            field.append(self._generate_field_realization())
        field = np.asarray(field)

        if divergence_free:
            fa = FourierAnalysis(self.width, self.ddims)
            k, kmag = fa.generate_waves("central")
            with np.errstate(divide="ignore", invalid="ignore"):
                field -= k * np.sum(k * field, axis=0) / (kmag * kmag)
            np.nan_to_num(field, copy=False)
            field *= {2: 2.0**0.5, 3: 1.5**0.5}[
                self.ndim
            ]  # renormalize to preserve power spectrum amplitude

        if fourier_space:
            return FFTArray(field * self.dV, delta=self.delta)
        else:
            field = np.fft.ifftn(
                field, axes=tuple(range(1, self.ndim + 1)), norm="forward"
            )
            return field.real
