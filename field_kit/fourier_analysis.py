import numpy as np
from scipy.fft import fftfreq, fftshift, fftn


def window_data(data, filter_function="tukey", **kwargs):
    """
    https://stackoverflow.com/questions/27345861/extending-1d-function-across-3-dimensions-for-data-windowing

    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
        Input data to be windowed, modified in place.
    filter_function : str
        Function can accept one argument: the window length, but
        all other keyword arguments are passed to the function.
        Default: tukey
    """
    import scipy.signal.windows

    filter_function = getattr(scipy.signal.windows, filter_function)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [
            1,
        ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        np.power(window, (1.0 / data.ndim), out=window)
        data *= window


class FFTArray:
    def __init__(self, x, delta, **kwargs):
        self.x = fftn(x, **kwargs)
        self.shape = x.shape
        self.delta = delta

    def __array__(self, dtype=None, copy=None):
        return self.x


class FourierAnalysis:
    def __init__(self, width, ddims):
        self.width = np.atleast_1d(width)
        self.ddims = np.atleast_1d(ddims).astype("int")
        self.delta = self.width / self.ddims
        self.ndims = self.ddims.size
        self.shape = tuple(np.insert(self.ddims, 0, self.ndims))

    def _make_wavenumbers(self):
        # Shift the wavenumbers so that the zero is at the center
        # of the transformed image and compute the grid
        kvec = [fftshift(fftfreq(self.ddims[0], d=self.delta[0]))]
        if self.ndims > 1:
            kvec.append(fftshift(fftfreq(self.ddims[1], d=self.delta[1])))
        if self.ndims > 2:
            kvec.append(fftshift(fftfreq(self.ddims[2], d=self.delta[2])))
        self._kvec = np.array(np.meshgrid(*kvec, indexing="ij"))
        self._kk = (self._kvec**2).sum(axis=0)
        self._kmag = np.sqrt(self._kk)

    _kvec = None
    _kmag = None

    @property
    def kvec(self):
        if self._kvec is None:
            self._make_wavenumbers()
        return self._kvec

    @property
    def kx(self):
        return self.kvec[0, ...]

    @property
    def ky(self):
        return self.kvec[1, ...]

    @property
    def kz(self):
        return self.kvec[2, ...]

    @property
    def kmag(self):
        if self._kmag is None:
            self._make_wavenumbers()
        return self._kmag

    def _check_data(self, data):
        if data.shape != self.shape[1:]:
            raise ValueError(
                "Incompatible array shape for this FourierAnalysis instance!"
            )
        if not np.isclose(data.delta, self.delta).all():
            raise ValueError(
                "Incompatible cell spacing for this FourierAnalysis instance!"
            )

    def fftn(self, x, **kwargs):
        if x.shape != self.shape[1:]:
            raise ValueError(
                "Incompatible array shape for this FourierAnalysis instance!"
            )
        return FFTArray(x, self.delta, **kwargs)

    def generate_fd_wvs(self, diff_type):
        if diff_type == "central":
            diff_func = lambda k, dx: np.sin(2.0 * np.pi * k * dx) / dx
        elif diff_type == "forward":
            diff_func = lambda k, dx: -1j * np.exp(2.0 * np.pi * 1j * k * dx - 1.0) / dx
        else:
            raise NotImplementedError()
        kd = diff_func(
            self.kmag, np.expand_dims(self.delta, axis=tuple(range(1, self.ndims + 1)))
        )
        kkd = np.sqrt((kd * np.conj(kd)).sum(axis=0))
        return kd, kkd

    def divergence_component(self, datax, datay, dataz=None):
        if not isinstance(datax, FFTArray):
            datax = self.fftn(datax)
        self._check_data(datax)
        if not isinstance(datay, FFTArray):
            datay = self.fftn(datay)
        self._check_data(datay)
        kdata = self.kx * datax + self.ky * datay
        if dataz is not None:
            if not isinstance(dataz, FFTArray):
                dataz = self.fftn(dataz)
            self._check_data(dataz)
            kdata += self.kz * dataz
        kdata /= self._kk
        if dataz is None:
            return self.kx * kdata, self.ky * kdata
        else:
            return self.kx * kdata, self.ky * kdata, self.kz * kdata

    def make_powerspec(self, data, nbins):
        if not isinstance(data, FFTArray):
            data = self.fftn(data)
        self._check_data(data)

        P = np.abs(np.prod(self.delta) * fftshift(data)) ** 2 / np.prod(self.width)

        # Set the maximum and minimum limits on the wavenumber bins

        kmin = 1.0 / self.width.max()
        kmax = 1.0 / self.delta.min()

        # Bin up the gridded power spectrum into a 1-D power spectrum

        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins)
        k = np.sqrt(kbins[1:] * kbins[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            Pk = (
                np.histogram(self.kmag, kbins, weights=P)[0]
                / np.histogram(self.kmag, kbins)[0]
            )

        return k[Pk > 0], Pk[Pk > 0]
