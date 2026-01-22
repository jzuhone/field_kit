import numpy as np
from scipy.fft import fftfreq, fftshift, fftn, ifftn


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


class FFTArray(np.ndarray):

    def __new__(cls, input_array, delta=None):
        # Create the standard ndarray instance
        obj = np.asarray(input_array).view(cls)

        # Attach the metadata to the new instance
        obj.delta = delta
        return obj

    def __array_finalize__(self, obj):
        # Standard check to see if we are being called from explicit constructor
        if obj is None: return

        # Copy properties from the parent (obj) to the child (self)
        # We use getattr(..., None) because 'obj' might be a plain numpy array
        # (e.g. if you do: plain_arr * custom_arr)
        self.delta = getattr(obj, 'delta', None)


class FourierAnalysis:
    def __init__(self, width, ddims):
        self.width = np.atleast_1d(width)
        self.ddims = np.atleast_1d(ddims).astype("int")
        self.delta = self.width / self.ddims
        self.dV = np.prod(self.delta)
        self.ndim = self.ddims.size
        self.shape = tuple(np.insert(self.ddims, 0, self.ndim))

    def _make_wavenumbers(self):
        # Shift the wavenumbers so that the zero is at the center
        # of the transformed image and compute the grid
        kvec = [fftshift(fftfreq(self.ddims[0], d=self.delta[0]))]
        if self.ndim > 1:
            kvec.append(fftshift(fftfreq(self.ddims[1], d=self.delta[1])))
        if self.ndim > 2:
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

    @property
    def khat(self):
        return np.nan_to_num(self._kvec/self._kmag)

    def _check_data(self, data):
        if len(data.shape) == self.ndim+1:
            self_shape = self.shape
        elif len(data.shape) == self.ndim:
            self_shape = self.shape[1:]
        else:
            raise ValueError("Incompatible array dimensions for this FourierAnalysis instance!")
        if data.shape != self_shape:
            raise ValueError(
                "Incompatible array shape for this FourierAnalysis instance!"
            )
        if hasattr(data, "delta"):
            if not np.isclose(data.delta, self.delta).all():
                raise ValueError(
                    "Incompatible cell spacing for this FourierAnalysis instance!"
                )

    def fftn(self, x, **kwargs):
        x = np.asarray(x)
        self._check_data(x)
        if len(x.shape) == self.ndim+1:
            axes = tuple(range(1, self.ndim+1))
        else:
            axes = None
        return FFTArray(fftn(x, axes=axes, **kwargs), delta=self.delta)

    def ifftn(self, x, **kwargs):
        if not isinstance(x, FFTArray):
            raise TypeError("Input must be an FFTArray!")
        self._check_data(x)
        if len(x.shape) == self.ndim+1:
            axes = tuple(range(1, self.ndim+1))
        else:
            axes = None
        return ifftn(np.array(x), axes=axes, **kwargs).real
        
    def generate_waves(self, diff_type):
        if diff_type == "continuum":
            return self.kvec, self.kmag
        elif diff_type == "central":
            diff_func = lambda k, dx: np.sin(2.0 * np.pi * k * dx) / dx
        elif diff_type == "forward":
            diff_func = lambda k, dx: -1j * np.exp(2.0 * np.pi * 1j * k * dx - 1.0) / dx
        else:
            raise NotImplementedError()
        k = diff_func(
            self.kvec, np.expand_dims(self.delta, axis=tuple(range(1, self.ndim + 1)))
        )
        kmag = np.sqrt((k * np.conj(k)).sum(axis=0))
        return k, kmag

    def divergence_component(self, data_vec, diff_type="central", return_fft=False):
        if not isinstance(data_vec, FFTArray):
            data_vec = self.fftn(data_vec)
        else:
            self._check_data(data_vec)
        k, kmag = self.generate_waves(diff_type)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = k*np.sum(k * data_vec, axis=0)/(kmag*kmag)
        np.nan_to_num(ret, copy=False)
        if return_fft:
            return ret
        else:
            return ifftn(ret, axes=tuple(range(1, self.ndim+1))).real
    
    def divergence_of_field(self, data_vec):
        if data_vec.shape != self.shape:
            raise ValueError("Incompatible array dimensions for this FourierAnalysis instance!")
        div = np.gradient(data_vec[0], self.delta[0], axis=0, edge_order=2)
        if self.ndim > 1:
            div += np.gradient(data_vec[1], self.delta[1], axis=1, edge_order=2)
        if self.ndim == 3:
            div += np.gradient(data_vec[2], self.delta[2], axis=2, edge_order=2)
        return div

    def curl_of_field(self, data_vec):
        if data_vec.shape != self.shape:
            raise ValueError("Incompatible array dimensions for this FourierAnalysis instance!")
        if self.ndim == 1:
            raise NotImplementedError("You cannot compute the curl in one dimension!")
        curl = np.empty_like(data_vec)
        dvydx = np.gradient(data_vec[1], self.delta[0], axis=0, edge_order=2)
        dvxdy = np.gradient(data_vec[0], self.delta[1], axis=1, edge_order=2)
        curl[2] = dvydx - dvxdy
        if self.ndim == 3:
            dvzdy = np.gradient(data_vec[2], self.delta[1], axis=1, edge_order=2)
            dvydz = np.gradient(data_vec[1], self.delta[2], axis=2, edge_order=2)
            curl[0] = dvzdy - dvydz
            dvxdz = np.gradient(data_vec[0], self.delta[2], axis=2, edge_order=2)
            dvzdx = np.gradient(data_vec[2], self.delta[0], axis=0, edge_order=2)
            curl[1] = dvxdz - dvzdx
        return curl

    def make_powerspec(self, data, nbins):
        if not isinstance(data, FFTArray):
            data = self.fftn(data)
        else:
            self._check_data(data)

        P = np.abs(self.dV * fftshift(data)) ** 2 / np.prod(self.width)

        # Set the maximum and minimum limits on the wavenumber bins

        kmin = 1.0 / self.width.max()
        kmax = 1.0 / self.delta.min()

        # Bin up the gridded power spectrum into a 1-D power spectrum

        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins)
        kmid = np.sqrt(kbins[1:] * kbins[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            Pk = (
                np.histogram(self.kmag, kbins, weights=P)[0]
                / np.histogram(self.kmag, kbins)[0]
            )

        return kmid[Pk > 0], Pk[Pk > 0]
