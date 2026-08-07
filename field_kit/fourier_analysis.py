import numpy as np
import numpy.ma as ma
from scipy.fft import fftfreq, fftn, ifftn


class FFTArray(np.ndarray):
    def __new__(cls, input_array, delta=None):
        # Create the standard ndarray instance
        obj = np.asarray(input_array).view(cls)

        # Attach the metadata to the new instance
        obj.delta = delta
        return obj

    def __array_finalize__(self, obj):
        # Standard check to see if we are being called from explicit constructor
        if obj is None:
            return

        # Copy properties from the parent (obj) to the child (self)
        # We use getattr(..., None) because 'obj' might be a plain numpy array
        # (e.g. if you do: plain_arr * custom_arr)
        self.delta = getattr(obj, "delta", None)

    def average_symmetric_k(self, axis=None):
        """
        Averages the +k and -k modes of an array along multiple axes, using
        the same (unshifted) layout as scipy.fft.fftn/ifftn: k=0 at index 0,
        and the mode at index i mirrors the one at index (-i) % N. The
        standalone Nyquist mode (index N/2 for even N, which has no
        distinct -k mirror) is dropped from the output, same as before.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            The axes along which to fold. If None, folds over all axes.

        Returns
        -------
        ndarray
            The symmetrically folded array.
        """
        # If no axes specified, fold over all of them
        if axis is None:
            axis = tuple(range(self.ndim))
        # If a single integer is passed, make it a tuple
        elif isinstance(axis, int):
            axis = (axis,)

        folded_arr = self.copy()

        # Sequentially fold over each requested axis
        for ax in axis:
            N = folded_arr.shape[ax]
            n_pos = N - N // 2  # number of k >= 0 modes, excluding Nyquist

            pos_indices = np.arange(n_pos)
            neg_indices = (-pos_indices) % N

            pos = np.take(folded_arr, pos_indices, axis=ax)
            neg = np.take(folded_arr, neg_indices, axis=ax)

            folded_arr = 0.5 * (pos + neg)

        return folded_arr


class FourierAnalysis:
    def __init__(self, width, ddims):
        self.width = np.atleast_1d(width)
        self.ddims = np.atleast_1d(ddims).astype("int")
        self.delta = self.width / self.ddims
        self.dV = np.prod(self.delta)
        self.ndim = self.ddims.size
        self.shape = tuple(np.insert(self.ddims, 0, self.ndim))
        self.dk = 2.0 * np.pi / self.width
        self.dVk = np.prod(self.dk)
        self.geom_factor = 1.0 / (2.0 * np.pi) ** self.ndim

    def _make_wavenumbers(self):
        # Same (unshifted) layout as scipy.fft.fftn/ifftn: zero-frequency
        # at index 0, matching the FFTArray data these are combined with.
        kvec = [2.0 * np.pi * fftfreq(self.ddims[0], d=self.delta[0])]
        if self.ndim > 1:
            kvec.append(2.0 * np.pi * fftfreq(self.ddims[1], d=self.delta[1]))
        if self.ndim > 2:
            kvec.append(2.0 * np.pi * fftfreq(self.ddims[2], d=self.delta[2]))
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
        return np.nan_to_num(self._kvec / self._kmag)

    def _check_data(self, data):
        if data.ndim == self.ndim + 1:
            self_shape = self.shape
        elif data.ndim == self.ndim:
            self_shape = self.shape[1:]
        else:
            raise ValueError(
                "Incompatible array dimensions for this FourierAnalysis instance!"
            )
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
        if x.ndim == self.ndim + 1:
            axes = tuple(range(1, self.ndim + 1))
        else:
            axes = None
        return FFTArray(fftn(x * self.dV, axes=axes, **kwargs), delta=self.delta)

    def ifftn(self, x, **kwargs):
        if not isinstance(x, FFTArray):
            raise TypeError("Input must be an FFTArray!")
        self._check_data(x)
        if x.ndim == self.ndim + 1:
            axes = tuple(range(1, self.ndim + 1))
        else:
            axes = None
        return ifftn(np.array(x) / self.dV, axes=axes, **kwargs).real

    def generate_waves(self, diff_type):
        if diff_type == "continuum":
            return self.kvec, self.kmag
        elif diff_type == "central":

            def diff_func(k, dx):
                return np.sin(k * dx) / dx
        elif diff_type == "forward":

            def diff_func(k, dx):
                return -1j * np.exp(1j * k * dx - 1.0) / dx
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
            ret = k * np.sum(k * data_vec, axis=0) / (kmag * kmag)
        ret = FFTArray(np.nan_to_num(ret), delta=self.delta)
        if return_fft:
            return ret
        else:
            return self.ifftn(ret)

    def solenoidal_component(self, data_vec, diff_type="central", return_fft=False):
        vc = self.divergence_component(data_vec, diff_type=diff_type, return_fft=return_fft)
        if return_fft:
            return self.fftn(data_vec)-vc
        else:
            return data_vec-vc

    def divergence_of_field(self, data_vec):
        if data_vec.shape != self.shape:
            raise ValueError(
                "Incompatible array dimensions for this FourierAnalysis instance!"
            )
        div = np.gradient(data_vec[0], self.delta[0], axis=0, edge_order=2)
        if self.ndim > 1:
            div += np.gradient(data_vec[1], self.delta[1], axis=1, edge_order=2)
        if self.ndim == 3:
            div += np.gradient(data_vec[2], self.delta[2], axis=2, edge_order=2)
        return div

    def curl_of_field(self, data_vec):
        if data_vec.shape != self.shape:
            raise ValueError(
                "Incompatible array dimensions for this FourierAnalysis instance!"
            )
        if self.ndim == 1:
            raise NotImplementedError("You cannot compute the curl in one dimension!")
        dvydx = np.gradient(data_vec[1], self.delta[0], axis=0, edge_order=2)
        dvxdy = np.gradient(data_vec[0], self.delta[1], axis=1, edge_order=2)
        if self.ndim == 2:
            # The curl of a 2D vector field is a scalar (its out-of-plane component).
            return dvydx - dvxdy
        curl = np.empty_like(data_vec)
        curl[2] = dvydx - dvxdy
        dvzdy = np.gradient(data_vec[2], self.delta[1], axis=1, edge_order=2)
        dvydz = np.gradient(data_vec[1], self.delta[2], axis=2, edge_order=2)
        curl[0] = dvzdy - dvydz
        dvxdz = np.gradient(data_vec[0], self.delta[2], axis=2, edge_order=2)
        dvzdx = np.gradient(data_vec[2], self.delta[0], axis=0, edge_order=2)
        curl[1] = dvxdz - dvzdx
        return curl

    def potential_of_field(self, data_vec, diff_type="continuum", return_fft=False):
        r"""
        Invert B = curl(A) to recover the vector potential A for a
        divergence-free vector field B, in the Coulomb gauge (div(A) = 0):
        A_hat(k) = i(k x B_hat(k)) / (k . k).

        In 3D, A is returned as a 3-component vector field. In 2D, k and
        B_hat are both 2-vectors, so their cross product is a scalar and A
        is just the out-of-plane component.

        Parameters
        ----------
        data_vec : ndarray or FFTArray
            The divergence-free vector field B, of shape (ndim, \*ddims).
        diff_type : str, optional
            Which wavenumbers to use for the inversion (passed to
            generate_waves). Default is "continuum", the exact FFT
            wavenumbers -- this is the exact algebraic inverse of a curl
            computed the same way, not of curl_of_field, which uses
            real-space finite differences and will only agree with this
            at low k.
        return_fft : boolean, optional
            If True, return the potential in Fourier space. Default is False.
        """
        if self.ndim == 1:
            raise NotImplementedError("The vector potential is not defined in 1D.")
        if not isinstance(data_vec, FFTArray):
            data_vec = self.fftn(data_vec)
        else:
            self._check_data(data_vec)
        k, _ = self.generate_waves(diff_type)
        # k . k (unconjugated self-dot-product), not |k|^2 = k . conj(k):
        # the two coincide for real k (continuum/central) but not for the
        # complex-valued k produced by e.g. diff_type="forward", and it's
        # k . k that the underlying vector identity k x (k x B) = k(k.B)
        # - (k.k)B actually requires.
        kk = np.sum(k * k, axis=0)
        if self.ndim == 2:
            cross = k[0] * data_vec[1] - k[1] * data_vec[0]
        else:
            cross = np.cross(k, data_vec, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_hat = 1j * cross / kk
        a_hat = FFTArray(np.nan_to_num(a_hat), delta=self.delta)
        if return_fft:
            return a_hat
        else:
            return self.ifftn(a_hat)

    def window_data(self, data, filter_function="tukey", **kwargs):
        """
        https://stackoverflow.com/questions/27345861/extending-1d-function-across-3-dimensions-for-data-windowing

        Performs an in-place windowing on N-dimensional spatial-domain data.
        This is done to mitigate boundary effects in the FFT.

        Parameters
        ----------
        data : ndarray
            Input data to be windowed, modified in place. Must be either
            the same shape as the FourierAnalysis instance, or be a 2-D or
            3-D vector field with the same shape
        filter_function : str
            Function can accept one argument: the window length, but
            all other keyword arguments are passed to the function.
            Default: tukey
        """
        import scipy.signal.windows

        self._check_data(data)

        filter_function = getattr(scipy.signal.windows, filter_function)
        for axis, axis_size in enumerate(self.shape[1:]):
            # set up shape for numpy broadcasting
            filter_shape = [
                1,
            ] * self.ndim
            filter_shape[axis] = axis_size
            window = filter_function(axis_size, **kwargs).reshape(filter_shape)
            # scale the window intensities to maintain image intensity
            np.power(window, (1.0 / self.ndim), out=window)
            data *= window

    def make_powerspec(self, data):
        if not isinstance(data, FFTArray):
            data = self.fftn(data)
        else:
            self._check_data(data)

        P = np.abs(data) ** 2 / np.prod(self.width)

        return FFTArray(P, delta=self.delta)

    def integrate_kspace(self, x, axis=None):
        if not isinstance(x, FFTArray):
            raise TypeError("Input must be an FFTArray!")
        self._check_data(x)
        if axis is None:
            axis = tuple(range(self.ndim))
        if x.ndim == self.ndim + 1:
            iaxis = tuple(ax + 1 for ax in axis)
        else:
            iaxis = axis
        naxis = len(axis)
        geom_factor = 1.0 / (2.0 * np.pi) ** naxis
        return np.sum(x, axis=iaxis) * np.prod(self.dk[list(axis)]) * geom_factor

    def make_binned_powerspec(self, data, bins):

        P = self.make_powerspec(data)

        # Bin up the gridded power spectrum into a 1-D power spectrum
        if isinstance(bins, int):
            kmin = 2.0 * np.pi * (2.0 / self.width.max())
            kmax = 2.0 * np.pi * (0.5 / self.delta.min())
            kbins = np.logspace(np.log10(kmin), np.log10(kmax), bins + 1)
        elif isinstance(bins, np.ndarray):
            kbins = bins
        n = np.histogram(self.kmag, kbins)[0]

        if P.ndim == self.ndim + 1:
            Pk = []
            for Pi in P:
                Pk.append(np.histogram(self.kmag, kbins, weights=Pi)[0])
            Pk = np.array(Pk)
        else:
            Pk = np.histogram(self.kmag, kbins, weights=P)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            Pk /= n

        return kbins, ma.masked_invalid(Pk)
