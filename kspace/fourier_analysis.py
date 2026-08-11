import numpy as np
import numpy.ma as ma
from scipy.fft import fftfreq, fftn, ifftn


class FFTArray(np.ndarray):
    """
    An ndarray subclass that carries the grid spacing `delta` it was
    transformed with as metadata, so downstream methods (e.g.
    FourierAnalysis.ifftn) can validate that arrays being combined share
    the same grid.

    Parameters
    ----------
    input_array : array-like
        The array data.
    delta : array-like, optional
        The grid spacing along each dimension.
    """

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
        distinct -k mirror) is dropped from the output.

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
    """
    FFT-based analysis of scalar and vector fields on a regular grid:
    power spectra, divergence/curl, and vector-potential inversion.

    Fourier transforms are always taken with the k = 2*pi*f convention
    and normalized by the cell volume dV (see fftn/ifftn) rather than
    scipy.fft's default normalization.

    Parameters
    ----------
    width : array-like
        The physical size of the box [kpc] along each dimension.
    ddims : array-like
        The number of grid cells along each dimension.
    """

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
        """
        ndarray : The wavenumber vector k at each grid point, of shape
        (ndim, \\*ddims), using the k = 2*pi*f convention. Computed
        lazily on first access.
        """
        if self._kvec is None:
            self._make_wavenumbers()
        return self._kvec

    @property
    def kx(self):
        """
        ndarray : The x-component of the wavenumber vector at each grid
        point (kvec[0]).
        """
        return self.kvec[0, ...]

    @property
    def ky(self):
        """
        ndarray : The y-component of the wavenumber vector at each grid
        point (kvec[1]).
        """
        return self.kvec[1, ...]

    @property
    def kz(self):
        """
        ndarray : The z-component of the wavenumber vector at each grid
        point (kvec[2]).
        """
        return self.kvec[2, ...]

    @property
    def kmag(self):
        """
        ndarray : The magnitude of the wavenumber vector at each grid
        point, sqrt(kvec . kvec). Computed lazily on first access.
        """
        if self._kmag is None:
            self._make_wavenumbers()
        return self._kmag

    @property
    def khat(self):
        """
        ndarray : The unit wavenumber vector at each grid point,
        kvec / kmag, with the k=0 mode (division by zero) set to zero.
        """
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
        """
        Forward FFT of real-space data, normalized by the cell volume dV
        (F(k) = dV * sum_x f(x) exp(-i k.x)) rather than scipy.fft's
        default normalization.

        Parameters
        ----------
        x : ndarray
            The data to transform, either a scalar field of shape
            `ddims` or a vector field of shape (ndim, \\*ddims).
        **kwargs
            Additional keyword arguments passed to scipy.fft.fftn.

        Returns
        -------
        FFTArray
            The transformed data.
        """
        x = np.asarray(x)
        self._check_data(x)
        if x.ndim == self.ndim + 1:
            axes = tuple(range(1, self.ndim + 1))
        else:
            axes = None
        return FFTArray(fftn(x * self.dV, axes=axes, **kwargs), delta=self.delta)

    def ifftn(self, x, **kwargs):
        """
        Inverse FFT of Fourier-space data, normalized by the cell volume
        dV (the exact inverse of fftn).

        Parameters
        ----------
        x : FFTArray
            The Fourier-space data to transform, either a scalar field
            of shape `ddims` or a vector field of shape
            (ndim, \\*ddims). Must be an FFTArray (not a plain ndarray).
        **kwargs
            Additional keyword arguments passed to scipy.fft.ifftn.

        Returns
        -------
        ndarray
            The real part of the transformed, real-space data.
        """
        if not isinstance(x, FFTArray):
            raise TypeError("Input must be an FFTArray!")
        self._check_data(x)
        if x.ndim == self.ndim + 1:
            axes = tuple(range(1, self.ndim + 1))
        else:
            axes = None
        return ifftn(np.array(x) / self.dV, axes=axes, **kwargs).real

    def generate_waves(self, diff_type):
        """
        Generate the wavenumber array used for spectral differentiation,
        for a choice of finite-difference approximation.

        Parameters
        ----------
        diff_type : str
            Which differencing scheme's wavenumber to use:

            - "continuum": the exact FFT wavenumbers (kvec), i.e. an
              exact spectral derivative.
            - "central": the wavenumber corresponding to a periodic
              central-difference derivative, sin(k*dx)/dx.
            - "forward": the (complex-valued) wavenumber corresponding
              to a periodic forward-difference derivative.

        Returns
        -------
        k : ndarray
            The wavenumber array, of shape (ndim, \\*ddims).
        kmag : ndarray
            The Hermitian modulus sqrt(k . conj(k)) at each grid point.
        """
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
        """
        Project out the compressional (curl-free, longitudinal)
        component of a vector field:
        v_compressional_hat(k) = k_hat (k_hat . v_hat(k)).

        Parameters
        ----------
        data_vec : ndarray or FFTArray
            The vector field to project, of shape (ndim, \\*ddims).
        diff_type : str, optional
            Which wavenumbers to use for the projection (passed to
            generate_waves). Default is "central".
        return_fft : boolean, optional
            If True, return the compressional component in Fourier
            space. Default is False.
        """
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
        """
        Project out the divergence-free (solenoidal) component of a vector
        field: v_solenoidal = v - v_compressional, where v_compressional is
        the component returned by divergence_component.

        Parameters
        ----------
        data_vec : ndarray or FFTArray
            The vector field to project, of shape (ndim, \\*ddims).
        diff_type : str, optional
            Which wavenumbers to use (passed to generate_waves). Default
            is "central".
        return_fft : boolean, optional
            If True, return the solenoidal component in Fourier space.
            Default is False.
        """
        if not isinstance(data_vec, FFTArray):
            data_vec = self.fftn(data_vec)
        else:
            self._check_data(data_vec)
        vc = self.divergence_component(data_vec, diff_type=diff_type, return_fft=True)
        vs = FFTArray(data_vec - vc, delta=self.delta)
        if return_fft:
            return vs
        else:
            return self.ifftn(vs)

    def _spatial_gradient(self, data, delta, axis, periodic):
        if periodic:
            # Central difference with wraparound, i.e. the real-space
            # operator whose Fourier multiplier is generate_waves's
            # "central" wavenumber (sin(k*dx)/dx). Only valid for data on
            # a periodic domain, e.g. an FFT-generated field.
            return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (
                2.0 * delta
            )
        else:
            return np.gradient(data, delta, axis=axis, edge_order=2)

    def divergence_of_field(self, data_vec, periodic=False):
        """
        Compute the divergence of a vector field using real-space finite
        differences.

        Parameters
        ----------
        data_vec : ndarray
            The vector field, of shape (ndim, \\*ddims).
        periodic : boolean, optional
            If True, use periodic (wraparound) central differences instead
            of one-sided differences at the domain boundary. Only
            appropriate for data on a periodic domain (e.g. a field
            generated via FFT) -- for general, non-periodic data the
            default (False) is correct. Default is False.
        """
        if data_vec.shape != self.shape:
            raise ValueError(
                "Incompatible array dimensions for this FourierAnalysis instance!"
            )
        div = self._spatial_gradient(data_vec[0], self.delta[0], 0, periodic)
        if self.ndim > 1:
            div += self._spatial_gradient(data_vec[1], self.delta[1], 1, periodic)
        if self.ndim == 3:
            div += self._spatial_gradient(data_vec[2], self.delta[2], 2, periodic)
        return div

    def curl_of_field(self, data_vec, periodic=False):
        """
        Compute the curl of a vector field using real-space finite
        differences.

        Parameters
        ----------
        data_vec : ndarray
            The vector field, of shape (ndim, \\*ddims).
        periodic : boolean, optional
            If True, use periodic (wraparound) central differences instead
            of one-sided differences at the domain boundary. Only
            appropriate for data on a periodic domain (e.g. a field
            generated via FFT) -- for general, non-periodic data the
            default (False) is correct. Default is False.
        """
        if data_vec.shape != self.shape:
            raise ValueError(
                "Incompatible array dimensions for this FourierAnalysis instance!"
            )
        if self.ndim == 1:
            raise NotImplementedError("You cannot compute the curl in one dimension!")
        dvydx = self._spatial_gradient(data_vec[1], self.delta[0], 0, periodic)
        dvxdy = self._spatial_gradient(data_vec[0], self.delta[1], 1, periodic)
        if self.ndim == 2:
            # The curl of a 2D vector field is a scalar (its out-of-plane component).
            return dvydx - dvxdy
        curl = np.empty_like(data_vec)
        curl[2] = dvydx - dvxdy
        dvzdy = self._spatial_gradient(data_vec[2], self.delta[1], 1, periodic)
        dvydz = self._spatial_gradient(data_vec[1], self.delta[2], 2, periodic)
        curl[0] = dvzdy - dvydz
        dvxdz = self._spatial_gradient(data_vec[0], self.delta[2], 2, periodic)
        dvzdx = self._spatial_gradient(data_vec[2], self.delta[0], 0, periodic)
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
        """
        Compute the gridded power spectrum P(k) = abs(data_hat(k))**2 / V
        of a scalar or vector field.

        Parameters
        ----------
        data : ndarray or FFTArray
            The field, either real-space or already Fourier-transformed.

        Returns
        -------
        FFTArray
            The power spectrum, same shape as the (Fourier-space) input.
        """
        if not isinstance(data, FFTArray):
            data = self.fftn(data)
        else:
            self._check_data(data)

        P = np.abs(data) ** 2 / np.prod(self.width)

        return FFTArray(P, delta=self.delta)

    def integrate_kspace(self, x, axis=None):
        """
        Integrate a gridded quantity over k-space (e.g. to recover a
        total variance from a power spectrum),
        sum(x) * dVk / (2*pi)**ndim.

        Parameters
        ----------
        x : FFTArray
            The gridded quantity to integrate.
        axis : int or tuple of ints, optional
            The axis or axes to integrate over. If None, integrates over
            all spatial axes. Default is None.

        Returns
        -------
        ndarray or float
            The integrated quantity.
        """
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
        """
        Compute the power spectrum of a field and bin it radially in
        the wavenumber magnitude into a 1D power spectrum.

        Parameters
        ----------
        data : ndarray or FFTArray
            The field, either real-space or already Fourier-transformed.
        bins : int or ndarray
            If an int, the number of log-spaced bins to use between the
            smallest and largest wavenumbers resolved by the grid. If an
            ndarray, the bin edges to use directly.

        Returns
        -------
        kbins : ndarray
            The bin edges used.
        Pk : numpy.ma.MaskedArray
            The binned power spectrum, with empty bins masked.
        """
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
