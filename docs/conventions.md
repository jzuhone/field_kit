# Conventions

`kspace` makes a few specific choices about Fourier conventions and grid
layout that matter if you're combining its output with your own FFT code,
or extending the package itself.

## The $2\pi$ wavenumber convention

Wavenumbers are defined as $k = 2\pi f$, where $f$ is the ordinary
(cycles-per-unit-length) frequency from `numpy`/`scipy`'s `fftfreq`. This is
the convention used throughout physics for spatial power spectra. Concretely:

```python
k = 2 * np.pi * np.fft.fftfreq(n, d=delta)
```

and correspondingly `dk = 2π / width` for the spacing between discrete
wavenumbers in a box of a given `width`. If you generate your own arrays to
feed into {class}`~kspace.FourierAnalysis` methods, make sure they use
this same convention — mixing $k$ and $f$ conventions will silently give
wrong amplitudes and mode locations.

## Normal vs. modified wavenumbers

Most of `kspace` — power spectra, `fftn`/`ifftn`, GRF generation — works
with the "normal" wavenumbers described above: the exact FFT frequencies
scaled by $2\pi$, with no dependence on how a derivative might be
discretized.

A few methods that compute spatial derivatives spectrally (divergence,
curl, vector potentials — see
{meth}`~kspace.FourierAnalysis.generate_waves` and the `diff_type`
argument below) instead offer **modified** wavenumbers: $k$ values altered
so that multiplying by $ik$ in Fourier space reproduces a specific
real-space finite-difference stencil exactly, rather than the exact
spectral derivative. `kspace` calls this choice `diff_type`:

- `"continuum"`: the normal, unmodified FFT wavenumbers — an exact
  spectral derivative.
- `"central"`: wavenumbers rescaled to $\sin(k\,dx)/dx$, matching a
  periodic central-difference stencil.
- `"forward"`: wavenumbers rescaled to a (complex-valued) periodic
  forward-difference stencil.

Modified wavenumbers exist so that a spectral derivative can be made to
agree with a real-space finite-difference calculation done elsewhere in a
pipeline (e.g. on a simulation grid); reach for `"continuum"` unless you
specifically need that agreement. See
[Vector potentials and curl inversion](#vector-potentials-and-curl-inversion)
below for the full set of caveats
 — in particular, `"forward"` wavenumbers
are complex and change which dot product (`k·k` vs. `k·k̄`) is correct in
downstream formulas.

## FFT normalization

{meth}`FourierAnalysis.fftn() <kspace.FourierAnalysis.fftn>` and
{meth}`~kspace.FourierAnalysis.ifftn` don't use `scipy.fft`'s default
normalization. They multiply/divide by the cell volume `dV` so that the
discrete transform approximates a continuum Fourier transform:

$$
\hat{f}(\mathbf{k}) \approx \sum_n f(\mathbf{x}_n)\, e^{-i\mathbf{k}\cdot\mathbf{x}_n} \, dV
$$

Don't mix raw `scipy.fft` calls with these wrapped methods on the same
data — the amplitudes won't match.

## `FFTArray`

{class}`~kspace.fourier_analysis.FFTArray` is a thin `ndarray` subclass
that carries the grid spacing (`delta`) as metadata, produced by
`fftn`/`make_powerspec`/etc. Some `FourierAnalysis` methods (`ifftn`,
`integrate_kspace`) require an `FFTArray` specifically and will raise
`TypeError` on a plain array, so that operations aren't silently applied to
data on an incompatible grid.

(power-energy-and-amplitude-spectra)=
## Power, energy, and amplitude spectra

A {class}`~kspace.PowerSpectrum` (`PowerLaw`, `PowerLawBetaModel`,
`DoublePowerLaw`) parametrizes the isotropic power spectrum $P(k)$
directly: calling `power_spec(k)` evaluates $P(k)$. Two related spectra are
also available, and both depend on `ndim` ($d$) because they involve
integrating $P(k)$ over the surface of a $k$-space sphere of radius $k$:

- {meth}`~kspace.PowerSpectrum.E` — the **energy spectrum**, $P(k)$
  integrated over that shell, so that $\int E(k)\,dk$ over some range of
  $k$ gives the variance contributed by that range of scales:

  $$
  E(k) = \frac{1}{(2\pi)^d} \times
  \begin{cases}
    P(k) & d = 1 \\
    2\pi k\, P(k) & d = 2 \\
    4\pi k^2\, P(k) & d = 3
  \end{cases}
  $$

  The $2\pi k$ and $4\pi k^2$ factors are the circumference/surface area of
  the shell (in 1D the "shell" is just the two points $\pm k$, so there's no
  extra geometric factor). {meth}`~kspace.PowerSpectrum.renormalize`
  uses this: it rescales $P(k)$'s normalization so that
  $\int E(k)\,dk = f_{\rm rms}^2$ over the requested range.
- {meth}`~kspace.PowerSpectrum.A` — the **amplitude spectrum**,
  $A(k) = \sqrt{E(k)\,k}$, the Fourier amplitude associated with
  wavenumber $k$.

$P(k)$ — not $E(k)$ or $A(k)$ — is the quantity that
{meth}`~kspace.FourierAnalysis.make_binned_powerspec` recovers from a
field realization; see the next section.

(gridded-vs-binned-power-spectra)=
## Gridded vs. binned power spectra

{meth}`~kspace.FourierAnalysis.make_powerspec` and
{meth}`~kspace.FourierAnalysis.make_binned_powerspec` both compute a
power spectrum from a field, but at different levels of reduction:

- `make_powerspec(data)` returns
  $P(\mathbf{k}) = |\hat{f}(\mathbf{k})|^2 / V$ on the full Fourier grid —
  one value per discrete mode $\mathbf{k}$, as an `FFTArray` of the same
  shape as the (transformed) input. It makes no isotropy assumption, so
  it's the right starting point if you need per-mode or directional power
  rather than a single curve — {meth}`~kspace.FourierAnalysis.make_binned_powerspec`
  and {meth}`~kspace.FourierAnalysis.integrate_kspace`-based
  calculations both build on it.
- `make_binned_powerspec(data, bins)` calls `make_powerspec` internally,
  then histograms the gridded values by $|\mathbf{k}|$ into 1-D bins,
  averaging over all modes in each shell — i.e. it assumes isotropy. The
  result is directly comparable to `power_spec(k)` on a `PowerSpectrum`
  instance (see
  [Power, energy, and amplitude spectra](#power-energy-and-amplitude-spectra)
  above), which is exactly $P(k)$, not $E(k)$ or $A(k)$ — as in the
  {doc}`quickstart` power-spectrum-recovery example. Bins with no grid
  modes in range come back masked (`numpy.ma.masked_invalid`) rather than
  as `0` or `NaN`.

## Divergence-free vector fields

`generate_vector_field_realization(divergence_free=True)` projects out the
longitudinal (along-$\mathbf{k}$) component of each Fourier mode:

$$
\mathbf{B}(\mathbf{k}) \mathrel{-}= \hat{\mathbf{k}}\,(\hat{\mathbf{k}} \cdot \mathbf{B}(\mathbf{k}))
$$

then rescales the result by $\sqrt{n/(n-1)}$ ($n$ = 2 or 3 dimensions) to
restore the per-component power lost by the projection, so the transverse
field still has the same power spectrum $P(k)$ as an unprojected component.

(vector-potentials-and-curl-inversion)=
## Vector potentials and curl inversion

Given a divergence-free vector field $\mathbf{B}$,
{meth}`~kspace.FourierAnalysis.potential_of_field` recovers a vector
potential $\mathbf{A}$ with $\mathbf{B} = \nabla \times \mathbf{A}$ in the
Coulomb gauge ($\nabla \cdot \mathbf{A} = 0$), using

$$
\hat{\mathbf{A}}(\mathbf{k}) = \frac{i\,\mathbf{k} \times \hat{\mathbf{B}}(\mathbf{k})}{\mathbf{k}\cdot\mathbf{k}}
$$

In 2D, $\mathbf{k}$ and $\hat{\mathbf{B}}$ are both 2-vectors, so their cross
product collapses to a scalar — $\mathbf{A}$ is just that out-of-plane
component.

This inversion, and the corresponding {meth}`~kspace.FourierAnalysis.curl_of_field`
/ {meth}`~kspace.FourierAnalysis.divergence_component`, accept a
`diff_type` argument selecting which wavenumbers to use in the underlying
formula (see {meth}`~kspace.FourierAnalysis.generate_waves`):

- `"continuum"` (default): the exact FFT wavenumbers. This is the exact
  algebraic inverse of a curl computed the same way — *not* of
  {meth}`~kspace.FourierAnalysis.curl_of_field`, which differentiates in
  real space via `numpy.gradient` and only agrees with the spectral result
  at low $k$.
- `"central"`: wavenumbers modified to match a real-space central-difference
  stencil ($\sin(k\,dx)/dx$). These remain real-valued, so the formulas above
  apply unchanged.
- `"forward"`: wavenumbers modified to match a forward-difference stencil.
  These are *complex*-valued, and the $\mathbf{k}\cdot\mathbf{k}$ above is
  the plain (unconjugated) self-dot-product $\sum k_i^2$ — not
  $\sum |k_i|^2$ — since that's what the underlying vector identity
  $\mathbf{k}\times(\mathbf{k}\times\mathbf{B}) = \mathbf{k}(\mathbf{k}\cdot\mathbf{B}) - (\mathbf{k}\cdot\mathbf{k})\mathbf{B}$
  requires. The two coincide for the real-valued `"continuum"`/`"central"`
  cases but not for `"forward"`.
