# Conventions

`field_kit` makes a few specific choices about Fourier conventions and grid
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
feed into {class}`~field_kit.FourierAnalysis` methods, make sure they use
this same convention — mixing $k$ and $f$ conventions will silently give
wrong amplitudes and mode locations.

## FFT normalization

{meth}`FourierAnalysis.fftn() <field_kit.FourierAnalysis.fftn>` and
{meth}`~field_kit.FourierAnalysis.ifftn` don't use `scipy.fft`'s default
normalization. They multiply/divide by the cell volume `dV` so that the
discrete transform approximates a continuum Fourier transform:

$$
\hat{f}(\mathbf{k}) \approx \sum_n f(\mathbf{x}_n)\, e^{-i\mathbf{k}\cdot\mathbf{x}_n} \, dV
$$

Don't mix raw `scipy.fft` calls with these wrapped methods on the same
data — the amplitudes won't match.

## `FFTArray`

{class}`~field_kit.fourier_analysis.FFTArray` is a thin `ndarray` subclass
that carries the grid spacing (`delta`) as metadata, produced by
`fftn`/`make_powerspec`/etc. Some `FourierAnalysis` methods (`ifftn`,
`integrate_kspace`) require an `FFTArray` specifically and will raise
`TypeError` on a plain array, so that operations aren't silently applied to
data on an incompatible grid.

## Divergence-free vector fields

`generate_vector_field_realization(divergence_free=True)` projects out the
longitudinal (along-$\mathbf{k}$) component of each Fourier mode:

$$
\mathbf{B}(\mathbf{k}) \mathrel{-}= \hat{\mathbf{k}}\,(\hat{\mathbf{k}} \cdot \mathbf{B}(\mathbf{k}))
$$

then rescales the result by $\sqrt{n/(n-1)}$ ($n$ = 2 or 3 dimensions) to
restore the per-component power lost by the projection, so the transverse
field still has the same power spectrum $P(k)$ as an unprojected component.

## Vector potentials and curl inversion

Given a divergence-free vector field $\mathbf{B}$,
{meth}`~field_kit.FourierAnalysis.potential_of_field` recovers a vector
potential $\mathbf{A}$ with $\mathbf{B} = \nabla \times \mathbf{A}$ in the
Coulomb gauge ($\nabla \cdot \mathbf{A} = 0$), using

$$
\hat{\mathbf{A}}(\mathbf{k}) = \frac{i\,\mathbf{k} \times \hat{\mathbf{B}}(\mathbf{k})}{\mathbf{k}\cdot\mathbf{k}}
$$

In 2D, $\mathbf{k}$ and $\hat{\mathbf{B}}$ are both 2-vectors, so their cross
product collapses to a scalar — $\mathbf{A}$ is just that out-of-plane
component.

This inversion, and the corresponding {meth}`~field_kit.FourierAnalysis.curl_of_field`
/ {meth}`~field_kit.FourierAnalysis.divergence_component`, accept a
`diff_type` argument selecting which wavenumbers to use in the underlying
formula (see {meth}`~field_kit.FourierAnalysis.generate_waves`):

- `"continuum"` (default): the exact FFT wavenumbers. This is the exact
  algebraic inverse of a curl computed the same way — *not* of
  {meth}`~field_kit.FourierAnalysis.curl_of_field`, which differentiates in
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
