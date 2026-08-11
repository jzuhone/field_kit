# Quickstart

This walks through generating a Gaussian random field with a prescribed
power spectrum, then recovering its power spectrum from the realization to
check that the two match.

## Define a power spectrum

```python
from kspace import PowerLawBetaModel

# A power-law power spectrum with large- and small-scale cutoffs:
#   l_min, l_max set the cutoff scales; alpha is the power-law index.
power_spec = PowerLawBetaModel(l_min=10.0, l_max=200.0, alpha=-11.0 / 3.0)

# Normalize so the field has a given RMS amplitude
power_spec.renormalize(f_rms=10.0)
```

`l_min`/`l_max` and the grid coordinates below are in whatever length unit
you choose (kpc is a common choice for the astrophysical use case this
package targets, but nothing enforces that).

## Generate a field realization

```python
import numpy as np
from kspace import GaussianRandomField

left_edge = np.array([0.0, 0.0, 0.0])
right_edge = np.array([750.0, 750.0, 750.0])
ddims = [256, 256, 256]

grf = GaussianRandomField(left_edge, right_edge, ddims, power_spec, seed=10)
field = grf.generate_scalar_field_realization()
```

`field` is a real-valued NumPy array of shape `(256, 256, 256)`.

## Recover and check the power spectrum

```python
from kspace import FourierAnalysis

fa = FourierAnalysis(right_edge - left_edge, ddims)
kbins, pk = fa.make_binned_powerspec(field, nbins=60)

# Compare to the input spectrum at the bin centers
k_centers = np.sqrt(kbins[1:] * kbins[:-1])
measured = pk
expected = power_spec(k_centers)
```

Plotting `measured` and `expected` against `k_centers` on a log-log axis
should show good agreement away from the largest and smallest scales
(where the finite grid and box size limit the achievable dynamic range).

## Forward and inverse FFTs

`fa` above already wraps the FFT machinery used internally by
`make_binned_powerspec`; you can also call it directly. `fftn`/`ifftn`
aren't drop-in replacements for `scipy.fft`'s functions of the same name —
they use the $k = 2\pi f$ convention and normalize by the cell volume `dV`
(see {doc}`conventions`), and `ifftn` requires its input to be the
`FFTArray` that `fftn` returns, not a plain array:

```python
# Forward transform: a real-space field -> an FFTArray in Fourier space
field_hat = fa.fftn(field)

# Inverse transform: exact inverse of fftn, back to real space
field_recovered = fa.ifftn(field_hat)

np.allclose(field, field_recovered)  # True
```

`fftn` also accepts vector fields, of shape `(ndim, *ddims)`, transforming
each component along the spatial axes.

## Vector fields

```python
# A 3-component vector field, e.g. a synthetic turbulent velocity field
vfield = grf.generate_vector_field_realization()

# Or a divergence-free ("solenoidal") one
vfield_solenoidal = grf.generate_vector_field_realization(divergence_free=True)
```

## Divergence and curl

`FourierAnalysis.divergence_of_field`/`curl_of_field` compute these in real
space via finite differences. Since `vfield` came from an FFT-based
realization, its domain is periodic, so pass `periodic=True` to use
wraparound (rather than one-sided) differences at the box edges:

```python
div_v = fa.divergence_of_field(vfield, periodic=True)
curl_v = fa.curl_of_field(vfield, periodic=True)

# vfield_solenoidal was constructed to be divergence-free, so this should
# be close to zero everywhere (up to finite-difference truncation error)
div_v_solenoidal = fa.divergence_of_field(vfield_solenoidal, periodic=True)
```

There are also spectral counterparts, `divergence_component`/
`solenoidal_component`, which project a vector field into its compressional
and solenoidal parts in Fourier space rather than differentiating in real
space — see {doc}`conventions` for how they relate to `divergence_of_field`/
`curl_of_field` and the choice of wavenumber (`diff_type`) they use.

See {doc}`conventions` for the Fourier conventions this package uses, and
{doc}`tutorials/index` for complete, runnable examples.
