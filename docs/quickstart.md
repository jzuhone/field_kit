# Quickstart

This walks through generating a Gaussian random field with a prescribed
power spectrum, then recovering its power spectrum from the realization to
check that the two match.

## Define a power spectrum

```python
from field_kit import PowerLawBetaModel

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
from field_kit import GaussianRandomField

left_edge = np.array([0.0, 0.0, 0.0])
right_edge = np.array([750.0, 750.0, 750.0])
ddims = [256, 256, 256]

grf = GaussianRandomField(left_edge, right_edge, ddims, power_spec, seed=10)
field = grf.generate_scalar_field_realization()
```

`field` is a real-valued NumPy array of shape `(256, 256, 256)`.

## Recover and check the power spectrum

```python
from field_kit import FourierAnalysis

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

## Vector fields

```python
# A 3-component vector field, e.g. a synthetic turbulent velocity field
vfield = grf.generate_vector_field_realization()

# Or a divergence-free ("solenoidal") one
vfield_solenoidal = grf.generate_vector_field_realization(divergence_free=True)
```

See {doc}`conventions` for the Fourier conventions this package uses, and
{doc}`tutorials/index` for complete, runnable examples.
