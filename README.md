<img src="https://raw.githubusercontent.com/jzuhone/kspace/main/docs/_static/logo.svg" alt="kspace" height="72">

Tools for generating and analyzing Gaussian random fields (GRFs) and their
power spectra on regular grids, in 1, 2, or 3 dimensions. Built for
astrophysical/cosmological applications where fields are specified by a
power spectrum in Fourier space and realized on a real-space grid.

Features:

- Generate scalar or vector Gaussian random field realizations from an
  arbitrary power spectrum (`GaussianRandomField`), including
  divergence-free vector fields.
- Built-in power spectrum models (`PowerLaw`, `PowerLawBetaModel`), or
  supply your own callable.
- FFT-based analysis (`FourierAnalysis`): binned power spectra, divergence
  and curl of vector fields, windowing to reduce FFT boundary effects.

## Install

```bash
pip install kspace
```

For development, clone the repo and use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

To also install the heavier dependencies used by the example notebooks
(`matplotlib`, `pandas`, `yt`, `h5py`, `pooch`):

```bash
uv sync --group docs
```

## Quick example

```python
import numpy as np
from kspace import GaussianRandomField, FourierAnalysis, PowerLawBetaModel

# A power-law power spectrum with large- and small-scale cutoffs
power_spec = PowerLawBetaModel(l_min=10.0, l_max=200.0, alpha=-11.0 / 3.0)
power_spec.renormalize(f_rms=10.0)

le = np.array([0.0, 0.0, 0.0])
re = np.array([750.0, 750.0, 750.0])
ddims = [256, 256, 256]

grf = GaussianRandomField(le, re, ddims, power_spec, seed=10)
field = grf.generate_scalar_field_realization()

fa = FourierAnalysis(re - le, ddims)
kbins, pk = fa.make_binned_powerspec(field, nbins=60)
```

See `docs/examples/` for more complete, runnable notebooks (GRF generation,
vector field decomposition, power spectrum estimation, and analysis of
simulation data).
