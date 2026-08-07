# field_kit

`field_kit` generates and analyzes Gaussian random fields (GRFs) and their
power spectra on regular grids, in 1, 2, or 3 dimensions. It's built for
astrophysical and cosmological applications where a field is specified by a
power spectrum in Fourier space and realized on a real-space grid — for
example, synthetic turbulent velocity or magnetic fields with a prescribed
power-law spectrum.

Core features:

- Generate scalar or vector GRF realizations from an arbitrary power
  spectrum ({class}`~field_kit.GaussianRandomField`), including
  divergence-free vector fields.
- Built-in power spectrum models ({class}`~field_kit.PowerLaw`,
  {class}`~field_kit.PowerLawBetaModel`, {class}`~field_kit.DoublePowerLaw`),
  or supply your own callable via {class}`~field_kit.PowerSpectrum`.
- FFT-based analysis ({class}`~field_kit.FourierAnalysis`): binned power
  spectra, divergence and curl of vector fields, windowing to reduce FFT
  boundary effects, and vector-potential/curl inversion.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
conventions
tutorials/index
api/index
```
