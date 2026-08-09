# Tutorials

These notebooks walk through `field_kit` usage end-to-end.

```{toctree}
:maxdepth: 1

Power Spectra and Gaussian Random Fields <../examples/grf_example>
Vector Field Decomposition <../examples/decomp_example>
More Power Spectra Examples <../examples/more_power_spectra>
Computing a Power Spectrum from a Hydrodynamic Simulation <../examples/sim_example>
```

- **[Power Spectra and Gaussian Random Fields](../examples/grf_example)** — build
  power spectra with different dissipation scales, generate GRF
  realizations, and check the recovered power spectrum against the input
  model.
- **[Vector Field Decomposition](../examples/decomp_example)** — generate
  vector field realizations and project out the divergence-free
  (solenoidal) component.
- **[More Power Spectra Examples](../examples/more_power_spectra)** — reduce a full 3D
  gridded power spectrum to a 1D radially-binned spectrum and compare to
  the analytic integral.
- **[Computing a Power Spectrum from a Hydrodynamic Simulation](../examples/sim_example)** —
  load a `yt` sample dataset, extract a uniform-grid field, and compute its
  power spectrum with {class}`~field_kit.FourierAnalysis`.
