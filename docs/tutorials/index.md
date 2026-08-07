# Tutorials

These notebooks (from `examples/` in the repository) walk through
`field_kit` usage end-to-end, using their previously-saved outputs rather
than being re-executed as part of the doc build (some require downloading
simulation data via `yt`/`pooch`).

```{toctree}
:maxdepth: 1

Generating Gaussian random fields <../examples/grf_example>
Vector field decomposition <../examples/decomp_example>
3D to 1D power spectra <../examples/P3D_to_P1D>
Sloshing cluster power spectra <../examples/sloshing_example>
IllustrisTNG halo power spectra <../examples/tng_example>
```

- **[Generating Gaussian random fields](../examples/grf_example)** — build
  power spectra with different dissipation scales, generate GRF
  realizations, and check the recovered power spectrum against the input
  model.
- **[Vector field decomposition](../examples/decomp_example)** — generate
  vector field realizations and project out the divergence-free
  (solenoidal) component.
- **[3D to 1D power spectra](../examples/P3D_to_P1D)** — reduce a full 3D
  gridded power spectrum to a 1D radially-binned spectrum and compare to
  the analytic integral.
- **[Power spectra from a sloshing cluster simulation](../examples/sloshing_example)** —
  load a `yt` sample dataset, extract a uniform-grid field, and compute its
  power spectrum with {class}`~field_kit.FourierAnalysis`.
- **[Power spectra from an IllustrisTNG halo](../examples/tng_example)** —
  same as above, applied to a magnetic field from a cosmological
  simulation.
