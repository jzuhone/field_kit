# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Python library for generating Gaussian random fields (GRFs) with a
specified power spectrum and analyzing fields via FFT-based power spectra,
divergence/curl, and windowing — in 1, 2, or 3 dimensions. Used for
astrophysical/cosmological applications (docstrings use kpc for grid
coordinates). Core package: `field_kit/` (flat layout, no subpackages):
`base_field.py`, `gaussian_random_field.py`, `power_spectra.py`,
`fourier_analysis.py`, `utils.py`, `constants.py`.

## Commands

- Install deps: `uv sync` (add `--group docs` for the heavier deps needed by
  `examples/` notebooks and the documentation build: matplotlib, pandas, yt,
  h5py, pooch, sphinx, myst-nb, pydata-sphinx-theme)
- Run tests: `uv run pytest`
- Lint: `uv run ruff check field_kit/`
- Format: `uv run ruff format field_kit/`
- Build docs: `uv run --group docs sphinx-build -b html docs docs/_build/html`

## Conventions and gotchas

- **Wavenumber convention**: this codebase uses the `k = 2π·f` convention
  throughout (`constants.two_pi`, `dk = 2π/width`), not the bare-frequency
  convention. Any new k-space code must stay consistent with this.
- **Custom FFT normalization**: `FourierAnalysis.fftn`/`ifftn` multiply/divide
  by the cell volume `dV` before calling `scipy.fft`. Don't mix raw
  `scipy.fft` calls with these wrapped methods — the amplitudes won't match.
- **`FFTArray`** is an `ndarray` subclass carrying a `delta` (grid spacing)
  attribute. Several `FourierAnalysis` methods (`ifftn`, `integrate_kspace`)
  raise `TypeError` if given a plain array instead of an `FFTArray`.
- **numba is a hard dependency**, not optional — `gaussian_random_field.py`
  and `utils.py` use `@njit(parallel=True, fastmath=True)` with no
  pure-Python fallback. Hermitian symmetry for real-valued GRF output is
  enforced manually via jitted loops (`utils.py`), not via `rfft`/`irfft`.
- **Divergence-free projection** in `base_field.py` renormalizes by a
  hardcoded, dimension-specific factor (`{2: 2**0.5, 3: 1.5**0.5}`) to
  preserve power spectrum amplitude — this is a derived constant, not
  arbitrary; don't "simplify" it away.
- Docstrings are NumPy-style (`Parameters\n----------`) where present, but
  many methods have none — match existing style when adding new ones rather
  than introducing a different convention.
- No type hints are used anywhere in the source; don't add them piecemeal to
  individual functions unless asked.
- `examples/*.ipynb` are illustrative, manually-run notebooks (not executed
  in CI/tests). Use `/verify-examples` to run them headlessly after changes
  that touch `field_kit/` internals.
- `examples/*.ipynb` are also rendered as tutorial pages in `docs/` (via a
  `docs/examples -> ../examples` symlink and `nb_execution_mode = "off"` in
  `docs/conf.py`, so the doc build uses each notebook's saved outputs rather
  than re-running it). Each notebook needs a top-level markdown title cell
  (`# Some Title`) — without one, Sphinx's toctree can't link to it by name.
  Add new example notebooks to the toctree in `docs/tutorials/index.md`.
