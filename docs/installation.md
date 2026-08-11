# Installation

```bash
pip install kspace
```

## Developing `kspace`

`kspace` uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

This installs the core dependencies (`numpy`, `scipy`, `numba`). The example
notebooks under `examples/` need a few heavier, optional dependencies
(`matplotlib`, `pandas`, `yt`, `h5py`, `pooch`) to run — install those with:

```bash
uv sync --group docs
```

The same dependency group also includes everything needed to build this
documentation site (see {doc}`conventions` for the underlying assumptions
`kspace` makes about grids and Fourier conventions before diving into
the API).

## Building the documentation locally

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
