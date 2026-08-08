---
name: verify-examples
description: Execute the example notebooks in examples/ headlessly to check that changes to field_kit internals haven't broken them. User-triggered only (not auto-invoked) since it may download simulation data and can be slow.
disable-model-invocation: true
---

Run each notebook in `examples/` headlessly with `nbconvert` using the
project's environment (which needs the `docs` dependency group for
matplotlib/yt/h5py/pooch):

```bash
uv sync --group docs
uv run --group docs jupyter nbconvert --to notebook --execute --output-dir /tmp/verify-examples examples/*.ipynb
```

Report which notebooks succeeded and which raised errors, including the
exception traceback for any failure (found in the executed notebook's output
cells, or in nbconvert's stderr). Do not modify the notebooks in `examples/`
themselves — write executed copies to `/tmp/verify-examples` only.

Notes:
- `tng_example.ipynb` and `sloshing_example.ipynb` may download simulation
  data via `pooch`/`yt` on first run — this can take a while and requires
  network access.
- If a notebook fails, check first whether it's a real regression in
  `field_kit/` vs. a stale/outdated notebook (e.g. referencing a renamed
  function or removed argument) before treating it as a bug in the library.
