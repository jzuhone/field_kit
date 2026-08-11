from importlib.metadata import version as _pkg_version

project = "kspace"
copyright = "2026, John ZuHone"
author = "John ZuHone"
version = _pkg_version("kspace")
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "examples/.jupytext-sync-ipynb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath", "amsmath"]

# Notebooks already carry their own saved outputs (some require downloading
# simulation data via yt/pooch, which we don't want the doc build to do);
# render those outputs as-is rather than re-executing.
nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/jzuhone/kspace",
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/logo-icon.svg",
        "image_dark": "_static/logo-icon.svg",
        "text": f"kspace {version}",
    },
}
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
