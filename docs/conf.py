"""Sphinx configuration for the BacLCT documentation."""

from __future__ import annotations

import os
from importlib.metadata import version as _distribution_version

# napari, magicgui and qtpy import Qt at module level, and RTD and CI have no display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

project = "BacLCT"
author = "Moritz Kunzmann"
copyright = "2026, Moritz Kunzmann"
release = _distribution_version("baclct")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# pull types and defaults from the signature into each Args entry, so
# docstrings stay free of redundant ``(type, default: X)`` boilerplate.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True
typehints_defaults = "comma"
typehints_use_signature = False
typehints_use_signature_return = True
always_use_bars_union = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
