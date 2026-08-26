"""Sphinx configuration for the BacLCT documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "BacLCT"
author = "Moritz Kunzmann"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
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

# allow local builds without the full runtime stack installed; autodoc only
# needs to import modules far enough to read signatures and docstrings. the
# baclct env (linux-64 via pixi) has all of these natively, so listing them
# here only matters for previews on machines without the runtime deps.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torch_geometric",
    "lightning",
    "torchmetrics",
    "tensorboard",
    "polars",
    "numpy",
    "pyarrow",
    "rustworkx",
    "dask",
    "scipy",
    "skimage",
    "sklearn",
    "fastremap",
    "joblib",
    "networkx",
    "tifffile",
    "tqdm",
    "yaml",
    "hydra",
    "omegaconf",
    "matplotlib",
    "seaborn",
    "cmap",
    "ctc_metrics",
    # napari stack, which the docs env does not install
    "napari",
    "magicgui",
    "qtpy",
    "superqt",
]

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
