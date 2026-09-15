"""Test configuration and shared inputs for the napari plugin."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile

# these tests need a Qt platform; default to offscreen so a headless run works
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def masks() -> np.ndarray:
    """Two rectangular cells, held still over three frames."""
    masks = np.zeros((3, 60, 60), dtype=np.uint16)
    for t in range(3):
        masks[t, 10:20, 10:18] = 1
        masks[t, 30:40, 30:38] = 2
    return masks


@pytest.fixture
def images(masks) -> np.ndarray:
    return (masks > 0).astype(np.uint16) * 100


@pytest.fixture
def sequence(tmp_path: Path, images, masks) -> tuple[Path, Path]:
    """The same pair on disk, as the CLI takes it."""
    image_path, mask_path = tmp_path / "movie.tif", tmp_path / "movie_masks.tif"
    tifffile.imwrite(image_path, images, photometric="minisblack")
    tifffile.imwrite(mask_path, masks, photometric="minisblack")
    return image_path, mask_path


@pytest.fixture
def tracks() -> pl.DataFrame:
    """One two-frame track, as `TrackResult` carries it."""
    return pl.DataFrame(
        {
            "label": [1, 1],
            "t": [0, 1],
            "centroid-0": [15.0, 15.0],
            "centroid-1": [14.0, 14.0],
            "parent": [0, 0],
        }
    )
