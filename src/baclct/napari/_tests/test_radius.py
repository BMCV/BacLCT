"""Tests for the search-radius preview."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from skimage.measure import regionprops_table

from baclct.napari._radius import (
    DILATED_OVERLAP_RADIUS_MULTIPLIER,
    PRUNE_COLOR,
    SEARCH_COLOR,
    radius_preview_shapes,
    resolve_radius_px,
)
from baclct.utils.graph_params import resolve_search_radius


def _masks(n_cells: int = 4, n_frames: int = 3) -> np.ndarray:
    """Frames of evenly spaced rectangular cells."""
    masks = np.zeros((n_frames, 100, 20 * n_cells + 20), dtype=np.uint16)
    for t in range(n_frames):
        for i in range(n_cells):
            x = 10 + 20 * i
            masks[t, 30:50, x : x + 8] = i + 1
    return masks


def test_resolve_radius_px_matches_the_dataset():
    """The preview must resolve "2.5x" exactly as the pipeline does."""
    masks = _masks()

    # what the dataset itself would compute, from the same first-frame region props
    props = regionprops_table(masks[0], properties=("axis_major_length",))
    node_feats = pl.DataFrame(
        {
            "t": np.zeros(len(props["axis_major_length"]), dtype=int),
            "axis_major_length": props["axis_major_length"],
        }
    )
    expected = resolve_search_radius("2.5x", node_feats)
    assert resolve_radius_px(masks, "2.5x") == expected
    assert resolve_radius_px(masks, 42) == 42

    # an empty first frame must not collapse the expected cell size
    masks[0] = 0
    assert resolve_radius_px(masks, "2.5x") == expected

    with pytest.raises(ValueError, match="no labelled cells"):
        resolve_radius_px(np.zeros((2, 10, 10), dtype=np.uint16), "2.5x")


@pytest.mark.parametrize(
    ("prune_edges_by", "shape_types"),
    [
        (None, ["ellipse"]),
        (("ellipse", 7), ["ellipse", "ellipse"]),
        # 'overlap' links intersecting masks, so its region is the mask itself
        (("overlap", 0.1), ["ellipse", "polygon"]),
    ],
    ids=["off", "ellipse", "overlap"],
)
def test_preview_draws_the_search_region_and_the_pruned_one(prune_edges_by, shape_types):
    shapes, types, colors = radius_preview_shapes(
        _masks(), t=1, radius_px=30, prune_edges_by=prune_edges_by
    )

    assert types == shape_types
    assert colors == [SEARCH_COLOR, PRUNE_COLOR][: len(shapes)]
    # every vertex is pinned to the previewed frame, so the shape renders there only
    assert shapes[0].shape == (4, 3)
    assert np.all(shapes[0][:, 0] == 1)


def test_preview_sizes_the_dilated_overlap_region():
    """The dilation radius comes from the parameter, or from a per-cell column."""
    masks = _masks()
    param = 5
    by_pixels, types, _ = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", param)
    )
    assert types == ["ellipse", "polygon"]
    # the cell spans rows 30 to 49, and the outline sits one dilation radius outside it
    expected = 2 * param * DILATED_OVERLAP_RADIUS_MULTIPLIER
    assert by_pixels[1][:, 1].min() == pytest.approx(29 - expected, abs=1.0)
    assert by_pixels[1][:, 1].max() == pytest.approx(49 + expected, abs=1.0)

    # the shipped spores model dilates by each cell's thickness, 8 px here
    by_column, _, _ = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", "thickness")
    )
    expected = 2 * 8 * DILATED_OVERLAP_RADIUS_MULTIPLIER
    assert by_column[1][:, 1].max() == pytest.approx(49 + expected, abs=1.0)

    with pytest.raises(ValueError, match="Cannot preview"):
        radius_preview_shapes(
            masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", "area")
        )


def test_preview_selects_the_cell_to_outline():
    """`label` picks a cell, 0 takes the one closest to the frame center."""
    masks = _masks()
    edge, _, _ = radius_preview_shapes(masks, t=0, radius_px=10, label=1)
    center, _, _ = radius_preview_shapes(masks, t=0, radius_px=10)
    center_x = masks.shape[2] / 2.0

    assert abs(center[0][:, 2].mean() - center_x) < abs(edge[0][:, 2].mean() - center_x)

    with pytest.raises(ValueError, match="No cell with label 999"):
        radius_preview_shapes(masks, t=0, radius_px=10, label=999)
