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


def test_resolve_radius_px_passes_through_pixels():
    assert resolve_radius_px(_masks(), 42) == 42


def test_resolve_radius_px_uses_the_first_nonempty_frame():
    """An empty first frame must not collapse the expected cell size."""
    masks = _masks()
    expected = resolve_radius_px(masks, "2.5x")
    masks[0] = 0

    assert resolve_radius_px(masks, "2.5x") == expected


def test_preview_draws_one_circle_for_a_single_cell():
    masks = _masks(n_cells=4)
    shapes, shape_types, colors = radius_preview_shapes(masks, t=1, radius_px=30)

    # a single cell, so one search circle only
    assert len(shapes) == 1
    assert shape_types == ["ellipse"]
    assert colors == [SEARCH_COLOR]
    # every vertex is pinned to the previewed frame, so the shape renders there only
    assert shapes[0].shape == (4, 3)
    assert np.all(shapes[0][:, 0] == 1)


def test_preview_adds_the_pruning_ellipse_in_a_distinct_color():
    masks = _masks(n_cells=4)
    shapes, _, colors = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("ellipse", 7)
    )
    # the search circle plus the pruning ellipse for the one cell, colored apart
    assert len(shapes) == 2
    assert colors == [SEARCH_COLOR, PRUNE_COLOR]


def test_preview_draws_the_cell_outline_for_plain_overlap():
    """'overlap' links cells whose masks intersect, so its region is the mask itself."""
    masks = _masks(n_cells=4)
    shapes, shape_types, colors = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("overlap", 0.1)
    )
    assert shape_types == ["ellipse", "polygon"]
    assert colors == [SEARCH_COLOR, PRUNE_COLOR]

    # the outline is the undilated cell, so it stays inside the 20x8 px mask bounds
    outline = shapes[1]
    assert outline[:, 1].min() >= 29 and outline[:, 1].max() <= 50
    dilated, _, _ = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", 5)
    )
    assert dilated[1][:, 1].max() > outline[:, 1].max()


def test_preview_draws_the_dilated_overlap_region():
    """The region has to cover the neighbor too, so both cells' dilations are drawn."""
    masks = _masks(n_cells=4)
    param = 5
    shapes, shape_types, colors = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", param)
    )
    # the search circle plus the dilated-mask outline of the one cell
    assert shape_types == ["ellipse", "polygon"]
    assert colors == [SEARCH_COLOR, PRUNE_COLOR]
    # the dilated boundary is a polygon, so more than the ellipse's four corners
    assert len(shapes[1]) > 4

    # the cell spans rows 30 to 49, and the outline sits one dilation radius outside it
    expected = 2 * param * DILATED_OVERLAP_RADIUS_MULTIPLIER
    assert shapes[1][:, 1].min() == pytest.approx(29 - expected, abs=1.0)
    assert shapes[1][:, 1].max() == pytest.approx(49 + expected, abs=1.0)


def test_preview_sizes_dilated_overlap_by_a_per_cell_column():
    """The shipped spores model dilates by each cell's thickness, not a fixed radius."""
    masks = _masks(n_cells=4)
    by_column, _, _ = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", "thickness")
    )
    # the cell is 8 px wide, so its thickness dilates further than a 5 px parameter
    by_pixels, _, _ = radius_preview_shapes(
        masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", 5)
    )
    assert by_column[1][:, 1].max() > by_pixels[1][:, 1].max()
    expected = 2 * 8 * DILATED_OVERLAP_RADIUS_MULTIPLIER
    assert by_column[1][:, 1].max() == pytest.approx(49 + expected, abs=1.0)

    with pytest.raises(ValueError, match="Cannot preview"):
        radius_preview_shapes(
            masks, t=0, radius_px=30, prune_edges_by=("dilated_overlap", "area")
        )


def test_preview_targets_the_requested_label():
    masks = _masks(n_cells=4)
    first, _, _ = radius_preview_shapes(masks, t=0, radius_px=10, label=1)
    last, _, _ = radius_preview_shapes(masks, t=0, radius_px=10, label=4)
    # the search circle is centered on the chosen cell, so the two differ along x
    assert last[0][:, 2].mean() > first[0][:, 2].mean()


def test_preview_defaults_to_the_centermost_cell():
    masks = _masks(n_cells=4)
    center, _, _ = radius_preview_shapes(masks, t=0, radius_px=10)
    edge, _, _ = radius_preview_shapes(masks, t=0, radius_px=10, label=1)
    center_x = masks.shape[2] / 2.0
    assert abs(center[0][:, 2].mean() - center_x) < abs(edge[0][:, 2].mean() - center_x)


def test_preview_unknown_label_raises():
    masks = _masks(n_cells=4)
    with pytest.raises(ValueError, match="No cell with label 999"):
        radius_preview_shapes(masks, t=0, radius_px=10, label=999)


def test_preview_of_empty_segmentation_raises():
    with pytest.raises(ValueError, match="no labelled cells"):
        resolve_radius_px(np.zeros((2, 10, 10), dtype=np.uint16), "2.5x")
