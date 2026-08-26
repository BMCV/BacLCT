"""Preview of the candidate-edge search region for the cells of one frame.

Every shape mirrors the geometry `EdgeFinder` applies, so the preview cannot drift from
the tracking result. Two consequences are visible to a user: a `'2.5x'` radius is a
multiple of the median major axis length in the first frame, not the frame on screen, and
shapes are drawn around the centroid while the edge finder measures from `center`, the
medial-axis midpoint, which differs slightly for strongly bent cells.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from skimage.measure import find_contours, regionprops_table
from skimage.morphology import isotropic_dilation

from baclct.features.graph import DILATED_OVERLAP_RADIUS_MULTIPLIER
from baclct.utils.graph_params import resolve_search_radius

# dilated_overlap takes its dilation radius from a per-cell column rather than a factor.
# only 'thickness' can be recomputed from the mask alone, which is all the preview has
PREVIEWABLE_PRUNE_COLUMNS = ("thickness",)

# the search radius and the tighter pruned region get distinct colors so pruning is
# visible even where the two regions are close in size
SEARCH_COLOR = "yellow"
PRUNE_COLOR = "cyan"

_PREVIEW_PROPS = (
    "label",
    "centroid",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
)


def _first_nonempty_frame(masks: np.ndarray) -> int:
    """Index of the first frame that contains any labelled cell."""
    for t, frame in enumerate(masks):
        if frame.any():
            return t
    raise ValueError("Segmentation contains no labelled cells.")


def resolve_radius_px(masks: np.ndarray, graph_search_radius: int | str) -> int:
    """Resolve a search radius spec to pixels, as the dataset does.

    Args:
        masks: Instance segmentation of shape `(T, ...)`.
        graph_search_radius: Radius in pixels, or a multiple of the expected cell size
            (e.g. `'2.5x'`).

    Returns:
        The radius in pixels.
    """
    if not isinstance(graph_search_radius, str):
        return int(graph_search_radius)

    t = _first_nonempty_frame(masks)
    props = regionprops_table(masks[t], properties=("axis_major_length",))
    node_feats = pl.DataFrame(
        {
            "t": np.full(len(props["axis_major_length"]), t),
            "axis_major_length": props["axis_major_length"],
        }
    )
    resolved = resolve_search_radius(graph_search_radius, node_feats)
    if not isinstance(resolved, int):
        # a (min, max) range is a training-time sweep and has no single circle to draw
        raise ValueError(
            f"Cannot preview a search radius range ({resolved!r}). Enter a single "
            "radius in pixels, or a multiple of the cell size such as '2.5x'."
        )
    return resolved


def _select_cell(props: dict, frame_shape: tuple[int, ...], label: int | None) -> int:
    """Index into `props` of the cell to preview: `label`, else the centermost cell."""
    labels = props["label"]
    if label is not None:
        matches = np.flatnonzero(labels == label)
        if len(matches) == 0:
            raise ValueError(f"No cell with label {label} in this frame.")
        return int(matches[0])

    cy, cx = frame_shape[0] / 2.0, frame_shape[1] / 2.0
    dy = props["centroid-0"] - cy
    dx = props["centroid-1"] - cx
    return int(np.argmin(dy * dy + dx * dx))


def _ellipse_corners(
    centroid: np.ndarray, major: np.ndarray, minor: np.ndarray
) -> np.ndarray:
    """The four corners of the rotated bounding box napari renders an ellipse in."""
    return np.stack(
        [
            centroid - major - minor,
            centroid - major + minor,
            centroid + major + minor,
            centroid + major - minor,
        ]
    )


def _dilation_radius(cell: np.ndarray, param: float | str) -> float:
    """Dilation radius `dilated_overlap` applies to one cell, in pixels.

    Raises:
        ValueError: If `param` names a column the preview cannot recompute.
    """
    if isinstance(param, str):
        if param not in PREVIEWABLE_PRUNE_COLUMNS:
            raise ValueError(
                f"Cannot preview expanded overlap sized by {param!r}. Supported: "
                f"{', '.join(PREVIEWABLE_PRUNE_COLUMNS)}."
            )
        from baclct.features.custom_features import thickness

        param = thickness(cell)
    # the region has to cover the neighbour too, and both cells are dilated
    return 2 * float(param) * DILATED_OVERLAP_RADIUS_MULTIPLIER


def radius_preview_shapes(
    masks: np.ndarray,
    t: int,
    radius_px: int,
    prune_edges_by: tuple[str, float | str] | None = None,
    label: int | None = None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Build napari `Shapes` data outlining the search region of one cell in frame `t`.

    Radii are large and frames dense, so drawing every cell is unreadable; a single cell
    conveys the region. The centermost cell is used unless `label` picks another.

    Args:
        masks: Instance segmentation of shape `(T, Y, X)`.
        t: Frame to preview.
        radius_px: Search radius in pixels (see `resolve_radius_px`).
        prune_edges_by: Pruning method and parameter, e.g. `('ellipse', 7)`. The pruned
            region is drawn on top of the search circle in a distinct color.
        label: Label of the cell to preview. Falls back to the centermost cell if `None`.

    Returns:
        The shape vertex arrays, their napari shape types, and a per-shape edge color
        (`SEARCH_COLOR` for the search circle, `PRUNE_COLOR` for a pruned region).

    Raises:
        ValueError: If `label` is given but no cell carries it in this frame.
    """
    props = regionprops_table(masks[t], properties=_PREVIEW_PROPS)
    if len(props["label"]) == 0:
        return [], [], []

    i = _select_cell(props, masks[t].shape, label)
    method, param = prune_edges_by or (None, 0.0)
    centroid = np.array([props["centroid-0"][i], props["centroid-1"][i]])

    # the search radius is an isotropic circle around the cell
    shapes = [
        _ellipse_corners(centroid, np.array([radius_px, 0.0]), np.array([0.0, radius_px]))
    ]
    shape_types = ["ellipse"]
    colors = [SEARCH_COLOR]

    if method == "ellipse":
        # mirrors _prune_edges_ellipse: a rotated ellipse whose semi-axes are the cell's
        # own axes scaled by `param`
        theta = props["orientation"][i]
        a = props["axis_major_length"][i] / 2.0 * float(param)
        b = props["axis_minor_length"][i] / 2.0 * float(param)
        major = np.array([np.cos(theta), np.sin(theta)]) * a
        minor = np.array([np.sin(theta), -np.cos(theta)]) * b
        shapes.append(_ellipse_corners(centroid, major, minor))
        shape_types.append("ellipse")
        colors.append(PRUNE_COLOR)
    elif method == "radius":
        # mirrors _prune_edges_radius: a circle scaled by the cell's major axis
        r = props["axis_major_length"][i] / 2.0 * float(param)
        shapes.append(_ellipse_corners(centroid, np.array([r, 0.0]), np.array([0.0, r])))
        shape_types.append("ellipse")
        colors.append(PRUNE_COLOR)
    elif method in ("overlap", "dilated_overlap"):
        # 'overlap' links intersecting masks, so its region is the mask itself
        cell = masks[t] == int(props["label"][i])
        radius = 0.0 if method == "overlap" else _dilation_radius(cell, param)
        region = isotropic_dilation(cell, radius) if radius else cell
        contours = find_contours(region.astype(float), 0.5)
        if contours:
            shapes.append(max(contours, key=len))
            shape_types.append("polygon")
            colors.append(PRUNE_COLOR)

    # pin every shape to the previewed frame so it only renders there
    shapes = [np.column_stack([np.full(len(s), t), s]) for s in shapes]
    return shapes, shape_types, colors
