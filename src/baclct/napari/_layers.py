"""Conversion of tracking results into napari layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import polars as pl

from baclct.io import coordinate_columns

if TYPE_CHECKING:
    import napari


def tracks_to_napari(
    tracks: pl.DataFrame,
) -> tuple[np.ndarray, dict[int, list[int]], dict[str, np.ndarray]]:
    """Convert a cleaned tracks frame into napari `Tracks` layer arguments.

    Args:
        tracks: Cleaned tracks (`label`, `t`, the center position as midpoint on medial
        axis or centroid, `parent`, optionally `state`), as returned by `BacLCT.track()`.

    Returns:
        `data` of shape `(N, 2 + ndim)` as `[track_id, t, (z), y, x]` sorted by
        `(track_id, t)` as napari requires, the lineage `graph` mapping each daughter
        track to its parent, and per-point `features`.
    """
    coords = coordinate_columns(tracks)
    ordered = tracks.sort("label", "t")

    data = ordered.select("label", "t", *coords).to_numpy()

    graph: dict[int, list[int]] = {}
    if "parent" in ordered.columns:
        lineage = ordered.select("label", "parent").unique()
        for child, parent in lineage.iter_rows():
            # a track without a parent is its own root: the exporters encode this as
            # parent 0 or as parent == label, neither of which is an edge in the lineage
            if parent is None or parent <= 0 or parent == child:
                continue
            graph[int(child)] = [int(parent)]

    features: dict[str, np.ndarray] = {}
    if "state" in ordered.columns:
        features["state"] = ordered.get_column("state").to_numpy()

    return data, graph, features


def add_result_layers(
    viewer: napari.Viewer,
    tracks: pl.DataFrame,
    masks_tracked: np.ndarray | da.Array,
    name: str,
    scale: tuple[float, ...] | None = None,
) -> list[Any]:
    """Add the tracked masks and the lineage graph to the viewer.

    Both layers are named after `name`, the segmentation they were tracked from, so
    results of different segmentations stay apart and read as a pair. Existing layers of
    the same name are replaced.
    """
    data, graph, features = tracks_to_napari(tracks)

    tracked_name, lineage_name = f"{name} (tracked)", f"{name} (lineage)"
    layers = []
    for layer_name in (tracked_name, lineage_name):
        if layer_name in viewer.layers:
            viewer.layers.remove(layer_name)

    layers.append(
        viewer.add_labels(masks_tracked, name=tracked_name, scale=scale, opacity=0.6)
    )

    tracks_layer = viewer.add_tracks(
        data,
        graph=graph,
        features=features or None,
        name=lineage_name,
        scale=scale,
        tail_length=10,
    )
    if "state" in features:
        tracks_layer.color_by = "state"
    layers.append(tracks_layer)

    return layers
