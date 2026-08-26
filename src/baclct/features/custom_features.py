"""Custom single-cell node features.

Either passed to `skimage.regionprops` as `extra_properties` if present in
`CUSTOM_NODE_PROPS`, or applied to existing node feature dataframe if present
in `CUSTOM_NODE_TRANSFORMS`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def center_local(regionmask: np.ndarray) -> tuple:
    """Compute center point.

    Midpoint on medial axis. Always within cell, even for elongated objects. Determined by
    order of graph traversal from one pole to other. If multiple poles exist (e.g.,
    through segmentation errors), the longest connected segment of the medial axis is
    chosen.

    Args:
        regionmask: Binary segmentation mask of single cell.
    """
    skeleton = skeletonize(regionmask)
    coords = np.argwhere(skeleton).astype(np.float32)
    num_pixels = len(coords)

    if num_pixels == 0:
        raise ValueError("Cannot compute center for empty mask.")
    if num_pixels == 1:
        return tuple(coords[0].tolist())

    # brute force adjecency, fine since few pixels in object
    thresh = np.sqrt(coords.shape[1]) + 0.05
    G = rx.PyGraph()
    G.add_nodes_from(range(num_pixels))

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    ii, jj = np.where((dist_matrix > 0) & (dist_matrix < thresh))
    G.add_edges_from(
        [(i, j, float(dist_matrix[i, j])) for i, j in zip(ii, jj, strict=True) if i < j]
    )

    poles = [v for v in G.node_indices() if G.degree(v) == 1]

    best_path = []
    max_len = -1.0
    if not poles:
        # loop or boundary artifact: fallback to diameter of the largest component
        components = rx.connected_components(G)
        largest_comp = max(components, key=len)
        subgraph = G.subgraph(list(largest_comp))  # type: ignore
        # get endpoints of the diameter in the largest component
        path_indices = rx.graph_longest_simple_path(subgraph)
        if path_indices:
            best_path = [list(largest_comp)[i] for i in path_indices]  # type: ignore
    else:
        for start_node in poles:
            path_dict = rx.dijkstra_shortest_paths(G, start_node, weight_fn=float)
            for target, path in path_dict.items():
                if G.degree(target) == 1 and target != start_node:
                    cur_len = sum(
                        G.get_edge_data(path[k], path[k + 1])
                        for k in range(len(path) - 1)
                    )
                    if cur_len > max_len:
                        max_len = cur_len
                        best_path = path

    # midpoint selection
    if not best_path:
        # if graph has no edges/paths return mean of largest component to stay inside mask
        comp = max(rx.connected_components(G), key=len)
        center = np.mean(coords[list(comp)], axis=0)  # type: ignore
    else:
        center = coords[best_path[len(best_path) // 2]]

    return tuple(center.tolist())


def thickness(regionmask: np.ndarray) -> float:
    """Cell thickness.

    Compute the maximum distance between points on the medial axis to the boundaries of a
    cell.
    """
    distances = distance_transform_edt(regionmask)

    return distances.max() * 2


def septum_width(regionmask: np.ndarray) -> float:
    """Calculate the cell thickness at the center of a cell."""
    distances = distance_transform_edt(regionmask)
    y, x = center_local(regionmask)

    return distances[int(y), int(x)] * 2


CUSTOM_NODE_PROPS: dict[str, Callable[..., Any]] = {
    "center_local": center_local,
    "thickness": thickness,
    "septum_width": septum_width,
}


def aspect_ratio(features: pl.DataFrame) -> pl.DataFrame:
    """Log ratio between major and minor axis length."""
    return features.with_columns(
        # clip minor axis to 1 pixel minimum to avoid log(inf) for degenerate 1D masks
        aspect_ratio=(
            pl.col("axis_major_length")
            / pl.col("axis_minor_length").clip(lower_bound=1.0)
        ).log()
    )


# dataFrame-level transforms applied after regionprops (e.g., derived columns).
CUSTOM_NODE_TRANSFORMS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "aspect_ratio": aspect_ratio,
}
