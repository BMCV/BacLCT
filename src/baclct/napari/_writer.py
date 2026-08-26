"""Writer for napari `Tracks` layers.

napari's built-in writers cover Labels layers (as TIFF) but cannot save a Tracks layer at
all, so the lineage would otherwise be lost on save. The CSV written here matches the
schema of `clean_tracks_df`, i.e. what `BacLCT.track()` returns and exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def write_tracks_csv(path: str, data: Any, meta: dict) -> list[str]:
    """Write a napari Tracks layer as `label,t,centroid-*,parent[,state]`.

    Args:
        path: Destination CSV path.
        data: Tracks array of shape `(N, 2 + ndim)`, i.e. `[track_id, t, (z), y, x]`.
        meta: Layer metadata. `graph` supplies the parent of each track and `features`
            may carry the predicted `state`.

    Returns:
        The paths written, as npe2 requires.
    """
    data = np.asarray(data)
    n_spatial = data.shape[1] - 2

    frame = pl.DataFrame(
        {
            "label": data[:, 0].astype(np.int32),
            "t": data[:, 1].astype(np.int32),
            **{f"centroid-{i}": data[:, 2 + i] for i in range(n_spatial)},
        }
    )

    # napari stores the lineage as {daughter: [parent]}; tracks without a parent get 0,
    # matching the CTC convention the rest of the pipeline uses
    graph: dict[int, list[int]] = meta.get("graph") or {}
    parents = {int(child): int(parent[0]) for child, parent in graph.items() if parent}
    frame = frame.with_columns(
        parent=pl.col("label").replace_strict(parents, default=0).cast(pl.Int32)
    )

    features = meta.get("features")
    if features is not None and "state" in features:
        frame = frame.with_columns(
            state=pl.Series(np.asarray(features["state"])).cast(pl.Int32)
        )

    path = str(Path(path).with_suffix(".csv"))
    frame.write_csv(path)
    return [path]
