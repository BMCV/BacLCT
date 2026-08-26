"""Convert tracking results into masks, frames, and on-disk export formats."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import dask.array as da
import fastremap
import numpy as np
import polars as pl
from tifffile import imwrite

from baclct.io.lineage import tracks_to_lbep
from baclct.utils.logger import get_pylogger
from baclct.utils.spacing import position_columns

logger = get_pylogger(__name__)

_TRACK_RENAMES = {"label_track": "label", "parent_track": "parent"}


def _predicted_state() -> pl.Expr:
    """Index of the highest-probability class across the per-class `p{i}` columns."""
    return pl.concat_list(r"^p\d+$").list.arg_max().cast(pl.Int32)


def coordinate_columns(tracks: pl.DataFrame) -> list[str]:
    """Cell position columns of `tracks`, in axis order, empty if there are none.

    Prefers `center-*` (medial axis) over `centroid-*`, since it stays inside elongated
    cells, and pixel-space `_px` columns over the physical-unit ones a non-unit `spacing`
    produces.
    """
    return position_columns(tracks.columns)


def _rename_coordinate_columns(cols: list[str]) -> dict[str, str]:
    """Map ordered position columns (`center-*`/`centroid-*`) to `cz`/`cy`/`cx`."""
    axes = ["z", "y", "x"][-len(cols) :]
    return dict(zip(cols, (f"c{axis}" for axis in axes), strict=True))


def _minimal_tracks_for_export(cleaned: pl.DataFrame) -> pl.DataFrame:
    """Reduce a tracking result df to label, position, parent, and state.

    Drops single-cell features, renames position columns to `cz`/`cy`/`cx`, and orders
    columns and rows (`label` then `t`) like a napari Tracks layer.
    """
    renames = _rename_coordinate_columns(coordinate_columns(cleaned))
    out = cleaned.rename(renames)
    cols = ["label", "t", *renames.values(), "parent"]
    if "state" in out.columns:
        cols.append("state")
    return out.select(cols).sort("label", "t")


def create_trajectory_masks(
    tracks: pl.DataFrame,
    masks: da.Array | np.ndarray,
    label_old: str = "label",
    label_new: str = "label_track",
) -> np.ndarray:
    """Convert tracks to trajectory masks."""
    max_label = int(max(tracks[label_new]) or 0)
    needed_dtype = np.min_scalar_type(max_label)
    masks_out = []
    for t, msk in enumerate(masks):  # type: ignore
        if isinstance(msk, da.Array):
            msk = msk.compute()
        assert isinstance(msk, np.ndarray)

        # cast to minimum type to safe memory, does not matter for
        # saving, since data is compressed, anyway
        out_dtype = np.result_type(msk.dtype, needed_dtype)
        if msk.dtype != out_dtype:
            msk = msk.astype(out_dtype)  # type: ignore

        label_map = tracks.filter(t=t).select(label_old, label_new)
        mapping = dict(zip(*label_map.to_numpy().T, strict=True))
        mapping.update({0: 0})

        # drop untracked masks, e.g., small fragments and relabel
        msk_traj = fastremap.mask_except(msk, list(mapping.keys()))
        masks_out.append(
            fastremap.remap(msk_traj, mapping, preserve_missing_labels=False)
        )

    return np.stack(masks_out)


def create_track_df(
    node_features: pl.DataFrame,
    trajectory_labels: dict[int, int],
    parent_labels: dict[int, int],
) -> pl.DataFrame:
    """Convert node features and lineage dicts to track dataframe."""
    tracks = node_features.join(
        pl.DataFrame(
            {
                "index": trajectory_labels.keys(),
                "label_track": trajectory_labels.values(),
            }
        ),
        on="index",
    )

    if parent_labels:
        tracks = tracks.join(
            pl.DataFrame(
                {
                    "label_track": parent_labels.keys(),
                    "parent_track": parent_labels.values(),
                }
            ),
            on="label_track",
            how="left",
        ).with_columns(pl.col("parent_track").fill_null(0))
    else:
        tracks = tracks.with_columns(parent_track=pl.lit(0))

    tracks = tracks.with_columns(cell_source=pl.lit("original"))

    return tracks


def _closest_background_pixel(frame: np.ndarray, centroid: np.ndarray) -> int:
    """Return flat index of background pixel closest to centroid (any dimensionality)."""
    zero_coords = np.argwhere(frame == 0)
    if zero_coords.size == 0:
        return -1
    distances = np.linalg.norm(zero_coords - centroid, axis=1)
    closest = zero_coords[np.argmin(distances)]
    return int(np.ravel_multi_index(tuple(closest), frame.shape))


def export_tracking_results_ctc(
    tracks: pl.DataFrame,
    masks_tracked: np.ndarray,
    res_dir: Path | None,
    is_gt: bool = False,
    fill_gaps: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Export tracking results to CTC format."""
    lbep = tracks_to_lbep(tracks, suffix="" if is_gt else "_track")
    if fill_gaps:
        # place a single-pixel marker within gaps to not raise CTC errors
        label_col = "label_track" if "label_track" in tracks.columns else "label"
        coord_cols = coordinate_columns(tracks)

        active_labels_by_time = defaultdict(list)
        for lbl, b, e, _ in lbep:
            for t in range(int(b), int(e) + 1):
                active_labels_by_time[t].append(lbl)

        for t in range(masks_tracked.shape[0]):
            expected_labels = active_labels_by_time.get(t, [])
            if not expected_labels:
                continue

            current_frame = masks_tracked[t]
            present_labels = np.unique(current_frame)
            missing_labels = np.setdiff1d(expected_labels, present_labels)

            if missing_labels.size > 0:
                logger.warning(
                    f"Frame {t}: {len(missing_labels)} label(s) absent from mask, "
                    f"placing fallback marker(s): {missing_labels.tolist()}"
                )
                for missing_lbl in missing_labels:
                    label_rows = tracks.filter(pl.col(label_col) == int(missing_lbl))
                    if label_rows.height > 0 and coord_cols:
                        # use last known position before the gap; fall back to first after
                        before = label_rows.filter(pl.col("t") < t)
                        ref = (
                            before.sort("t").tail(1)
                            if before.height > 0
                            else label_rows.sort("t").head(1)
                        )
                        centroid = np.array(ref.select(pl.col(coord_cols)).row(0))
                        flat_idx = _closest_background_pixel(current_frame, centroid)
                    else:
                        zeros = np.flatnonzero(current_frame == 0)
                        flat_idx = int(zeros[0]) if zeros.size > 0 else -1

                    if flat_idx < 0:
                        raise ValueError(
                            f"Could not place marker for label {missing_lbl} at t={t}: "
                            "no background pixel available."
                        )
                    current_frame.flat[flat_idx] = missing_lbl

    if res_dir is not None:
        res_dir.mkdir(exist_ok=True, parents=True)

        for t, msk in enumerate(masks_tracked):
            imwrite(res_dir / f"t{t:03d}.tif", msk, compression="deflate")
        np.savetxt(res_dir / f"{'man' if is_gt else 'res'}_track.txt", lbep, fmt="%0d")

    return lbep, masks_tracked


def export_tracking_results_flat(
    tracks: pl.DataFrame, masks_tracked: np.ndarray, res_dir: Path | None
) -> tuple[pl.DataFrame, np.ndarray]:
    """Export tracking results to flat format.

    Creates single .tif "masks_tracked.tif" and "tracks.parquet".
    """
    if res_dir is not None:
        res_dir.mkdir(exist_ok=True, parents=True)

        imwrite(
            res_dir / "masks_tracked.tif",
            masks_tracked,
            compression="deflate",
            photometric="minisblack",
        )
        tracks.write_parquet(res_dir / "tracks.parquet")

    return tracks, masks_tracked


def export_tracking_results_simple(
    tracks: pl.DataFrame,
    masks_tracked: np.ndarray,
    sequence_id: str,
    output_dir: Path,
    suffix: str = "_tracks",
) -> tuple[pl.DataFrame, np.ndarray]:
    """Save tracking results as a flat CSV and a deflate-compressed TIF.

    Files are written directly into `output_dir` (no per-sequence subdirectory) as
    '{sequence_id}{suffix}.csv' and '{sequence_id}{suffix}.tif'. The CSV contains label,
    t, position, parent, and state (when present). The TIF uses deflate compression, which
    tifffile can read and write without the `imagecodecs` extra.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _minimal_tracks_for_export(tracks).write_csv(
        output_dir / f"{sequence_id}{suffix}.csv"
    )
    imwrite(
        output_dir / f"{sequence_id}{suffix}.tif",
        masks_tracked,
        compression="deflate",
        photometric="minisblack",
    )
    return tracks, masks_tracked


def node_preds_to_df(
    node_preds: np.ndarray,
    node_index: np.ndarray,
    y: np.ndarray | None = None,
) -> pl.DataFrame:
    """Convert node predictions to dataframe.

    Args:
        node_preds: Per-node class probabilities.
        node_index: Global node index of each prediction.
        y: Node labels. The column is omitted when `None`, as it is at inference.
    """
    node_preds, node_index = np.asarray(node_preds), np.asarray(node_index)

    # built column by column, for the same reason as `edge_preds_to_df`
    columns: dict[str, pl.Series] = {
        "index": pl.Series("index", node_index, dtype=pl.UInt32)
    }
    if y is not None:
        columns["y"] = pl.Series("y", np.asarray(y), dtype=pl.Int64)
    for i in range(node_preds.shape[-1]):
        columns[f"p{i}"] = pl.Series(f"p{i}", node_preds[:, i], dtype=pl.Float32)

    return pl.DataFrame(columns)


def export_classification_results(
    node_preds_df: pl.DataFrame,
    output_dir: Path,
    tracks: pl.DataFrame | None = None,
) -> None:
    """Export node classification results."""
    out = node_preds_df.with_columns(
        predicted_label=_predicted_state(),
        label_source=pl.lit("tracked"),
    )
    if tracks is not None:
        meta_cols = ["index", "t", "label_track"]
        if "cell_source" in tracks.columns:
            meta_cols.append("cell_source")
        meta_cols += sorted(c for c in tracks.columns if c.startswith("centroid-"))
        out = out.join(
            tracks.select(meta_cols).unique("index"),
            on="index",
            how="left",
        ).rename({"label_track": "label"})
    out.write_csv(output_dir / "res_states.csv")


def clean_tracks_df(
    tracks: pl.DataFrame,
    node_preds_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return a user-facing tracks frame with standardised columns.

    Renames the internal `label_track` / `parent_track` to `label` / `parent`, drops the
    bookkeeping columns (`index`, `cell_source`, `y`), and optionally joins the predicted
    life cycle state as `state`. Single-cell features are kept, so the frame is as wide as
    the node features it came from. Integer columns are cast to `Int32`.
    """
    # drop any cached/GT state before joining predictions to avoid a suffix collision
    out = tracks.drop("state") if "state" in tracks.columns else tracks

    # join predicted state before dropping index (index is the join key)
    if node_preds_df is not None and "index" in out.columns:
        state = node_preds_df.with_columns(state=_predicted_state()).select(
            "index", "state"
        )
        out = out.join(state, on="index", how="left")

    drop_cols = [
        c for c in ("label", "parent", "index", "cell_source", "y") if c in out.columns
    ]
    out = out.drop(drop_cols).rename(_TRACK_RENAMES)

    int_cols = [c for c in ("label", "t", "parent", "state") if c in out.columns]
    return out.with_columns([pl.col(c).cast(pl.Int32) for c in int_cols])


def export_combined_tracks(
    tracks: pl.DataFrame,
    output_dir: Path,
    node_preds_df: pl.DataFrame | None = None,
) -> None:
    """Export unified CSV combining lineage and classification results for single cells.

    Writes res_tracks.csv with one row per detection: label, t, position (`cy`/`cx`, or
    `cz`/`cy`/`cx`), parent, and if available state (predicted class index from
    node_preds_df). Unlike `clean_tracks_df`, single-cell features are left out.
    """
    cleaned = clean_tracks_df(tracks, node_preds_df)
    _minimal_tracks_for_export(cleaned).write_csv(output_dir / "res_tracks.csv")
