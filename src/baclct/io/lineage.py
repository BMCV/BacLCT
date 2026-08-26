"""Locate and read lineage files, with or without life cycle states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def find_lineage_file(
    data_dir: Path,
    seq_id: str,
    data_format: str = "ctc",
    segmentation_name: str | None = None,
    with_states: bool = True,
) -> tuple[Path | None, bool]:
    """Locate the lineage file for a sequence, and whether it carries life cycle states.

    Searches next to the masks, then upwards: for CTC that is
    `{seq_id}_{segmentation_name}/TRA`, then `{seq_id}_{segmentation_name}`, then
    `data_dir`. A `states.txt` is preferred over a CTC `res_track.txt` / `man_track.txt`
    when `with_states` is set. Returns `(None, False)` when nothing is found.
    """
    if data_format == "ctc":
        seg_suffix = segmentation_name or "GT"
        search_dirs = [
            data_dir / f"{seq_id}_{seg_suffix}" / "TRA",
            data_dir / f"{seq_id}_{seg_suffix}",
            data_dir,
        ]
    elif data_format == "flat":
        search_dirs = [data_dir]
    elif data_format == "dirs":
        seg_suffix = segmentation_name or "Segmentation"
        search_dirs = [data_dir / seq_id / seg_suffix, data_dir / seq_id, data_dir]
    else:
        search_dirs = [data_dir / seq_id, data_dir]

    if with_states:
        for pdir in search_dirs:
            file = pdir / "states.txt"
            if file.exists():
                logger.debug(f"Loading lineage file: {file}.")
                return file, True

        logger.debug(
            f"Could not find `states.txt` in {search_dirs}. "
            "Trying lineage without states."
        )

    for pdir in search_dirs:
        for fname in ["res_track.txt", "man_track.txt"]:
            file = pdir / fname
            if file.exists():
                logger.debug(f"Loading lineage file: {file}.")
                return file, False

    logger.warning(
        f"Could not find lineage file in {data_dir} for {seq_id}. "
        "Continuing without lineage information."
    )
    return None, False


def _read_lineage_ctc(
    lineage_file: Path, as_numpy: bool = False
) -> pl.DataFrame | np.ndarray:
    """Read CTC lineage file."""
    if lineage_file.stem == "states":
        raise ValueError(
            "The filename `states.*` is reserved for state annotations. "
            f"Please use `man_track` or `res_track` (file: {lineage_file})."
        )

    lbep = pl.read_csv(
        lineage_file,
        separator=" ",
        has_header=False,
        schema=dict.fromkeys("lbep", pl.Int32),
    ).select(pl.col("l").alias("label"), "b", "e", pl.col("p").alias("parent"))

    if as_numpy:
        return lbep.to_numpy()

    return lbep


def _read_lineage_states(lineage_file: Path, seq_id: str | None) -> pl.DataFrame:
    """Read file defining single-cell lineage and life cycle states."""
    all_states = pl.read_csv(lineage_file)
    required = ["t", "state", "parent"]
    if seq_id is not None:
        required.append("sequence_id")

    if any(it not in all_states.columns for it in required):
        raise pl.exceptions.ColumnNotFoundError(
            "Requested lineage file does not contain required state information "
            f"({lineage_file})\n{all_states.glimpse(return_type='string')}"
        )

    if seq_id is not None:
        # polars infers sequence_id as integer when all values are numeric, stripping
        # leading zeros (e.g. "09" to 9). cast both sides to int for robust comparison or
        # fall back to string equality for non-numeric ids.
        try:
            states = all_states.filter(
                pl.col("sequence_id").cast(pl.Int64) == int(seq_id)
            )
        except ValueError:
            states = all_states.filter(sequence_id=seq_id)
    else:
        states = all_states.select(pl.exclude("sequence_id"))

    assert not states.is_empty(), (
        f"Could not load states for sequence {seq_id}:\nAll states:\n{all_states}"
    )
    return states


def load_lineage(
    lineage_file: Path,
    with_states: bool = False,
    seq_id: str | None = None,
    as_numpy: bool = False,
) -> pl.DataFrame | np.ndarray:
    """Read lineage file, either with or without states."""
    if with_states:
        lineage = _read_lineage_states(lineage_file, seq_id)
        if as_numpy:
            return tracks_to_lbep(lineage.select("label", "t", "parent"))
        return lineage

    # fallback, load ctc format without states
    return _read_lineage_ctc(lineage_file, as_numpy)


def tracks_to_lbep(tracks: pl.DataFrame, suffix: str = "") -> np.ndarray:
    """Convert tracks to CTC lineage format."""
    label_col = "label" + suffix
    parent_col = "parent" + suffix

    # parent_col is taken from the earliest `t` row per label so that corrections that
    # append rows with a different parent_track (e.g. relabeled gap-only detections) never
    # break the original lineage assignment (e.g., if being empty).
    return (
        tracks.group_by(label_col)
        .agg(
            pl.col("t").min().alias("tmin"),
            pl.col("t").max().alias("tmax"),
            pl.col(parent_col).sort_by("t").first(),
        )
        .select(label_col, "tmin", "tmax", parent_col)
        .cast(pl.Int64)
        .sort("tmin", label_col)
        .to_numpy()
    )
