"""Custom edge features derived from existing edge feature dataframe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def relative_size(
    edge_data: pl.DataFrame | pl.LazyFrame, based_on: str = "area"
) -> pl.DataFrame | pl.LazyFrame:
    """Calculates the log ratio of sizes between destination and source objects.

    Args:
        edge_data: Edge features containing indices of connected nodes (`src`, `dst`)
            and their relational features. Must contain `{based_on}_src` and
            `{based_on}_dst` columns.
        based_on: The base feature to use for size comparison (e.g., 'area',
            'axis_major_length').

    Returns:
        Input with an added 'relative_size' column.
    """
    src_col = f"{based_on}_src"
    dst_col = f"{based_on}_dst"

    columns = (
        edge_data.collect_schema().names()
        if isinstance(edge_data, pl.LazyFrame)
        else edge_data.columns
    )
    if src_col not in columns or dst_col not in columns:
        raise ValueError(
            f"Columns {src_col} or {dst_col} not found for 'relative_size' calculation.\n"
            f"Available columns: {columns}"
        )

    # add a small epsilon to avoid log(0) or division by zero.
    eps = 1e-6
    return edge_data.with_columns(
        relative_size=((pl.col(dst_col) + eps) / (pl.col(src_col) + eps)).log()
    )


def intensity_mean_diff(
    edge_data: pl.DataFrame | pl.LazyFrame, based_on: str = "intensity_mean"
) -> pl.DataFrame | pl.LazyFrame:
    """Difference in mean intensity between source and destination cells (src - dst)."""
    src_col = f"{based_on}_src"
    dst_col = f"{based_on}_dst"

    columns = (
        edge_data.collect_schema().names()
        if isinstance(edge_data, pl.LazyFrame)
        else edge_data.columns
    )
    if src_col not in columns or dst_col not in columns:
        raise ValueError(
            f"Columns {src_col} or {dst_col} not found for 'intensity_mean_diff'.\n"
            f"Available columns: {columns}"
        )

    return edge_data.with_columns(intensity_mean_diff=pl.col(src_col) - pl.col(dst_col))


CUSTOM_EDGE_PROPS: dict[str, Callable[..., Any]] = {
    "relative_size": relative_size,
    "intensity_mean_diff": intensity_mean_diff,
}
