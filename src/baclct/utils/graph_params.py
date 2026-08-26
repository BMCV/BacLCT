"""Resolve graph parameter specs to the concrete values graph construction needs."""

from __future__ import annotations

import polars as pl
from omegaconf import ListConfig


def expand_param_range(val: int | list | tuple) -> list[int]:
    """Expand a dynamic graph-property spec to all integer values it can take.

    Supported forms: an int (single value), a `(min, max)` tuple (inclusive range,
    step 1), or a `(min, max, step)` tuple (inclusive range with the given step).
    """
    if isinstance(val, int):
        return [val]
    if isinstance(val, list | tuple | ListConfig) and len(val) in (2, 3):
        lo, hi = int(val[0]), int(val[1])
        step = int(val[2]) if len(val) == 3 else 1
        return list(range(lo, hi + 1, step))
    raise TypeError(
        f"Parameter '{val}' is invalid. Supported: int, (min, max), or "
        "(min, max, step) lists/tuples."
    )


def _expected_cell_size(node_feats: pl.DataFrame) -> float:
    """Median major-axis length at the first frame.

    Prefers the cached `len_init` column (computed during normalization with
    `scale_relative_size`) and falls back to recomputing it from `axis_major_length`.
    """
    if "len_init" in node_feats.columns:
        val = node_feats.get_column("len_init").first()
        if val:
            return float(val)  # type: ignore
    init = node_feats.filter(pl.col("t") == pl.col("t").min())
    val = init.get_column("axis_major_length").median()
    return float(val) if val else 1.0  # type: ignore


def resolve_search_radius(
    graph_search_radius: int | str | tuple | list, node_feats: pl.DataFrame
) -> int | tuple | list:
    """Convert relative search radius to pixels."""
    if not isinstance(graph_search_radius, str):
        return graph_search_radius

    spec = graph_search_radius.strip()
    if not spec.endswith("x"):
        raise ValueError(
            f"Invalid graph_search_radius {graph_search_radius!r}: a string radius must "
            "end with 'x' (a multiple of expected cell size), e.g. '2.5x'."
        )
    try:
        mult = float(spec[:-1])
    except ValueError as e:
        raise ValueError(
            f"Invalid graph_search_radius {graph_search_radius!r}: expected a number "
            "before 'x', e.g. '2.5x'."
        ) from e
    if mult <= 0:
        raise ValueError(
            f"Invalid graph_search_radius {graph_search_radius!r}: the multiplier must "
            "be positive."
        )
    return max(1, round(mult * _expected_cell_size(node_feats)))


def max_search_radius(
    graph_search_radius: int | str | tuple | list, node_feats: pl.DataFrame
) -> int:
    """Largest radius the spec can reach, in pixels.

    The edge store is built at this radius so smaller values are reachable by filtering
    `dist_spat` on load.
    """
    resolved = resolve_search_radius(graph_search_radius, node_feats)
    return max(expand_param_range(resolved))
