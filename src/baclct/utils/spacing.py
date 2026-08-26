"""Physical frame and voxel spacing used to make distances acquisition-invariant."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

# spatial axes ordered slowest to fastest, matching numpy axes after time
SPACING_AXES = ("t", "z", "y", "x")


def normalize_spacing(spacing: SpacingLike = None) -> dict[str, float]:
    """Convert a spacing to a `{t, z, y, x}` dict.

    Each value is the physical size of one step along that axis: the voxel size for `z, y,
    x` (µm) and the interval between frames for `t` (min). Unspecified axes default to
    `1.0`.

    Args:
        spacing: `None` (all `1.0`), a mapping over any subset of 't', 'z', 'y', 'x', or a
            length-4 sequence in `(t, z, y, x)` order with `1.0` for unused axes. Shorter
            sequences are rejected as ambiguous.
    """
    out = dict.fromkeys(SPACING_AXES, 1.0)
    if spacing is None:
        return out

    if isinstance(spacing, Mapping):
        for key, val in spacing.items():
            if key not in out:
                raise ValueError(
                    f"Unknown spacing axis {key!r}, expected {SPACING_AXES}."
                )
            out[key] = float(val)  # type: ignore
        return out

    if isinstance(spacing, Sequence) and not isinstance(spacing, str):
        seq = list(spacing)
        if len(seq) != len(SPACING_AXES):
            raise ValueError(
                f"A spacing sequence must be full (t, z, y, x); got length {len(seq)}. "
                "Use a mapping to set only some axes."
            )
        return {axis: float(val) for axis, val in zip(SPACING_AXES, seq, strict=True)}

    raise TypeError(f"Cannot interpret spacing of type {type(spacing)!r}.")


def spatial_spacing(spacing: dict[str, float], ndim: int) -> tuple[float, ...]:
    """Spatial spacing as a `(z, y, x)`-ordered tuple for an `ndim`-dimensional image."""
    return tuple(spacing[axis] for axis in SPACING_AXES[1:][-ndim:])


def needs_spacing(spacing: Sequence[float] | None) -> bool:
    """Whether spacing must be applied: present and not unit (1.0 on every axis).

    Absent or unit spacing is a no-op that leaves coordinates and sizes in pixel units, so
    no rescaling, `_px` columns, or cache metadata are needed.
    """
    return spacing is not None and any(s != 1.0 for s in spacing)


def position_columns(cols: Iterable[str]) -> list[str]:
    """Position column names in axis order, or empty if no complete family is present.

    Prefers pixel space (`_px`) and center over centroid. Under a non-unit spacing the
    unsuffixed columns hold physical units, so anything indexing an array must go through
    here. Any iterable of column names works, so a single `tracks` row as a dict can be
    passed directly.
    """
    for suffix in ("_px", ""):
        for prefix in ("center", "centroid"):
            ordered = _ordered_axis_cols(cols, prefix, suffix)
            if ordered:
                return ordered
    return []


def _ordered_axis_cols(cols: Iterable[str], prefix: str, suffix: str) -> list[str]:
    """Position columns of one family in axis order, or empty if incomplete.

    Matches `{prefix}-{axis}{suffix}` and returns them ordered by axis only when the
    axes form a contiguous 0..n-1 range.
    """
    pat = rf"{re.escape(prefix)}-(\d+){re.escape(suffix)}"
    found = {int(m.group(1)): c for c in cols if (m := re.fullmatch(pat, c))}
    if found and set(found) == set(range(len(found))):
        return [found[i] for i in range(len(found))]
    return []


SpacingLike = None | Mapping[str, float] | Sequence[float]
