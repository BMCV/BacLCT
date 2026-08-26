"""Node feature normalization functions."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs


def scale_relative_size(
    features: pl.DataFrame, ignore_border: bool = True
) -> pl.DataFrame:
    """Normalize image features based on initial cell size.

    Normalizes size-based features (e.g., axis lengths) with respect to the median cell
    size at the first time point (t=0) and applies a log scale. This is useful for
    datasets with growing cells.

    Other bounded features such as intensity, eccentricity, orientation, etc., are not
    considered here.

    Args:
        features: Unscaled single-cell features.
        ignore_border: If `True`, ignores cells touching the image border when computing
            median.

    Returns:
        Original features and normalized feature columns suffixed with `_norm`.
    """
    feats_init = features.filter(t=pl.col.t.min())
    if "touches_border" in features.columns and ignore_border:
        feats_init = feats_init.filter(~pl.col("touches_border"))

    # ensure we don't divide by zero if the initial frame is empty or lacks features
    if feats_init.height == 0:
        return features.with_columns(
            len_init=pl.lit(1.0, dtype=pl.Float32),
            area_init=pl.lit(1.0, dtype=pl.Float32),
        )

    len_init = feats_init.get_column("axis_major_length").median()
    if len_init is None or len_init == 0:
        len_init = 1.0

    area_init = feats_init.get_column("area").median()
    if area_init is None or area_init == 0:
        area_init = 1.0

    # define columns to be normalized by length
    length_dependent_cols = cs.contains(
        "axis", "thickness", "septum_width"
    ) - cs.ends_with("_norm")

    eps = 1e-6
    return features.with_columns(
        ((length_dependent_cols + eps) / (pl.lit(len_init) + eps))
        .log()
        .name.suffix("_norm"),
        area_norm=((pl.col("area") + eps) / (pl.lit(area_init) + eps)).log(),
        len_init=pl.lit(len_init, dtype=pl.Float32),
        area_init=pl.lit(area_init, dtype=pl.Float32),
    )
