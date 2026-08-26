"""Test custom edge features."""

from __future__ import annotations

import numpy as np
import polars as pl

from baclct.features.custom_edge_features import relative_size


def test_relative_size():
    """relative_size is the log ratio of dst to src size."""
    edge_data = pl.DataFrame(
        {
            "area_src": [10.0, 20.0],
            "area_dst": [20.0, 10.0],
        }
    )

    featured_data = relative_size(edge_data, based_on="area")

    assert "relative_size" in featured_data.columns
    assert np.isclose(featured_data["relative_size"][0], np.log(2.0))  # type: ignore
    assert np.isclose(featured_data["relative_size"][1], np.log(0.5))  # type: ignore
