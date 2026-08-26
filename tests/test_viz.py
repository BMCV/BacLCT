"""Test trajectory visualization."""

from __future__ import annotations

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

from baclct.viz import TrajectoryVisualizer  # noqa: E402


def test_prepare_tracks_normalizes_to_the_public_schema():
    """Track ids replace mask labels, so a corrected frame plots its trajectories."""
    df = TrajectoryVisualizer._prepare_tracks(
        pl.DataFrame(
            {
                "t": [0, 1],
                "label": [5, 5],
                "label_track": [7, 7],
                "parent_track": [3, 3],
                "center-0": [1.0, 2.0],
                "center-1": [3.0, 4.0],
                "centroid-0": [9.0, 9.0],
                "centroid-1": [9.0, 9.0],
            }
        )
    )

    assert df["label"].to_list() == [7, 7]
    assert df["parent"].to_list() == [3, 3]
    assert "label_track" not in df.columns


def test_prepare_tracks_requires_positions():
    """A frame without any position column is rejected rather than plotted empty."""
    with pytest.raises(ValueError, match="center-"):
        TrajectoryVisualizer._prepare_tracks(pl.DataFrame({"t": [0], "label": [1]}))


def test_show_draws_lineage_and_divisions(toy_images, toy_masks, toy_tracks_df):
    """Divisions render as white lines alongside the per-track lineage."""
    viz = TrajectoryVisualizer(toy_images, toy_masks, toy_tracks_df, name="toy")
    _, ax = viz.show(t=3, n_frames=3, plot_contours=False, verbose=False)

    # the tracks carry both families, and the medial center is what the model tracks on
    assert viz.coords == ["center-0", "center-1"]

    # lineage lines carry a per-track rgb triple, divisions the literal 'white'
    divisions = [ln for ln in ax.lines if isinstance(ln.get_color(), str)]
    lineage = [ln for ln in ax.lines if not isinstance(ln.get_color(), str)]
    assert divisions, "no division lines drawn"
    assert lineage, "no lineage lines drawn"
