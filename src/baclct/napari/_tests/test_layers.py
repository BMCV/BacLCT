"""Tests for the tracks-to-napari conversion and the Tracks writer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from baclct.napari._layers import tracks_to_napari
from baclct.napari._writer import write_tracks_csv


@pytest.fixture
def tracks() -> pl.DataFrame:
    """Two tracks, where track 3 is a daughter of track 1."""
    return pl.DataFrame(
        {
            "label": [1, 1, 3, 2],
            "t": [0, 1, 2, 0],
            "centroid-0": [10.0, 11.0, 12.0, 50.0],
            "centroid-1": [20.0, 21.0, 22.0, 60.0],
            "parent": [0, 0, 1, 0],
            "state": [0, 1, 1, 2],
        }
    )


def test_tracks_to_napari_sorts_and_builds_lineage(tracks):
    data, graph, features = tracks_to_napari(tracks)

    assert data.shape == (4, 4)  # [track_id, t, y, x]
    # napari requires the data sorted by (track_id, t)
    order = np.lexsort((data[:, 1], data[:, 0]))
    assert np.array_equal(order, np.arange(len(data)))

    # only the real division edge is an edge; roots (parent 0) are not
    assert graph == {3: [1]}
    assert np.array_equal(features["state"], np.array([0, 1, 2, 1]))


def test_tracks_to_napari_ignores_self_parents():
    tracks = pl.DataFrame(
        {
            "label": [1, 2],
            "t": [0, 0],
            "centroid-0": [1.0, 2.0],
            "centroid-1": [1.0, 2.0],
            "parent": [1, 0],  # a track that is its own parent is a root, not an edge
        }
    )
    _, graph, _ = tracks_to_napari(tracks)
    assert graph == {}


def test_tracks_to_napari_supports_3d():
    tracks = pl.DataFrame(
        {
            "label": [1],
            "t": [0],
            "centroid-0": [1.0],
            "centroid-1": [2.0],
            "centroid-2": [3.0],
            "parent": [0],
        }
    )
    data, _, _ = tracks_to_napari(tracks)
    assert data.shape == (1, 5)  # [track_id, t, z, y, x]
    assert np.array_equal(data[0], [1, 0, 1.0, 2.0, 3.0])


def test_write_tracks_csv_roundtrip(tracks, tmp_path):
    data, graph, features = tracks_to_napari(tracks)
    path = tmp_path / "out.csv"

    written = write_tracks_csv(str(path), data, {"graph": graph, "features": features})

    assert written == [str(path)]
    out = pl.read_csv(path)
    assert out.columns == ["label", "t", "centroid-0", "centroid-1", "parent", "state"]
    # the lineage survives the round trip, roots included
    assert out.sort("label", "t")["parent"].to_list() == [0, 0, 0, 1]
