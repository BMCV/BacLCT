"""LAPTracker behaviour on degenerate graphs: gaps, false positives, lost daughters."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import _find_division

from baclct.tracking.tracker import LAPTracker


@pytest.mark.parametrize(
    "missing_frames, time_steps, bridge",
    [
        ([2], [1, 2], (1, 3)),
        ([1, 2], [1, 2, 3], (0, 3)),
    ],
)
def test_lap_tracker_bridges_empty_frame(
    toy_features_with_empty_frame_factory,
    perfect_predictions_factory,
    missing_frames,
    time_steps,
    bridge,
    mock_dataset_factory,
):
    """A trajectory survives frames that hold no detections at all."""
    node_feats = toy_features_with_empty_frame_factory(remove_frames=missing_frames)
    predictions = perfect_predictions_factory(node_feats, time_steps)

    tracker = LAPTracker(
        dataset=mock_dataset_factory(node_feats=node_feats, masks=None),
        predictions=predictions,
        thr_corr=0.9,
        segmentation_correction=None,
    )
    tracks = tracker.track()

    # label 13 spans all four frames of the toy data
    track_13 = tracks.filter(pl.col("label") == 13)
    t_start, t_end = bridge
    start = track_13.filter(pl.col("t") == t_start)
    end = track_13.filter(pl.col("t") == t_end)

    assert not start.is_empty()
    assert not end.is_empty()
    assert start["label_track"][0] == end["label_track"][0]


def test_lap_tracker_bridges_trajectory_gap(
    toy_handcrafted_features,
    perfect_predictions_factory,
    mock_dataset_factory,
):
    """A cell missing from two frames its neighbours still occupy keeps its track."""
    node_feats = toy_handcrafted_features.filter(
        ~(pl.col("label").eq(13) & pl.col("t").is_in([1, 2]))
    )
    predictions = perfect_predictions_factory(node_feats, [1, 2, 3])

    tracker = LAPTracker(
        dataset=mock_dataset_factory(node_feats=node_feats, masks=None),
        predictions=predictions,
        thr_corr=0.9,
        segmentation_correction=None,
    )
    tracks = tracker.track()

    # every frame is populated, so the tracker has to carry label 13 across two of them
    assert sorted(np.unique(tracks["t"]).tolist()) == [0, 1, 2, 3]
    track_13 = tracks.filter(pl.col("label") == 13).sort("t")
    assert track_13["t"].to_list() == [0, 3]
    assert track_13["label_track"].n_unique() == 1


def test_lap_tracker_isolates_spurious_detections(
    toy_features_with_spurious_detections,
    predictions_for_spurious_detections,
    mock_dataset_factory,
):
    """False positives stay in single-node trajectories instead of stealing a link."""
    node_feats, spurious_labels = toy_features_with_spurious_detections
    tracker = LAPTracker(
        dataset=mock_dataset_factory(node_feats=node_feats, masks=None),
        predictions=predictions_for_spurious_detections,
        thr_corr=0.9,
        segmentation_correction=None,
    )
    tracks = tracker.track()

    spurious = tracks.filter(pl.col("label").is_in(spurious_labels))
    assert spurious.height == len(spurious_labels)
    assert spurious["label_track"].n_unique() == len(spurious_labels)

    real = tracks.filter(~pl.col("label").is_in(spurious_labels))
    assert real.filter(
        pl.col("label_track").is_in(spurious["label_track"].to_list())
    ).is_empty()


@pytest.mark.parametrize("relabel_single_daughter_divs", [True, False])
def test_lap_tracker_handles_missing_daughter(
    toy_features_with_missing_daughter,
    predictions_for_missing_daughter,
    relabel_single_daughter_divs,
    mock_dataset_factory,
):
    """A division with one daughter lost becomes a correspondence, or nothing."""
    node_feats, parent_label, daughter_label = toy_features_with_missing_daughter
    tracker = LAPTracker(
        dataset=mock_dataset_factory(node_feats=node_feats, masks=None),
        predictions=predictions_for_missing_daughter,
        thr_corr=0.9,
        segmentation_correction=None,
        relabel_single_daughter_divs=relabel_single_daughter_divs,
    )
    tracks = tracker.track()

    parent = tracks.filter(pl.col("label") == parent_label)
    daughter = tracks.filter(pl.col("label") == daughter_label)
    assert not parent.is_empty()
    assert not daughter.is_empty()

    # the single daughter is never a division, so it either continues the parent or
    # starts a trajectory of its own
    same_track = daughter["label_track"][0] == parent["label_track"][0]
    assert same_track is relabel_single_daughter_divs
    assert daughter["parent_track"][0] == 0


def test_lap_tracker_detects_division_across_a_gap(
    toy_handcrafted_features,
    perfect_predictions_factory,
    mock_dataset_factory,
):
    """A division whose first frame is missing still reaches both daughters."""
    parent_label, daughters = _find_division(toy_handcrafted_features)
    is_daughter = pl.col("label").is_in(daughters)
    t_birth = int(toy_handcrafted_features.filter(is_daughter)["t"].to_numpy().min())
    node_feats = toy_handcrafted_features.filter(~(is_daughter & pl.col("t").eq(t_birth)))

    tracker = LAPTracker(
        dataset=mock_dataset_factory(node_feats=node_feats, masks=None),
        predictions=perfect_predictions_factory(node_feats, [1, 2]),
        thr_corr=0.9,
        segmentation_correction=None,
    )
    tracks = tracker.track()

    parent = tracks.filter(pl.col("label") == parent_label)
    both = tracks.filter(is_daughter, pl.col("t") == t_birth + 1)
    assert both.height == len(daughters)
    assert both["parent_track"].unique().to_list() == [parent["label_track"][0]]
