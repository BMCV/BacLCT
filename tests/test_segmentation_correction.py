"""Test segmentation error correction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import EDGE_PRED_SCHEMA

from baclct.tracking.segmentation_correction import (
    CloseDetectionGaps,
    MergeEarlyDivisions,
    SplitFalseMerges,
)
from baclct.utils import segmentation_correction as sc_utils

_SCHEMA = {
    "index": pl.Int64,
    "t": pl.Int64,
    "label": pl.Int64,
    "label_track": pl.Int64,
    "parent_track": pl.Int64,
    "cell_source": pl.Utf8,
    "center-0": pl.Float64,
    "center-1": pl.Float64,
}


def _make_tracks(**col_lists) -> pl.DataFrame:
    """Build minimal `tracks` rows, defaulting every column that is not passed.

    Defaults are index=range, t=0, label=1, label_track=1, parent_track=0,
    cell_source='original' and centers 7.0. All lists must have the same length.
    """
    n = len(next(iter(col_lists.values())))
    data = {
        "index": list(range(n)),
        "t": [0] * n,
        "label": [1] * n,
        "label_track": [1] * n,
        "parent_track": [0] * n,
        "cell_source": ["original"] * n,
        "center-0": [7.0] * n,
        "center-1": [7.0] * n,
    }
    data.update(col_lists)
    return pl.DataFrame(data, schema=_SCHEMA)


def _edge_preds(edges: list[tuple[int, int, float, float]]) -> pl.DataFrame:
    """Build edge predictions from `(src, dst, p1, p2)` rows, p0 taking the remainder."""
    src, dst, p1, p2 = zip(*edges, strict=True) if edges else ((), (), (), ())
    return pl.DataFrame(
        {
            "src": src,
            "dst": dst,
            "p0": [1.0 - a - b for a, b in zip(p1, p2, strict=True)],
            "p1": p1,
            "p2": p2,
        },
        schema=EDGE_PRED_SCHEMA,
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_close_detection_gaps(ndim):
    """A detection gap is closed with an unchanged copy of the source cell, 2D or 3D.

    The copy is drawn where the source mask sits. The interpolated position only
    reaches the new row.
    """
    schema = {
        "index": pl.Int64,
        "t": pl.Int64,
        "label_track": pl.Int64,
        "parent_track": pl.Int64,
        "cell_source": pl.Utf8,
        "center-0": pl.Float64,
        "center-1": pl.Float64,
    }
    if ndim == 2:
        masks = np.zeros((3, 24, 24), dtype=np.uint16)
        masks[0, 4:9, 4:9] = 1
        masks[2, 6:11, 6:11] = 1
        centers = {"center-0": [6.0, 8.0], "center-1": [6.0, 8.0]}
        expected_pos = [7.0, 7.0]
    else:
        masks = np.zeros((3, 6, 24, 24), dtype=np.uint16)
        masks[0, 1:4, 4:9, 4:9] = 1
        masks[2, 1:4, 6:11, 6:11] = 1
        schema["center-2"] = pl.Float64
        centers = {
            "center-0": [2.0, 2.0],
            "center-1": [6.0, 8.0],
            "center-2": [6.0, 8.0],
        }
        expected_pos = [2.0, 7.0, 7.0]

    tracks = pl.DataFrame(
        {
            "index": [0, 1],
            "t": [0, 2],
            "label_track": [1, 1],
            "parent_track": [0, 0],
            "cell_source": ["original", "original"],
            **centers,
        },
        schema=schema,
    )

    corrector = CloseDetectionGaps(max_gap=2)
    corrected_tracks, corrected_masks = corrector.correct(tracks, masks)

    assert corrected_tracks.height == tracks.height + 1
    assert np.array_equal(corrected_masks[1] == 1, corrected_masks[0] == 1)
    new_row = corrected_tracks.filter(pl.col("t") == 1)
    assert new_row.height == 1
    assert new_row["cell_source"].item() == "interpolated"
    assert [new_row[c].item() for c in centers] == expected_pos


def test_find_nearest_background_pixel_3d():
    """The nearest background voxel is returned as a 3D index tuple."""
    frame = np.ones((4, 4, 4), dtype=np.uint16)
    frame[2, 1, 3] = 0
    frame[0, 0, 0] = 0  # farther from the center, so it must lose
    nearest = sc_utils.find_nearest_background_pixel(np.array([2.0, 1.0, 2.0]), frame)
    assert nearest == (2, 1, 3)


def _make_division_scenario(edges: list[tuple[int, int, float]] | None = None):
    """Return (tracks, masks, edge_preds) for the false split P(1) -> D1(2)/D2(3).

    D1 (short, t2-3) and the persistent sibling D2 (t2-5) touch as left/right halves.
    The short daughter ends before the sibling, so it clears the not-at-sequence-end
    filter in `get_short_tracks`. An appearing trajectory C'(4) at t6-7 covers the
    daughters' union, so the two merge back into it. `edges` holds (src, dst, p1)
    triples: the parent's last node is 1, the daughters' are 3 and 7, and the
    successor's first node is 8.
    """
    T = 8
    masks = np.zeros((T, 20, 20), dtype=int)
    for t in (0, 1):
        masks[t, 5:10, 6:14] = 1
    for t in (2, 3):
        masks[t, 5:10, 6:10] = 2  # D1 (left), short
    for t in (2, 3, 4, 5):
        masks[t, 5:10, 10:14] = 3  # D2 (right, adjacent -> touches D1), persistent
    for t in (6, 7):
        masks[t, 5:10, 6:14] = 4  # successor

    labels = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4]
    tracks = _make_tracks(
        **{
            "t": [0, 1, 2, 3, 2, 3, 4, 5, 6, 7],
            "label": labels,
            "label_track": labels,
            "parent_track": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            "center-1": [10.0, 10.0, 8.0, 8.0, 12.0, 12.0, 12.0, 12.0, 10.0, 10.0],
        }
    )
    return tracks, masks, _edge_preds([(s, d, p1, 0.0) for s, d, p1 in edges or []])


@pytest.mark.parametrize(
    ("edges", "merged"),
    [
        ([(7, 8, 0.9)], True),  # a daughter has a strong edge into the successor
        ([(1, 8, 0.9)], True),  # the parent does, straight across the daughters
        ([(7, 8, 0.9), (3, 8, 0.1)], True),  # one strong edge is enough
        ([(7, 8, 0.1), (1, 8, 0.2)], False),  # the edges exist and speak against it
        ([], True),  # nothing predicted: the mask coverage stands on its own
    ],
)
def test_merge_early_division(edges, merged):
    """A short division merging back into an appearing successor is undone if supported.

    The daughters cover the newly appearing cell, so they and it fold into the parent
    and the lineage becomes one continuous trajectory.
    """
    tracks, masks, edge_preds = _make_division_scenario(edges)

    corrected, corrected_masks = MergeEarlyDivisions().correct(
        tracks, masks, edge_predictions=edge_preds
    )

    if not merged:
        assert set(corrected["label_track"].to_list()) == {1, 2, 3, 4}
        assert corrected.filter(pl.col("parent_track") > 0).height > 0
        return

    # division gone, daughters and successor folded into parent label 1
    assert corrected.filter(pl.col("parent_track") > 0).height == 0
    assert set(corrected["label_track"].to_list()) == {1}
    assert set(np.unique(corrected_masks).tolist()) - {0} == {1}
    # the merged parent is now one continuous trajectory t0..t7
    assert sorted(corrected.filter(pl.col("label_track") == 1)["t"].to_list()) == (
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    )


def _make_split_scenario(
    p1_d1_to_x: float = 0.9,
    p1_d2_to_x: float = 0.9,
    p2_x_to_d1p: float = 0.9,
    p2_x_to_d2p: float = 0.9,
    p1_m_to_x: float = 0.05,
    p1_d_to_after: float = 0.05,
    wide_merge: bool = False,
):
    """Synthetic M to D1+D2(t=1) to X(t=2) to D1'+D2'(t=3) scenario.

    Labels and indices:
      M=1 (t=0, idx=0)
      D1=2 (t=1, idx=1), D2=3 (t=1, idx=2)   parent=M
      X=6  (t=2, idx=3), a distinct label for the merged cell
      D1'=4 (t=3, idx=4), D2'=5 (t=3, idx=5) parent=X_label=6

    Masks: D1 occupies [2:7, 2:8], D2 occupies [2:7, 8:14], X occupies [2:7, 2:14].
    D1 and D2 are adjacent (share the col-7/col-8 border) so the touch check passes.
    X uses a distinct label so neither D1 nor D2 appears in labels_with_children.
    `wide_merge` stretches X to [2:14, 2:14], which the daughters cover 0.42 of, so the
    geometry alone no longer carries the split and an edge has to support it.
    """
    T = 4
    masks = np.zeros((T, 20, 20), dtype=int)
    masks[0, 5:10, 5:10] = 1  # M
    masks[1, 2:7, 2:8] = 2  # D1 (cols 2-7)
    masks[1, 2:7, 8:14] = 3  # D2 (cols 8-13, adjacent to D1)
    masks[2, 2 : 14 if wide_merge else 7, 2:14] = 6  # X (merged, distinct label)
    masks[3, 2:7, 2:8] = 4  # D1'
    masks[3, 2:7, 8:14] = 5  # D2'

    labels = [1, 2, 3, 6, 4, 5]
    tracks = _make_tracks(
        **{
            "t": [0, 1, 1, 2, 3, 3],
            "label": labels,
            "label_track": labels,
            "parent_track": [0, 1, 1, 0, 6, 6],  # D1/D2 parent=M, D1'/D2' parent=X(=6)
            "center-0": [7.0, 4.5, 4.5, 4.5, 4.5, 4.5],
            "center-1": [7.0, 4.5, 10.5, 7.5, 4.5, 10.5],
        }
    )

    edge_preds = _edge_preds(
        [
            (1, 3, p1_d1_to_x, 0.05),  # D1 -> X
            (2, 3, p1_d2_to_x, 0.05),  # D2 -> X
            (3, 4, 0.05, p2_x_to_d1p),  # X -> D1'
            (3, 5, 0.05, p2_x_to_d2p),  # X -> D2'
            (0, 3, p1_m_to_x, 0.05),  # M -> X
            (1, 4, p1_d_to_after, 0.05),  # D1 -> D1'
            (2, 5, p1_d_to_after, 0.05),  # D2 -> D2'
        ]
    )
    return tracks, masks, edge_preds


def test_split_false_merges():
    """The merged mask is split back into D1/D2, and D1'/D2' are relabeled to them."""
    tracks, masks, edge_preds = _make_split_scenario()

    corrector = SplitFalseMerges(thr_corr=0.5, thr_div=0.5)
    corrected_tracks, corrected_masks = corrector.correct(
        tracks, masks, edge_predictions=edge_preds
    )

    # X is split: D1 (label 2) and D2 (label 3) reappear at the merge frame t=2
    t2_labels = corrected_tracks.filter(pl.col("t") == 2)["label_track"].to_list()
    assert 2 in t2_labels and 3 in t2_labels
    assert (corrected_masks[2] == 2).any() and (corrected_masks[2] == 3).any()

    # the post-merge daughters D1'(4)/D2'(5) are relabeled back to D1(2)/D2(3) at t=3
    t3_labels = corrected_tracks.filter(pl.col("t") == 3)["label_track"].to_list()
    assert 4 not in t3_labels and 5 not in t3_labels
    assert 2 in t3_labels and 3 in t3_labels
    assert (corrected_masks[3] == 4).sum() == 0 and (corrected_masks[3] == 5).sum() == 0
    assert (corrected_masks[3] == 2).any() and (corrected_masks[3] == 3).any()


@pytest.mark.parametrize(
    ("arm", "wide_merge", "scores", "expect_split"),
    [
        # the daughters cover X exactly, so the geometry decides on its own
        ("coverage", False, {}, True),
        # X is too wide to be carried by coverage, so one edge each has to support it
        ("parent", True, {"p1_m_to_x": 0.9}, True),
        ("division", True, {"p2_x_to_d1p": 0.9}, True),
        ("carries_on", True, {"p1_d_to_after": 0.9}, True),
        ("contradicted", True, {}, False),
        # the signal the step must not depend on: only the daughters link into X
        ("daughter_into_merge", True, {"p1_d1_to_x": 0.9}, False),
    ],
)
def test_split_evidence_arms(arm, wide_merge, scores, expect_split):
    """A split needs coverage or one supporting edge, never the daughter link."""
    quiet = {
        "p1_d1_to_x": 0.05,
        "p1_d2_to_x": 0.05,
        "p2_x_to_d1p": 0.05,
        "p2_x_to_d2p": 0.05,
    }
    tracks, masks, edge_preds = _make_split_scenario(
        wide_merge=wide_merge, **{**quiet, **scores}
    )

    corrector = SplitFalseMerges(thr_corr=0.5, thr_div=0.5)
    corrected_tracks, corrected_masks = corrector.correct(
        tracks, masks, edge_predictions=edge_preds
    )

    t2_labels = corrected_tracks.filter(pl.col("t") == 2)["label_track"].to_list()
    if expect_split:
        assert 2 in t2_labels and 3 in t2_labels, f"{arm}: expected a split"
        assert (corrected_masks[2] == 2).any() and (corrected_masks[2] == 3).any()
    else:
        assert t2_labels == [6], f"{arm}: expected the merge to be kept"
        assert (corrected_masks[2] == 6).any()


def test_split_skips_merge_longer_than_max_merge_frames():
    """A merge that outlasts `max_merge_frames` is left to the tracker."""
    tracks, masks, edge_preds = _make_split_scenario_multiframe()

    kept, kept_masks = SplitFalseMerges(max_merge_frames=2).correct(
        tracks.clone(), masks.copy(), edge_predictions=edge_preds
    )
    assert kept.filter(pl.col("label_track") == 6).height == 3
    assert (kept_masks[2:5] == 6).any()

    # the same merge is split once the bound admits its three frames
    split, _ = SplitFalseMerges(max_merge_frames=3).correct(
        tracks, masks, edge_predictions=edge_preds
    )
    assert split.filter(pl.col("label_track") == 6).height == 0


# D1/D2 -> X as correspondences, X -> nodes 6/7 as divisions. Shared by the two
# scenarios below, in which 6 and 7 are the cells that reappear after the merge.
_MERGE_EDGES = [
    (1, 3, 0.9, 0.05),
    (2, 3, 0.9, 0.05),
    (3, 6, 0.05, 0.9),
    (3, 7, 0.05, 0.9),
]


def _make_split_scenario_foreign_rediv():
    """Split scenario where X's top-p2 edges point at a foreign lineage's daughters.

    M=1 divides into D1=2/D2=3 (t=1) which merge into X=6 (t=2). Separately, an
    unrelated parent P=7 divides into A=8/B=9 (t=3). X's strongest p2 edges point at
    A and B, so they are selected as X's re-division products even though their real
    parent is 7, not X. Relabeling them back to D1/D2 would sever track 8/9 and leave
    P=7 with a corrupted division.

    Labels/indices:
      M=1 (t0, idx0), D1=2/D2=3 (t1, idx1/2) parent M, X=6 (t2, idx3)
      P=7 (t1/t2, idx4/5), A=8/B=9 (t3, idx6/7) parent P=7
    """
    T = 4
    masks = np.zeros((T, 20, 20), dtype=int)
    masks[0, 5:10, 5:10] = 1  # M
    masks[1, 2:7, 2:8] = 2  # D1
    masks[1, 2:7, 8:14] = 3  # D2 (adjacent to D1)
    masks[2, 2:7, 2:14] = 6  # X (merged)
    masks[1, 13:18, 2:8] = 7  # P (foreign lineage, away from X)
    masks[2, 13:18, 2:8] = 7
    masks[3, 13:18, 2:8] = 8  # A (foreign daughter)
    masks[3, 13:18, 8:14] = 9  # B (foreign daughter, adjacent to A)

    labels = [1, 2, 3, 6, 7, 7, 8, 9]
    tracks = _make_tracks(
        **{
            "t": [0, 1, 1, 2, 1, 2, 3, 3],
            "label": labels,
            "label_track": labels,
            "parent_track": [0, 1, 1, 0, 0, 0, 7, 7],
            "center-0": [7.0, 4.5, 4.5, 4.5, 15.5, 15.5, 15.5, 15.5],
            "center-1": [7.0, 4.5, 10.5, 7.5, 4.5, 4.5, 4.5, 10.5],
        }
    )
    return tracks, masks, _edge_preds(_MERGE_EDGES)


def test_split_does_not_relabel_foreign_redivision_products():
    """A high-p2 edge into another lineage must not sever that lineage's tracks."""
    tracks, masks, edge_preds = _make_split_scenario_foreign_rediv()

    corrector = SplitFalseMerges(thr_corr=0.5, thr_div=0.5)
    corrected_tracks, _ = corrector.correct(tracks, masks, edge_predictions=edge_preds)

    # the foreign daughters keep their label and their real parent P=7. They are
    # never relabeled into D1/D2.
    foreign = corrected_tracks.filter(pl.col("label_track").is_in([8, 9]))
    assert sorted(foreign["label_track"].unique().to_list()) == [8, 9]
    assert foreign["parent_track"].unique().to_list() == [7]
    # P=7 still has exactly its two original daughters.
    children_of_7 = (
        corrected_tracks.filter(pl.col("parent_track") == 7)["label_track"]
        .unique()
        .to_list()
    )
    assert sorted(children_of_7) == [8, 9]


def _make_split_scenario_multiframe():
    """False merge that stays merged for several frames before genuinely re-dividing.

    M=1 divides into D1=2/D2=3 (t=1) which merge into X=6 spanning t=2,3,4, then X
    genuinely re-divides back into D1'=4/D2'=5 (t=5, parent X=6). The split must delete
    X across the whole [2,4] stretch and leave D1/D2 under their real parent M=1, so no
    phantom X is left behind for the following gap-closing step to bridge.
    """
    T = 6
    masks = np.zeros((T, 20, 20), dtype=int)
    masks[0, 5:10, 5:10] = 1  # M
    masks[1, 2:7, 2:8] = 2  # D1
    masks[1, 2:7, 8:14] = 3  # D2 (adjacent to D1)
    for t in (2, 3, 4):
        masks[t, 2:7, 2:14] = 6  # X (merged, multi-frame)
    masks[5, 2:7, 2:8] = 4  # D1'
    masks[5, 2:7, 8:14] = 5  # D2'

    labels = [1, 2, 3, 6, 6, 6, 4, 5]
    tracks = _make_tracks(
        **{
            "t": [0, 1, 1, 2, 3, 4, 5, 5],
            "label": labels,
            "label_track": labels,
            "parent_track": [0, 1, 1, 0, 0, 0, 6, 6],
            "center-0": [7.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
            "center-1": [7.0, 4.5, 10.5, 7.5, 7.5, 7.5, 4.5, 10.5],
        }
    )
    return tracks, masks, _edge_preds(_MERGE_EDGES)


def test_split_multiframe_merge_deletes_whole_stretch():
    """A multi-frame merge is removed across its whole span, not just the merge frame."""
    tracks, masks, edge_preds = _make_split_scenario_multiframe()

    corrector = SplitFalseMerges(thr_corr=0.5, thr_div=0.5)
    corrected_tracks, corrected_masks = corrector.correct(
        tracks, masks, edge_predictions=edge_preds
    )

    # X=6 is gone from tracks and masks across the entire merged stretch.
    assert corrected_tracks.filter(pl.col("label_track") == 6).height == 0
    assert not (corrected_masks[2:5] == 6).any()
    # the re-division products inherit M=1, not the deleted X, with no parent split.
    for label in (2, 3):
        parents = (
            corrected_tracks.filter(pl.col("label_track") == label)["parent_track"]
            .unique()
            .to_list()
        )
        assert parents == [1]


def _make_single_gap_scenario():
    """Return (tracks, masks) for one track present at t=0 and t=2, gap at t=1.

    The cell sits at (20, 20) and has no mask at t=1, a one-frame gap.
    """
    T, H, W = 3, 40, 40
    masks = np.zeros((T, H, W), dtype=int)
    masks[0, 15:25, 15:25] = 1
    masks[2, 15:25, 15:25] = 1
    tracks = _make_tracks(
        **{"t": [0, 2], "center-0": [20.0, 20.0], "center-1": [20.0, 20.0]}
    )
    return tracks, masks


def _add_gap_only_cell(tracks, masks, rows: slice, cols: slice):
    """Add a spurious one-frame cell (label 2) in the gap of the single-gap scenario."""
    masks[1, rows, cols] = 2
    spurious = _make_tracks(
        **{
            "index": [2],
            "t": [1],
            "label": [2],
            "label_track": [2],
            "center-0": [float((rows.start + rows.stop - 1) / 2)],
            "center-1": [float((cols.start + cols.stop - 1) / 2)],
        }
    )
    return pl.concat([tracks, spurious]), masks


@pytest.mark.parametrize("outcome", ["relabel", "remove", "draw_over"])
def test_close_gaps_resolves_conflicting_mask(outcome):
    """Each of the three conflict outcomes fires on the situation it was written for.

    The conflicting mask is a one-frame trajectory inside the gap, as an intensity flip
    or a temporary undersegmentation leaves behind. No integration case reaches these
    branches, so this is the only coverage they have.
    """
    tracks, masks = _make_single_gap_scenario()
    if outcome == "draw_over":
        # overlaps the copy on three columns: too little to remove, enough to conflict
        tracks, masks = _add_gap_only_cell(tracks, masks, slice(15, 25), slice(22, 35))
    else:
        tracks, masks = _add_gap_only_cell(tracks, masks, slice(17, 23), slice(17, 23))

    # relabel needs the spurious cell to link to the cell after the gap, the other two
    # need every edge it has to stay weak
    p1 = 0.9 if outcome == "relabel" else 0.2
    edge_predictions = _edge_preds([(2, 1, p1, 0.0)])

    corrector = CloseDetectionGaps(max_gap=2, thr_corr=0.5)
    corrected, corrected_masks = corrector.correct(
        tracks, masks, edge_predictions=edge_predictions
    )

    gap_frame = corrected_masks[1]
    sources = corrected.filter(pl.col("t") == 1)["cell_source"].to_list()
    if outcome == "relabel":
        # the spurious mask itself becomes the missing detection, nothing is drawn
        assert 2 not in corrected["label_track"].to_list()
        assert (gap_frame == 1).sum() == 36
        assert sources == ["original"]
    elif outcome == "remove":
        # the spurious mask is deleted and the copy takes its place
        assert 2 not in corrected["label_track"].to_list()
        assert (gap_frame == 1).sum() == 100
        assert sources == ["interpolated"]
    else:
        # the copy is drawn over the overlap, the rest of the conflicting mask survives
        assert (gap_frame == 1).sum() == 100
        assert (gap_frame == 2).sum() == 100
        assert sorted(sources) == ["interpolated", "original"]


def _textured_images(gap_is_flat: bool):
    """Return a (3, 40, 40) stack: textured cell at t=0,2 over flat-ish background.

    The cell region [15:25, 15:25] carries strong texture (high std) at the bracket
    frames. When `gap_is_flat`, the same region at the gap frame t=1 is constant (std
    0, background) so the detection gate sees a true negative. Otherwise it is textured
    too (a real missed detection).
    """
    rng = np.random.default_rng(0)
    images = rng.normal(100.0, 4.0, size=(3, 40, 40))  # light background texture
    for t in (0, 2):
        images[t, 15:25, 15:25] = rng.normal(100.0, 60.0, size=(10, 10))
    if gap_is_flat:
        images[1, 15:25, 15:25] = 100.0  # cell gone: flat patch
    else:
        images[1, 15:25, 15:25] = rng.normal(100.0, 60.0, size=(10, 10))
    return images


def test_detection_gate_fills_gap_with_detection_candidate():
    """A gap frame as textured as the cell is a real FN: it is interpolated (filled)."""
    tracks, masks = _make_single_gap_scenario()
    images = _textured_images(gap_is_flat=False)

    corrector = CloseDetectionGaps(max_gap=2, require_detection=True)
    corrected, corrected_masks = corrector.correct(tracks, masks, images=images)

    # gap bridged: one cell added, still a single continuous trajectory
    assert corrected.height == tracks.height + 1
    assert corrected["label_track"].n_unique() == 1
    assert corrected_masks[1].max() == 1, "Gap frame was not filled."


def test_detection_gate_leaves_gap_unbridged_then_consolidation_splits():
    """A flat (background) gap footprint is a TN: the gate leaves the gap unbridged.

    Splitting the trajectory is deferred to the final consolidation step
    (`split_unbridged_gaps`), so the gate alone invents no cell and keeps the gap.
    """
    tracks, masks = _make_single_gap_scenario()
    images = _textured_images(gap_is_flat=True)

    corrector = CloseDetectionGaps(max_gap=2, require_detection=True)
    gapped, gapped_masks = corrector.correct(tracks, masks, images=images)

    # no cell invented: the trajectory keeps its label and its (now unbridged) gap
    assert gapped.height == tracks.height
    assert gapped["label_track"].n_unique() == 1
    assert gapped_masks[1].max() == 0

    # consolidation splits the unbridged gap into two appearing trajectories
    corrected, corrected_masks = sc_utils.split_unbridged_gaps(gapped, gapped_masks)
    assert corrected.height == tracks.height
    assert corrected["label_track"].n_unique() == 2
    new_label = next(lbl for lbl in corrected["label_track"].to_list() if lbl != 1)
    new_track = corrected.filter(pl.col("label_track") == new_label)
    assert new_track["parent_track"].to_list() == [0], "Post-gap track must appear."
    assert new_track["t"].to_list() == [2]
    # masks relabeled accordingly, gap frame still empty, both tracks gap-free
    assert corrected_masks[1].max() == 0
    assert set(np.unique(corrected_masks).tolist()) - {0} == {1, new_label}


def test_drop_orphan_rows_splits_clobbered_marker():
    """A marker overwritten by a neighbor leaves a track row with no mask pixel.

    Regression for the TOIAM-subsampling crash: a 1-pixel interpolated marker at t=1 is
    overwritten by a cell placed over the same region later in the pass, so the row has no
    matching mask. drop_orphan_rows removes it and the consolidation splits the gap, so
    the final state is CTC-valid.
    """
    masks = np.zeros((3, 10, 10), dtype=np.int64)
    masks[0, 1:3, 1:3] = 1  # label 1 present at t=0
    masks[2, 1:3, 1:3] = 1  # label 1 present at t=2
    masks[1, 5:7, 5:7] = 2  # at t=1 a neighbor owns the region, clobbering the marker

    tracks = _make_tracks(
        t=[0, 1, 2, 1],
        label_track=[1, 1, 1, 2],
        cell_source=["original", "interpolated_marker", "original", "original"],
    )

    pruned = sc_utils.drop_orphan_rows(tracks, sc_utils.build_mask_label_map(masks))
    assert pruned.filter((pl.col("t") == 1) & (pl.col("label_track") == 1)).height == 0

    corrected, corrected_masks = sc_utils.split_unbridged_gaps(pruned, masks)
    # must not raise
    sc_utils.assert_ctc_valid(corrected, sc_utils.build_mask_label_map(corrected_masks))
    # label 1's post-gap piece becomes a new appearing track (1, 2, and the new piece)
    assert corrected["label_track"].n_unique() == 3


def test_drop_orphan_rows_resets_dangling_parent():
    """A child of a parent that vanishes entirely has its parent_track reset to 0."""
    masks = np.zeros((2, 10, 10), dtype=np.int64)
    masks[0, 1:3, 1:3] = 2  # only the child (label 2) is ever segmented
    masks[1, 1:3, 1:3] = 2

    tracks = _make_tracks(
        t=[0, 0, 1],
        label_track=[1, 2, 2],
        parent_track=[0, 1, 1],
        cell_source=["interpolated_marker", "original", "original"],
    )

    pruned = sc_utils.drop_orphan_rows(tracks, sc_utils.build_mask_label_map(masks))
    assert 1 not in pruned["label_track"].to_list()
    child_parents = pruned.filter(pl.col("label_track") == 2)["parent_track"].unique()
    assert child_parents.to_list() == [0]
