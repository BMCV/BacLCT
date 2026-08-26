"""Shared helpers for segmentation correction.

In file order:

    1. Geometry and masks. Contains checks and computation for gap closing and conflict
       identification, e.g., computing overlap, connectivity, and interpolated positions.
    2. Edge scores. Retrieve the correspondence (`p1`) and division (`p2`) predictions
       and check whether two cells are potentially linked over time.
    3. Trajectory queries. Identify a cell's parent, daughters, lifetime, and trajectory
       gaps.
    4. Track edits. Add, drop, relabel, and split trajectories.
    5. Detection score. Decide from image intensity whether a gap frame might come from a
       missed detection or from a tracking error.
    6. Validation. Check that corrected `tracks` follow CTC rules: single detection for
       each frame, two daughters starting the frame after their parent ends, and a mask
       per row.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from skimage.measure import label as label_connected_components

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# how far around a footprint to look for unsegmented pixels when scoring it
_BACKGROUND_MARGIN_PX = 15


# geometry and masks


def ensure_connected(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a boolean mask."""
    labeled = label_connected_components(mask)
    if labeled.max() <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == int(sizes.argmax())


def would_fragment(
    frame: np.ndarray,
    conflict_label: int,
    new_mask: np.ndarray,
    max_fragment_frac: float,
) -> bool:
    """Return True if drawing new_mask would break conflict_label.

    What is left of `conflict_label` after `new_mask` is drawn over it must stay in one
    piece, or split off no more than `max_fragment_frac` of those pixels.
    """
    remaining = (frame == conflict_label) & ~new_mask
    if not remaining.any():
        return False
    labeled = label_connected_components(remaining)
    if labeled.max() <= 1:
        return False
    sizes = np.bincount(labeled.ravel())[1:]  # drop background
    return float(sizes.min()) / float(sizes.sum()) > max_fragment_frac


def source_mask(masks: np.ndarray, t: int, label: int) -> np.ndarray | None:
    """Return a copy of label's pixels at frame t, or None if absent."""
    cell_mask = masks[t] == label
    return cell_mask.copy() if np.any(cell_mask) else None


def mask_coverage(mask: np.ndarray, cover: np.ndarray) -> float:
    """Fraction of `mask` that `cover` overlaps, or 0.0 for an empty mask."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(cover, mask).sum()) / area


def find_nearest_background_pixel(
    center: np.ndarray, frame_mask: np.ndarray
) -> tuple[int, ...] | None:
    """Find the background pixel (value 0) closest to center."""
    bg_pixels = np.argwhere(frame_mask == 0)
    if len(bg_pixels) == 0:
        return None
    distances = np.linalg.norm(bg_pixels - center, axis=1)
    return tuple(int(v) for v in bg_pixels[int(distances.argmin())])


def interpolate_position(
    pos1: np.ndarray, pos2: np.ndarray, step: int, total_steps: int
) -> np.ndarray:
    """Return the position `step` of `total_steps` along the line from pos1 to pos2."""
    return pos1 + (pos2 - pos1) * (step / total_steps)


def analyze_conflict(
    new_mask: np.ndarray,
    t_interp: int,
    masks: np.ndarray,
    iou_threshold: float,
) -> dict:
    """Find the masks at t_interp that the new mask would overlap.

    Returns coverage per label. A label counts as a conflict when the larger of its IoU
    and its coverage exceeds iou_threshold.
    """
    frame = masks[t_interp]
    new_bool = new_mask > 0
    conflict_labels: list[int] = []
    coverage: dict[int, float] = {}

    for label in np.unique(frame[new_bool]):
        if label == 0:
            continue
        existing_bool = frame == label
        intersection = np.logical_and(new_bool, existing_bool).sum()
        union = np.logical_or(new_bool, existing_bool).sum()
        iou_val = float(intersection / union) if union > 0 else 0.0
        coverage_val = (
            float(intersection / existing_bool.sum()) if existing_bool.sum() > 0 else 0.0
        )
        coverage[int(label)] = coverage_val
        if max(iou_val, coverage_val) > iou_threshold:
            conflict_labels.append(int(label))

    return {"conflict_labels": conflict_labels, "coverage": coverage}


def cells_touch(
    tracks: pl.DataFrame,
    masks: np.ndarray,
    label_a: int,
    label_b: int,
) -> bool:
    """Return True if label_a and label_b touch at every frame they share.

    They must form a single connected region whenever both are present.
    """
    frames = (
        tracks.filter(pl.col("label_track").is_in([label_a, label_b]))
        .get_column("t")
        .unique()
        .to_list()
    )
    for t in frames:
        if t >= masks.shape[0]:
            continue
        mask_t = masks[t]
        relevant = np.isin(mask_t, [label_a, label_b])
        if not np.any(relevant):
            continue
        if label_connected_components(relevant).max() > 1:
            return False
    return True


def build_mask_label_map(masks: np.ndarray) -> dict[int, set[int]]:
    """Return a mapping {frame_index: set(non-zero labels)} for fast look-ups."""
    return {t: set(np.unique(masks[t]).tolist()) - {0} for t in range(masks.shape[0])}


# edge scores


def get_score_col(edge_predictions: pl.DataFrame) -> str:
    """Return normalized prediction column for correspondence score.

    Uses `p1` if available and non-zero, otherwise `p2` (for model without learned
    division detection).
    """
    if (
        "p1" in edge_predictions.columns
        and "p2" in edge_predictions.columns
        and edge_predictions["p1"].max() == 0.0
    ):
        return "p2"
    return "p1"


def check_for_correspondence(
    src_indices: list[int],
    dst_indices: list[int],
    edge_predictions: pl.DataFrame,
    thr: float,
) -> bool:
    """Return True if every src has a correspondence score (`p1`) above thr to a dst."""
    if "p1" not in edge_predictions.columns:
        return False
    for src in src_indices:
        edges = edge_predictions.filter(
            (pl.col("src") == src) & pl.col("dst").is_in(dst_indices)
        )
        if edges.height == 0 or float(edges["p1"].to_numpy().max()) <= thr:
            return False
    return True


def correspondence_to(
    edge_predictions: pl.DataFrame, src_indices: list[int], dst_idx: int
) -> float:
    """Return the strongest correspondence score (`p1`) from src_indices to dst_idx."""
    if "p1" not in edge_predictions.columns:
        return 0.0
    edges = edge_predictions.filter(
        pl.col("src").is_in(src_indices) & (pl.col("dst") == dst_idx)
    )
    return float(edges["p1"].to_numpy().max()) if edges.height > 0 else 0.0


def best_outgoing_correspondence(
    edge_predictions: pl.DataFrame, src_indices: list[int]
) -> float:
    """Return the strongest correspondence score (`p1`) from src_indices."""
    if "p1" not in edge_predictions.columns:
        return 0.0
    edges = edge_predictions.filter(pl.col("src").is_in(src_indices))
    return float(edges["p1"].to_numpy().max()) if edges.height > 0 else 0.0


def has_strong_external_edge(
    indices: list[int],
    edge_predictions: pl.DataFrame | None,
    thr: float,
) -> bool:
    """Return True if a cell has a strong edge to a cell outside its own trajectory."""
    if edge_predictions is None:
        return False
    cols = [c for c in ["p1", "p2"] if c in edge_predictions.columns]
    if not cols:
        return False
    ext_edges = edge_predictions.filter(
        (pl.col("src").is_in(indices) & ~pl.col("dst").is_in(indices))
        | (~pl.col("src").is_in(indices) & pl.col("dst").is_in(indices))
    )
    if ext_edges.height == 0:
        return False
    scores = ext_edges.select(pl.max_horizontal(*[pl.col(c) for c in cols]).alias("s"))
    return float(scores["s"].to_numpy().max()) > thr


# trajectory queries


def has_parent(label: int, tracks: pl.DataFrame) -> bool:
    """Return True if the label has any row with a non-zero parent_track."""
    rows = tracks.filter(pl.col("label_track") == label)
    if rows.height == 0:
        return False
    return int(rows["parent_track"].to_numpy().max()) > 0


def has_children(label: int, tracks: pl.DataFrame) -> bool:
    """Return True if any track lists label as its parent."""
    return tracks.filter(pl.col("parent_track") == label).height > 0


def find_sibling(tracks: pl.DataFrame, label: int, parent: int) -> int | None:
    """Return the label of the one other daughter of parent, or None if none exists."""
    sibling = tracks.filter(
        (pl.col("label_track") != label) & (pl.col("parent_track") == parent)
    )
    if sibling.height == 0:
        return None
    return sibling.get_column("label_track").item(0)


def get_short_tracks(
    tracks: pl.DataFrame, max_lifetime: int = 0, division_only: bool = False
) -> pl.DataFrame:
    """Return short-lived tracks, optionally restricted to daughters of a division."""
    parent_filter = (
        pl.col("parent_track") > 0 if division_only else pl.col("parent_track") >= 0
    )
    return (
        tracks.filter(parent_filter)
        .with_columns(
            lifetime=pl.col("t").max().over("label_track")
            - pl.col("t").min().over("label_track")
            + 1,
        )
        .sort("t", descending=True)
        .unique("label_track", maintain_order=True)
        # a track running out at the end of the sequence may simply be cut off, so
        # require another lifetime's worth of frames after it before calling it short
        .filter(
            (pl.col("lifetime") <= max_lifetime)
            & (pl.sum_horizontal("t", "lifetime") <= pl.col("t").max())
        )
    )


def is_gap_only_trajectory(
    label: int, t_gap_start: int, t_gap_end: int, tracks: pl.DataFrame
) -> bool:
    """Return True if every detection of label falls within [t_gap_start, t_gap_end]."""
    label_times = tracks.filter(pl.col("label_track") == label)["t"]
    if label_times.len() == 0:
        return False
    label_times_np = label_times.to_numpy()
    return (
        int(label_times_np.min()) >= t_gap_start
        and int(label_times_np.max()) <= t_gap_end
    )


def shares_frames(label: int, others: list[int], tracks: pl.DataFrame) -> bool:
    """Return True if label appears in any frame that one of others appears in."""
    label_times = set(tracks.filter(pl.col("label_track") == label)["t"].to_list())
    if not label_times:
        return False
    other_times = set(tracks.filter(pl.col("label_track").is_in(others))["t"].to_list())
    return bool(label_times & other_times)


def find_trajectories_with_gaps(
    tracks: pl.DataFrame, max_gap: int | None = None
) -> pl.DataFrame:
    """One row per gap, holding the cells that bracket it in `_prev` and `_next` columns.

    `t_diff` is the frame distance between them, so `max_gap` bounds the missing frames
    in between, not that distance.
    """
    sorted_tracks = tracks.sort("label_track", "t")

    consecutive_cells = sorted_tracks.select(
        [pl.col(c).alias(f"{c}_prev") for c in sorted_tracks.columns]
        + [pl.col(c).shift(-1).alias(f"{c}_next") for c in sorted_tracks.columns]
    )
    same_track_pairs = consecutive_cells.filter(
        (pl.col("label_track_prev") == pl.col("label_track_next"))
        & (pl.col("label_track_prev").is_not_null())
    )

    gaps = same_track_pairs.with_columns(
        (pl.col("t_next") - pl.col("t_prev")).alias("t_diff")
    ).filter(pl.col("t_diff") > 1)

    if max_gap is None:
        return gaps
    return gaps.filter(pl.col("t_diff") - 1 <= max_gap)


# track edits


def make_track_row(
    index: int,
    t: int,
    track_label: int,
    parent_track: int,
    pos: np.ndarray,
    pos_cols: list[str],
    schema_cols: list[str],
    cell_source: str,
) -> dict:
    """Build a new tracks row with all schema columns (unknowns set to None).

    `pos` carries one coordinate per entry in `pos_cols`, in matching axis order.
    """
    new_cell: dict = {
        "index": index,
        "t": t,
        "label_track": track_label,
        "parent_track": parent_track,
        "cell_source": cell_source,
    }
    for col, value in zip(pos_cols, pos, strict=False):
        new_cell[col] = value
    for col in schema_cols:
        if col not in new_cell:
            new_cell[col] = None
    return new_cell


def remove_rows(
    tracks: pl.DataFrame,
    remove_pairs: list[tuple[int, int]],
    zero_dangling_parents: bool = False,
) -> pl.DataFrame:
    """Drop rows identified by (t, label_track) pairs.

    When `zero_dangling_parents` is True, any track still pointing to a removed label
    as its parent has that `parent_track` reset to 0.
    """
    if not remove_pairs:
        return tracks
    removes_df = pl.DataFrame(
        {
            "t": [t for t, _ in remove_pairs],
            "label_track": [lbl for _, lbl in remove_pairs],
        },
        schema={"t": tracks.schema["t"], "label_track": tracks.schema["label_track"]},
    ).with_columns(pl.lit(True).alias("_remove"))
    result = (
        tracks.join(removes_df, on=["t", "label_track"], how="left")
        .filter(pl.col("_remove").is_null())
        .drop("_remove")
    )
    if not zero_dangling_parents:
        return result
    removed_labels = list({lbl for _, lbl in remove_pairs})
    return result.with_columns(
        pl.when(pl.col("parent_track").is_in(removed_labels))
        .then(pl.lit(0))
        .otherwise(pl.col("parent_track"))
        .alias("parent_track")
    )


def apply_relabels(
    tracks: pl.DataFrame, all_relabels: list[tuple[int, int, int]]
) -> pl.DataFrame:
    """Replace old label values with new values in label_track and parent_track.

    Gap-only labels are globally unique, so a direct replace is safe.
    """
    if not all_relabels:
        return tracks
    relabel_map = {old: new for _t, old, new in all_relabels}
    old_labels = list(relabel_map.keys())
    new_labels = list(relabel_map.values())
    return tracks.with_columns(
        label_track=pl.col("label_track").replace(old_labels, new_labels),
        parent_track=pl.col("parent_track").replace(old_labels, new_labels),
    )


def relabel_track(
    tracks: pl.DataFrame,
    old_labels: list[int],
    new_label: int,
    target_parent: int,
    t_start: int | None = None,
) -> pl.DataFrame:
    """Merge old_labels into new_label and repoint the affected lineage.

    Rewrites `label_track` from each old label to `new_label` (only at `t >= t_start`
    when given), repoints any daughter whose `parent_track` is an old label to
    `new_label`, then gives the merged rows, which now point at themselves,
    `target_parent` instead. Used by `MergeEarlyDivisions` to merge daughters into their
    parent, and by `SplitFalseMerges` to relabel the cells after a merge back to the
    daughters they came from.
    """
    label_when = pl.col("label_track").is_in(old_labels)
    if t_start is not None:
        label_when = label_when & (pl.col("t") >= t_start)
    tracks = tracks.with_columns(
        label_track=pl.when(label_when)
        .then(pl.lit(new_label))
        .otherwise(pl.col("label_track")),
        # grandchildren: cells whose parent was a merged label now point to new_label
        parent_track=pl.when(pl.col("parent_track").is_in(old_labels))
        .then(pl.lit(new_label))
        .otherwise(pl.col("parent_track")),
    )
    # the merged rows (now labeled new_label) still point at themselves. give them the
    # correct parent, restricted to t >= t_start when the relabel was time-scoped.
    self_ref = (pl.col("label_track") == new_label) & pl.col("parent_track").is_in(
        [0, new_label, *old_labels]
    )
    if t_start is not None:
        self_ref = self_ref & (pl.col("t") >= t_start)
    return tracks.with_columns(
        parent_track=pl.when(self_ref)
        .then(pl.lit(target_parent))
        .otherwise(pl.col("parent_track"))
    )


def drop_orphan_rows(
    tracks: pl.DataFrame, mask_label_map: dict[int, set[int]]
) -> pl.DataFrame:
    """Drop track rows with no matching mask pixel and detach parents that are gone.

    A 1-pixel marker can be overwritten by a cell drawn over the same region later in
    the same step, leaving a track row with no mask. That frame holds no real detection,
    so the row is removed and `split_unbridged_gaps` starts a new trajectory after it.
    Any track pointing at a label that no longer exists is reset to no parent.
    """
    present = pl.Series(
        [
            int(lbl) in mask_label_map.get(int(t), set())
            for t, lbl in tracks.select(["t", "label_track"]).iter_rows()
        ]
    )
    if not present.all():
        logger.info(
            f"Dropped {int((~present).sum())} orphan track row(s) with no mask pixel."
        )

    kept = tracks.filter(present)
    # an earlier step can remove a parent's rows while a daughter still points at it, so
    # this runs whether or not a row was dropped here
    surviving = kept.get_column("label_track").unique().to_list()
    return kept.with_columns(
        parent_track=pl.when(pl.col("parent_track").is_in(surviving))
        .then(pl.col("parent_track"))
        .otherwise(pl.lit(0))
    )


def split_trajectory_at_gaps(
    tracks: pl.DataFrame,
    masks: np.ndarray,
    label: int,
    cut_frames: list[int],
    new_labels: list[int],
) -> tuple[pl.DataFrame, np.ndarray]:
    """Cut a trajectory at one or more gaps into pieces without gaps.

    `cut_frames` are the frames each gap ends at, in ascending order. The piece before
    the first cut keeps `label`, every later piece takes the matching entry in
    `new_labels` and appears with no parent. Each daughter of `label` is repointed to
    whichever piece was there in the frame before it was born, so divisions stay
    attached to the right piece.
    """
    T = masks.shape[0]
    # piece boundaries [start, cut0, ..., T], where piece i spans [bounds[i], bounds[i+1])
    bounds = [0, *cut_frames, T]
    piece_label = [label, *new_labels]

    def piece_of(t: int) -> int:
        for i in range(len(piece_label)):
            if bounds[i] <= t < bounds[i + 1]:
                return piece_label[i]
        return label

    # relabel masks per piece (only the later pieces change)
    for i in range(1, len(piece_label)):
        for t in range(bounds[i], bounds[i + 1]):
            sel = masks[t] == label
            if sel.any():
                masks[t][sel] = piece_label[i]

    # repoint children to the piece active just before each child's birth frame
    child_first = (
        tracks.filter(pl.col("parent_track") == label)
        .group_by("label_track")
        .agg(pl.col("t").min().alias("t0"))
    )
    child_parent = {
        int(r["label_track"]): piece_of(int(r["t0"]) - 1)
        for r in child_first.iter_rows(named=True)
    }

    # relabel the trajectory rows by piece
    t_expr = pl.col("t")
    new_label_expr = pl.col("label_track")
    for i in range(1, len(piece_label)):
        new_label_expr = (
            pl.when(
                (pl.col("label_track") == label)
                & (t_expr >= bounds[i])
                & (t_expr < bounds[i + 1])
            )
            .then(pl.lit(piece_label[i]))
            .otherwise(new_label_expr)
        )
    tracks = tracks.with_columns(label_track=new_label_expr)

    # the later pieces appear (no parent)
    tracks = tracks.with_columns(
        parent_track=pl.when(pl.col("label_track").is_in(new_labels))
        .then(pl.lit(0))
        .otherwise(pl.col("parent_track"))
    )
    # repoint each child label to its piece
    parent_expr = pl.col("parent_track")
    for child_label, parent_label in child_parent.items():
        parent_expr = (
            pl.when(pl.col("label_track") == child_label)
            .then(pl.lit(parent_label))
            .otherwise(parent_expr)
        )
    tracks = tracks.with_columns(parent_track=parent_expr)
    return tracks, masks


def split_unbridged_gaps(
    tracks: pl.DataFrame, masks: np.ndarray
) -> tuple[pl.DataFrame, np.ndarray]:
    """Cut every trajectory that still has a gap into separate appearing trajectories.

    Run once after all correction steps. A gap left at this point was either wider than
    the steps' `max_gap` or refused by the detection score, because the images show no
    cell there. The trajectory is cut and the pieces after each gap become new
    trajectories with no parent.
    """
    gaps = find_trajectories_with_gaps(tracks, max_gap=None)
    if gaps.height == 0:
        return tracks, masks

    cuts: dict[int, list[int]] = {}
    for gap in gaps.iter_rows(named=True):
        cuts.setdefault(int(gap["label_track_prev"]), []).append(int(gap["t_next"]))

    next_label = int(tracks.get_column("label_track").to_numpy().max() or 0) + 1
    for label, points in cuts.items():
        cut_frames = sorted(set(points))
        new_labels = list(range(next_label, next_label + len(cut_frames)))
        next_label += len(cut_frames)
        tracks, masks = split_trajectory_at_gaps(
            tracks, masks, label, cut_frames, new_labels
        )
    logger.info(f"Split {len(cuts)} trajectory(ies) at true-negative gaps.")
    return tracks, masks


# detection score


def region_std(frame: np.ndarray, mask: np.ndarray) -> float:
    """Intensity std of frame pixels inside a boolean mask (0 if empty)."""
    vals = frame[mask]
    return float(vals.std()) if vals.size else 0.0


def reference_cell_std(
    images: np.ndarray,
    masks: np.ndarray,
    track_label: int,
    t_prev: int,
    t_next: int,
) -> float | None:
    """Mean intensity std of the cell at the two real frames bracketing the gap.

    Returns None when the source cell has no pixels (footprint not assessable).
    """
    stds = []
    for t in (t_prev, t_next):
        if t < 0 or t >= len(images):
            continue
        cell = source_mask(masks, t, track_label)
        if cell is not None and cell.any():
            stds.append(region_std(np.asarray(images[t]), cell))
    return float(np.mean(stds)) if stds else None


def footprint_detection_score(
    frame: np.ndarray,
    masks_frame: np.ndarray,
    footprint: np.ndarray,
    ref_std: float,
) -> float:
    """Score a footprint by intensity spread: 1 is the source cell, 0 the background.

    The local background is the unsegmented pixels in a margin around the footprint. A
    footprint over a real cell is textured, one over background is flat. When the cell
    is no more textured than its background there is nothing to tell them apart, so the
    score is 1 and the gap is closed.
    """
    coords = np.nonzero(footprint)
    if len(coords[0]) == 0:
        return 0.0
    sl = tuple(
        slice(
            max(0, int(coords[d].min()) - _BACKGROUND_MARGIN_PX),
            min(frame.shape[d], int(coords[d].max()) + _BACKGROUND_MARGIN_PX + 1),
        )
        for d in range(footprint.ndim)
    )
    sub_img = frame[sl].astype(float)
    bg_std = region_std(sub_img, masks_frame[sl] == 0)
    gap_std = region_std(frame.astype(float), footprint)

    denom = ref_std - bg_std
    if denom <= 1e-6:  # cell no more textured than background: cannot discriminate
        return 1.0
    return (gap_std - bg_std) / denom


def gap_has_candidate(
    gap_info: dict,
    images: np.ndarray,
    masks: np.ndarray,
    detection_threshold: float,
) -> bool:
    """Return True if the image shows a cell in any frame of the gap.

    Real cells are textured, background is flatter. The footprint is the source mask
    placed as-is, the same in every gap frame, so only the image underneath it changes.
    A frame scoring at least `detection_threshold` holds a real, missed detection.

    Args:
        gap_info: One row of `find_trajectories_with_gaps`, so the cells bracketing the
            gap are its `_prev` and `_next` columns and `t_diff` is their distance.
        images: Image sequence with shape `(T, H, W)`.
        masks: Instance-segmentation masks for `images`.
        detection_threshold: Footprint score at which a gap frame counts as holding a
            cell, on the scale `footprint_detection_score` returns.
    """
    prev = {k[:-5]: v for k, v in gap_info.items() if k.endswith("_prev")}
    nxt = {k[:-5]: v for k, v in gap_info.items() if k.endswith("_next")}
    t_prev = int(prev["t"])
    t_next = int(nxt["t"])
    t_diff = int(gap_info["t_diff"])
    track_label = int(prev["label_track"])

    ref_std = reference_cell_std(images, masks, track_label, t_prev, t_next)
    if ref_std is None:
        return True  # cannot assess footprint: fall back to filling

    footprint = source_mask(masks, t_prev, track_label)
    if footprint is None or not footprint.any():
        return False  # no footprint to assess: treat as a true negative (no fill)

    for i in range(1, t_diff):
        t = t_prev + i
        if t < 0 or t >= len(images):
            continue
        score = footprint_detection_score(
            np.asarray(images[t]), masks[t], footprint, ref_std
        )
        if score >= detection_threshold:
            return True
    return False


# validation


def assert_ctc_valid(tracks: pl.DataFrame, mask_label_map: dict[int, set[int]]) -> None:
    """Check full CTC compliance, raising ValueError with a diagnostic summary."""
    issues: list[str] = []

    per_label = tracks.group_by("label_track").agg(
        pl.col("t").min().alias("t_min"),
        pl.col("t").max().alias("t_max"),
        pl.col("t").n_unique().alias("n_frames"),
    )

    # 1. no gaps
    gap_labels = per_label.filter(
        pl.col("t_max") - pl.col("t_min") + 1 != pl.col("n_frames")
    )["label_track"].to_list()
    if gap_labels:
        issues.append(f"Gaps in trajectories for labels: {gap_labels}")

    # 2. no missing parents
    existing_labels = set(tracks["label_track"].unique().to_list())
    parent_labels = set(
        tracks.filter(pl.col("parent_track") > 0)["parent_track"].unique().to_list()
    )
    missing_parents = parent_labels - existing_labels
    if missing_parents:
        issues.append(f"Missing parent labels (not in tracks): {sorted(missing_parents)}")

    # 3/4/5/6: division checks
    children = tracks.filter(pl.col("parent_track") > 0)
    if children.height > 0:
        div_stats = children.group_by("parent_track").agg(
            pl.col("label_track").n_unique().alias("n_daughters"),
            pl.col("t").min().alias("daughter_first_t"),
        )
        # 3. exactly 2 daughters
        wrong_count = div_stats.filter(pl.col("n_daughters") != 2)[
            "parent_track"
        ].to_list()
        if wrong_count:
            issues.append(f"Parents with != 2 daughters: {wrong_count}")

        # 4. daughters start together
        mismatched = (
            children.group_by(["parent_track", "label_track"])
            .agg(pl.col("t").min().alias("t_first"))
            .group_by("parent_track")
            .agg(pl.col("t_first").n_unique().alias("n_starts"))
            .filter(pl.col("n_starts") > 1)
        )["parent_track"].to_list()
        if mismatched:
            issues.append(
                f"Daughters of same parent start at different frames: {mismatched}"
            )

        # 5. daughters start at parent_last + 1
        parent_last = per_label.select(["label_track", "t_max"]).rename(
            {"label_track": "parent_track", "t_max": "parent_t_max"}
        )
        div_check = (
            div_stats.join(parent_last, on="parent_track", how="left")
            .filter(pl.col("daughter_first_t") != pl.col("parent_t_max") + 1)[
                "parent_track"
            ]
            .to_list()
        )
        if div_check:
            issues.append(
                f"Daughters don't start at parent_last+1 for parents: {div_check}"
            )

        # 6. no parent-child frame overlap
        overlap = tracks.join(
            div_stats.select(["parent_track", "daughter_first_t"]).rename(
                {"parent_track": "label_track"}
            ),
            on="label_track",
            how="inner",
        ).filter(pl.col("t") >= pl.col("daughter_first_t"))
        if overlap.height > 0:
            issues.append(
                f"Parent rows at t >= child start for labels: "
                f"{overlap['label_track'].unique().to_list()}"
            )

    # 7. mask consistency
    mask_issues: list[str] = []
    for row in tracks.select(["t", "label_track"]).iter_rows(named=True):
        t = int(row["t"])
        lbl = int(row["label_track"])
        if lbl not in mask_label_map.get(t, set()):
            mask_issues.append(f"label {lbl} at t={t}")
    if mask_issues:
        issues.append(
            f"Labels in tracks missing from masks ({len(mask_issues)} total): "
            + "; ".join(mask_issues[:10])
            + ("..." if len(mask_issues) > 10 else "")
        )

    if issues:
        summary = "\n".join(f"  - {i}" for i in issues)
        raise ValueError(
            f"CTC validation failed after segmentation correction "
            f"({len(issues)} issue(s)):\n{summary}"
        )
