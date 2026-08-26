"""Correction of segmentation errors: missing, merged, and prematurely split cells.

Bacteria divide at their center over several frames, so often (especially without stable
fluo reporters) a division timepoint can't be accurately determined. Hence, a segmentation
model without temporal context often detects divisions prematurely (false split) or merges
cells after a detected division (false merge). In addition, in case of focus issues, some
cells might not be detected for a single (or few) frames. This module aims to fix the 3
kinds of errors using multi-frame network predictions. In principle, the correction can
also be applied to errors unrelated to divisions, e.g., temporary over- or
undersegmentation, as long as a cell is not matched with any of the false segments but
tracked over the gap.

Whether a division is classified as correct or incorrect is determined by the lifetime of
the daughter cells and the tracking predictions. If two daughter cells merge for one (or
few) frames but reappear as two cells afterwards, the division is assumed to be correct
and the merge is removed and replaced by the interpolated daughter masks. Contrary, if a
division is first detected, but the daughter cells merge again, both daughters and the
cell they reappear as are relabeled to the parent. For both cases and for missed
detections, multi-frame edge predictions are used to correct the segmentation error.

To not interfere with correct trajectories, there are several guards. Removed cell(s) must
only appear within the gap and must not have any strong external edges (i.e. should be
isolated). Merged daughters must touch over the full duration they share (no migration;
assumes an instance segmentation model that allows cell contact, e.g., Omnipose) and must
cover the cell they merge into or reappear as, which the parent or one of them must also
link to by a strong edge. The daughter -> merge correspondence is deliberately not among
those edges: a single daughter is not the whole merged mask, so a trained model scores it
low (0.02 to 0.19 where the parent -> merge edge scores 0.82 to 0.88). When cells are
placed, the position is checked for existing cells using the segmentation masks, and
optionally also using the image (to ensure that the image actually shows a cell there).

Correction runs in four steps, chained by `SegmentationCorrector`:

    1. Close the gaps no other mask conflicts with (`CloseDetectionGaps`). A cell tracked
       before and after a frame in which it has no mask gets a copy of its previous mask
       placed in the missing frame.
    2. Merge early divisions (`MergeEarlyDivisions`). A cell splits into two daughters
       that stay next to each other and appear as a single cell again a few frames
       later. The daughters and that reappearing cell are relabeled to the parent.
    3. Split false merges (`SplitFalseMerges`). One mask covers two cells. If they are
       the daughters of one parent, the merged mask is deleted and both are drawn back
       into the frame. If it only sits inside the gap of cells that the tracker follows
       across it, the merged mask is deleted and the cells it covered are put back in
       step 4.
    4. Close the remaining gaps (`CloseDetectionGaps` again), including the ones step 3
       created, and resolve the conflicts step 1 left alone.

The corrected masks are checked against the CTC rules, and every cell this module added is
marked in `cell_source` so it can be dropped downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import polars as pl
from tqdm import tqdm

from baclct.utils import segmentation_correction as sc_utils
from baclct.utils.logger import get_pylogger
from baclct.utils.spacing import position_columns

logger = get_pylogger(__name__)


class BaseCorrection(ABC):
    """Base class for one segmentation error correction step.

    Subclasses implement `correct()`. The checks and geometry they have in common live
    in `baclct.utils.segmentation_correction`, so every step evaluates them the same way.
    """

    @abstractmethod
    def correct(
        self, tracks: pl.DataFrame, masks: np.ndarray, **kwargs
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Apply the correction step, returning the corrected tracks and masks.

        `masks` is modified in place.

        Args:
            tracks: Trajectories from the tracker, one row per cell-timepoint.
            masks: Instance-segmentation masks of the tracked sequence.
            **kwargs: Step-specific inputs, e.g. `edge_predictions` or `images`.
        """


class SegmentationCorrector(BaseCorrection):
    """Runs the four correction steps in order.

    1. `CloseDetectionGaps` with `skip_conflicts=True`: closes the gaps no other mask
       conflicts with. A frame is left open if an existing mask overlaps the interpolated
       one above `iou_threshold`.
    2. `MergeEarlyDivisions`: reverses divisions whose daughters merge again.
    3. `SplitFalseMerges`: removes a merge of two daughters that reappear as two cells.
    4. `CloseDetectionGaps` with `fill_division_gaps=True`: closes the gaps step 3
       opened and resolves the frames step 1 skipped.

    Any gap left afterwards (wider than `max_gap`, or refused by the detection gate) is
    cut into separate trajectories (`split_unbridged_gaps`), so the result satisfies the
    CTC rules.

    With no arguments every threshold keeps its class default. `skip_conflicts` and
    `fill_division_gaps` are set per step and override anything passed in.
    """

    def __init__(
        self,
        close_gaps_kwargs: dict | None = None,
        merge_divisions_kwargs: dict | None = None,
        split_merges_kwargs: dict | None = None,
    ):
        """Initialize the four steps with optional per-step arguments.

        Args:
            close_gaps_kwargs: Passed to both `CloseDetectionGaps` steps.
                `skip_conflicts` and `fill_division_gaps` are set per step, so values
                given here for those two are ignored.
            merge_divisions_kwargs: Passed to `MergeEarlyDivisions`.
            split_merges_kwargs: Passed to `SplitFalseMerges`.
        """
        gap_kwargs = {
            k: v
            for k, v in (close_gaps_kwargs or {}).items()
            if k not in ("skip_conflicts", "fill_division_gaps")
        }
        self._close_gaps = CloseDetectionGaps(
            skip_conflicts=True, fill_division_gaps=False, **gap_kwargs
        )
        self._close_remaining_gaps = CloseDetectionGaps(
            skip_conflicts=False, fill_division_gaps=True, **gap_kwargs
        )
        self._merge = MergeEarlyDivisions(**(merge_divisions_kwargs or {}))
        self._split = SplitFalseMerges(**(split_merges_kwargs or {}))

    @property
    def named_steps(self) -> list[tuple[str, BaseCorrection]]:
        """The four steps in the order `correct` applies them, each with its name.

        Util for naming steps during testing and logging.
        """
        return [
            ("1. close gaps", self._close_gaps),
            ("2. merge early divisions", self._merge),
            ("3. split false merges", self._split),
            ("4. close remaining gaps", self._close_remaining_gaps),
        ]

    def correct(
        self, tracks: pl.DataFrame, masks: np.ndarray, **kwargs
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Apply the four steps in sequence, then consolidate and validate.

        Intermediate states may break the CTC rules: step 3 deletes a merged mask before
        step 4 closes the gap it leaves, and the detection gate can leave a gap open.
        Only the final state is validated, and a ValueError names every violation that
        is left.
        """
        for _name, step in self.named_steps:
            tracks, masks = step.correct(tracks, masks, **kwargs)

        # a marker pixel can be overwritten by a cell placed later in the same step,
        # leaving a track row with no mask
        mask_label_map = sc_utils.build_mask_label_map(masks)
        tracks = sc_utils.drop_orphan_rows(tracks, mask_label_map)
        # a trajectory that still holds a gap was rejected by the detection gate. cut it
        # into separate appearing trajectories, so the CTC rules are met in one place.
        tracks, masks = sc_utils.split_unbridged_gaps(tracks, masks)
        mask_label_map = sc_utils.build_mask_label_map(masks)
        sc_utils.assert_ctc_valid(tracks, mask_label_map)
        return tracks, masks


@dataclass
class _CellPlacer:
    """Places interpolated cells of one trajectory into masks and track df.

    Holds what each placement needs (the masks to draw into, the trajectory's label and
    parent, the position and schema column names) and hands out the next free row index,
    so the gap-closing loops add a cell with a single call.
    """

    masks: np.ndarray
    schema_cols: list[str]
    pos_cols: list[str]
    label: int
    parent: int
    next_index: int

    def place_mask(self, mask: np.ndarray, t: int, pos: np.ndarray) -> dict:
        """Draw the interpolated mask at frame t, returning its track row.

        Only its largest connected component is drawn, so a disconnected copy cannot put
        two cells under one label.
        """
        self.masks[t][sc_utils.ensure_connected(mask)] = self.label
        return self._row(t, pos, "interpolated")

    def place_marker(self, t: int, pos: np.ndarray) -> dict | None:
        """Place a 1-pixel marker at the nearest background pixel."""
        nearest_bg = sc_utils.find_nearest_background_pixel(pos, self.masks[t])
        if nearest_bg is None:
            logger.debug(
                f"Track {self.label}: no background pixel at t={t}, skipping marker."
            )
            return None
        self.masks[t][nearest_bg] = self.label
        return self._row(t, pos, "interpolated_marker")

    def _row(self, t: int, pos: np.ndarray, cell_source: str) -> dict:
        cell = sc_utils.make_track_row(
            self.next_index,
            t,
            self.label,
            self.parent,
            pos,
            self.pos_cols,
            self.schema_cols,
            cell_source,
        )
        self.next_index += 1
        return cell


class CloseDetectionGaps(BaseCorrection):
    """Close detection gaps by interpolating the missing masks.

    The last mask before the gap is copied unchanged into every missing frame. The
    position linearly interpolated between the two ends of the gap is what the tracks row
    records and where a marker goes, not where the copy is drawn.

    A mask overlapping that copy by more than `iou_threshold` is resolved in one of
    three ways, tried in order. All three require the conflicting mask to lie entirely
    within the gap and to have no parent and no daughters, so an established trajectory
    is never modified:

    - Relabel: its strongest outgoing correspondence points at the cell after the gap
      and exceeds `thr_corr`. It is relabeled to that trajectory.
    - Remove: no strong edge links it to another trajectory, and the copy covers more
      than `min_removal_coverage` of it. It is a duplicate of the missing detection, so
      it is deleted in all its frames and the copy takes its place.
    - Draw over: as for remove, but the copy covers less of it. It is kept and only its
      overlapping pixels are overwritten, unless what remains of it would fragment.

    An unresolved conflict yields a 1-pixel marker at the nearest background pixel, so
    the trajectory stays continuous. Under `skip_conflicts` no marker is placed and only
    conflict-free frames are filled, since a conflicting mask is often the false merge
    that `SplitFalseMerges` removes afterwards.
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_gap: int = 2,
        thr_corr: float = 0.5,
        skip_conflicts: bool = False,
        fill_division_gaps: bool = False,
        min_removal_coverage: float = 0.8,
        max_fragment_frac: float = 0.1,
        require_detection: bool = False,
        detection_threshold: float = 0.0,
    ):
        """Initialize correction.

        Args:
            iou_threshold: Overlap (the larger of IoU and coverage) above which an
                existing mask counts as conflicting.
            max_gap: Widest gap, in frames, to close. The graph bounds it as well, since
                an edge has to span the gap. It does not bound `fill_division_gaps`.
            thr_corr: Minimum correspondence score for the relabel outcome, and the
                score above which an external edge counts as strong.
            skip_conflicts: If True, fill only the frames nothing conflicts with.
            fill_division_gaps: If True, also extend a parent up to the frame where its
                daughters start.
            min_removal_coverage: Coverage above which the remove outcome deletes the
                conflicting mask instead of drawing over it.
            max_fragment_frac: Largest fraction of a conflicting mask that may be split
                off when the copy is drawn over it, so elongated cells stay in one piece.
            require_detection: If True and `correct` receives `images`, close only the
                gaps whose footprint scores at least `detection_threshold` in one of
                their frames.
            detection_threshold: Footprint score above which a gap frame counts as a
                detection, on the scale `footprint_detection_score` returns. Higher
                values leave more gaps open and risk cutting cells with little contrast.
        """
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self.thr_corr = thr_corr
        self.skip_conflicts = skip_conflicts
        self.fill_division_gaps = fill_division_gaps
        self.min_removal_coverage = min_removal_coverage
        self.max_fragment_frac = max_fragment_frac
        self.require_detection = require_detection
        self.detection_threshold = detection_threshold

    def correct(
        self, tracks: pl.DataFrame, masks: np.ndarray, **kwargs
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Close gaps in trajectories by interpolating masks and resolving conflicts."""
        edge_predictions: pl.DataFrame | None = kwargs.get("edge_predictions")
        images = kwargs.get("images")
        gate = self.require_detection and images is not None
        gaps = sc_utils.find_trajectories_with_gaps(tracks, self.max_gap)

        new_cells: list[dict] = []
        all_relabels: list[tuple[int, int, int]] = []  # (t, old_label, new_label)
        all_removes: list[tuple[int, int]] = []  # (t, label)
        index_col = tracks.get_column("index")
        current_index = int(index_col.to_numpy().max()) + 1 if index_col.len() > 0 else 0

        if gaps.height > 0:
            pbar = tqdm(
                gaps.iter_rows(named=True), total=len(gaps), desc="Interpolating gaps"
            )
            for gap in pbar:
                # if placement looks like background, i.e. a gap is "hallucinated", skip
                # placement and initialize new trajectory after gap. this can happen in
                # case of long-range FP correspondences in sparse regions. cell does not
                # contend with foreground, i.e. is placed without check. in dense regions
                # placement would have been rejected since it likely interferes with
                # existing, valid trajectories. only tested with bright field.
                if gate and not sc_utils.gap_has_candidate(
                    gap, images, masks, self.detection_threshold
                ):
                    continue

                interpolated_cells, track_modifications, current_index = (
                    self._interpolate_gap(
                        gap,
                        tracks.columns,
                        masks,
                        current_index,
                        tracks,
                        edge_predictions,
                    )
                )
                new_cells.extend(interpolated_cells)
                all_relabels.extend(track_modifications["relabel"])
                all_removes.extend(track_modifications["remove"])

            # deduplicate: a gap-only label may appear across multiple gap frames
            all_relabels = list(dict.fromkeys(all_relabels))
            all_removes = list(dict.fromkeys(all_removes))

            tracks = sc_utils.remove_rows(tracks, all_removes, zero_dangling_parents=True)
            tracks = sc_utils.apply_relabels(tracks, all_relabels)

            if new_cells:
                new_cells_df = pl.DataFrame(new_cells, schema=tracks.schema)
                logger.info(f"Added {new_cells_df.height} cells.")
                tracks = pl.concat([tracks, new_cells_df])

        if self.fill_division_gaps:
            div_cells, current_index, orphans = self._fill_division_gaps(
                tracks,
                masks,
                edge_predictions,
                tracks.columns,
                current_index,
                images if gate else None,
            )
            if orphans:
                logger.info(f"Dropped {len(orphans)} spurious division daughter(s).")
                tracks = tracks.with_columns(
                    parent_track=pl.when(pl.col("label_track").is_in(orphans))
                    .then(pl.lit(0))
                    .otherwise(pl.col("parent_track"))
                )
            if div_cells:
                div_df = pl.DataFrame(div_cells, schema=tracks.schema)
                logger.info(f"Filled {div_df.height} division-gap cell(s).")
                tracks = pl.concat([tracks, div_df])

        return tracks, masks

    def _interpolate_gap(
        self,
        gap_info: dict,
        schema_cols: list[str],
        masks: np.ndarray,
        start_index: int,
        tracks: pl.DataFrame,
        edge_predictions: pl.DataFrame | None,
    ) -> tuple[list[dict], dict, int]:
        """Close a single gap by copying the source mask and resolving conflicts.

        Args:
            gap_info: One row of `find_trajectories_with_gaps`.
            schema_cols: Columns of `tracks`, so new cells carry the full schema.
            masks: Segmentation masks, mutated in-place.
            start_index: First available cell index for newly created rows.
            tracks: Trajectories from the tracker, used to identify a conflicting mask.
            edge_predictions: Per-edge class probabilities, used to validate a
                spurious cell.

        Returns:
            interpolated_cells: New cells to add.
            track_modifications: The "relabel" and "remove" edits the caller applies.
            current_index: Next available cell index.
        """
        prev_cell = {
            k.replace("_prev", ""): v for k, v in gap_info.items() if "_prev" in k
        }
        next_cell = {
            k.replace("_next", ""): v for k, v in gap_info.items() if "_next" in k
        }
        t_diff = gap_info["t_diff"]
        t_prev = prev_cell["t"]
        t_gap_start = t_prev + 1
        t_gap_end = next_cell["t"] - 1
        track_label = prev_cell["label_track"]
        next_idx = next_cell.get("index")  # row index of the cell after the gap

        interpolated_cells: list[dict] = []
        track_modifications: dict = {"relabel": [], "remove": []}

        pos_cols = position_columns(prev_cell)
        pos_prev = np.array([prev_cell.get(c, 0) for c in pos_cols])
        pos_next = np.array([next_cell.get(c, 0) for c in pos_cols])
        placer = _CellPlacer(
            masks,
            schema_cols,
            pos_cols,
            track_label,
            prev_cell["parent_track"],
            start_index,
        )

        for i in range(1, t_diff):
            t_interp = t_prev + i
            pos_interp = sc_utils.interpolate_position(pos_prev, pos_next, i, t_diff)

            # skip frames a relabeled conflicting mask already filled in this gap
            if np.any(masks[t_interp] == track_label):
                continue

            new_mask_frame = sc_utils.source_mask(masks, t_prev, track_label)
            if new_mask_frame is None:
                continue

            conflict_info = sc_utils.analyze_conflict(
                new_mask_frame, t_interp, masks, self.iou_threshold
            )
            if not conflict_info["conflict_labels"]:
                interpolated_cells.append(
                    placer.place_mask(new_mask_frame, t_interp, pos_interp)
                )
                continue

            if self.skip_conflicts:
                continue

            handled, cell = self._resolve_conflicting_mask(
                conflict_info,
                new_mask_frame,
                t_interp,
                pos_interp,
                placer,
                tracks,
                masks,
                edge_predictions,
                track_modifications,
                t_gap_start,
                t_gap_end,
                next_idx,
            )
            if not handled:
                cell = placer.place_marker(t_interp, pos_interp)
            if cell is not None:
                interpolated_cells.append(cell)

        return interpolated_cells, track_modifications, placer.next_index

    def _resolve_conflicting_mask(
        self,
        conflict_info: dict,
        new_mask_frame: np.ndarray,
        t_interp: int,
        pos_interp: np.ndarray,
        placer: _CellPlacer,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        edge_predictions: pl.DataFrame | None,
        track_modifications: dict,
        t_gap_start: int,
        t_gap_end: int,
        next_idx: int | None,
    ) -> tuple[bool, dict | None]:
        """Resolve a conflicting mask with the first outcome that applies.

        Only a mask lying entirely within the gap and having no parent and no daughters
        is eligible. See the class docstring for the conditions of each outcome. Remove
        and draw over write the copy over every mask under it, so a second conflicting
        label is overwritten unchecked.

        Returns:
            handled: True when one of the three outcomes fired.
            cell: The cell placed by remove or draw over, None otherwise.
        """
        track_label = placer.label
        coverage_scores = conflict_info["coverage"]
        for conflict_label in conflict_info["conflict_labels"]:
            if not sc_utils.is_gap_only_trajectory(
                conflict_label, t_gap_start, t_gap_end, tracks
            ):
                continue
            if sc_utils.has_parent(conflict_label, tracks) or sc_utils.has_children(
                conflict_label, tracks
            ):
                continue

            conflict_indices = tracks.filter(pl.col("label_track") == conflict_label)[
                "index"
            ].to_list()

            # relabel: conflicting mask has correspondence with cell after gap
            if edge_predictions is not None and next_idx is not None:
                corr_to_next = sc_utils.correspondence_to(
                    edge_predictions, conflict_indices, next_idx
                )
                best_corr = sc_utils.best_outgoing_correspondence(
                    edge_predictions, conflict_indices
                )
                if corr_to_next > self.thr_corr and corr_to_next >= best_corr:
                    logger.debug(
                        f"Track {track_label}: relabel gap-only {conflict_label} "
                        f"to it (p1={corr_to_next:.3f}/{best_corr:.3f})."
                    )
                    self._relabel_or_remove(
                        conflict_label,
                        track_label,
                        tracks,
                        masks,
                        track_modifications,
                    )
                    return True, None

            no_outside_edges = edge_predictions is None or not (
                sc_utils.has_strong_external_edge(
                    conflict_indices, edge_predictions, self.thr_corr
                )
            )
            coverage_val = coverage_scores.get(conflict_label, 0.0)

            # remove: nothing outside links to it and the copy covers most of it
            if no_outside_edges and coverage_val > self.min_removal_coverage:
                logger.debug(
                    f"Track {track_label}: remove spurious gap-only {conflict_label} "
                    f"at t={t_interp} (coverage={coverage_val:.2f})."
                )
                self._relabel_or_remove(
                    conflict_label, 0, tracks, masks, track_modifications
                )
                return True, placer.place_mask(new_mask_frame, t_interp, pos_interp)

            # draw over: minor overlap, keep the conflicting mask unless it would break
            if no_outside_edges and coverage_val > self.iou_threshold:
                if sc_utils.would_fragment(
                    masks[t_interp],
                    conflict_label,
                    new_mask_frame,
                    self.max_fragment_frac,
                ):
                    continue
                logger.debug(
                    f"Track {track_label}: draw over conflict {conflict_label} "
                    f"at t={t_interp} (coverage={coverage_val:.2f})."
                )
                return True, placer.place_mask(new_mask_frame, t_interp, pos_interp)

        return False, None

    def _fill_division_gaps(
        self,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        edge_predictions: pl.DataFrame | None,
        schema_cols: list[str],
        start_index: int,
        images: np.ndarray | None = None,
    ) -> tuple[list[dict], int, list[int]]:
        """Extend parents up to the frame where their daughters start.

        Finds parents whose daughters start more than one frame after the parent's last
        frame and fills the frames in between with the parent's last mask, or with a
        1-pixel marker where another mask conflicts with it.

        When `images` is given and no frame of the gap has a detection candidate, the
        division is treated as spurious: the parent is not extended and its daughters
        are returned as orphans (parent reset to 0).
        """
        parent_last_t = tracks.group_by("label_track").agg(
            pl.col("t").max().alias("t_last")
        )
        daughter_first_t = (
            tracks.filter(pl.col("parent_track") > 0)
            .group_by("parent_track")
            .agg(pl.col("t").min().alias("t_first"))
            .rename({"parent_track": "label_track"})
        )
        division_gaps = parent_last_t.join(daughter_first_t, on="label_track").filter(
            pl.col("t_first") - pl.col("t_last") > 1
        )

        if division_gaps.is_empty():
            return [], start_index, []

        new_cells: list[dict] = []
        orphans: list[int] = []
        current_index = start_index

        pos_cols = position_columns(schema_cols)

        for row in division_gaps.iter_rows(named=True):
            label = row["label_track"]
            t_last = row["t_last"]
            t_first = row["t_first"]

            parent_row = tracks.filter(
                (pl.col("label_track") == label) & (pl.col("t") == t_last)
            )
            if parent_row.height == 0:
                continue
            parent_dict = parent_row.row(0, named=True)
            pos_src = np.array([parent_dict.get(c, 0.0) for c in pos_cols])

            # the parent keeps moving through the gap, so fills interpolate towards where
            # the daughters appear. both continue the parent, hence their midpoint
            daughters = tracks.filter(
                (pl.col("parent_track") == label) & (pl.col("t") == t_first)
            )
            pos_dst = (
                daughters.select(pos_cols).to_numpy().mean(axis=0)
                if daughters.height
                else pos_src
            )

            # no candidate anywhere in the gap: drop the division, orphan the daughters
            source = sc_utils.source_mask(masks, t_last, label)
            gap_supported = True
            if images is not None and source is not None and source.any():
                ref_std = sc_utils.region_std(np.asarray(images[t_last]), source)
                gap_supported = any(
                    sc_utils.footprint_detection_score(
                        np.asarray(images[t]), masks[t], source, ref_std
                    )
                    >= self.detection_threshold
                    for t in range(t_last + 1, t_first)
                )
            if not gap_supported:
                orphans.extend(
                    int(d)
                    for d in tracks.filter(pl.col("parent_track") == label)["label_track"]
                    .unique()
                    .to_list()
                )
                continue

            placer = _CellPlacer(
                masks,
                schema_cols,
                pos_cols,
                label,
                parent_dict["parent_track"],
                current_index,
            )
            for t_fill in range(t_last + 1, t_first):
                if np.any(masks[t_fill] == label):
                    continue

                new_mask_frame = sc_utils.source_mask(masks, t_last, label)
                if new_mask_frame is None:
                    break

                pos_fill = sc_utils.interpolate_position(
                    pos_src, pos_dst, t_fill - t_last, t_first - t_last
                )
                conflict_info = sc_utils.analyze_conflict(
                    new_mask_frame, t_fill, masks, self.iou_threshold
                )
                if not conflict_info["conflict_labels"]:
                    new_cells.append(placer.place_mask(new_mask_frame, t_fill, pos_fill))
                else:
                    cell = placer.place_marker(t_fill, pos_fill)
                    if cell is not None:
                        new_cells.append(cell)
            current_index = placer.next_index

        return new_cells, current_index, orphans

    def _relabel_or_remove(
        self,
        conflict_label: int,
        new_val: int,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        track_modifications: dict,
    ) -> None:
        """Relabel a gap-only trajectory to `new_val`, or delete it when that is 0.

        Only the masks are edited, in every frame the trajectory appears in. The
        matching tracks edit is recorded in `track_modifications`.
        """
        times = tracks.filter(pl.col("label_track") == conflict_label)["t"].to_list()
        for t in times:
            masks[t, masks[t] == conflict_label] = new_val
        if new_val == 0:
            track_modifications["remove"].extend((t, conflict_label) for t in times)
        else:
            track_modifications["relabel"].extend(
                (t, conflict_label, new_val) for t in times
            )


class MergeEarlyDivisions(BaseCorrection):
    """Undo prematurely detected divisions.

    In the tracks, this shows as a cell splitting into two daughters that stay side by
    side and appear as a single cell again a few frames later (`A -> A -> B/C -> B/C ->
    A`). The daughters and that reappearing cell are relabeled to the parent.

    A division is undone when all of the following hold:

    - One daughter lives at most `max_lifetime` frames and does not run to the end of
      the sequence, while the other spans at least `min_sibling_frames` frames. A
      division whose daughters are both short-lived is left alone.
    - Neither daughter divides again.
    - The two daughters touch in every frame they share, and neither of them shares a
      frame with the parent.
    - A cell with no parent appears in the frame after the daughters end and is covered
      by their masks (`_find_merge_successor`).
    - A correspondence above `thr_corr` links the parent or one of the daughters to that
      appearing cell (`_merge_is_supported`). All three edges may be missing (e.g., if
      gap is larger than maximum temporal distance for graph edges, or if nodes were
      added during the correction and don't have edges). In that case, only mask coverage
      is used.
    """

    def __init__(
        self,
        max_lifetime: int = 5,
        thr_corr: float = 0.5,
        min_sibling_frames: int = 3,
        min_merge_coverage: float = 0.5,
    ):
        """Initialize correction.

        Args:
            max_lifetime: Longest a daughter may live for its division to count as early.
            thr_corr: Minimum correspondence score on an edge from the parent or a
                daughter into the cell they merge back into.
            min_sibling_frames: Fewest frames the other daughter must span. Two
                short-lived daughters followed by a single cell are ambiguous, an early
                division or a false merge of a real one, so this step declines them.
            min_merge_coverage: Fraction of the appearing cell's first mask the daughters
                must cover for them to count as merged.
        """
        self.max_lifetime = max_lifetime
        self.thr_corr = thr_corr
        self.min_sibling_frames = min_sibling_frames
        self.min_merge_coverage = min_merge_coverage

    def correct(
        self, tracks: pl.DataFrame, masks: np.ndarray, **kwargs
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Merge short-lived daughters back into their parent."""
        edge_predictions = kwargs.get("edge_predictions")
        assert edge_predictions is not None

        tracks_corrected = tracks.clone()
        n_relabeled = 0
        col_name = sc_utils.get_score_col(edge_predictions)

        # a daughter that divides again is a real division, so merging it into the
        # parent would drop that division event
        labels_with_children: set[int] = set(
            tracks_corrected.filter(pl.col("parent_track") > 0)
            .get_column("parent_track")
            .to_list()
        )

        short_tracks = sc_utils.get_short_tracks(
            tracks_corrected, max_lifetime=self.max_lifetime, division_only=True
        )
        logger.info(f"Checking {len(short_tracks)} tracks for early divisions.")

        for label in short_tracks.get_column("label_track").unique():
            track_rows = tracks_corrected.filter(pl.col("label_track") == label).sort("t")
            if track_rows.height == 0:
                continue

            first_row = track_rows.row(0, named=True)
            parent = first_row["parent_track"]
            t_start = first_row["t"]

            if parent == 0:
                continue

            sibling_label = sc_utils.find_sibling(tracks_corrected, label, parent)
            if sibling_label is None:
                continue

            if label in labels_with_children or sibling_label in labels_with_children:
                logger.debug(
                    f"Keeping division {label}/{sibling_label}: a daughter divides again."
                )
                continue

            sibling = tracks_corrected.filter(
                pl.col("label_track") == sibling_label
            ).sort("t")
            t_end_sibling = sibling.get_column("t").max()
            persistent_sibling = (t_end_sibling > t_start) and (
                sibling.height >= self.min_sibling_frames
            )

            if not persistent_sibling:
                continue

            if not sc_utils.cells_touch(tracks_corrected, masks, label, sibling_label):
                continue

            if sc_utils.shares_frames(parent, [label, sibling_label], tracks_corrected):
                continue

            successor = self._find_merge_successor(
                tracks_corrected, masks, label, sibling_label
            )
            if successor is None:
                logger.debug(
                    f"Keeping division {label}/{sibling_label}: the daughters do "
                    "not come back together."
                )
                continue

            if not self._merge_is_supported(
                tracks_corrected,
                edge_predictions,
                parent,
                [label, sibling_label],
                successor,
                col_name,
            ):
                logger.debug(
                    f"Keeping division {label}/{sibling_label}: no correspondence "
                    f"reaches {successor}."
                )
                continue

            merge_labels = [label, sibling_label, successor]
            logger.info(f"Relabeling {merge_labels} to {parent}.")

            # capture indices before relabeling to mark them as merged
            merged_indices = tracks_corrected.filter(
                pl.col("label_track").is_in(merge_labels)
            ).get_column("index")

            # daughters and successor become one parent trajectory. a division edge may
            # span several frames, so the gap it leaves is closed by the following step.
            tracks_corrected, masks = self._merge_labels_into_parent(
                tracks_corrected, masks, merge_labels, parent
            )

            if "cell_source" in tracks_corrected.columns:
                merged_set = merged_indices.to_list()
                tracks_corrected = tracks_corrected.with_columns(
                    cell_source=pl.when(pl.col("index").is_in(merged_set))
                    .then(pl.lit("merged"))
                    .otherwise(pl.col("cell_source"))
                )

            n_relabeled += 1

        if n_relabeled > 0:
            logger.info(f"Relabeled {n_relabeled} tracks.")

        return tracks_corrected, masks

    def _merge_is_supported(
        self,
        tracks: pl.DataFrame,
        edge_predictions: pl.DataFrame,
        parent: int,
        daughters: list[int],
        successor: int,
        col_name: str,
    ) -> bool:
        """True if correspondence to successor above `thr_corr`.

        It may come from the parent or from either daughter. All three edges can be
        absent, pruned by radius or interpolated during gap closing, in which case
        only the mask coverage is used.
        """
        successor_rows = tracks.filter(pl.col("label_track") == successor).sort("t")
        dst = successor_rows.row(0, named=True)["index"]
        srcs = (
            tracks.filter(pl.col("label_track").is_in([parent, *daughters]))
            .group_by("label_track")
            .agg(pl.col("index").sort_by("t").last())
            .get_column("index")
            .to_list()
        )
        preds = edge_predictions.filter(
            pl.col("src").is_in(srcs) & (pl.col("dst") == dst)
        )
        if preds.height == 0:
            return True
        return float(preds.get_column(col_name).to_numpy().max()) > self.thr_corr

    def _relabel_tracks(
        self, tracks: pl.DataFrame, merge_labels: list[int], parent_label: int
    ) -> pl.DataFrame:
        """Merge merge_labels into parent_label, inheriting parent_label's own parent.

        The merged rows (the daughters, and the appearing successor when there is one)
        join the parent's trajectory and must inherit the parent's own parent, rather
        than pointing at the parent itself.
        """
        # look up parent_label's actual parent before any relabeling
        parent_rows = tracks.filter(
            (pl.col("label_track") == parent_label)
            & (pl.col("parent_track") != parent_label)
        )
        parent_of_parent = (
            int(parent_rows["parent_track"].to_numpy().max() or 0)
            if parent_rows.height > 0
            else 0
        )
        return sc_utils.relabel_track(
            tracks, merge_labels, parent_label, parent_of_parent
        )

    def _find_merge_successor(
        self,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        label: int,
        sibling_label: int,
    ) -> int | None:
        """Return the cell the two daughters merge back into, or None.

        That cell has no parent, starts exactly one frame after the last daughter frame
        so the merged trajectory has no gap, and its first mask is covered by the union
        of the daughters' last masks (at least `min_merge_coverage`). A real division
        leaves no such cell: its daughters go on as two trajectories, or disappear
        without one taking their place.
        """
        daughters = [label, sibling_label]
        d_frames = set(
            tracks.filter(pl.col("label_track").is_in(daughters))["t"].to_list()
        )
        if not d_frames:
            return None
        d_t_max = max(d_frames)

        # union of the daughters' masks at their respective last frames
        union = np.zeros(masks.shape[1:], dtype=bool)
        for d in daughters:
            d_last = int(max(tracks.filter(pl.col("label_track") == d)["t"].to_list()))
            union |= masks[d_last] == d
        if not union.any():
            return None

        t_succ = d_t_max + 1
        if t_succ >= masks.shape[0]:
            return None
        candidates = (
            tracks.filter((pl.col("parent_track") == 0) & (pl.col("t") == t_succ))
            .get_column("label_track")
            .unique()
            .sort()
        )
        for cand in (int(c) for c in candidates):
            if cand in daughters:
                continue
            cand_frames = set(tracks.filter(pl.col("label_track") == cand)["t"].to_list())
            if cand_frames & d_frames or min(cand_frames) != t_succ:
                continue
            if sc_utils.mask_coverage(masks[t_succ] == cand, union) >= (
                self.min_merge_coverage
            ):
                return cand
        return None

    def _merge_labels_into_parent(
        self,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        merge_labels: list[int],
        parent: int,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Relabel every label in merge_labels to parent, in both masks and tracks.

        The daughters touch, so their union stays one connected region per frame. A gap
        left in front of the successor is closed by the next gap-closing step.
        """
        for t in range(masks.shape[0]):
            sel = np.isin(masks[t], merge_labels)
            if sel.any():
                masks[t][sel] = parent
        tracks = self._relabel_tracks(tracks, merge_labels, parent)
        return tracks, masks


class _TrajectoryInfo(NamedTuple):
    """First frame, last frame, all frames, row indices, and whether it has a parent."""

    t_min: int
    t_max: int
    frames: set[int]
    indices: list[int]
    has_parent: bool


class SplitFalseMerges(BaseCorrection):
    """Remove masks that cover two cells at once.

    Two patterns are handled:

    - Merged daughters: two daughters that touch are covered by a single mask for up to
      `max_merge_frames` frames and reappear as two cells afterwards. The merged mask is
      deleted, the two daughters are drawn back into the merge frame, and the cells that
      reappear are relabeled to them. The daughters are read off the lineage, so the
      division must have been detected for the pattern to be found at all.
    - A mask inside another cell's gap: a short-lived mask exists only within the gap of
      one or more cells that the tracker follows across it, and no strong correspondence
      leaves it. It is deleted, and the last gap-closing step fills the gap.
    """

    def __init__(
        self,
        thr_corr: float = 0.5,
        thr_div: float = 0.5,
        min_merge_coverage: float = 0.5,
        max_merge_frames: int = 3,
    ):
        """Initialize correction.

        Args:
            thr_corr: Minimum correspondence score for an edge from the parent into the
                merged mask, from a daughter into the cell that carries it on, and for a
                cell to count as tracked across a merge sitting in its gap.
            thr_div: Minimum division score for an edge from the merged mask to the
                cells that reappear. Supports a split, and never blocks one: a division
                already detected upstream needs no second confirmation here.
            min_merge_coverage: Fraction of the merged mask the two daughters' masks
                must cover for the geometry alone to carry the split. Below it, an edge
                has to support the merge instead.
            max_merge_frames: Longest a merge may last to be split. The daughters are
                redrawn from their masks in the frame before it, which goes stale as the
                merge runs on, so a long one is left to the tracker.
        """
        self.thr_corr = thr_corr
        self.thr_div = thr_div
        self.min_merge_coverage = min_merge_coverage
        self.max_merge_frames = max_merge_frames

    def correct(
        self, tracks: pl.DataFrame, masks: np.ndarray, **kwargs
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Split merged daughters and delete merged masks sitting inside a gap."""
        edge_predictions = kwargs.get("edge_predictions")
        if edge_predictions is None:
            return tracks, masks
        if "p1" not in edge_predictions.columns or "p2" not in edge_predictions.columns:
            return tracks, masks

        # one candidate per division, so a pair of daughters cannot appear twice
        candidates = self._find_merged_daughters(tracks, masks, edge_predictions)
        if candidates:
            logger.info(f"Found {len(candidates)} merged-daughter candidate(s).")

        tracks_corrected = tracks.clone()
        new_rows: list[dict] = []
        rows_to_remove: list[tuple[int, int]] = []
        # (old_label, new_label, t_start, target_parent)
        all_daughter_relabelings: list[tuple[int, int, int, int]] = []

        processed_merges: set[int] = set()
        for candidate in candidates:
            (
                daughter_a_idx,
                daughter_b_idx,
                merged_idx,
                after_a_idx,
                after_b_idx,
                t_merge,
            ) = candidate
            if merged_idx in processed_merges:
                continue
            # the two daughters must touch at all shared frames
            daughter_a_rows = tracks_corrected.filter(pl.col("index") == daughter_a_idx)
            daughter_b_rows = tracks_corrected.filter(pl.col("index") == daughter_b_idx)
            if daughter_a_rows.height == 0 or daughter_b_rows.height == 0:
                continue
            daughter_a_label = int(daughter_a_rows[0, "label_track"])
            daughter_b_label = int(daughter_b_rows[0, "label_track"])
            if not sc_utils.cells_touch(
                tracks_corrected,
                masks,
                daughter_a_label,
                daughter_b_label,
            ):
                logger.debug(
                    f"Skipping merge at t={t_merge}: daughters {daughter_a_label} and "
                    f"{daughter_b_label} do not touch."
                )
                continue
            result = self._apply_split(
                tracks_corrected,
                masks,
                daughter_a_idx,
                daughter_b_idx,
                merged_idx,
                after_a_idx,
                after_b_idx,
                t_merge,
                tracks_corrected.columns,
            )
            if result is not None:
                new_cells, remove_pairs, daughter_relabelings = result
                new_rows.extend(new_cells)
                rows_to_remove.extend(remove_pairs)
                all_daughter_relabelings.extend(daughter_relabelings)
                processed_merges.add(merged_idx)

        if rows_to_remove:
            removed_labels = {lbl for _, lbl in rows_to_remove}
            tracks_corrected = sc_utils.remove_rows(tracks_corrected, rows_to_remove)
            fully_removed = {
                lbl
                for lbl in removed_labels
                if tracks_corrected.filter(pl.col("label_track") == lbl).height == 0
            }
            if fully_removed:
                # a cell whose relabeling _apply_split skipped still points at the mask
                # that is now gone. zero those, so the lineage stays valid.
                tracks_corrected = tracks_corrected.with_columns(
                    parent_track=pl.when(
                        pl.col("parent_track").is_in(list(fully_removed))
                    )
                    .then(pl.lit(0))
                    .otherwise(pl.col("parent_track"))
                )
        if new_rows:
            new_df = pl.DataFrame(new_rows, schema=tracks_corrected.schema)
            tracks_corrected = pl.concat([tracks_corrected, new_df])

        # relabel the cells after the merge back to the daughters (masks were already
        # done in place), pointing them at the daughters' parent rather than the merge
        for old_label, new_label, t_start, target_parent in all_daughter_relabelings:
            if old_label == new_label:
                continue
            tracks_corrected = sc_utils.relabel_track(
                tracks_corrected, [old_label], new_label, target_parent, t_start
            )

        gap_removes = self._remove_merges_inside_gaps(
            tracks_corrected, masks, edge_predictions
        )
        if gap_removes:
            tracks_corrected = sc_utils.remove_rows(
                tracks_corrected, gap_removes, zero_dangling_parents=True
            )

        return tracks_corrected, masks

    def _find_merged_daughters(
        self, tracks: pl.DataFrame, masks: np.ndarray, edge_predictions: pl.DataFrame
    ) -> list[tuple[int, int, int, int, int, int]]:
        """Find masks that cover two daughters of the same parent.

        The daughters come from the lineage, not from the edges: two daughters of one
        parent that end in the same frame and never divide again. The mask that covers
        them is the one they overlap most at the next frame, and it must start there.
        Either their coverage of it or the edge evidence then decides.

        Returns:
            `(daughter_a_idx, daughter_b_idx, merged_idx, after_a_idx, after_b_idx,
            t_merge)` tuples, earliest merge first.
        """
        spans: dict[int, tuple[int, int]] = {
            int(row["label_track"]): (int(row["t_min"]), int(row["t_max"]))
            for row in tracks.group_by("label_track")
            .agg(pl.col("t").min().alias("t_min"), pl.col("t").max().alias("t_max"))
            .iter_rows(named=True)
        }
        index_at: dict[tuple[int, int], int] = {
            (int(row["label_track"]), int(row["t"])): int(row["index"])
            for row in tracks.select(["label_track", "t", "index"]).iter_rows(named=True)
        }
        labels_with_children: set[int] = set(
            tracks.filter(pl.col("parent_track") > 0).get_column("parent_track").to_list()
        )
        daughters_of: dict[int, list[int]] = {}
        for row in (
            tracks.filter(pl.col("parent_track") > 0)
            .select("label_track", "parent_track")
            .unique()
            .iter_rows(named=True)
        ):
            daughters_of.setdefault(int(row["parent_track"]), []).append(
                int(row["label_track"])
            )

        candidates: list[tuple[int, int, int, int, int, int]] = []
        for parent, pair in daughters_of.items():
            if len(pair) != 2:
                continue
            daughter_a, daughter_b = pair
            # a daughter that divides again is a real division, not half of a merge
            if daughter_a in labels_with_children or daughter_b in labels_with_children:
                continue
            t_last = spans[daughter_a][1]
            if t_last != spans[daughter_b][1] or t_last + 1 >= masks.shape[0]:
                continue
            t_merge = t_last + 1

            union = (masks[t_last] == daughter_a) | (masks[t_last] == daughter_b)
            if not union.any():
                continue
            merged_label, coverage = self._best_covered_label(
                masks[t_merge], union, exclude=(daughter_a, daughter_b)
            )
            # a mask that predates the merge frame is an established cell, not the
            # daughters under one label
            if merged_label is None or spans[merged_label][0] != t_merge:
                continue
            merged_span = spans[merged_label][1] - t_merge + 1
            if merged_span > self.max_merge_frames:
                logger.debug(
                    f"Keeping mask {merged_label} at t={t_merge}: the merge lasts "
                    f"{merged_span} frames, over max_merge_frames."
                )
                continue

            merged_idx = index_at.get((merged_label, t_merge))
            daughter_a_idx = index_at.get((daughter_a, t_last))
            daughter_b_idx = index_at.get((daughter_b, t_last))
            if merged_idx is None or daughter_a_idx is None or daughter_b_idx is None:
                continue

            outgoing = edge_predictions.filter(pl.col("src") == merged_idx).sort(
                "p2", descending=True
            )
            if outgoing.height < 2:
                continue
            after_a_idx = int(outgoing[0, "dst"])
            after_b_idx = int(outgoing[1, "dst"])

            if coverage < self.min_merge_coverage and not self._split_is_supported(
                edge_predictions,
                index_at.get((parent, spans[parent][1])),
                [daughter_a_idx, daughter_b_idx],
                merged_idx,
                [after_a_idx, after_b_idx],
            ):
                logger.debug(
                    f"Keeping mask {merged_label} at t={t_merge}: daughters "
                    f"{daughter_a}/{daughter_b} cover {coverage:.2f} and no edge "
                    "supports the merge."
                )
                continue

            candidates.append(
                (
                    daughter_a_idx,
                    daughter_b_idx,
                    merged_idx,
                    after_a_idx,
                    after_b_idx,
                    t_merge,
                )
            )

        return sorted(candidates, key=lambda c: c[5])

    @staticmethod
    def _best_covered_label(
        frame: np.ndarray, cover: np.ndarray, exclude: tuple[int, ...]
    ) -> tuple[int | None, float]:
        """Label in `frame` that `cover` overlaps the largest fraction of."""
        best, best_coverage = None, 0.0
        for label in np.unique(frame[cover]):
            label = int(label)
            if label == 0 or label in exclude:
                continue
            coverage = sc_utils.mask_coverage(frame == label, cover)
            if coverage > best_coverage:
                best, best_coverage = label, coverage
        return best, best_coverage

    def _split_is_supported(
        self,
        edge_predictions: pl.DataFrame,
        parent_idx: int | None,
        daughter_indices: list[int],
        merged_idx: int,
        after_indices: list[int],
    ) -> bool:
        """True if an edge supports reading the mask as the two daughters merged.

        The daughters themselves are not asked: a single daughter is not the whole
        merged mask, so the model has no reason to score that as a correspondence
        (measured at 0.02 to 0.19 on real merges). What it does predict is the parent
        continuing into the merge, the merge dividing into the cells that reappear, and
        each daughter continuing into the one that carries it on. Every edge may be
        missing, pruned by radius or interpolated during gap closing, in which case the
        mask coverage decides alone.
        """
        if (
            parent_idx is not None
            and sc_utils.correspondence_to(edge_predictions, [parent_idx], merged_idx)
            > self.thr_corr
        ):
            return True
        divides = edge_predictions.filter(
            (pl.col("src") == merged_idx) & pl.col("dst").is_in(after_indices)
        )
        if divides.height and float(divides["p2"].to_numpy().max()) > self.thr_div:
            return True
        carries_on = edge_predictions.filter(
            pl.col("src").is_in(daughter_indices) & pl.col("dst").is_in(after_indices)
        )
        return bool(
            carries_on.height and float(carries_on["p1"].to_numpy().max()) > self.thr_corr
        )

    def _remove_merges_inside_gaps(
        self,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        edge_predictions: pl.DataFrame,
    ) -> list[tuple[int, int]]:
        """Delete merged masks that sit inside the gap of the cells they cover.

        The step requires the mask to be lineage-free, which a merge need not be in
        general: it deletes only masks the tracker left out of every lineage, so a merge
        it did give a parent or daughters to is left to the merged-daughter pattern
        above. All of the following must hold:

        1. It exists only within a gap and shares no frame with the covering cells.
        2. It has no parent and no daughters.
        3. No correspondence above `thr_corr` leaves it for a cell outside it.
        4. The covering cells either all link across the gap by correspondence, or their
           masks together cover more than `min_merge_coverage` of it in every frame.

        Covering cells are the ones that exist before and after the merged mask but
        never during it. Their mask is the union of their pixels in the last frame
        before and the first frame after.

        Returns:
            `(t, label_track)` pairs for the caller to drop from the tracks.
        """
        traj_info = {
            int(row["label_track"]): _TrajectoryInfo(
                int(row["t_min"]),
                int(row["t_max"]),
                {int(t) for t in row["t_list"]},
                [int(i) for i in row["indices"]],
                int(row["parent_max"]) > 0,
            )
            for row in tracks.group_by("label_track")
            .agg(
                pl.col("t").min().alias("t_min"),
                pl.col("t").max().alias("t_max"),
                pl.col("t").alias("t_list"),
                pl.col("index").alias("indices"),
                pl.col("parent_track").max().alias("parent_max"),
            )
            .iter_rows(named=True)
        }
        labels_with_children: set[int] = {
            int(v)
            for v in tracks.filter(pl.col("parent_track") > 0)["parent_track"].to_list()
        }

        gap_windows = [
            (int(g["t_prev"]) + 1, int(g["t_next"]) - 1)
            for g in sc_utils.find_trajectories_with_gaps(tracks).iter_rows(named=True)
        ]
        if not gap_windows:
            return []
        gap_only = {
            lbl
            for lbl, info in traj_info.items()
            if any(gs <= info.t_min and info.t_max <= ge for gs, ge in gap_windows)
        }

        removals: list[tuple[int, int]] = []
        for merged_label in gap_only:
            merged = traj_info[merged_label]
            if merged.has_parent or merged_label in labels_with_children:
                continue

            # a mask the tracker follows onwards is a real cell, not a merge
            outgoing = edge_predictions.filter(
                pl.col("src").is_in(merged.indices) & ~pl.col("dst").is_in(merged.indices)
            )
            if (
                outgoing.height > 0
                and float(outgoing["p1"].to_numpy().max()) > self.thr_corr
            ):
                continue

            covering = self._covering_cells(merged_label, merged, traj_info)
            if not covering:
                continue

            if not (
                self._covered_by_correspondence(
                    tracks, covering, merged, edge_predictions
                )
                or self._covered_by_overlap(
                    masks, covering, merged_label, merged, traj_info
                )
            ):
                continue

            logger.debug(
                f"Removing merged mask {merged_label} "
                f"(t={merged.t_min}-{merged.t_max}, covers {len(covering)} cells)."
            )
            for t in sorted(merged.frames):
                masks[t, masks[t] == merged_label] = 0
                removals.append((t, merged_label))

        if removals:
            n = len({lbl for _, lbl in removals})
            logger.info(f"Removed {n} merged mask(s) sitting inside a gap.")
        return removals

    @staticmethod
    def _covering_cells(
        merged_label: int,
        merged: _TrajectoryInfo,
        traj_info: dict[int, _TrajectoryInfo],
    ) -> list[int]:
        """Labels that exist before and after the merged mask but never during it."""
        return [
            lbl
            for lbl, info in traj_info.items()
            if lbl != merged_label
            and info.t_min < merged.t_min
            and info.t_max > merged.t_max
            and not (info.frames & merged.frames)
        ]

    def _covered_by_correspondence(
        self,
        tracks: pl.DataFrame,
        covering: list[int],
        merged: _TrajectoryInfo,
        edge_predictions: pl.DataFrame,
    ) -> bool:
        """True if every covering cell links across the merge by correspondence."""
        before: list[int] = []
        after: list[int] = []
        for lbl in covering:
            pre = (
                tracks.filter(
                    (pl.col("label_track") == lbl) & (pl.col("t") < merged.t_min)
                )
                .sort("t", descending=True)
                .head(1)
            )
            post = (
                tracks.filter(
                    (pl.col("label_track") == lbl) & (pl.col("t") > merged.t_max)
                )
                .sort("t")
                .head(1)
            )
            if pre.height:
                before.append(int(pre[0, "index"]))
            if post.height:
                after.append(int(post[0, "index"]))
        return bool(
            before
            and after
            and sc_utils.check_for_correspondence(
                before, after, edge_predictions, self.thr_corr
            )
        )

    def _covered_by_overlap(
        self,
        masks: np.ndarray,
        covering: list[int],
        merged_label: int,
        merged: _TrajectoryInfo,
        traj_info: dict[int, _TrajectoryInfo],
    ) -> bool:
        """True if the covering cells' masks cover the merged mask in every frame."""
        union_mask = np.zeros(masks.shape[1:], dtype=bool)
        for lbl in covering:
            frames = traj_info[lbl].frames
            t_before = max((t for t in frames if t < merged.t_min), default=None)
            t_after = min((t for t in frames if t > merged.t_max), default=None)
            if t_before is not None:
                union_mask |= masks[t_before] == lbl
            if t_after is not None:
                union_mask |= masks[t_after] == lbl
        if not union_mask.any():
            return False
        for t in sorted(merged.frames):
            merged_pix = masks[t] == merged_label
            if not merged_pix.any():
                continue
            if sc_utils.mask_coverage(merged_pix, union_mask) <= self.min_merge_coverage:
                return False
        return True

    def _apply_split(
        self,
        tracks: pl.DataFrame,
        masks: np.ndarray,
        daughter_a_idx: int,
        daughter_b_idx: int,
        merged_idx: int,
        after_a_idx: int,
        after_b_idx: int,
        t_merge: int,
        schema_cols: list[str],
    ) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
        """Delete the merged mask, draw both daughters back in, relabel what follows.

        Returns:
            `(new_cells, rows_to_remove, daughter_relabelings)` or `None` if aborted.
            `daughter_relabelings` contains `(old_label, new_label, t_start,
            target_parent)` tuples that the caller applies to `tracks`. Masks are
            relabeled in-place here.
        """
        merged_rows = tracks.filter(pl.col("index") == merged_idx)
        daughter_a_rows = tracks.filter(pl.col("index") == daughter_a_idx)
        daughter_b_rows = tracks.filter(pl.col("index") == daughter_b_idx)

        if (
            merged_rows.height == 0
            or daughter_a_rows.height == 0
            or daughter_b_rows.height == 0
        ):
            return None

        merged_label = int(merged_rows[0, "label_track"])
        daughter_a_label = int(daughter_a_rows[0, "label_track"])
        daughter_b_label = int(daughter_b_rows[0, "label_track"])
        daughter_a_parent = int(daughter_a_rows[0, "parent_track"])
        daughter_b_parent = int(daughter_b_rows[0, "parent_track"])
        t_last = int(daughter_a_rows[0, "t"])

        pos_cols = position_columns(tracks.columns)

        def _row_pos(rows: pl.DataFrame) -> np.ndarray:
            return np.array([float(rows[0, c] or 0.0) for c in pos_cols])

        pos_a = _row_pos(daughter_a_rows)
        pos_b = _row_pos(daughter_b_rows)

        after_a_rows = tracks.filter(pl.col("index") == after_a_idx)
        after_b_rows = tracks.filter(pl.col("index") == after_b_idx)
        pos_after_a = _row_pos(after_a_rows) if after_a_rows.height > 0 else pos_a
        pos_after_b = _row_pos(after_b_rows) if after_b_rows.height > 0 else pos_b
        pos_interp_a = (pos_a + pos_after_a) / 2.0
        pos_interp_b = (pos_b + pos_after_b) / 2.0

        mask_src_a = sc_utils.source_mask(masks, t_last, daughter_a_label)
        mask_src_b = sc_utils.source_mask(masks, t_last, daughter_b_label)

        if mask_src_a is None and mask_src_b is None:
            logger.debug(
                f"Split skipped for merged mask {merged_label} at t={t_merge}: "
                "neither daughter has source pixels."
            )
            return None

        masks[t_merge, masks[t_merge] == merged_label] = 0

        max_index = int(tracks.get_column("index").to_numpy().max()) + 1
        new_cells: list[dict] = []

        def _would_violate_ctc(label: int, t_merge: int) -> bool:
            child_ts = tracks.filter(pl.col("parent_track") == label).get_column("t")
            return len(child_ts) > 0 and int(child_ts.to_numpy().min()) <= t_merge

        for mask_src, label, parent, pos in (
            (mask_src_a, daughter_a_label, daughter_a_parent, pos_interp_a),
            (mask_src_b, daughter_b_label, daughter_b_parent, pos_interp_b),
        ):
            if mask_src is None or _would_violate_ctc(label, t_merge):
                continue
            masks[t_merge, sc_utils.ensure_connected(mask_src)] = label
            new_cells.append(
                sc_utils.make_track_row(
                    max_index,
                    t_merge,
                    label,
                    parent,
                    pos,
                    pos_cols,
                    schema_cols,
                    "split",
                )
            )
            max_index += 1

        logger.info(
            f"Split merged mask {merged_label} at t={t_merge} back into "
            f"{daughter_a_label} and {daughter_b_label}."
        )

        labels_being_relabeled: set[int] = {
            int(r[0, "label_track"]) for r in [after_a_rows, after_b_rows] if r.height > 0
        }
        daughter_relabelings: list[tuple[int, int, int, int]] = []
        for after_rows, daughter_label, daughter_parent in [
            (after_a_rows, daughter_a_label, daughter_a_parent),
            (after_b_rows, daughter_b_label, daughter_b_parent),
        ]:
            if after_rows.height == 0:
                continue
            after_label = int(after_rows[0, "label_track"])
            if after_label == daughter_label:
                continue
            # a strong division edge can point at a neighbour from another lineage.
            # relabeling that would cut its track and give its parent three daughters.
            if int(after_rows[0, "parent_track"]) != merged_label:
                logger.debug(
                    f"Skipping relabel of {after_label} to {daughter_label}: it does "
                    f"not come from merged mask {merged_label} "
                    f"(parent={int(after_rows[0, 'parent_track'])})."
                )
                continue
            t_rediv = int(after_rows[0, "t"])
            after_max_t = int(after_rows.get_column("t").to_numpy().max())
            persistent_children = tracks.filter(
                (pl.col("parent_track") == daughter_label)
                & (~pl.col("label_track").is_in(list(labels_being_relabeled)))
                & (pl.col("t") <= after_max_t)
            )
            if persistent_children.height > 0:
                logger.debug(
                    f"Skipping relabel of {after_label} to {daughter_label}: "
                    f"{daughter_label} still has daughters in that frame range."
                )
                continue
            for t_frame in range(t_rediv, masks.shape[0]):
                frame = masks[t_frame]
                frame[frame == after_label] = daughter_label
            daughter_relabelings.append(
                (after_label, daughter_label, t_rediv, daughter_parent)
            )

        # delete a multi-frame merge across its whole stretch, so the reappearing cells
        # can be repointed rather than left under a mask that is gone
        rows_to_remove = [(t_merge, merged_label)]
        if daughter_relabelings:
            t_rediv_min = min(t_rediv for *_, t_rediv, _ in daughter_relabelings)
            for t in range(t_merge + 1, t_rediv_min):
                if np.any(masks[t] == merged_label):
                    masks[t, masks[t] == merged_label] = 0
                    rows_to_remove.append((t, merged_label))

        return new_cells, rows_to_remove, daughter_relabelings
