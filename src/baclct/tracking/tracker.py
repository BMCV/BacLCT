"""Reconstruct trajectories from multi-frame edge predictions using LAP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import polars as pl
import rustworkx as rx
from tqdm import tqdm

from baclct.data.dataset import GraphDataset
from baclct.io import (
    create_track_df,
    create_trajectory_masks,
    export_tracking_results_ctc,
    export_tracking_results_flat,
)
from baclct.tracking.graph import (
    edge_df_to_graph,
    label_trajectories,
)
from baclct.tracking.postprocessing import (
    edge_preds_to_sigmoid,
    edge_preds_to_softmax,
    expand_binary_predictions,
    merge_edge_predictions,
    resolve_duplicate_predictions,
)
from baclct.tracking.segmentation_correction import BaseCorrection
from baclct.utils.logger import get_pylogger
from baclct.utils.progress import ProgressCallback, track_iter

logger = get_pylogger(__name__)


class BaseTracker(ABC):
    """Base class for graph-based cell trackers.

    The GNN classifies each candidate edge independently and does not enforce biological
    tracking constraints (1-to-1 correspondences, 1-to-2 divisions). Because edges may
    span multiple frames, a single cell can also end up with several competing
    correspondence or division candidates. From these predictions, the tracker
    reconstructs trajectories.

    This base class provides the shared interface:
      - Prediction sanitization (duplicate resolution, edge direction, binary-to-3-class
        expansion for models without learned divisions) and sigmoid/softmax normalization.
      - Candidate selection: Edge is a correspondence candidate if `p1` is larger than
        `p0` and `p2` and above `thr_corr` (analogously for division candidates with `p2`
        and `thr_div`). When `ignore_negative_edges=True`, `p0` is dropped from the
        comparison.
      - Connected-component trajectory labeling with in-/out-degree sanity checks against
        1-to-1 correspondences and `max_division_targets`.
      - Optional segmentation correction and export to CTC or flat formats.

    Subclasses implement `track()` to define how candidate edges are matched.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        predictions: pl.DataFrame,
        segmentation_correction: BaseCorrection | list[BaseCorrection] | None = None,
        max_division_targets: int = 2,
        norm_fn: Literal["sigmoid", "softmax", "none"] = "softmax",
        thr_corr: float | None = None,
        thr_div: float | None = None,
        ignore_negative_edges: bool = False,
        edge_direction: Literal["both", "future"] = "both",
    ) -> None:
        """Initialize tracker.

        Args:
            dataset: Dataset containing node and edge information as well as paths to
                features, images, and masks.
            predictions: Per-edge GNN output probabilities, one column per class
                (e.g., `p0` to `p2` for tracking with division detection).
            segmentation_correction: Segmentation error correction class.
            max_division_targets: Maximum number of daughter cells per parent. Beyond
                that, assignments are ignored.
            norm_fn: Normalization function applied to class logits.
            thr_corr: Minimum threshold for correspondence (after norm).
            thr_div: Minimum threshold for division (after norm).
            ignore_negative_edges: If `True`, correspondences and divisions are always
                considered, even if the pseudoprobability for the inactive class is
                higher. Otherwise, `p0` acts as suppression.
            edge_direction: How to reduce a bidirectional graph. Both modes keep the
                forward edges (`src < dst`), 'both' first averages each pair's backward
                prediction into its forward one.
        """
        self.dataset = dataset
        predictions, norm_fn = self._sanitize_predictions(
            predictions, norm_fn, edge_direction
        )
        self.predictions = self._add_node_metadata(predictions).cast(
            dict.fromkeys(["src", "dst"], pl.UInt32)
        )

        self.norm_fn = norm_fn
        self.max_divisions = max_division_targets

        if isinstance(segmentation_correction, BaseCorrection):
            self.segmentation_correction = [segmentation_correction]
        else:
            self.segmentation_correction = segmentation_correction

        self.corrected_masks = None

        self.thr_corr = thr_corr
        self.thr_div = thr_div
        self.ignore_negative_edges = ignore_negative_edges

        if self.ignore_negative_edges:
            assert thr_corr is not None, "Please provide a threshold for `thr_corr`."
            assert thr_div is not None, "Please provide a threshold for `thr_div`."

        self.tracks = None

        # optional progress sink, attached by callers (e.g. the napari plugin). set as an
        # attribute rather than a constructor arg so hydra tracker configs stay unchanged.
        self.progress: ProgressCallback | None = None
        self._tracked_masks: np.ndarray | None = None

    @staticmethod
    def _sanitize_predictions(
        predictions: pl.DataFrame,
        norm_fn: Literal["sigmoid", "softmax", "none"],
        edge_direction: Literal["both", "future"],
    ) -> tuple[pl.DataFrame, Literal["sigmoid", "softmax", "none"]]:
        """Resolve duplicates, filter direction, and expand binary predictions.

        Applies three sanitization steps in order:
        1. Average same-edge duplicates (e.g. from overlapping inference windows).
        2. Merge or filter bidirectional edges according to `edge_direction`.
        3. If only a single prediction column exists (i.e. during binary classification),
           expand via sigmoid into 3-class format and override `norm_fn` to `"none"` so
           sigmoid is applied only once.
        """
        predictions = resolve_duplicate_predictions(predictions)
        predictions = merge_edge_predictions(predictions, edge_direction)
        if len(predictions.select(r"^p\d$").columns) == 1:
            predictions = expand_binary_predictions(predictions)
            norm_fn = "none"

        return predictions, norm_fn

    def _add_node_metadata(self, edge_predictions: pl.DataFrame) -> pl.DataFrame:
        """Add node data to edges.

        Returns:
            Edge predictions with node data suffixed with `_src` and `dst`, e.g., `t_src`.
        """
        timepoints = self.dataset.node_feats.select("index", "t")

        return (
            edge_predictions.join(timepoints, left_on="src", right_on="index")
            .join(timepoints, left_on="dst", right_on="index", suffix="_dst")
            .select(
                "src", "dst", r"^p\d$", "t", "t_dst", td=pl.col("t_dst") - pl.col("t")
            )
        )

    def normalize_edges(
        self,
        how: Literal["sigmoid", "softmax", "none"] = "softmax",
    ) -> pl.DataFrame:
        """Normalize edge prediction logits.

        Args:
            how: Normalization function name. Use "none" when predictions are
                already normalized (e.g. after expand_binary_predictions).

        Returns:
            Dataframe with normalized edge pseudoprobabilities.
        """
        if how == "none":
            preds = self.predictions
        elif how == "sigmoid":
            preds = edge_preds_to_sigmoid(self.predictions)
        elif how == "softmax":
            preds = edge_preds_to_softmax(self.predictions)
        else:
            raise ValueError(
                f"{how!r} is not a valid normalization. "
                "Please use one of 'sigmoid', 'softmax', 'none'."
            )
        # nan predictions (e.g. from gnn numerical failures caused by single nan feature
        # which is in term distributed via message passing and batch norm) must be
        # dropped: in polars, nan > threshold is true, so nan edges would otherwise pass
        # all candidate filters and be greedily assigned as divisions (nan appears first
        # when sorting descending). should not be an issue, since nan features are
        # sanitized or raise.
        return preds.filter(~pl.col("p0").is_nan())

    def _find_corr_and_div_candidates(
        self, preds_norm
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Find active candidate edges.

        Applies tresholds and suppression (if `ignore_negative_edges=False`).

        Args:
            preds_norm: Normalized edge pseudoprobabilities.

        Returns:
            Active edges.
        """
        if "p2" not in preds_norm:
            raise ValueError(
                "Could not find division predictions `p2` in tracks. If using "
                "predictions without learned division detection, please expand using "
                "`expand_binary_predictions` first."
            )

        filt_corr = pl.col("p1") > (self.thr_corr or 0)
        filt_div = pl.col("p2") > (self.thr_div or 0)

        cols = ["p1", "p2"]
        if not self.ignore_negative_edges:
            cols.insert(0, "p0")
        pmax = pl.max_horizontal(*cols)

        corr = preds_norm.filter(filt_corr, p1=pmax)
        div = preds_norm.filter(filt_div, p2=pmax)

        return corr, div

    def find_trajectories(self, edge_data: pl.DataFrame) -> pl.DataFrame:
        """Construct trajectories from edge assignments.

        Employs sanity checks (1-to-1 correspondence, 1-to-2 divisions) and labels
        connected components in graph.

        Args:
            edge_data: Edges with `src`, `dst`, and predicted type ('traj' or 'div').

        Returns:
            Edges annotated with trajectory label and parent label.
        """
        G = edge_df_to_graph(edge_data, self.dataset.node_feats)
        assert (k := max([d[1] for d in G.in_degree()])) <= 1, (
            f"Edge data contains one-to-many assignments (1-to-{k})"
        )
        assert (m := max([d[1] for d in G.out_degree()])) <= self.max_divisions, (
            f"Edge data contains division with more than {self.max_divisions} children "
            f"(1-to-{m})."
        )

        traj_labels, parent_labels, _roots = label_trajectories(G)
        self.tracks = create_track_df(self.dataset.node_feats, traj_labels, parent_labels)

        return self.tracks

    def apply_corrections(self) -> None:
        """Apply segmentation error correction.

        A corrector that fails leaves the run uncorrected rather than aborting it: the
        tracks are restored and the masks are rebuilt from them by `tracked_masks()`.
        """
        if not self.segmentation_correction:
            return

        if self.tracks is None:
            raise ValueError("Cannot apply corrections without tracks.")

        # load and remap masks to track labels. default to label_old='label' as in
        # export_results
        masks = create_trajectory_masks(
            self.tracks,
            self.dataset.masks,
            label_old="label",
            label_new="label_track",
        )

        # correctors threshold on probabilities (thr_corr / thr_div), so they must
        # receive normalized predictions, not the raw logits stored in self.predictions.
        edge_predictions = self.normalize_edges(self.norm_fn)

        # images are optional: passed through for image-aware correction (e.g. gating
        # gap interpolation on a detection candidate). None for mask-only datasets.
        # accept eager (numpy) or lazy (dask) arrays. guard against non-array attributes
        # such as Mock datasets in tests.
        images = getattr(self.dataset, "images", None)
        if not isinstance(images, (np.ndarray, da.Array)):
            images = None

        uncorrected = self.tracks
        for correction in self.segmentation_correction:
            try:
                self.tracks, masks = correction.correct(
                    self.tracks,
                    masks,
                    edge_predictions=edge_predictions,
                    images=images,
                )
            except Exception as err:
                logger.warning(
                    f"Segmentation correction ({type(correction).__name__}) failed, "
                    f"keeping the uncorrected tracks and masks: {err}"
                )
                # the steps edit `masks` in place, so it is already partly corrected and
                # must not be paired with the restored tracks
                self.tracks = uncorrected
                self.corrected_masks = None
                return

        self.corrected_masks = masks

    @abstractmethod
    def track(self) -> pl.DataFrame:
        """Implements tracking logic."""
        raise NotImplementedError

    def tracked_masks(
        self,
        tracks: pl.DataFrame | None = None,
        label_old: str = "label",
        label_new: str = "label_track",
    ) -> np.ndarray:
        """Masks relabelled with trajectory labels.

        Returns the segmentation-corrected masks when a corrector produced them (they are
        already relabelled), otherwise remaps the original masks onto trajectory labels.

        Args:
            tracks: Tracks to relabel by. Defaults to the tracks from `track()`.
            label_old: Column name for the label in the original masks.
            label_new: Column name for the label in the tracked masks.

        Returns:
            Masks with trajectory labels, same shape as the input masks.
        """
        if self.corrected_masks is not None:
            return self.corrected_masks

        if tracks is None:
            assert self.tracks is not None, (
                "Please run tracker.track() or provide valid track df."
            )
            tracks = self.tracks
            if self._tracked_masks is not None:
                return self._tracked_masks

        masks = self.dataset.masks
        assert masks is not None, "Please provide valid segmentation masks."
        tracked = create_trajectory_masks(
            tracks, masks, label_old=label_old, label_new=label_new
        )
        if tracks is self.tracks:
            self._tracked_masks = tracked
        return tracked

    def export_results(
        self,
        tracks: pl.DataFrame | None,
        res_dir: Path,
        format: Literal["ctc", "flat"],
        label_old: str = "label",
        label_new: str = "label_track",
    ) -> tuple[np.ndarray | pl.DataFrame, np.ndarray]:
        """Export tracking results.

        Args:
            tracks: Dataframe with edge indices, trajectory label, and parent label.
            res_dir: Output directory.
            format: Format for saving outputs. Either CTC or flat (single stacks for
                images and labels, and a .csv file containing tracks).
            label_old: Column name for label in original masks.
            label_new: Column name for label in tracked masks.

        Returns:
            lineage: CTC-formatted (lbep) or single-object lineage.
            tracked_masks: Masks with tracked labels.

        """
        if tracks is None:
            assert self.tracks is not None, (
                "Please run tracker.track() or provide valid track df."
            )
            tracks = self.tracks

        masks = self.tracked_masks(tracks, label_old=label_old, label_new=label_new)

        if format.lower() == "ctc":
            return export_tracking_results_ctc(tracks, masks, res_dir, fill_gaps=True)
        elif format.lower() == "flat":
            return export_tracking_results_flat(tracks, masks, res_dir)
        else:
            raise ValueError(
                f"{format.lower()=} is not supported. Please use 'CTC' or 'flat'."
            )


class LAPTracker(BaseTracker):
    """LAP-based cell tracker.

    Walks frame by frame. For each consecutive pair, correspondence candidates are
    resolved by maximum-weight bipartite matching using `p1` as the edge weight. Sources
    without a correspondence are then checked for divisions: Candidates are sorted by `p2`
    in descending order and assigned greedily, capped at `max_division_targets` daughters
    per source. If `relabel_single_daughter_divs` is set, a source with only one valid
    division candidate is relabeled as a correspondence. Sources still unmatched after
    these steps are carried over and matched against later frames via candidate edges that
    span multiple time steps, allowing tracks to bridge gaps.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        predictions: pl.DataFrame,
        segmentation_correction: BaseCorrection | list[BaseCorrection] | None = None,
        max_division_targets: int = 2,
        norm_fn: Literal["sigmoid", "softmax", "none"] = "sigmoid",
        thr_corr: float | None = 0.5,
        thr_div: float | None = 0.5,
        ignore_negative_edges: bool = True,
        relabel_single_daughter_divs: bool = True,
        edge_direction: Literal["both", "future"] = "both",
    ) -> None:
        """Initialize tracker.

        Args:
            dataset: See `BaseTracker`.
            predictions: See `BaseTracker`.
            segmentation_correction: See `BaseTracker`.
            max_division_targets: See `BaseTracker`.
            norm_fn: See `BaseTracker`.
            thr_corr: See `BaseTracker`.
            thr_div: See `BaseTracker`.
            ignore_negative_edges: See `BaseTracker`.
            relabel_single_daughter_divs: If `True`, division candidates with exactly
                one remaining target (after real divisions and correspondences are
                matched) are assigned as correspondences. Required for division
                detection with binary predictions.
            edge_direction: See `BaseTracker`.
        """
        super().__init__(
            dataset=dataset,
            predictions=predictions,
            segmentation_correction=segmentation_correction,
            max_division_targets=max_division_targets,
            norm_fn=norm_fn,
            thr_corr=thr_corr,
            thr_div=thr_div,
            ignore_negative_edges=ignore_negative_edges,
            edge_direction=edge_direction,
        )

        self.relabel_single_daughter_divs = relabel_single_daughter_divs
        self.unmatched, self.matched_in, self.matched_out = set(), set(), set()

    def _get_matching_candidates_frame(
        self, candidates: pl.DataFrame, t: int, t_next: int
    ):
        """Get candidates for matching.

        Candidates are all nodes on frame `t` above that are predicted as active, and
        previously unmatched nodes.

        Args:
            candidates: Candidate edges starting at frame `ti <= t` and pointing to `t+1`.
            t: Source frame for matching. All "new" cells are here, previously unmatched
               cells are at `t-n`.
            t_next: Target frame for matching.

        Returns:
            Candidates for matching and their predicted weight.

        """
        candidates_frame = candidates.filter(t=t, t_dst=t_next)
        if self.unmatched:
            candidates_frame = candidates_frame.vstack(
                candidates.filter(pl.col("src").is_in(self.unmatched), t_dst=t_next)
            )
        candidates_frame = candidates_frame.filter(
            ~pl.col("src").is_in(self.matched_out),
            ~pl.col("dst").is_in(self.matched_in),
        )

        return (
            candidates_frame.filter(
                ~pl.col("dst").is_in(pl.col("src").implode()),
            )
            .unique(["src", "dst"])  # required due to vstack
            .sort("src", "dst")
        )

    @staticmethod
    def _build_pygraph(candidates: pl.DataFrame) -> tuple[rx.PyGraph, set[int]]:
        """Construct rx graph."""
        G = rx.PyGraph(multigraph=False)
        sources = {src: G.add_node(src) for src in np.unique(candidates["src"])}
        targets = {dst: G.add_node(dst) for dst in np.unique(candidates["dst"])}
        sources_mapped = set(
            sources.values()
        )  # keep track of source nodes, since rx does not preserve order

        G.add_edges_from(
            [(sources[u], targets[v], w) for u, v, w in candidates.iter_rows()]
        )
        assert rx.is_bipartite(G)

        return G, sources_mapped

    @staticmethod
    def _resolve_undirected_edges(
        matching: set[tuple[int, int]], graph: rx.PyGraph, source_nodes: set[int]
    ) -> tuple[list[int], list[int]]:
        """Order matches by time.

        Matchings by rx are not sorted by their direction. Hence, predictions might be
        present for `src -> dst` and `dst -> src`. For track reconstruction, we sort them.

        Args:
            matching: Unordered `src -> dst` correspondence matching.
            graph: Local trajectory graph.
            source_nodes: Global set of source nodes.

        Returns:
            Ordered matching.
        """
        src, dst = [], []
        for u, v in sorted(matching):
            if u in source_nodes:
                src.append(graph[u])
                dst.append(graph[v])
            else:
                src.append(graph[v])
                dst.append(graph[u])

        return src, dst

    def _update_matched_unmatched(
        self,
        src_available: list[int],
        src_matched: list[int],
        dst_matched: list[int],
    ) -> None:
        """Keep track of nodes without outgoing edges."""
        self.matched_in.update(dst_matched)
        self.matched_out.update(src_matched)

        self.unmatched.update(src_available)
        self.unmatched.difference_update(self.matched_out)

    def _solve_lap_frame(
        self, candidates: pl.DataFrame, frame: int, frame_next: int
    ) -> pl.DataFrame:
        """Solve assignment between candidates on two frames.

        Solves correspondence assignment between `t`, `t-n` (unmatched), and `t+1`.

        Returns:
            3-col dataframe with `src`, `dst`, and `type==traj`.
        """
        candidates_frame = self._get_matching_candidates_frame(
            candidates, frame, frame_next
        ).select("src", "dst", "p1")
        G, source_nodes = self._build_pygraph(candidates_frame)
        matching = rx.max_weight_matching(
            G,
            max_cardinality=False,
            weight_fn=lambda x: int(x * 1e9) if not np.isnan(x) else 0,
        )
        src_matched, dst_matched = self._resolve_undirected_edges(
            matching, G, source_nodes
        )
        src_available = self.dataset.node_feats.filter(t=frame)["index"].to_list()
        self._update_matched_unmatched(src_available, src_matched, dst_matched)

        return (
            pl.DataFrame(
                np.stack([src_matched, dst_matched], 1),
                schema=dict.fromkeys(["src", "dst"], pl.UInt32),
            )
            .with_columns(type=pl.lit("traj"))
            .unique(maintain_order=True)
        )

    def _match_divisions(self, candidates: pl.DataFrame, frame: int, frame_next: int):
        """Greedily match divisions for unmatched nodes."""
        candidates_frame = (
            self._get_matching_candidates_frame(candidates, frame, frame_next)
            .with_columns(n_targets=pl.len().over("src"))
            .filter(
                pl.col("n_targets") > 1,
            )
            # have to sort by index, otherwise identical p2 will cause non-deterministic
            .sort("p2", "src", "dst", descending=[True, False, False])
            .select("src", "dst", "p2")
        )

        divisions = []
        out_degree = dict.fromkeys(np.unique(candidates_frame["src"]), 0)
        in_degree = dict.fromkeys(np.unique(candidates_frame["dst"]), 0)
        for u, v, _w in candidates_frame.iter_rows():
            if (out_degree[u] >= 2) or (in_degree[v] >= 1):
                continue

            out_degree[u] += 1
            in_degree[v] += 1
            divisions.append({"src": u, "dst": v, "type": "div"})

        divisions = pl.DataFrame(divisions)
        if divisions.height > 0:
            divisions = divisions.cast(dict.fromkeys(["src", "dst"], pl.UInt32))
            src_matched = divisions["src"].to_list()
            dst_matched = divisions["dst"].to_list()
        else:
            src_matched, dst_matched = [], []

        src_available = np.unique(candidates_frame["src"]).tolist()
        self._update_matched_unmatched(src_available, src_matched, dst_matched)

        return divisions

    def _match_single_daughter_divs(
        self, candidates: pl.DataFrame, frame: int, frame_next: int
    ) -> pl.DataFrame:
        """Assign single-daughter division candidates as correspondences.

        Called after `_match_divisions`. Any remaining unmatched source with exactly one
        candidate target is assigned as a trajectory edge. Respects existing
        `matched_in`/`matched_out` state, so real divisions and correspondences always
        have precedence. Edges are assigned in descending order of `p2`.
        """
        _empty = pl.DataFrame(
            schema={"src": pl.UInt32, "dst": pl.UInt32, "type": pl.Utf8}
        )

        candidates_frame = (
            self._get_matching_candidates_frame(candidates, frame, frame_next)
            .with_columns(n_targets=pl.len().over("src"))
            .filter(pl.col("n_targets") == 1)
            .sort("p2", "src", "dst", descending=[True, False, False])
        )

        if candidates_frame.is_empty():
            return _empty

        corr = []
        assigned_dst: set[int] = set()
        for u, v in candidates_frame.select("src", "dst").iter_rows():
            if v in assigned_dst:
                continue
            corr.append({"src": u, "dst": v, "type": "traj"})
            assigned_dst.add(v)

        if not corr:
            return _empty

        result = pl.DataFrame(corr).cast(dict.fromkeys(["src", "dst"], pl.UInt32))
        self._update_matched_unmatched(
            np.unique(candidates_frame["src"]).tolist(),
            result["src"].to_list(),
            result["dst"].to_list(),
        )
        return result

    def track(self):
        """Run tracking.

        Predictions are normalized. LAP is solved for correspondences, unmatched nodes
        are considered for division, and tracks are reconstructed.

        Returns:
            Trajectories determined by `index`, position, label/parent on image `label`
            and label/parent in tracks `label_track`.
        """
        preds_norm = self.normalize_edges(self.norm_fn)
        candidates_corr, candidates_div = self._find_corr_and_div_candidates(preds_norm)

        matches = []

        timepoints = np.unique(self.dataset.node_feats["t"])
        pbar = tqdm(
            zip(timepoints[:-1], timepoints[1:], strict=True), total=len(timepoints) - 1
        )
        for t, t_next in track_iter(
            pbar,
            self.progress,
            stage="tracking",
            total=len(timepoints) - 1,
            message="Linking trajectories",
        ):
            matched_corr = self._solve_lap_frame(candidates_corr, t, t_next)
            if matched_corr.height > 0:
                matches.append(matched_corr)

            matched_div = self._match_divisions(candidates_div, t, t_next)
            if matched_div.height > 0:
                matches.append(matched_div)

            if self.relabel_single_daughter_divs:
                matched_single = self._match_single_daughter_divs(
                    candidates_div, t, t_next
                )
                if matched_single.height > 0:
                    matches.append(matched_single)

        if len(matches) > 0:
            matches = pl.concat(matches).sort("src", "dst")
        else:
            matches = pl.DataFrame(
                None, schema={"src": pl.UInt32, "dst": pl.UInt32, "type": pl.Utf8}
            )

        self.tracks = self.find_trajectories(matches)
        self.apply_corrections()
        return self.tracks
