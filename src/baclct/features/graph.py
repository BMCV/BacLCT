"""Graph construction, pruning, and extraction of relational features."""

from __future__ import annotations

import itertools
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias

import dask.array as da
import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from omegaconf import ListConfig
from scipy.spatial import cKDTree  # type: ignore
from skimage.measure import regionprops
from skimage.metrics import contingency_table
from skimage.segmentation import find_boundaries
from sklearn.neighbors import KDTree
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from baclct.features.custom_edge_features import CUSTOM_EDGE_PROPS
from baclct.io import dataset_identity_matches
from baclct.utils import edge_cache
from baclct.utils.data import collect
from baclct.utils.logger import get_pylogger
from baclct.utils.progress import ProgressCallback, track_iter

logger = get_pylogger(__name__)

# `dilated_overlap` dilates both cells by this multiple of their per-cell radius column.
# It is part of the edge store's name, so lowering or raising it invalidates the cache.
DILATED_OVERLAP_RADIUS_MULTIPLIER = 2.5

EdgePruneArg: TypeAlias = Literal["dilated_overlap", "overlap", "ellipse", "radius", "gt"]
EdgeFeatName: TypeAlias = Literal[
    "dist_temp", "dist_spat", "overlap", "iou", "cosine_similarity", "relative_size"
]

# edge features that negate when the edge direction is reversed (src <-> dst)
ANTISYMMETRIC_EDGE_FEATS = (
    "dist_temp",
    "relative_size",
    "intensity_mean_diff",
)


class NoEdgesError(ValueError):
    """No edges could be built for the requested strides, e.g. dt exceeds the sequence."""


def build_lineage_gt_edges(node_feats: pl.DataFrame) -> pl.DataFrame:
    """Create GT lineage graph.

    Args:
        node_feats: Position (tyx) and handcrafted single-cell features, including the
            `index`, `t`, `label`, and `parent` lineage columns.
    """
    nf = node_feats.select("index", "t", "label", "parent").filter(
        pl.col("label").is_not_null()
    )

    correspondence = (
        nf.sort("label", "t")
        .with_columns(dst=pl.col("index").shift(-1).over("label"))
        .filter(pl.col("dst").is_not_null())
        .select(src="index", dst="dst", y=pl.lit(1, dtype=pl.Int64))
    )

    parent_last = (
        nf.sort("t").group_by("label").last().select("label", parent_idx="index")
    )
    daughter_first = (
        nf.sort("t")
        .group_by("label")
        .first()
        .select("label", "parent", daughter_idx="index")
        .filter(pl.col("parent").is_not_null() & (pl.col("parent") != 0))
        .filter(pl.col("parent") != pl.col("label"))
    )
    division = daughter_first.join(
        parent_last, left_on="parent", right_on="label", how="inner"
    ).select(src="parent_idx", dst="daughter_idx", y=pl.lit(2, dtype=pl.Int64))

    return pl.concat([correspondence, division]).sort("src", "dst")


class EdgeFinder:
    """Construct the spatio-temporal graph linking cells across frames.

    For each source frame and each requested temporal stride, candidate edges to nearby
    cells in target frames are found by a KDTree query within `search_radius` around the
    chosen center point (`centroid` or medial-axis `center`). Candidates may then be
    pruned by elongated mask overlap (`dilated_overlap`, with a dilation radius scaled by
    each cell's diameter to account for object movement), by plain mask overlap or IoU
    threshold, by an ellipse or circle scaled to each cell's axes, or restricted to
    ground-truth pairs.

    Surviving edges are labeled (`inactive`, `correspondence`, or `division`, with
    division relabelable via `treat_divs_as`) and annotated with the features listed in
    `feature_names` (e.g., temporal distance, Euclidean center distance, mask overlap,
    IoU) plus any registered custom edge features. Features that depend on deep node
    embeddings (e.g., `cosine_similarity`) are deferred to `GraphDataset`.
    """

    def __init__(
        self,
        feature_names: list[EdgeFeatName] | tuple[EdgeFeatName, ...] = (
            "dist_temp",
            "dist_spat",
        ),
        treat_divs_as: Literal["correspondence", "division", "negative"] = "division",
        prune_edges_by: EdgePruneArg
        | tuple[EdgePruneArg, int | float | str]
        | None = None,
        bidirectional=True,
        edge_normalization: str | float = "cell_size",
        extra_features: list[str] | None = None,
        center_name: Literal["centroid", "center"] = "centroid",
        n_jobs: int = -1,
    ):
        """Initialize edge finder.

        Args:
            feature_names: Name of edge features for model.
            treat_divs_as: Edge class assigned to division edges. Mapped to 0 - inactive,
                1 - correspondence, and 2 - division.
            prune_edges_by: Determines if/how edges are pruned, e.g., by removing non-GT
                edges or restricting edges to a specific region. Passed as str of pruning
                method (e.g., `GT` or `overlap`), tuple with a pruning method and
                parameter (e.g., `(radius, 5)`). If `None`, will return all edges.
            bidirectional: If `True`, returned graph contains two edges per pair of nodes,
                i.e. `src -> dst` and `dst -> src`.
            edge_normalization: Name of edge normalization function or a scaling factor
                for positional features (e.g., to scale distance between cells based on
                magnification).
            extra_features: Names of additional edge features as defined by
                `custom_edge_features.py`.
            center_name: Type of center point to use, `centroid` (mean of coords) or
                `center` (midpoint on medial axis).
            n_jobs: Number of parallel threads (joblib) used during candidate edge
                construction and pruning across frame pairs.
        """
        self.feature_names = feature_names
        self.edge_normalization = edge_normalization
        self.extra_features = extra_features
        self.center_name = center_name
        self.n_jobs = n_jobs

        self.extra_features_fns = []
        if self.extra_features:
            for feat_name in self.extra_features:
                if feat_name in CUSTOM_EDGE_PROPS:
                    self.extra_features_fns.append(CUSTOM_EDGE_PROPS[feat_name])
                else:
                    logger.warning(f"Custom edge feature '{feat_name}' not found.")

        # get rid of features that can't be computed here, e.g., cosine similarity
        self.feature_cols = [
            f
            for f in feature_names
            if f
            in ["dist_temp", "dist_spat", "overlap", "iou"]
            + list(CUSTOM_EDGE_PROPS.keys())
        ]
        self.should_compute_overlap = "overlap" in feature_names
        self.should_compute_iou = "iou" in feature_names
        self.should_compute_dilated_overlap = (
            # tuple or str, so indexing should be save
            "dilated_overlap" in [prune_edges_by, prune_edges_by[0]]
            if prune_edges_by is not None
            else False
        )

        div_labels = {"correspondence": 1, "division": 2, "negative": 0}
        self.treat_divs_as = treat_divs_as
        self.div_label = div_labels[treat_divs_as]

        self.prune_edges = prune_edges_by is not None
        self.prune_method: EdgePruneArg | None = None
        self.prune_param: int | float | str | None = None

        if self.prune_edges:
            if isinstance(
                prune_edges_by, (tuple, list, ListConfig)
            ):  # listconfig required for hydra, "_convert_: all" works too
                self.prune_method, self.prune_param = prune_edges_by
            else:
                self.prune_method = prune_edges_by
                if self.prune_method == "overlap":
                    self.prune_param = 0.0

        # bidirectional is only used with post-processing, for saving, it is just dropped
        # that way, the lit datasets can work independently on directionality
        self.bidirectional = bidirectional

        # optional progress sink, attached by callers (see `baclct.utils.progress`)
        self.progress: ProgressCallback | None = None

    def __getstate__(self) -> dict:
        """Drop the progress sink when pickling (e.g. to joblib workers).

        The edge finder is pickled to worker processes because the per-frame tasks
        reference bound methods. A sink is typically a closure over a GUI (unpicklable),
        and only the parent process consumes progress.
        """
        state = self.__dict__.copy()
        state["progress"] = None
        return state

    def __repr__(self) -> str:
        """Print summary."""
        if self.prune_edges:
            prune_info = (
                f"prune_method={self.prune_method}, prune_param={self.prune_param!r}"
            )
        else:
            prune_info = "prune_edges=False"

        info = [
            prune_info,
            f"features={self.feature_names}",
            f"extra_features={self.extra_features}",
            f"edge_normalization={self.edge_normalization}",
            f"div_label={self.div_label} (treat_as='{self.treat_divs_as}')",
            f"bidirectional={self.bidirectional}",
        ]

        info_str = ",\n    ".join(info)
        return f"EdgeFinder(\n    {info_str}\n  )"

    @property
    def _prune_str(self) -> str:
        """Pruning identifier used in cache file and directory names."""
        if not self.prune_edges or self.prune_method is None:
            return "none"
        if self.prune_method == "dilated_overlap":
            return (
                f"{self.prune_method}-{self.prune_param}"
                f"-x{DILATED_OVERLAP_RADIUS_MULTIPLIER:g}"
            )
        if self.prune_param is not None:
            return f"{self.prune_method}-{self.prune_param}"
        return str(self.prune_method)

    def cache_dirname(self) -> str:
        """Return the partitioned edge store directory name.

        There is one store per prune configuration. It holds one `dt={k}/part.parquet`
        partition per frame stride, so a single directory covers any requested `time_step`
        and is extended with new strides without recomputing existing ones. The maximum
        radius the store was built at lives in `meta.json`. A smaller `dist_spat` filters
        instead of recomputing.
        """
        return f"edges_prune-{self._prune_str}"

    @property
    def _required_cols(self) -> list[str]:
        """Required column names for pruning method and features."""
        cols = [
            "t",
            "label",
            "parent",
            "area",
            "axis_major_length",
            f"{self.center_name}-0",
            f"{self.center_name}-1",
        ]
        if self.prune_method == "ellipse":
            cols += [
                "axis_minor_length",
                "orientation",
            ]
        elif self.prune_method == "radius":
            cols += ["axis_major_length"]

        if self.should_compute_iou:
            cols += ["area"]

        if self.edge_normalization == "cell_size":
            cols += ["len_init"]

        if self.extra_features and "intensity_mean_diff" in self.extra_features:
            cols += ["intensity_mean"]

        return sorted(set(cols))

    @property
    def prune_param_parsed(self) -> pl.Expr | int | float:
        """Parsed prune parameter or placeholder."""
        if self.prune_param is None:
            return 1.0
        elif isinstance(self.prune_param, str):
            return pl.col(self.prune_param)
        elif isinstance(self.prune_param, int | float):
            return self.prune_param
        else:
            raise ValueError(
                f"Unsupported {self.prune_param=} of type {type(self.prune_param)}. "
                "Supports: str | int | float."
            )

    def _prune_edges_ellipse(
        self, edge_data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Prune edges using scaled ellipse created by minor and major axes."""
        factor = self.prune_param_parsed

        # scaled semi-axes
        a = (pl.col("axis_major_length_src") / 2.0) * factor
        b = (pl.col("axis_minor_length_src") / 2.0) * factor

        # translate centers, so that source is origin
        dx = pl.col(f"{self.center_name}-1_dst") - pl.col(f"{self.center_name}-1_src")
        dy = pl.col(f"{self.center_name}-0_dst") - pl.col(f"{self.center_name}-0_src")

        # get orientation angles of src ellipse, orientation in y/x format
        cos_theta = pl.col("orientation_src").sin()
        sin_theta = pl.col("orientation_src").cos()

        # rotate reference point, i.e. "un-rotate" ellipse
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta

        # check if transformed center of dst satiesfies standard
        # equation of src ellipse: https://en.wikipedia.org/wiki/Ellipse#Standard_equation
        is_inside = (x_rot / a).pow(2) + (y_rot / b).pow(2) <= 1.0

        return edge_data.filter(is_inside)

    def _prune_edges_radius(
        self, edge_data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Prune edges based on scaled circle using major axis."""
        factor = self.prune_param_parsed

        r = (pl.col("axis_major_length_src") / 2.0) * factor
        return edge_data.filter(pl.col("dist_spat") <= r)

    def _prune_edges(
        self,
        edge_data: pl.DataFrame | pl.LazyFrame,
        node_feats: pl.DataFrame | None = None,
        masks: np.ndarray | da.Array | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Run edge pruning."""
        if not self.prune_edges:
            return edge_data

        if self.prune_method == "overlap":
            return edge_data.filter(pl.col("overlap") > self.prune_param_parsed)

        elif self.prune_method == "ellipse":
            return self._prune_edges_ellipse(edge_data)

        elif self.prune_method == "radius":
            return self._prune_edges_radius(edge_data)

        elif self.prune_method == "gt":
            return edge_data.filter(
                pl.col("y") > 0,
                # keeping closest edges is done using by filter in post or hparams
                # pl.col("dist_temp") == pl.col("dist_temp").abs().min(),
                pl.col("dist_temp") > 0,
            )
        elif self.prune_method == "dilated_overlap":
            assert node_feats is not None and masks is not None
            return self._prune_edges_dilated_overlap(
                edge_data,
                node_feats=node_feats,
                masks=masks,
                sampling_rate=3,
                radius_multiplier=DILATED_OVERLAP_RADIUS_MULTIPLIER,
            )
        else:
            raise ValueError(f"{self.prune_method} is not a valid method for pruing.")

    def _label_edges(
        self, edge_data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Label edges based on correspondence and division."""
        return edge_data.with_columns(
            y=pl.when(pl.col("label_src") == pl.col("label_dst"))
            .then(pl.lit(1))
            .otherwise(
                pl.when(
                    # both directions
                    (pl.col("parent_dst") == pl.col("label_src"))
                    | (pl.col("parent_src") == pl.col("label_dst"))
                )
                .then(pl.lit(2))
                .otherwise(pl.lit(0))
            )
        )

    def _remap_div_label(
        self, edge_data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Remap divisions to configured div_label.

        Cache stores divisions as 2, relabeling done based on treat_divs_as (e.g. 1 for
        correspondence, 0 for negative).
        """
        if self.div_label == 2:
            return edge_data
        return edge_data.with_columns(
            y=pl.when(pl.col("y") == 2)
            .then(pl.lit(self.div_label))
            .otherwise(pl.col("y"))
        )

    def _get_index_and_pos(
        self,
        node_feats: pl.DataFrame,
        t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get index and position for all cells on frame."""
        feats_t = node_feats.filter(t=t)

        idx = feats_t["index"].to_numpy()
        c = feats_t.select(rf"^{self.center_name}-\d$").to_numpy()

        return idx, c

    def _validate(
        self,
        loaded_data: pl.LazyFrame,
        node_feats: pl.DataFrame,
        search_radius: int,
        time_steps: list[int],
    ) -> pl.LazyFrame | None:
        """Validate loaded edge data.

        Validates that:
            - All requested features are present.
            - No spurious nodes in the graph (matching segmentation).

        Backward edges are not checked here: They are reconstructed on load by mirroring
        the positive-dt partitions, so they always exist for a bidirectional finder.
        """
        # columns for lazy frame are retrieved that way instead of df.columns
        # pl raises warning that df.columns might be expensive and proposes alternative
        columns = loaded_data.collect_schema().names()
        if not all(c in columns for c in self.feature_cols):
            logger.warning(
                "Not all features are present in data. "
                f"Required: {self.feature_cols}, Available: {columns}"
            )
            return

        # can't materialize full graph, so we just pull unique src and dst nodes
        src = collect(loaded_data.select("src").unique())["src"].to_numpy()
        dst = collect(loaded_data.select("dst").unique())["dst"].to_numpy()
        nodes_graph = np.unique(np.concatenate([src, dst]))
        nodes = np.unique(node_feats["index"])
        if (nodes_spurious := np.setdiff1d(nodes_graph, nodes).size) > 0:
            logger.warning(f"Found spurious nodes in graph: {nodes_spurious}.")
            return

        if (nodes_missing := np.setdiff1d(nodes, nodes_graph).size) > 0:
            logger.warning(
                "Some nodes are missing from graph and might be isolated: "
                f"{nodes_missing}."
            )

        return loaded_data

    def _store_meta(
        self,
        search_radius: int,
        positive_only: bool,
        strides: list[int] | None = None,
        dataset_meta: dict | None = None,
    ) -> dict:
        """Store metadata used to check compatibility on later loads."""
        meta = {
            "max_radius": int(search_radius),
            "feature_cols": list(self.feature_cols),
            "prune": self._prune_str,
            "positive_only": bool(positive_only),
            "bidirectional": bool(self.bidirectional),
        }
        if strides is not None:
            # record every computed stride, including empty ones (e.g., dt > seq length)
            # to prevent recomputing impossible framedists
            meta["strides"] = sorted(int(s) for s in strides)  # type: ignore
        if dataset_meta is not None:
            meta["dataset"] = dataset_meta  # type: ignore
        return meta

    def _store_compatible(
        self, meta: dict, search_radius: int, dataset_meta: dict | None = None
    ) -> bool:
        """Whether an existing store satisfies the requested features, radius, data."""
        compatible = (
            set(self.feature_cols) <= set(meta.get("feature_cols", []))
            and meta.get("max_radius", -1) >= search_radius
            and meta.get("bidirectional") == self.bidirectional
        )
        if dataset_meta is not None:
            compatible = compatible and dataset_identity_matches(
                dataset_meta, meta.get("dataset")
            )
        return compatible

    def _persist_store(
        self,
        edge_data: pl.DataFrame | pl.LazyFrame,
        cache_dir: Path,
        search_radius: int,
        positive_only: bool,
        strides: list[int],
        recorded_strides: list[int] | None = None,
        dataset_meta: dict | None = None,
    ):
        """Write the computed `strides` as partitions plus store metadata.

        Boundary mirroring emits a few edges with a smaller |dt| than the requested
        stride. Restricting to `strides` keeps a partition from being overwritten by a
        fragment of another stride (the mirror on load rebuilds those from their own
        forward partition). With `positive_only`, backward edges are dropped and mirrored
        on load instead.

        `recorded_strides` is the full set of computed strides written to metadata. It
        defaults to `strides` and differs only when extending an existing store, where it
        must also include the strides already present so they are not lost.
        """
        edge_data = edge_data.filter(pl.col("dist_temp").abs().is_in(strides))
        if positive_only and self.bidirectional:
            edge_data = edge_data.filter(pl.col("dist_temp") > 0)
        edge_cache.write_partitions(edge_data, cache_dir)
        edge_cache.write_meta(
            cache_dir,
            self._store_meta(
                search_radius, positive_only, recorded_strides or strides, dataset_meta
            ),
        )

    def mirror_edges(
        self, edge_data: pl.LazyFrame | pl.DataFrame
    ) -> pl.LazyFrame | pl.DataFrame:
        """Reconstruct backward edges (negative dt) from positive-dt partitions.

        Swaps endpoints, advances the source frame to the old destination, negates
        `dist_temp` and anti-symmetric features, and recomputes overlap against the new
        source area. Symmetric features are copied.
        """
        schema = edge_data.collect_schema()
        cols = schema.names()
        exprs = [
            pl.col("dst").alias("src"),
            pl.col("src").alias("dst"),
            # adding dist_temp (Int64) upcasts t, cast back to keep the stored schema
            (pl.col("t") + pl.col("dist_temp")).cast(schema["t"]).alias("t"),
            (-pl.col("dist_temp")).alias("dist_temp"),
        ]
        for c in cols:  # swap paired node columns, e.g. area_src <-> area_dst
            if c.endswith("_src") and c != "src" and (c[:-4] + "_dst") in cols:
                base = c[:-4]
                exprs += [
                    pl.col(f"{base}_dst").alias(f"{base}_src"),
                    pl.col(f"{base}_src").alias(f"{base}_dst"),
                ]
        exprs += [
            (-pl.col(c)).alias(c)
            for c in ANTISYMMETRIC_EDGE_FEATS
            if c != "dist_temp" and c in cols
        ]

        mirrored = edge_data.with_columns(exprs)
        if {"overlap", "intersection", "area_src"} <= set(cols):
            # null intersection (no overlap) fills to 0 like the forward partition
            mirrored = mirrored.with_columns(
                (pl.col("intersection") / pl.col("area_src"))
                .fill_null(0)
                .alias("overlap")
            )
        return mirrored

    def _load_store(
        self,
        cache_dir: Path,
        requested_dt: list[int],
        positive_only: bool,
        mirror: bool = True,
    ) -> pl.LazyFrame | None:
        """Load the requested strides, mirroring backward edges when stored positive.

        With `mirror=False`, the forward-only (positive-dt) scan is returned so callers
        can mirror just the rows they filter to. Mirroring on load redefines `t` and
        `dist_temp`, which blocks predicate pushdown of later per-frame filters on those
        columns and forces a full scan of the backward edges on every collect.
        """
        edge_data = edge_cache.load_partitions(cache_dir, requested_dt)
        if edge_data is not None and mirror and positive_only and self.bidirectional:
            edge_data = pl.concat([edge_data, self.mirror_edges(edge_data)])
        return edge_data

    def _compute(
        self,
        node_feats: pl.DataFrame,
        search_radius: int,
        time_steps: list[int],
        masks: da.Array | np.ndarray | None = None,
        connectivity: Literal["dense", "sequential", "star"] = "star",
    ) -> pl.LazyFrame:
        """Compute edges for all frames."""
        num_frames = int(node_feats["t"].to_numpy().max()) + 1
        logger.debug(
            f"Finding edges with radius={search_radius} and steps={time_steps} "
            f"({node_feats.height} nodes, {num_frames} frames.)"
        )

        num_edges = 0
        edge_data = []
        for t in track_iter(
            tqdm(
                range(num_frames),
                desc=f"Finding edges (r={search_radius}, steps={time_steps})",
            ),
            self.progress,
            stage="edges",
            total=num_frames,
            message="Building graph",
        ):
            # usually, we use star connectivity for precomputing edges
            # however, if we want to use sequential connectivity, we only need to
            # compute edges for the first step
            final_time_steps = (
                time_steps if connectivity != "sequential" else time_steps[:1]
            )
            # when used in loop, the final result for star and dense is identical
            # after deduplication, but star is much more efficient
            conn = "sequential" if connectivity == "sequential" else "star"

            edges = self.get_edges_for_frame(
                node_feats,
                t,
                search_radius=search_radius,
                time_steps=final_time_steps,
                masks=masks,
                connectivity=conn,
                mirror_at_last_frames=True,
            )
            if edges is not None:
                num_edges += edges.height
                edge_data.append(edges.lazy())

        if len(edge_data) == 0:
            raise NoEdgesError("Could not create graph. All nodes are isolated.")

        graph = pl.concat(edge_data).unique(subset=["src", "dst"], maintain_order=True)
        logger.info(
            f"Created graph with {num_edges} edges for {node_feats.height} nodes "
            f"({num_frames} frames)."
        )
        return graph

    def find_edge_pairs(
        self, node_feats: pl.DataFrame, t_src: int, t_dst: int, radius: int
    ):
        """Find edges between two frames and compute spatial and temporal distance."""
        idx_src, c_src = self._get_index_and_pos(node_feats, t_src)
        idx_dst, c_dst = self._get_index_and_pos(node_feats, t_dst)
        if min(len(c_src), len(c_dst)) == 0:
            return

        tree = KDTree(c_dst)
        nbrs, dists = tree.query_radius(c_src, radius, return_distance=True)

        lens = [len(n) for n in nbrs]
        idx_src_tree = np.repeat(np.arange(len(nbrs)), lens)
        idx_dst_tree = np.concat(nbrs)
        if len(idx_dst_tree) == 0:
            return

        return pl.DataFrame(
            {
                "src": idx_src[idx_src_tree],
                "dst": idx_dst[idx_dst_tree],
                "dist_spat": np.concat(dists),
                "dist_temp": np.full(idx_src_tree.shape, t_dst - t_src),
            },
            schema={
                "src": pl.UInt32,
                "dst": pl.UInt32,
                # spatial distance is a continuous euclidean distance, not a frame
                # index, so keep the fractional part (unlike the integer dist_temp)
                "dist_spat": pl.Float64,
                "dist_temp": pl.Int64,
            },
        )

    def _transform(
        self,
        edge_data: pl.DataFrame,
        node_feats: pl.DataFrame | None = None,
        spacing_t: float = 1.0,
    ) -> torch.Tensor | None:
        """Normalize edge features and convert to tensor.

        Args:
            edge_data: Edge features containing indices of connected nodes (`src`, `dst`)
                and their relational features.
            node_feats: Position (tyx) and handcrafted single-cell features, used to
                derive the cell-size scaling factor for `dist_spat`.
            spacing_t: Physical duration of one frame. The cached `dist_temp` is a frame
                stride, so it is multiplied by `spacing_t` before the symlog transform.
                A sparsely sampled inference sequence (every k-th frame) recovers the
                training temporal distribution by passing `spacing_t=k`.
        """
        if self.feature_cols is None:
            return

        # NOTE: using starts_with to expand selection (e.g., suffix -\d) does not
        #       maintain order leading to incorrect normalization! either define
        #       feature cols explicitly or do expansion within __init__.
        x_df = edge_data.select(self.feature_cols)
        x_np = x_df.to_numpy(writable=True).astype(np.float32)

        # TODO: Refactor, so that time normalization stays consistent
        if self.edge_normalization == "min_max":
            return torch.as_tensor(minmax_scale(x_np), dtype=torch.float32)

        if self.edge_normalization == "none":
            return torch.as_tensor(x_np, dtype=torch.float32)

        # separate dist_spat for special normalization
        if "dist_spat" in self.feature_cols:
            dist_spat_idx = self.feature_cols.index("dist_spat")
            scaling_factor = None

            if self.edge_normalization == "cell_size":
                if node_feats is not None and "len_init" in node_feats.columns:
                    scaling_factor = node_feats.select(pl.col("len_init").first()).item()
            elif isinstance(self.edge_normalization, (float, int)):
                scaling_factor = self.edge_normalization

            if scaling_factor and scaling_factor > 0:
                dist_spat = x_np[:, dist_spat_idx] + 1e-6
                if np.any(dist_spat <= 0):
                    raise ValueError(
                        # did only occur when indices were not correctly assigned
                        f"Found negative spatial distances: {x_np[dist_spat <= 0]}"
                    )
                x_np[:, dist_spat_idx] = np.log(dist_spat / scaling_factor)

                for i, c in enumerate(self.feature_cols):
                    # don't need to normalize overlap, iou, cosine similarity, or
                    # custom edge features (they manage their own scale)
                    if (c == "dist_spat") or (
                        c in ["overlap", "iou", "cosine_similarity"]
                        or c in CUSTOM_EDGE_PROPS
                    ):
                        continue
                    # time is also normalized using log multiplied by sign for symmetry.
                    # spacing_t turns the integer frame stride into a physical duration so
                    # train and inference see the same dist_temp distribution
                    elif c == "dist_temp":
                        t = x_np[:, i] * spacing_t
                        x_np[:, i] = np.sign(t) * np.log(np.abs(t) + 1)
                    else:
                        raise NotImplementedError(
                            f"Could not normalize edge feature {c}."
                            "Please provide an implementation."
                        )

                return torch.as_tensor(x_np, dtype=torch.float32)

        raise ValueError(f"Unknown {self.edge_normalization=}.")

    def to_bidirectional(
        self, edge_data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Create directed graph from undirected edges.

        Mirrors (src, dst), copies symmetric features (such as distances and IoU),
        transforms or recomputes remaining features (dist_temp, overlap).
        """
        forward_graph = edge_data
        backward_graph = edge_data.with_columns(
            pl.col("^.*src$").name.replace("src", "dst"),
            pl.col("^.*dst$").name.replace("dst", "src"),
            pl.col("dist_temp") * -1,
        )
        if self.should_compute_overlap:
            backward_graph = backward_graph.with_columns(
                overlap=pl.col("intersection") / pl.col("area_src")
            )

        return pl.concat([forward_graph, backward_graph])

    @staticmethod
    def resolve_boundary_pairs(
        pairs: Iterable[tuple[int, int]], t_max: int, bidirectional: bool
    ) -> list[tuple[int, int]]:
        """Clip out-of-bounds frame pairs to in-bounds ones while preserving the stride.

        A forward edge whose destination runs past the last frame (`t -> t+n`) is
        replaced by the same-stride edge ending at the source (`t-n -> t`), so the
        boundary node keeps a neighbor at the configured stride. The start of the sequence
        is handled symmetrically. Pairs still out of bounds after substitution (sequence
        shorter than the stride) are dropped, as are any out-of-bounds pairs when the
        graph is not bidirectional.
        """
        resolved: list[tuple[int, int]] = []
        for t_src, t_dst in pairs:
            lo, hi = (t_src, t_dst) if t_src <= t_dst else (t_dst, t_src)
            n = hi - lo
            if n == 0:
                continue
            if 0 <= lo and hi < t_max:
                resolved.append((lo, hi))
            elif not bidirectional:
                continue
            elif hi >= t_max and 0 <= lo < t_max and lo - n >= 0:
                resolved.append((lo - n, lo))  # t -> t+n becomes t-n -> t
            elif lo < 0 and 0 <= hi < t_max and hi + n < t_max:
                resolved.append((hi, hi + n))  # t-n -> t becomes t -> t+n
        return resolved

    @staticmethod
    def _resolve_timesteps(
        t: int,
        time_steps: list[int],
        connectivity: Literal["dense", "sequential", "star"],
    ):
        if len(time_steps) == 1:
            return [(t, t + time_steps[0])]
        else:
            if time_steps[0] != 0:
                all_time_steps = [0] + time_steps
            else:
                all_time_steps = time_steps

            timepoints = [t + step for step in all_time_steps]
            if connectivity == "dense":
                # [t, t+1], [t, t+2], [t+1, t+2], [t+1, t+3], ...
                return itertools.combinations(timepoints, 2)
            elif connectivity == "sequential":
                # [t, t+1], [t+1, t+2], [t+2, t+3]
                return itertools.pairwise(timepoints)
            elif connectivity == "star":
                # [t, t+1], [t, t+2], [t, t+3]
                return itertools.product([t], timepoints)

            raise ValueError(
                f"Unknown {connectivity=}, use 'dense' (t->t+1, t->t+2, t+1->t+2), "
                "'sequential' (t->t+1, t+1->t+2), or 'star' (t->t+1, t->t+2)"
            )

    def postprocess_edges(
        self,
        edge_data: pl.DataFrame | pl.LazyFrame,
        node_feats: pl.DataFrame,
        masks: da.Array | np.ndarray | None = None,
        skip_pruning: bool = False,
    ):
        """Post-process edges.

        Adds lineage information, computes additional features, and prunes edges.
        """
        initial_cols = edge_data.collect_schema().names()
        additional_node_cols = (
            ["intensity_mean_src", "intensity_mean_dst"]
            if self.extra_features and "intensity_mean_diff" in self.extra_features
            else []
        )

        edge_data = _add_node_data(
            edge_data, node_feats, self._required_cols, available_cols=initial_cols
        )
        edge_data = self._label_edges(edge_data)

        # node-based pruning done pre-overlap computation to reduce number of edges
        # since this heavily impacts the lazy computation graph, we skip this when loading
        if (
            self.prune_edges
            and self.prune_method in ["dilated_overlap", "ellipse", "radius", "gt"]
            and not skip_pruning
        ):
            edge_data = self._prune_edges(edge_data, node_feats=node_feats, masks=masks)

        if self.should_compute_overlap:
            # when processing graph loaded from file, overlap is already present
            if "overlap" not in initial_cols:
                assert masks is not None, "Computing overlap requires passing `masks`."
                frame_pairs = collect(
                    edge_data.select("t_src", "t_dst").unique().sort("t_src")
                )
                overlap = self.compute_overlap(
                    node_feats,
                    frame_pairs.to_numpy(),
                    masks,
                    include_iou=self.should_compute_iou,
                )
                if overlap is not None:
                    if isinstance(edge_data, pl.LazyFrame):
                        overlap = overlap.lazy()

                    edge_data = edge_data.select(
                        *initial_cols, *additional_node_cols, "t_src", "t_dst", "y"
                    ).join(
                        # the left join re-introduces non-overlapping edges.
                        # drop t_src/t_dst from overlap: they are redundant with the
                        # values carried from _add_node_data above, and keeping overlap's
                        # copy would leave t_src/t_dst null for non-overlapping edges
                        # (only overlapping edges have a match in the overlap DF)
                        overlap.drop(["t_src", "t_dst"], strict=False),  # type: ignore
                        on=("src", "dst"),
                        how="left",
                    )

            # edge-based pruning is done immediately after overlap computation
            if self.prune_edges and self.prune_method == "overlap" and not skip_pruning:
                edge_data = self._prune_edges(edge_data)

        n_rows = (
            edge_data.select(pl.len()).collect().item()  # type: ignore
            if isinstance(edge_data, pl.LazyFrame)
            else edge_data.height
        )
        if n_rows == 0:
            return

        # to save time, the graph is only created in one direction (past -> fut)
        # instead of computing all features twice, we just transform them
        if self.bidirectional:
            edge_data = self.to_bidirectional(edge_data).unique(
                ["src", "dst"], maintain_order=True
            )

        # custom features computed for final graph in case of to retain symmetry
        if self.extra_features_fns:
            for fn in self.extra_features_fns:
                edge_data = fn(edge_data)

        additional_columns = []
        if self.should_compute_overlap:
            # area_dst lets the backward overlap be recomputed when mirroring on load
            additional_columns += ["intersection", "area_src", "area_dst"]

        return edge_data.sort("src", "dst").select(
            "src",
            "dst",
            "y",
            pl.col("t_src").alias("t"),
            # zero-overlap is filled, here
            # (basically only for clarity and to not introduce 0-labels)
            pl.col(*self.feature_cols).fill_null(0),
            *additional_columns,
        )

    def get_edges_for_frame(
        self,
        node_feats: pl.DataFrame,
        t: int,
        search_radius: int,
        time_steps: list[int],
        masks: da.Array | np.ndarray | None = None,
        connectivity: Literal["dense", "sequential", "star"] = "sequential",
        mirror_at_last_frames: bool = False,
        combinations: list[tuple[int, int]] | None = None,
    ) -> pl.DataFrame | None:
        """Get all edges originating from single frame.

        Gets all edges between source frame and potential target frames.

        Args:
            node_feats: Position (tyx) and handcrafted single-cell features.
            t: Source frame.
            search_radius: Maximum distance between two connected nodes.
            time_steps: Time steps between source and target frames.
            masks: Optional segmentation masks for overlap computation.
            connectivity: Connectivity between graph edges. Depending on parameter,
                edges are constructed from one source frame (`star`, e.g., t -> t+1, t ->
                t+2, t -> t+3), between sequential frames ( `sequential`, e.g., t -> t+1,
                t+1 -> t+2, t+2 -> t+3) or between all provided frames (`dense`, e.g., t
                -> t+1, t+1 -> t+2, t+2 -> t+3, t -> t+2, t -> t+3). Star connectivity is
                only used to efficiently construct dense edges without redundancy (see
                `GraphDataset.create_graph`).
            mirror_at_last_frames: If `true`, substitute edges whose target falls out
                of bounds with the same-stride in-bounds edge (e.g., at the last frame
                `t -> t+1` becomes `t-1 -> t`), so boundary nodes keep a neighbor at the
                configured stride. See `resolve_boundary_pairs`.
            combinations: Explicit `(t_src, t_dst)` frame pairs to build edges for. When
                given, overrides the pairs derived from `time_steps` and `connectivity`,
                used to extend the temporal field of view for message passing.
        """
        try:
            node_feats.select(self._required_cols)
        except pl.exceptions.ColumnNotFoundError as err:
            raise ValueError(
                f"Could not find all of the requried features {self._required_cols} in "
                f"columns of node features: Available: {node_feats.columns}."
            ) from err

        t_max = (
            len(masks) if masks is not None else int(node_feats["t"].to_numpy().max()) + 1
        )
        if combinations is None:
            combinations = self._resolve_timesteps(t, time_steps, connectivity)

        if mirror_at_last_frames:
            combinations = self.resolve_boundary_pairs(
                combinations, t_max, self.bidirectional
            )

        edge_data = []
        for t_src, t_dst in combinations:
            if (t_src == t_dst) or not (0 <= t_src < t_max) or not (0 <= t_dst < t_max):
                continue

            edge_data_temp = self.find_edge_pairs(node_feats, t_src, t_dst, search_radius)
            if edge_data_temp is not None:
                edge_data.append(edge_data_temp)

        if len(edge_data) == 0:
            # if empty, return None and create empty template graph in `GraphDataset`
            return

        edge_data = pl.concat(edge_data).unique(["src", "dst"], maintain_order=True)
        return self.postprocess_edges(edge_data, node_feats, masks)

    def compute_overlap(
        self,
        node_features: pl.DataFrame,
        frame_pairs: np.ndarray,
        masks: da.Array | np.ndarray,
        include_iou: bool = False,
    ) -> pl.DataFrame | None:
        """Compute overlap between two object masks."""
        if len(frame_pairs) == 0:
            return

        masks_roi = {
            t: masks[t].compute() if isinstance(masks[t], da.Array) else masks[t]
            for t in np.unique(frame_pairs)
        }
        # bbox columns are (min per spatial axis, then max per spatial axis), so the
        # first half are lower corners and the second half upper corners in nD
        ndim = sum(c.startswith("bbox-") for c in node_features.columns) // 2
        lo = [f"bbox-{i}" for i in range(ndim)]
        hi = [f"bbox-{i}" for i in range(ndim, 2 * ndim)]
        bounding_boxes = [
            node_features.filter(pl.col("t").is_in([t_src, t_dst]))
            .select(pl.col(lo).min(), pl.col(hi).max())
            .row(0)
            for t_src, t_dst in frame_pairs
        ]

        tasks = []
        for (t_src, t_dst), bbox in zip(frame_pairs, bounding_boxes, strict=True):
            # a shared ROI spanning every cell on both frames, sliced per spatial axis
            sl = tuple(slice(bbox[i], bbox[ndim + i]) for i in range(ndim))

            tasks.append(
                delayed(EdgeFinder._compute_overlap_task)(
                    masks_roi[t_src][sl],
                    masks_roi[t_dst][sl],
                    t_src,
                    t_dst,
                )
            )

        overlap_data = Parallel(n_jobs=self.n_jobs, backend="loky")(tasks)

        lbl_map = node_features.select("index", "label", "t", "area")
        overlap_data = (
            _add_labels(pl.concat(overlap_data), lbl_map)
            .filter(pl.max_horizontal("^label_.+$") != 0)
            .with_columns(overlap=pl.col("intersection") / pl.col("area_src"))
        )

        if include_iou:
            overlap_data = overlap_data.with_columns(
                iou=pl.col("intersection")
                / (pl.col("area_src") + pl.col("area_dst") - pl.col("intersection")),
            )

        return overlap_data

    @staticmethod
    def _compute_overlap_task(mask_src_slice, mask_dst_slice, t_src, t_dst):
        overlap_frame = _compute_overlap_frames(mask_src_slice, mask_dst_slice)

        return pl.DataFrame(
            {k: np.hstack(v) for k, v in overlap_frame.items()},
            schema=dict.fromkeys(overlap_frame.keys(), pl.Int64),
        ).with_columns(t_src=pl.lit(t_src), t_dst=pl.lit(t_dst))

    def __call__(
        self,
        node_feats: pl.DataFrame,
        search_radius: int,
        time_steps: list[int],
        masks: da.Array | np.ndarray | None = None,
        filepath: Path | None = None,
        overwrite: bool = False,
        connectivity: Literal["dense", "sequential", "star"] = "star",
        mirror: bool = True,
        validate: bool = True,
        dataset_meta: dict | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Build the spatio-temporal edge graph for a full sequence.

        Maintains a hive-partitioned store at `filepath` (a directory) with one
        `dt={k}/part.parquet` partition per temporal stride. Strides already present are
        reused. Only missing strides are computed and appended, so the store extends
        incrementally to arbitrary `time_steps`. On load, the requested partitions are
        filtered to `search_radius` and backward edges are mirrored from the stored
        positive-dt edges. With `filepath=None`, edges are computed without caching.

        Args:
            node_feats: Position (tyx) and handcrafted single-cell features.
            search_radius: Maximum Euclidean distance between connected nodes.
            time_steps: Temporal strides between connected frames (e.g., `[1, 2]`
                connects each frame to the next two).
            masks: Instance-segmentation masks for the sequence. Required when
                `prune_edges_by` or any feature needs them (e.g., 'overlap',
                'iou', 'dilated_overlap').
            filepath: Directory of the partitioned edge store. Reused and extended if
                compatible, otherwise (re)built. If `None`, edges are not cached.
            overwrite: If `True`, ignore any existing store and rebuild from scratch.
            connectivity: How nodes within the temporal window are connected.
                'dense' connects all node pairs within the radius (including
                non-adjacent frames like `t -> t+2`). 'sequential' only connects
                consecutive frame pairs (`t -> t+1`, `t+1 -> t+2`). 'star'
                connects each source frame to every step ahead (`t -> t+1`,
                `t -> t+2`). Iterating 'star' over all source frames produces
                the same edge set as 'dense' after deduplication while avoiding
                redundant work, so it is the default for cache construction.
            mirror: If `True`, mirror backward edges from the stored positive-dt
                partitions on load. With `False`, return the forward-only scan so the
                caller can filter per frame and mirror only the matched rows, keeping
                predicate pushdown into the parquet scan.
            validate: If `True`, verify the loaded store covers the requested features
                and matches the segmentation nodes, rebuilding on mismatch.
            dataset_meta: `dataset_identity()` of the sequence's images/masks, compared
                against the store's `meta.json` before reusing it. `None` skips the check.
        """
        requested_dt = sorted({abs(s) for s in time_steps if s != 0})

        if filepath is None:
            edge_data = self._compute(
                node_feats, search_radius, requested_dt, masks, connectivity
            )
            if not mirror and self.bidirectional:
                # same trim the on-disk store applies before writing: drop the boundary
                # fragments belonging to other strides, then keep forward edges only.
                # halves what an in-memory store holds; the caller mirrors the window it
                # selects, exactly as it would when reading positive-dt partitions.
                edge_data = edge_data.filter(
                    pl.col("dist_temp").abs().is_in(requested_dt),
                    pl.col("dist_temp") > 0,
                )
            return self._remap_div_label(edge_data)

        cache_dir = Path(filepath)
        meta = None if overwrite else edge_cache.read_meta(cache_dir)
        if meta is not None and not self._store_compatible(
            meta, search_radius, dataset_meta
        ):
            if dataset_meta is not None and not dataset_identity_matches(
                dataset_meta, meta.get("dataset")
            ):
                logger.warning("Edge store was built from different images/masks.")
            else:
                logger.debug("Edge store incompatible with request, rebuilding.")
            overwrite, meta = True, None

        positive_only = self.bidirectional if meta is None else meta["positive_only"]
        recorded = set(meta.get("strides", [])) if meta else set()
        available = recorded | (
            set()
            if (overwrite or meta is None)
            else edge_cache.available_partitions(cache_dir)
        )
        missing = [dt for dt in requested_dt if dt not in available]

        if overwrite or missing:
            if overwrite and cache_dir.exists():
                shutil.rmtree(cache_dir)
            strides = requested_dt if (overwrite or meta is None) else missing
            recorded_strides = sorted((set() if overwrite else available) | set(strides))
            try:
                graph = self._compute(
                    node_feats, search_radius, strides, masks, connectivity
                )
                self._persist_store(
                    graph,
                    cache_dir,
                    search_radius,
                    positive_only,
                    strides,
                    recorded_strides,
                    dataset_meta,
                )
            except NoEdgesError as err:
                # recorded so an impossible stride is not recomputed on every load
                logger.warning(f"No edges computed for dt={strides}: {err}.")
                cache_dir.mkdir(parents=True, exist_ok=True)
                edge_cache.write_meta(
                    cache_dir,
                    self._store_meta(
                        search_radius, positive_only, recorded_strides, dataset_meta
                    ),
                )

        edge_data = self._load_store(cache_dir, requested_dt, positive_only, mirror)
        if edge_data is None:
            raise ValueError(
                f"Edge store {cache_dir} has no partitions for dt={requested_dt}."
            )
        edge_data = edge_data.filter(pl.col("dist_spat") <= search_radius)

        if validate and (
            self._validate(edge_data, node_feats, search_radius, time_steps) is None
        ):
            logger.debug("Edge store failed validation, rebuilding from scratch.")
            shutil.rmtree(cache_dir, ignore_errors=True)
            graph = self._compute(
                node_feats, search_radius, requested_dt, masks, connectivity
            )
            self._persist_store(
                graph,
                cache_dir,
                search_radius,
                positive_only,
                requested_dt,
                dataset_meta=dataset_meta,
            )
            edge_data = self._load_store(
                cache_dir, requested_dt, positive_only, mirror
            ).filter(pl.col("dist_spat") <= search_radius)  # type: ignore

        return self._remap_div_label(edge_data)

    @staticmethod
    def _dilated_overlap_task(src, dst_nodes, coords, radii):
        s = int(src)
        # a label without boundary pixels has no coords (only happens if full frame is
        # label), so its pairs cannot be measured and are dropped, like the pairs the
        # bounding-box pre-filter rejected. not relevant for cell tracking
        if s not in coords:
            return [True] * len(dst_nodes)

        tree = cKDTree(coords[s])

        overlap_res = []
        for d in dst_nodes:
            if d not in coords:
                overlap_res.append(True)
                continue

            rmax = radii[s] + radii[d]
            dists, _ = tree.query(coords[d], k=1, distance_upper_bound=rmax)

            # drop if outside of dilation radii
            if np.min(dists) > rmax:
                overlap_res.append(True)
            else:
                overlap_res.append(False)

        return overlap_res

    @staticmethod
    def _boundary_task(mask_path, sampling_rate, node_labels, node_indices):
        mask = mask_path.compute() if isinstance(mask_path, da.Array) else mask_path
        assert isinstance(mask, np.ndarray)

        # only use boundaries since inner pixels do not matter for distance check
        props = regionprops(mask * find_boundaries(mask, mode="inner").astype(int))
        # map by label so filtered small cells in the mask don't shift the pairing
        label_to_coords = {p.label: p.coords[::sampling_rate] for p in props}
        return {
            idx: label_to_coords[lbl]
            for lbl, idx in zip(node_labels, node_indices, strict=True)
            if lbl in label_to_coords
        }

    def _prune_edges_dilated_overlap(
        self,
        edge_data: pl.DataFrame | pl.LazyFrame,
        node_feats: pl.DataFrame,
        masks: da.Array | np.ndarray,
        sampling_rate: int | None = None,
        radius_multiplier: float = 1.0,
    ):
        """Compute dilated overlap between pair of masks.

        Since dilation is a discrete approximation of the closest distance between two
        sets of boundaries, we can reformulate the dilated overlap check as closest
        distance between the boundaries of two objects. Naming was chosen to be in line
        with paper, where actual dilation and overlap computation was used. Since this is
        extremely slow, replaced with the new implementation. See
        `test_features:test_edge_finder_prune_by_dilated_overlap`.

        Algorithm:
            1. Extract instance boundaries. Inner points can be discarded since closest
                point is always between boundary pixels.
            2. Pre-filter based on bounding boxes. Expand bounding boxes by dilation
                radius and check overlap for all src/dst pairs.
            3. Compute closest distance between boundaries and check if smaller than
                dilation radius.

        Args:
            edge_data: Edge features containing indices of connected nodes (`src`,
                `dst`) and their relational features. Pruned in place.
            node_feats: Position (tyx) and handcrafted single-cell features.
            masks: Segmentation masks or paths.
            sampling_rate: Only compute distance between every nth pixel/voxel.
            radius_multiplier: Multiply dilation radius.
        """
        if masks[0].ndim > 2:
            raise NotImplementedError(
                "dilated_overlap pruning is 2D only. For 3D data prune by 'radius' or "
                "'overlap' instead."
            )

        radius = self.prune_param_parsed
        radii = (
            np.repeat(radius, node_feats.height)
            if isinstance(radius, int | float)
            # otherwise is returned as pl.Expr to do things like pl.col("val") * radius in
            # other pruning methods, so we have to convert to array
            else node_feats.select(radius).to_numpy().flatten()
        ) * radius_multiplier

        src = edge_data.select("src").unique()
        dst = edge_data.select("dst").unique()
        if isinstance(src, pl.LazyFrame) and isinstance(dst, pl.LazyFrame):
            src = src.collect()
            dst = dst.collect()
        assert isinstance(src, pl.DataFrame) and isinstance(dst, pl.DataFrame)
        nodes = np.unique(np.hstack([np.unique(src["src"]), np.unique(dst["dst"])]))

        # store boundary coords
        tasks_boundaries = []
        unique_t = np.unique(node_feats.filter(pl.col("index").is_in(nodes))["t"])
        if unique_t.size == 0:
            return edge_data

        for t in unique_t:
            feats_t = node_feats.filter(t=t)
            idx = feats_t.get_column("index")
            lbl = feats_t.get_column("label")
            tasks_boundaries.append(
                delayed(EdgeFinder._boundary_task)(masks[t], sampling_rate, lbl, idx)
            )

        last_mask = (
            masks[unique_t[-1]].compute()
            if isinstance(masks[unique_t[-1]], da.Array)
            else masks[unique_t[-1]]
        )
        assert isinstance(last_mask, np.ndarray)
        h, w = last_mask.shape

        boundary_results = Parallel(n_jobs=self.n_jobs, backend="loky")(tasks_boundaries)

        coords = {}
        for res in boundary_results:
            coords.update(res)

        # pre-filter using bounding box overlap
        boxes_expanded = (
            node_feats.select("index", r"^bbox-\d$", pl.Series("radius", radii))
            .with_columns(
                pl.col(r"^bbox-[01]$") - pl.col("radius"),
                pl.col(r"^bbox-[23]$") + pl.col("radius"),
            )
            # clip should not be required, but fast and compatible with uint
            .with_columns(
                pl.col(r"^bbox-[02]$").clip(0, h),
                pl.col(r"^bbox-[13]$").clip(0, w),
            )
        )
        if isinstance(edge_data, pl.LazyFrame):
            boxes_expanded = boxes_expanded.lazy()

        candidates = (
            edge_data.select("src", "dst")
            .join(boxes_expanded, left_on="src", right_on="index")  # type: ignore
            .join(boxes_expanded, left_on="dst", right_on="index")  # type: ignore
            .filter(
                pl.col("bbox-0") <= pl.col("bbox-2_right"),
                pl.col("bbox-2") >= pl.col("bbox-0_right"),
                pl.col("bbox-1") <= pl.col("bbox-3_right"),
                pl.col("bbox-3") >= pl.col("bbox-1_right"),
            )
        )

        edge_index_candidates = collect(candidates.select("src", "dst"))
        assert isinstance(edge_index_candidates, pl.DataFrame)

        # check distances between boundaries
        tasks = []
        src_groups = []

        for (src,), grouped in edge_index_candidates.group_by("src", maintain_order=True):
            dst_nodes = grouped["dst"].to_list()
            src_groups.append((src, dst_nodes))
            tasks.append(
                delayed(EdgeFinder._dilated_overlap_task)(src, dst_nodes, coords, radii)
            )

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(tasks) if tasks else []

        # keep is the decision, not drop: a pair the bounding-box pre-filter rejected is
        # farther apart than the dilation radius, so it never reaches the distance check
        # and must not survive it either
        keep_srcs, keep_dsts = [], []
        for (src, dst_nodes), res in zip(src_groups, results, strict=True):
            for dst, should_drop in zip(dst_nodes, res, strict=True):
                if not should_drop:
                    keep_srcs.append(int(src))
                    keep_dsts.append(int(dst))

        schema = edge_data.collect_schema()
        keep_df = pl.DataFrame(
            {"src": keep_srcs, "dst": keep_dsts},
            schema={"src": schema["src"], "dst": schema["dst"]},
        )
        if isinstance(edge_data, pl.LazyFrame):
            keep_df = keep_df.lazy()

        return edge_data.join(keep_df, on=["src", "dst"], how="semi")  # type: ignore


def _compute_overlap_frames(ref_frame, next_frame):
    cont = contingency_table(ref_frame, next_frame).tocoo()

    overlap_data = defaultdict(list)
    overlap_data["label_src"].append(cont.row)
    overlap_data["label_dst"].append(cont.col)
    overlap_data["intersection"].append(cont.data)

    return overlap_data


def _add_labels(edge_data, node_data, on=("label", "t"), suffix=("_src", "_dst")):
    suffix = [suffix] if not isinstance(suffix, list | tuple) else suffix
    out = edge_data

    for suff in suffix:
        out = out.join(
            # ruff: function-uses-loop-variable (B023)
            node_data.rename(lambda n, suff=suff: n + suff),
            on=(c + suff for c in on),
        ).rename({f"index{suff}": suff[1:]})

    return out


def _add_node_data(
    edge_data: pl.DataFrame | pl.LazyFrame,
    node_feats: pl.DataFrame,
    val_cols: tuple[str, ...] | list[str] = ("t", "label", "^centroid.*$"),
    suffix_left: str = "_src",
    suffix_right: str = "_dst",
    available_cols: list[str] | None = None,
):
    add_cols = [
        c
        for c in val_cols
        if not available_cols
        or not (c + "_src" in available_cols and c + "_dst" in available_cols)
    ]
    if len(add_cols) == 0:
        return edge_data

    src = node_feats.select("index", pl.col(add_cols).name.suffix(suffix_left))
    dst = node_feats.select("index", pl.col(add_cols).name.suffix(suffix_right))

    if isinstance(edge_data, pl.LazyFrame):
        src = src.lazy()
        dst = dst.lazy()

    return edge_data.join(
        src,  # type: ignore
        left_on="src",
        right_on="index",
    ).join(
        dst,  # type: ignore
        left_on="dst",
        right_on="index",
    )
