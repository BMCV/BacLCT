"""Per-sequence `GraphDataset` and multi-sequence Lightning wrapper `TrackingDataset`."""

from __future__ import annotations

import itertools
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import dask.array as da
import lightning as L
import numpy as np
import polars as pl
import torch
import yaml
from lightning.pytorch.utilities.seed import isolate_rng
from omegaconf import ListConfig
from scipy.spatial.distance import cdist
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from baclct.features.extractors import (
    CellLevelExtractor,
    HandcraftedExtractor,
    _get_deep_dir,
    _get_seg_dir,
)
from baclct.features.graph import EdgeFinder, build_lineage_gt_edges
from baclct.io import (
    cached_percentiles,
    dataset_identity,
    find_lineage_file,
    load_images_and_masks,
)
from baclct.utils.data import (
    col_to_tensor,
    collect,
    default_multiprocessing_context,
)
from baclct.utils.feature_info import cache_signature, classify_cache
from baclct.utils.graph_params import (
    expand_param_range,
    max_search_radius,
    resolve_search_radius,
)
from baclct.utils.logger import get_pylogger
from baclct.utils.spacing import normalize_spacing, spatial_spacing

if TYPE_CHECKING:
    from baclct.utils.spacing import SpacingLike

logger = get_pylogger(__name__)


class GraphDataset(Dataset):
    """Trajectory graph dataset for a single image sequence.

    Contains the spatio-temporal graph of a single time-lapse. Single cells are
    represented as nodes, and their interactions between frames are represented as edges.
    On each `__getitem__` call, a temporal window (subgraph) centered on a single frame is
    returned as a `torch_geometric.data.Data` object.

    Responsibilities:

    - Loading or receiving images and masks (from disk via `data_dir`, lazy or eager,
      or directly via the `images` and `masks` arguments).
    - Computing or loading handcrafted and (optionally) deep node features.
    - Building or loading candidate edges with `EdgeFinder` (either precomputed for the
      full sequence or on-the-fly per window).
    - Assembling PyG `Data` objects with node features, edge features, edge indices,
      and ground-truth labels.

    Where the candidate edges live follows `feature_dir`. Without one they are built and
    kept in memory, which is the default for `BacLCT.track()`. With a directory (or
    `'temp'`) they are written to the partitioned edge store next to the node features and
    read back lazily, so an item scans parquet instead of holding the sequence in memory.

    This class handles a single sequence. For multi-sequence training with Lightning,
    use `TrackingDataset`, which wraps one `GraphDataset` per sequence and manages
    feature extraction, splits, and dataloaders.
    """

    def __init__(
        self,
        data_dir: Path | None,
        feature_dir: Path | Literal["temp"] | None,
        sequence_id: str | None,
        edge_finder: EdgeFinder,
        handcrafted_feature_extractor: HandcraftedExtractor,
        deep_feature_extractor: CellLevelExtractor | None,
        segmentation_name: str = "GT",
        data_format: str = "ctc",
        img_name: str | None = None,
        graph_search_radius: int | str | tuple[int, int] | tuple[int, int, int] = 200,
        graph_time_step: int | tuple[int, int] | tuple[int, int, int] = 1,
        graph_num_steps: int = 3,
        graph_connectivity: Literal["dense", "star", "sequential"] = "dense",
        graph_fov_size: int | float | None = None,
        graph_max_num_edges: int = 500_000,
        graph_edge_dropout: float = 0.0,
        graph_node_dropout: float = 0.0,
        graph_frame_dropout: float = 0.0,
        precompute_edges: bool = False,
        training: bool = False,
        images: da.Array | np.ndarray | None = None,
        masks: da.Array | np.ndarray | None = None,
        lazy: bool = True,
        spacing: SpacingLike = None,
        trust_cache: bool = False,
    ) -> None:
        """Initialize dataset for a single sequence.

        Args:
            data_dir: Root directory containing image and mask data.
            feature_dir: Base directory for cached features (node parquet, edge parquet,
                deep feature embeddings). Subdirectories are resolved automatically per
                sequence and segmentation. Pass `'temp'` for a temporary directory that
                is removed with the dataset.
            sequence_id: Identifier of the sequence, e.g., '01' or 'train'.
            edge_finder: Builds candidate edges between cells across frames.
            handcrafted_feature_extractor: Computes single-cell positions and features
                (area, intensity, shape descriptors, etc.). Handcrafted features are
                disabled for training by setting `feature_names` to `None`.
            deep_feature_extractor: Extractor for learned features (DINO, ResNet, etc.).
                Set to `None` for handcrafted-only mode.
            segmentation_name: Suffix identifying the segmentation variant or name,
                e.g., 'GT', 'SEG', 'masks'.
            data_format: On-disk layout of images and masks. One of 'ctc' (dirs, e.g.,
                '01' for images and '01_GT/TRA' for tracks), 'flat' (tifs, e.g.,
                '01_images.tif' and '01_masks.tif'), or 'dirs' (same as CTC, but
                dirs may contain tif stacks and does not check for 'TRA').
            img_name: Name of the image subdirectory (`data_format='dirs'`) or
                suffix (`data_format='flat'`).
            graph_search_radius: Maximum Euclidean distance for an edge to be created
                between two cells. An int is absolute pixels. A string ending in
                'x' (e.g. '2.5x') is a multiple of the expected cell size.
            graph_time_step: Temporal distance (in frames) between connected nodes.
            graph_num_steps: Number of consecutive frame gaps to consider. With
                `graph_time_step=1` and `graph_num_steps=3`, edges span `t -> t+1`,
                `t -> t+2`, and `t -> t+3`. Depending on `graph_connectivity`, edges
                may also be constructed between `t+1 -> t+2`, `t+1 -> t+3`, etc.
                Depending on `EdgeFinder.bidirectional`, edges are also constructed
                towards past frames, i.e. `t -> t-1`.
            graph_connectivity: How nodes within a subgraph window are connected.
                'dense' connects all node pairs within the radius (including
                non-adjacent frames like `t -> t+2`). 'star' connects the center frame
                to every other frame in the window but not those to each other.
                'sequential' only connects nodes between consecutive frame pairs
                (`t -> t+1`, `t+1 -> t+2`).
            graph_fov_size: Widens the temporal context for message passing by adding
                anchor frames around the center, spaced by `graph_time_step`. Every anchor
                keeps its full edges, so the maximum edge distance is unchanged and more
                anchors just grow the field of view. The number of anchors each side is
                `round(graph_fov_size * graph_num_steps)` (halves round down), so `1.0`
                adds `graph_num_steps` each side, `2.0` twice as many, and `0.5` half as
                many (one frame each side when `graph_num_steps=3`). `None`, or a value
                rounding to `0`, keeps only the center frame.
            graph_max_num_edges: Maximum number of edges per subgraph. When exceeded,
                edges are randomly sampled.
            graph_edge_dropout: Fraction of edges randomly dropped per sample (training
                only).
            graph_node_dropout: Fraction of nodes randomly dropped per sample (training
                only).
            graph_frame_dropout: Fraction of frames randomly dropped per sample (training
                only).
            precompute_edges: If `True`, construct (or load) all edges of the sequence
                during init and filter around a single frame or within a patch at
                `__getitem__`. If `False`, edges are computed on the fly.
            training: Enables augmentation (dropout, parameter sampling).
            images: Image sequence with shape `(T, H, W)`. If provided, `data_dir` is
                not used for image loading.
            masks: Instance-segmentation masks for `images`. If provided, `data_dir`
                is not used for mask loading.
            lazy: Use `pl.LazyFrame` for the edge data. Slower per query but much more
                memory-efficient for large sequences.
            spacing: Physical frame and voxel spacing as a `{t, z, y, x}` dict or a
                length-2/3/4 sequence. `t` expects minutes, `zyx` expect µm. Unspecified
                axes default to `1.0`.
            trust_cache: If `True`, skip hashing `images`/`masks` and never invalidate a
                cache whose name and structural metadata match the request. If `False`,
                a sampled per-frame content hash also guards each cache against being
                built from different images/masks under the same name.
        """
        self.spacing = normalize_spacing(spacing)
        self.data_dir = data_dir
        # the handle keeps a `feature_dir='temp'` cache alive for as long as the dataset
        feature_dir, self._feature_tempdir = _resolve_feature_dir(feature_dir)
        self.feature_dir = feature_dir
        self.sequence_id = sequence_id
        self.segmentation_name = segmentation_name
        self.data_format = data_format
        self.img_name = img_name

        self.edge_finder = edge_finder
        self.handcrafted_feature_extractor = handcrafted_feature_extractor
        self.deep_feature_extractor = deep_feature_extractor

        self.graph_search_radius = graph_search_radius
        self.graph_time_step = graph_time_step
        self.graph_num_steps = graph_num_steps
        self.graph_connectivity = graph_connectivity
        self.graph_fov_size = graph_fov_size
        self.graph_max_num_edges = graph_max_num_edges

        self.graph_edge_dropout = graph_edge_dropout
        self.graph_node_dropout = graph_node_dropout
        self.graph_frame_dropout = graph_frame_dropout

        if feature_dir is not None and self.sequence_id is not None:
            self.handcrafted_feat_dir = _get_seg_dir(
                feature_dir, self.sequence_id, segmentation_name
            )
            filepath_hc = self.handcrafted_feat_dir / "nodes.parquet"
            filepath_pct = self.handcrafted_feat_dir / "percentiles.json"
        else:
            self.handcrafted_feat_dir = None
            filepath_hc = None
            filepath_pct = None

        if masks is not None:
            self.images = images  # may be None for mask-only tracking
            self.masks = masks
            self.sequence_id = sequence_id or "pred"
            self.image_percentiles = cached_percentiles(images, filepath_pct)
        else:
            assert isinstance(data_dir, Path) and isinstance(sequence_id, str)
            self.images, self.masks, self.image_percentiles = load_images_and_masks(
                data_dir,
                sequence_id,
                data_format=data_format,
                lazy=True,
                return_percentiles=True,
                segmentation_name=segmentation_name,
                img_name=img_name,
                percentile_file=filepath_pct,
            )
        if self.images is not None:
            assert isinstance(self.images, (da.Array, np.ndarray)) and isinstance(
                self.masks, (da.Array, np.ndarray)
            )
            assert len(self.images) > 0, f"Could not find images at {self.data_dir}."
        assert len(self.masks) > 0, f"Could not find masks at {self.data_dir}."

        self.trust_cache = trust_cache
        self.dataset_meta = (
            None
            if trust_cache
            else dataset_identity(
                self.images,
                self.masks,
                data_dir=self.data_dir,
                sequence_id=self.sequence_id,
            )
        )

        self._spatial_spacing = spatial_spacing(self.spacing, self.masks.ndim - 1)

        # namespace the source with the dataset subdir (data_dir.name) so colliding
        # sequence ids across combined datasets (e.g. spores/01 vs van_vliet/01) stay
        # distinct in batched graphs and per-source aggregation
        self.data_source = (
            f"{self.data_dir.name}/{self.sequence_id}"
            if self.data_dir is not None
            else self.sequence_id
        )

        lineage_file, has_states = None, False
        if not isinstance(self.masks, np.ndarray) and self.data_dir and self.sequence_id:
            lineage_file, has_states = find_lineage_file(
                data_dir=self.data_dir,
                seq_id=self.sequence_id,
                data_format=self.data_format,
                segmentation_name=self.segmentation_name,
                with_states=True,
            )

        self.node_feats = self.handcrafted_feature_extractor(
            image=self.images,
            masks=self.masks,
            lineage_file=lineage_file,
            sequence_id=self.sequence_id,
            filepath=filepath_hc,
            validate=False,
            overwrite=False,
            image_percentiles=self.image_percentiles,
            spacing=self._spatial_spacing,
            dataset_meta=self.dataset_meta,
        )

        # if deep features are not cached, we compute them from scratch to prevent
        # duplication. depending on the feature size this is safe to do with multiple
        # workers (e.g., DINO-B: 768). here, we can also write a new cache.
        self.deep_feats = None
        if deep_feature_extractor:
            if feature_dir is not None and sequence_id is not None:
                self.deep_feat_dir = _get_deep_dir(
                    deep_feature_extractor, feature_dir, sequence_id, segmentation_name
                )
                self._build_deep_cache()
            else:
                self.deep_feat_dir = None
                self.deep_feats = self._get_deep_features(self.node_feats)

        # for sparsely annotated datasets, some trajectories do not have lineage
        # information. since we need all trajectories for message passing, unannotated
        # nodes will be kept here and only dropped during loss computation
        self.missing_trajectories = (
            self.node_feats.filter(pl.col("parent").is_null() | pl.col("state").is_null())
            .get_column("index")
            .to_list()
        )

        if not has_states:
            self.node_feats = self.node_feats.with_columns(y=pl.lit(0, dtype=pl.UInt32))
        else:
            assert "state" in self.node_feats
            self.node_feats = self.node_feats.with_columns(
                state=pl.col("state") - 1
            ).rename({"state": "y"})
            # missing_trajectories have null state. fill with 0 so col_to_tensor produces
            # a valid integer tensor. excluded for loss computation via
            # _validate_and_drop_edges.
            self.node_feats = self.node_feats.with_columns(y=pl.col("y").fill_null(0))

        self.graph_search_radius = resolve_search_radius(
            self.graph_search_radius, self.node_feats
        )

        # complete GT lineage graph (not determined by graph params), cached by
        # prepare_data when lineage is available. used only for validation logging to see
        # whether divisions are lost due to graph topology (e.g., too small search
        # radius). not used outside of logs/val.
        self.lineage_gt_edges: pl.DataFrame | None = None
        if self.handcrafted_feat_dir is not None:
            gt_edges_file = self.handcrafted_feat_dir / "lineage_gt_edges.parquet"
            if gt_edges_file.exists():
                self.lineage_gt_edges = pl.read_parquet(gt_edges_file)

        self.training = training
        self.precompute_edges = precompute_edges
        self.lazy = lazy

        self.valid_frames, self.missing_frames = self._build_index()
        self.num_frames = int(self.valid_frames.max()) + 1

        # keyed by (center frame, time steps): avoids resolving pairs and compiling the
        # filter again for every item sharing that key
        self._graph_pairs_cache: dict[
            tuple[int, tuple[int, ...]], list[tuple[int, int]]
        ] = {}
        self._graph_filter_cache: dict[tuple[int, tuple[int, ...]], pl.Expr | None] = {}
        # sorted once so a frame pair is a contiguous range, avoiding a full-table filter
        self._edge_slices: dict[tuple[int, int], tuple[int, int]] | None = None
        # avoids refetching when several `GraphPatchDataset` patches share a frame
        self._frame_edge_memo: tuple[tuple, pl.DataFrame | None] | None = None

        self._prepare_edges()
        self.template_graph = self._build_graph_template()

    def _prepare_edges(self) -> None:
        """Build or load the sequence's edges and set up the per-item lookups.

        The store keeps forward edges only and every item mirrors its own window: a global
        mirror doubles the store and, on a lazy one, redefines `t` and `dist_temp` so the
        per-frame predicate is no longer pushed down into the scan.

        Training samples a stride per item, so the store has to cover every value the
        configured range can take. Inference resolves to a single stride, so covering the
        whole range would build strides no item ever reads.
        """
        if not self.precompute_edges:
            self._mirror_per_item = False
            self._radius_aug = False
            self.edge_data = None
            self._edge_slices = None
            return

        if self.training:
            logger.debug(
                "Training graph uses precomputed edges. Spatial-radius sampling and "
                "dropout augmentation are applied per item."
            )

        filepath_edge = (
            self.handcrafted_feat_dir / self.edge_finder.cache_dirname()
            if self.handcrafted_feat_dir
            else None
        )

        self._mirror_per_item = self.edge_finder.bidirectional
        self.edge_data = self.edge_finder(
            node_feats=self.node_feats,
            search_radius=max_search_radius(self.graph_search_radius, self.node_feats),
            time_steps=self.time_steps(return_all_combinations=self.training),
            masks=self.masks,
            filepath=filepath_edge,
            # dense is inefficient during graph construction, star provides
            # identical results (when in loop) without duplication.
            connectivity="sequential"
            if self.graph_connectivity == "sequential"
            else "star",
            mirror=not self._mirror_per_item,
            validate=False,
            dataset_meta=self.dataset_meta,
        )
        if (filepath_edge is None) or (not self.lazy):
            self.edge_data = collect(self.edge_data)
        self._index_edges_by_frame_pair()

        # sampling a smaller radius per item filters on a column an older store may lack
        self._radius_aug = not isinstance(self.graph_search_radius, int)
        if self._radius_aug:
            names = (
                self.edge_data.collect_schema().names()
                if isinstance(self.edge_data, pl.LazyFrame)
                else self.edge_data.columns
            )
            if "dist_spat" not in names:
                logger.warning(
                    "graph_search_radius is a range but 'dist_spat' is absent from the "
                    "edge cache. Spatial augmentation is disabled for this dataset."
                )
                self._radius_aug = False

    @classmethod
    def from_images_and_masks(
        cls,
        images: da.Array | np.ndarray,
        masks: da.Array | np.ndarray,
        edge_finder: EdgeFinder,
        handcrafted_feature_extractor: HandcraftedExtractor,
        deep_feature_extractor: CellLevelExtractor | None = None,
        sequence_id: str = "pred",
        segmentation_name: str = "seg",
        feature_dir: Path | Literal["temp"] | None = None,
        graph_search_radius: int | str | tuple[int, int] = 200,
        graph_time_step: int | tuple[int, int] = 1,
        graph_num_steps: int = 3,
        graph_connectivity: Literal["dense", "star", "sequential"] = "dense",
        graph_fov_size: int | float | None = None,
        graph_max_num_edges: int = 500_000,
        precompute_edges: bool = True,
        patch_size: int | tuple[int, ...] | None = None,
        patch_overlap: float = 0.0,
        spacing: SpacingLike = None,
        trust_cache: bool = False,
    ) -> GraphDataset:
        """Instantiate dataset directly from images and masks without relying on files.

        When `patch_size` is given, tiles each frame into spatial patches
        (`GraphPatchDataset`) instead of building one whole-frame graph.

        `feature_dir` selects where features are cached, `'temp'` being a directory tied
        to the dataset's lifetime. `precompute_edges=False` skips the cache entirely and
        rebuilds each frame's edges once per temporal window it appears in.

        Dask arrays are accepted and preferred for large sequences, since deep feature
        extraction copies an in-memory stack into every worker.
        """
        assert isinstance(images, (da.Array, np.ndarray))
        assert isinstance(masks, (da.Array, np.ndarray))
        kwargs: dict[str, Any] = {
            "data_dir": None,
            "feature_dir": feature_dir,
            "sequence_id": sequence_id,
            "edge_finder": edge_finder,
            "handcrafted_feature_extractor": handcrafted_feature_extractor,
            "deep_feature_extractor": deep_feature_extractor,
            "graph_search_radius": graph_search_radius,
            "graph_time_step": graph_time_step,
            "graph_num_steps": graph_num_steps,
            "graph_connectivity": graph_connectivity,
            "graph_max_num_edges": graph_max_num_edges,
            "graph_fov_size": graph_fov_size,
            "precompute_edges": precompute_edges,
            "training": False,
            "images": images,
            "masks": masks,
            "segmentation_name": segmentation_name,
            "spacing": spacing,
            "trust_cache": trust_cache,
        }
        if patch_size is not None:
            return GraphPatchDataset(
                patch_size=patch_size, patch_overlap=patch_overlap, **kwargs
            )
        return cls(**kwargs)

    def __len__(self) -> int:
        """Number of frames in video."""
        return self.num_frames

    def __repr__(self) -> str:
        """Dataset summary."""
        info = [
            f"sequence_id={self.sequence_id!r}",
            f"num_nodes={self.node_feats.height}",
            f"training={self.training}",
            f"length={len(self)}",
            f"segmentation={self.segmentation_name!r}",
            f"handcrafted_features={self.handcrafted_feature_extractor!r}",
            f"deep_features={self.deep_feature_extractor!r}",
            f"edge_finder={self.edge_finder!r}",
            f"graph_search_radius={self.graph_search_radius!r}",
            f"graph_time_step={self.graph_time_step!r}",
            f"graph_num_steps={self.graph_num_steps!r}",
            f"graph_connectivity={self.graph_connectivity!r}",
            f"graph_fov_size={self.graph_fov_size!r}",
            f"graph_max_num_edges={self.graph_max_num_edges!r}",
            f"graph_time_steps={self.time_steps(return_all_combinations=False)!r}",
        ]

        info_str = ",\n  ".join(info)
        return f"GraphDataset(\n  {info_str}\n)"

    def __getstate__(self) -> dict:
        """Drop the temporary cache handle so workers never own its lifetime."""
        state = self.__dict__.copy()
        state["_feature_tempdir"] = None
        state["_frame_edge_memo"] = None
        return state

    def _build_deep_cache(self) -> None:
        """Extract deep features for the whole sequence into `deep_feat_dir`.

        Per-item extraction can only ever write a partial cache, which the next item then
        reads out of bounds, so the cache is always built for all nodes at once.
        """
        extractor = self.deep_feature_extractor
        assert extractor is not None
        num_nodes = int(self.node_feats["index"].to_numpy().max()) + 1
        if extractor.cache_covers(self.deep_feat_dir, num_nodes, self.dataset_meta):
            return

        _run_deep_extractor(
            extractor,
            image=self.images,
            masks=self.masks,
            node_feats=self.node_feats,
            output_dir=self.deep_feat_dir,
            image_percentiles=self.image_percentiles,
            spacing=self._spatial_spacing,
            dataset_meta=self.dataset_meta,
        )

    def _build_index(self):
        """Identify valid and missing frames."""
        valid = np.unique(self.node_feats["t"]).astype(int)
        all_frames = np.arange(valid.min(), valid.max() + 1)
        missing = all_frames[~np.isin(all_frames, valid)]

        return valid, missing

    def _build_graph_template(self):
        """Create empty graph template.

        Returned if no edges are found. This prevents downstream errors with empty batches
        (dataloader, skipping backprop on multi-gpu).
        """
        # hack to make __init__ run without providing edge finder makes it easier to mock
        # during testing, not used regularly. would raise outside of tests.
        if not self.edge_finder:
            return

        # getting graph for a single frame does not benefit from multi-threading and is
        # much faster single-threaded (no overhead). try-except for backwards
        # compatibility
        try:
            prev_jobs = self.edge_finder.n_jobs
            self.edge_finder.n_jobs = 1
            has_jobs = True
        except AttributeError:
            has_jobs = False

        all_possible_time_steps = self.time_steps(return_all_combinations=False)

        # find first time step with neighbors
        found_graph = False
        first_valid_seed = None
        first_valid_t = None

        for t in all_possible_time_steps:
            seed_mask = np.isin(self.valid_frames + t, self.valid_frames)
            if self.edge_finder.bidirectional:
                seed_mask = seed_mask | np.isin(self.valid_frames - t, self.valid_frames)

            seeds = self.valid_frames[seed_mask]
            if seeds.size > 0 and first_valid_seed is None:
                first_valid_seed = seeds[0].item()
                first_valid_t = t

            for seed in seeds:
                seed = seed.item()
                edge_data = self._get_graph_for_frame(seed, [t])
                if edge_data is not None:
                    found_graph = True
                    break

            if found_graph:
                break

        if has_jobs:
            self.edge_finder.n_jobs = prev_jobs

        # if we could not find graph using the current config, compute minimum required
        # distance and raise error with details.
        if not found_graph:
            if first_valid_seed is not None:
                seed = first_valid_seed
                t = first_valid_t
                feats_src = self.node_feats.filter(t=seed)
                feats_dst = self.node_feats.filter(t=seed + t)

                center_name = getattr(self.edge_finder, "center_name", "centroid")
                cc = rf"^{center_name}-\d$"
                try:
                    dists = cdist(
                        feats_src.select(pl.col(cc)).to_numpy(),
                        feats_dst.select(pl.col(cc)).to_numpy(),
                        metric="euclidean",
                    )
                    max_dist = np.max(dists) if dists.size > 0 else "N/A"
                    min_dist = np.min(dists) if dists.size > 0 else "N/A"
                except Exception:
                    max_dist = "Error computing"
                    min_dist = "Error computing"

                raise ValueError(
                    "Could not find edges for the requested configuration in any frame. "
                    "Consider increasing the radius or checking pruning parameters.\n"
                    f"Example frame: {seed} -> {seed + t}\n"
                    f"Distances: query={self.graph_search_radius}, "
                    f"max={max_dist}, "
                    f"min={min_dist}"
                )
            else:
                raise ValueError(
                    "Could not find non-empty frame pairs with the supplied parameters.\n"
                    f"Non-empty frames: {self.valid_frames}\n"
                    f"Possible steps: {self.graph_time_step}"
                )

        # get number of nodes for debug
        feats_src = self.node_feats.filter(t=seed)
        feats_dst = self.node_feats.filter(t=seed + t)

        # get first valid two-frame graph
        logger.debug(f"Creating initial dummy graph from frame {seed} with steps {[t]}.")
        logger.debug(f"Number of nodes: {feats_src.height} -> {feats_dst.height}")

        # edge_data is already computed in loop
        assert edge_data is not None

        # edge_data is small, so we can just materialize it
        edge_data = collect(edge_data)

        node_data = self._load_and_remap_nodes(edge_data.select("src", "dst").to_numpy())
        node_mapping = node_data.select("index", "new_index")
        edge_data = self._remap_edges(edge_data, node_mapping).sort("src_new", "dst_new")

        x_handcrafted, x_deep = self._get_node_features(node_data)
        edge_attr = self._get_edge_features(edge_data, self.node_feats)

        return {
            "x_handcrafted": None
            if x_handcrafted is None
            else torch.empty((0, *x_handcrafted.shape[1:]), dtype=x_handcrafted.dtype),
            "x_deep": None
            if x_deep is None
            else torch.empty((0, *x_deep.shape[1:]), dtype=x_deep.dtype),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, *edge_attr.shape[1:]), dtype=edge_attr.dtype),
            "y_nodes": torch.empty((0,), dtype=torch.long),
            "y_edges": torch.empty((0,), dtype=torch.long),
            "node_index": torch.empty((0,), dtype=torch.long),
            "node_mapping": torch.empty((0,), dtype=torch.long),
            "num_nodes": 0,
            "data_source": self.data_source,
            "empty_flag": True,
            "edge_ignore": torch.empty((0,), dtype=torch.bool),
        }

    def _load_and_remap_nodes(
        self, edge_index: np.ndarray, *, skip_augmentation: bool = False
    ) -> pl.DataFrame:
        """Load nodes and remap to continuous format starting at 0.

        Args:
            edge_index: Global edge index starting at 0 for first frame.
            skip_augmentation: If True, skip frame and node dropout. Used as
                a fallback when dropout accidentally empties the graph.

        Returns:
            Dataframe with `new_index` starting at 0 at sampled frame.
        """
        node_data = self.node_feats.filter(
            pl.col("index").is_in(np.unique(edge_index))
        ).sort("index")

        if self.graph_frame_dropout > 0.0 and self.training and not skip_augmentation:
            frames = np.unique(node_data["t"].to_numpy())
            if len(frames) > 2:
                num_frames = max(int(len(frames) * (1 - self.graph_frame_dropout)), 2)
                frames = np.random.choice(frames, num_frames, replace=False)  # noqa: NPY002, lightning seeds
                node_data = node_data.filter(pl.col("t").is_in(frames))

        if (
            self.graph_node_dropout > 0.0
            and self.training
            and node_data.height > 10
            and not skip_augmentation
        ):
            node_data = node_data.sample(fraction=1 - self.graph_node_dropout).sort(
                "index"
            )

        return node_data.with_row_index("new_index")

    def _remap_edges(
        self, edge_data: pl.DataFrame, node_mapping: pl.DataFrame
    ) -> pl.DataFrame:
        """Remap edges to continuous format starting at 0.

        Args:
            edge_data: Dataframe with global `src` indices starting at 0 at first frame.
            node_mapping: Nodes with continuous `new_index` and original `index`.

        Returns:
            Dataframe with continuous node and edge indices starting at requested frame.
            Original indices are retained, remapped indices are suffixed with `_new`.
        """
        return edge_data.join(
            node_mapping.rename({"new_index": "src_new"}),
            left_on="src",
            right_on="index",
        ).join(
            node_mapping.rename({"new_index": "dst_new"}),
            left_on="dst",
            right_on="index",
        )

    def _resolve_param(self, val, return_all_possible_values=False) -> list[int] | int:
        """Resolves a parameter that can be an integer or a [min, max(, step)] range.

        For examples see `tests/test_data.py:test_resolve_param_and_time_steps`.

        Args:
            val: Single integer, `(min, max)` inclusive range, or `(min, max, step)`
                inclusive range with the given step.
            return_all_possible_values: If `true` will turn all reachable time steps
                for the supplied `val`.

        Returns:
            Single value if given integer or `return_all_possible_values`, else a list
            of all possible time steps (int will be wrapped in list).
        """
        if isinstance(val, int):
            return [val] if return_all_possible_values else val

        all_vals = expand_param_range(val)
        if return_all_possible_values:
            return all_vals

        if self.training:
            return np.random.choice(all_vals).item()  # noqa: NPY002, lightning seeds

        # return the lower bound for deterministic validation/testing
        return all_vals[0]

    def time_steps(self, return_all_combinations: bool = True):
        """Time steps used for graph construction.

        Finds time steps for the graph construction params of the dataset
        `graph_time_steps` and `graph_num_steps`. Graph parameters might be sampled from
        a range of possible params. Optionally returns all possible combinations, e.g.,
        unique time steps for multiple param ranges (e.g., for step size `1:4` and for
        `1:4` steps).

        Args:
            return_all_combinations: If `True`, returns all possible combinations of
                time steps. Otherwise will return time steps for single (optionally
                sampled) params. Ignored if graph parameters are single values.

        Returns:
            Single time steps or combination of time steps for graph construction.
        """
        step = self._resolve_param(self.graph_time_step, return_all_combinations)
        num = self.graph_num_steps

        if return_all_combinations:
            assert isinstance(step, list)
            num = [num]

            final_steps = set()
            for ti, ni in itertools.product(step, num):
                assert isinstance(ti, int) and isinstance(ni, int), (  # redundant
                    f"Expected integers. Got {type(ti)=} and {type(ni)=}."
                )
                if ni > 0:
                    final_steps.update(list(range(ti, ti + ti * ni, ti)))
            return sorted(final_steps)

        # one sampled value per range, so an item gets a single graph (e.g. [1, 2, 3])
        num = self._resolve_param(self.graph_num_steps, return_all_possible_values=False)
        assert isinstance(step, int), f"Found step with {type(step)}, expected `int`."
        assert isinstance(num, int), f"Found num with {type(num)}, expected `int`."
        if step == 0:
            return []

        return list(range(step, step + step * num, step))

    def _resolve_fov_combinations(self, t: int, time_steps) -> list[tuple[int, int]]:
        """Resolve the `(t_src, t_dst)` frame pairs to connect around the center frame.

        Shared by the on-the-fly and the precomputed path so both see the same window.
        `graph_fov_size` widens the message-passing context by adding anchor frames around
        the center, spaced by the sampled step (not contiguous frames, which would add
        isolated frames for large steps). Every anchor contributes its full edges, so each
        anchor keeps the same maximum stride and the field of view grows with the anchor
        count. No added anchors leaves only the center frame (the base behavior).
        """
        fov_hops = (
            0
            if self.graph_fov_size is None
            else max(0, math.ceil(self.graph_fov_size * (len(time_steps) or 1) - 0.5))
        )
        step = min(time_steps) if time_steps else 1
        return [
            pair
            for k in range(-fov_hops, fov_hops + 1)
            for pair in self.edge_finder._resolve_timesteps(
                t + k * step, time_steps, self.graph_connectivity
            )
        ]

    def _resolve_edge_pairs(self, t: int, time_steps) -> set[tuple[int, int]]:
        """Resolve the `(t_src, dist_temp)` frame pairs reachable from the center frame.

        Out-of-bounds frame pairs are substituted with same-stride in-bounds pairs (see
        `EdgeFinder.resolve_boundary_pairs`), matching the on-the-fly edge construction so
        the cached base strides cover every request.
        """
        candidates = self._resolve_fov_combinations(t, time_steps)
        resolved = self.edge_finder.resolve_boundary_pairs(
            candidates, self.num_frames, self.edge_finder.bidirectional
        )

        pairs: set[tuple[int, int]] = set()
        for t_src, t_dst in resolved:
            pairs.add((t_src, t_dst - t_src))
            if self.edge_finder.bidirectional:
                pairs.add((t_dst, t_src - t_dst))
        return pairs

    def _frame_pairs(self, key: tuple[int, tuple[int, ...]]) -> list[tuple[int, int]]:
        """The `(t_src, dist_temp)` pairs an item reads, memoized per `(t, time_steps)`.

        Forward pairs only under `_mirror_per_item`, since the backward half of the window
        is mirrored from them after the lookup.
        """
        if key not in self._graph_pairs_cache:
            t, time_steps = key
            self._graph_pairs_cache[key] = [
                (t_src, dt)
                for t_src, dt in self._resolve_edge_pairs(t, list(time_steps))
                if not self._mirror_per_item or dt > 0
            ]
        return self._graph_pairs_cache[key]

    def _index_edges_by_frame_pair(self) -> None:
        """Index an in-memory edge table by `(t, dist_temp)` so items can slice it.

        Without this every `__getitem__` filters the whole table once per frame pair.
        Sorting once makes each pair a contiguous range, so an item concatenates zero-copy
        slices instead. A `LazyFrame` keeps the filter, whose predicate pushdown does the
        same job against the parquet store.
        """
        if not isinstance(self.edge_data, pl.DataFrame):
            self._edge_slices = None
            return

        self.edge_data = self.edge_data.sort("t", "dist_temp")
        counts = self.edge_data.group_by(["t", "dist_temp"], maintain_order=True).len()
        slices: dict[tuple[int, int], tuple[int, int]] = {}
        offset = 0
        for t, dist_temp, height in counts.iter_rows():
            slices[(t, dist_temp)] = (offset, height)
            offset += height
        self._edge_slices = slices

    def _slice_edges_for_pairs(self, pairs) -> pl.DataFrame:
        """Concatenate the indexed `(t, dist_temp)` ranges for `pairs`, empty if none."""
        assert self._edge_slices is not None and isinstance(self.edge_data, pl.DataFrame)
        parts = [
            self.edge_data.slice(*self._edge_slices[key])
            for key in pairs
            if key in self._edge_slices
        ]
        if not parts:
            return self.edge_data.clear()
        return parts[0] if len(parts) == 1 else pl.concat(parts)

    def _build_edge_filter(self, t: int, time_steps) -> pl.Expr | None:
        """Compile the frame pairs of `(t, time_steps)` into one predicate.

        Only needed where the edges are scanned rather than sliced, so that the parquet
        store can push the predicate down.
        """
        pairs = self._frame_pairs((t, tuple(time_steps)))
        if not pairs:
            return None

        # any_horizontal over per-pair equality conditions is faster than a join for the
        # small number of (t, dist_temp) pairs produced here (<20).
        conds = [
            (pl.col("t") == t_src) & (pl.col("dist_temp") == dt) for t_src, dt in pairs
        ]
        return pl.any_horizontal(*conds)

    def _get_graph_for_frame(self, t, time_steps) -> pl.DataFrame | None:
        """Get graph around single frame.

        Args:
            t: Center frame.
            time_steps: Time steps to connect nodes, e.g., [1, 2, 3]. Positive values
                will be mirrored if `self.edge_finder.bidirectional`.

        Returns:
            Edge dataframe and features or None if no edges exist.
        """
        assert isinstance(t, int) and isinstance(step := next(iter(time_steps)), int), (
            f"Expected integers. Got {t=} and type(time_steps={type(step)})"
        )

        # the patches of a frame share one lookup. not during training, where the radius
        # is resampled per item and two items of the same frame differ
        memo_key = (t, tuple(time_steps))
        if not self.training and self._frame_edge_memo is not None:
            cached_key, cached_edges = self._frame_edge_memo
            if cached_key == memo_key:
                return cached_edges

        if self.precompute_edges:
            assert self.edge_data is not None and isinstance(
                self.edge_data, pl.DataFrame | pl.LazyFrame
            )

            pairs = self._frame_pairs(memo_key)
            if not pairs:
                self._frame_edge_memo = (memo_key, None)
                return None

            # spatial augmentation: sample a smaller spatial radius per item in training
            radius_filter = (
                pl.col("dist_spat") <= self._resolve_param(self.graph_search_radius)
                if self._radius_aug and self.training
                else None
            )

            # an indexed in-memory table slices the frame pairs directly; otherwise fall
            # back to filtering, which is what a LazyFrame wants anyway (pushdown)
            if self._edge_slices is not None:
                selected = self._slice_edges_for_pairs(pairs)
                if radius_filter is not None:
                    selected = selected.filter(radius_filter)
                edge_data = (
                    pl.concat([selected, self.edge_finder.mirror_edges(selected)])
                    if self._mirror_per_item
                    else selected
                )
            else:
                if memo_key not in self._graph_filter_cache:
                    self._graph_filter_cache[memo_key] = self._build_edge_filter(
                        t, time_steps
                    )
                predicate = self._graph_filter_cache[memo_key]
                assert predicate is not None  # non-empty pairs always compile
                edge_data = self.edge_data.filter(predicate)
                if radius_filter is not None:
                    edge_data = edge_data.filter(radius_filter)
                edge_data = collect(edge_data)
                if self._mirror_per_item:
                    edge_data = pl.concat(
                        [edge_data, self.edge_finder.mirror_edges(edge_data)]
                    )

        else:
            # CAVEAT: Here, returns minimum radius instead of maximum if used with
            #         validation. Warns, so should be fine.
            search_radius = self._resolve_param(self.graph_search_radius)
            assert isinstance(search_radius, int)

            edge_data = self.edge_finder.get_edges_for_frame(
                self.node_feats,
                t,
                search_radius=search_radius,
                time_steps=time_steps,
                masks=self.masks,
                connectivity=self.graph_connectivity,
                mirror_at_last_frames=True,
                combinations=self._resolve_fov_combinations(t, time_steps),
            )

        assert edge_data is None or isinstance(edge_data, pl.DataFrame)
        self._frame_edge_memo = (memo_key, edge_data)
        return edge_data

    def _get_deep_features(self, node_data: pl.DataFrame) -> torch.Tensor | None:
        if self.deep_feature_extractor is None:
            return

        if self.deep_feats is not None:
            return self.deep_feats[node_data["index"]]  # type: ignore

        # the cache was built for the whole sequence in `_build_deep_cache`, so this only
        # ever reads it back
        return _run_deep_extractor(
            self.deep_feature_extractor,
            image=self.images,
            masks=self.masks,
            node_feats=node_data,
            output_dir=self.deep_feat_dir,
            image_percentiles=self.image_percentiles,
            spacing=self._spatial_spacing,
            timepoints=np.unique(node_data["t"]),
            dataset_meta=self.dataset_meta,
        )

    def _get_node_features(
        self, node_data: pl.DataFrame
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get handcrafted and deep single-cell node features.

        Features are normalized and converted to tensors.

        Args:
            node_data: Node positions and raw handcrafted features.

        Returns:
            Handcrafted and/or deep node features. If no features are extracted, the item
            returns None.
        """
        x_handcrafted = (
            self.handcrafted_feature_extractor._transform(node_data)
            if self.handcrafted_feature_extractor is not None
            else None
        )
        x_deep = self._get_deep_features(node_data)

        return x_handcrafted, x_deep

    def _get_edge_features(
        self, edge_data: pl.DataFrame, node_feats: pl.DataFrame
    ) -> torch.Tensor:
        """Get normalized edge features as a tensor.

        Args:
            edge_data: Edge features containing indices of connected nodes (`src`,
                `dst`) and their relational features.
            node_feats: Position (tyx) and handcrafted single-cell features.
        """
        edge_attr = self.edge_finder._transform(
            edge_data, node_feats, spacing_t=self.spacing["t"]
        )
        # can be none if the edge finder is only used for positions
        assert edge_attr is not None
        return edge_attr

    def _maybe_sample_edges(self, edge_data: pl.DataFrame) -> pl.DataFrame:
        """Cap number of edges during training.

        If number of edges exceeds `self.graph_max_num_edges`, randomly discards edges
        independent of their class. Only applied during training.
        """
        if self.graph_edge_dropout > 0 and self.training and edge_data.height > 10:
            edge_data = edge_data.sample(fraction=1 - self.graph_edge_dropout)

        if (
            self.training
            and (self.graph_max_num_edges != -1)
            and (self.graph_max_num_edges is not None)
            and (edge_data.height > self.graph_max_num_edges)
        ):
            logger.info(
                f"Batch had {edge_data.height} edges. "
                f"Limiting to {self.graph_max_num_edges} edges."
            )
            edge_data = edge_data.sample(self.graph_max_num_edges)

        return edge_data.sort("src", "dst")

    def _validate_and_drop_edges(
        self, edge_data: pl.DataFrame | None, t: int
    ) -> pl.DataFrame | None:
        """Validate edge data and drop missing trajectories."""
        min_edges = 2 if self.training else 1  # BN needs at least two edges
        if (edge_data is None) or (edge_data.height < min_edges):
            logger.warning(f"Could not find graph for frame {t}.")
            return None

        edge_data = edge_data.with_columns(
            drop=pl.col("src").is_in(self.missing_trajectories)
            | pl.col("dst").is_in(self.missing_trajectories)
        )
        if edge_data.filter(~pl.col("drop")).height < min_edges:
            logger.warning(f"Could not find annotated samples for frame {t}.")
            return None
        return edge_data

    def _construct_graph(self, edge_data: pl.DataFrame) -> Data:
        """Construct graph from edge data."""
        original_edge_data = edge_data  # retain for retry without augmentation

        # must be before edge filtering, otherwise isolated nodes are dropped
        node_data = self._load_and_remap_nodes(edge_data.select("src", "dst").to_numpy())
        node_mapping = node_data.select("index", "new_index")

        # sampling multiplicative: frames - nodes - edges, might drop too much
        #   e.g.: 0.1 ->     0.9    - 0.9^2 - 0.9  = 65% of edges kept
        #         0.3 ->     0.7    - 0.49  - 0.7  = 24% of edges kept
        edge_data_sampled = self._maybe_sample_edges(original_edge_data)
        edge_data_remapped = self._remap_edges(edge_data_sampled, node_mapping).sort(
            "src_new", "dst_new"
        )

        min_edges = 2 if self.training else 1  # BN needs at least two edges
        _augmentation_active = self.training and (
            self.graph_frame_dropout > 0.0
            or self.graph_node_dropout > 0.0
            or self.graph_edge_dropout > 0.0
        )

        if (
            edge_data_remapped.height < min_edges or node_data.height < min_edges
        ) and _augmentation_active:
            # dropout caused the graph to fall below minimum, retry without augmentation
            # so genuine data is not discarded as an empty template graph
            logger.debug(
                f"Dropout reduced graph below minimum edges ({min_edges}). "
                f"Retrying without augmentation for sequence {self.sequence_id}."
            )
            node_data = self._load_and_remap_nodes(
                original_edge_data.select("src", "dst").to_numpy(),
                skip_augmentation=True,
            )
            node_mapping = node_data.select("index", "new_index")
            edge_data_remapped = self._remap_edges(original_edge_data, node_mapping).sort(
                "src_new", "dst_new"
            )

        if edge_data_remapped.height < min_edges or node_data.height < min_edges:
            return Data(**self.template_graph)  # genuinely empty even without dropout

        edge_data = edge_data_remapped  # rebind for remainder of method

        edge_index = col_to_tensor(
            edge_data, ["src_new", "dst_new"], dtype=torch.long, transpose=True
        )
        node_index = col_to_tensor(node_data, "new_index", dtype=torch.long)
        node_index_orig = col_to_tensor(node_data, "index", dtype=torch.long)

        if edge_data["y"].null_count() > 0:
            raise ValueError(
                f"Edge label column 'y' contains {edge_data['y'].null_count()} null "
                "values, which indicates a bug in edge labeling."
            )
        edge_y = col_to_tensor(edge_data, "y", dtype=torch.long)
        node_y = col_to_tensor(node_data, "y", dtype=torch.long)
        edge_ignore = col_to_tensor(edge_data, "drop", dtype=torch.bool)

        x_handcrafted, x_deep = self._get_node_features(node_data)
        edge_attr = self._get_edge_features(edge_data, self.node_feats)

        graph = Data(
            x_handcrafted=x_handcrafted,
            x_deep=x_deep,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # name can't contain "index" or it will be remapped by pyg dataloader
            node_mapping=node_index_orig,
            node_index=node_index,
            num_nodes=len(node_index),
            data_source=self.data_source,
            y_edges=edge_y,
            y_nodes=node_y,
            edge_ignore=edge_ignore,
            empty_flag=False,
        )

        return graph

    def __getitem__(self, idx: int) -> Data:  # type: ignore
        """Get final graph data for single frame.

        Args:
            idx: Frame.

        Returns:
            Pytorch Geometric graph around requested time point. If no edges are found,
            returns an empty graph.
        """
        edge_data = self._get_graph_for_frame(
            idx, self.time_steps(return_all_combinations=False)
        )

        edge_data = self._validate_and_drop_edges(edge_data, idx)
        if edge_data is None:
            return Data(**self.template_graph)

        return self._construct_graph(edge_data)


class GraphPatchDataset(GraphDataset):
    """`GraphDataset` variant that splits each frame into spatial patches.

    Instead of returning a subgraph for an entire frame, each `__getitem__` call returns
    the subgraph for a single spatial patch. Patches are defined by a regular grid over
    the image with configurable size and overlap. Nodes are assigned to the patch that
    contains their centroid, edges connect nodes across patches as usual.

    This is useful for very large images where the full-frame graph would exceed memory
    or edge count limits. Indexing is over `(frame, patch)` pairs rather than frames
    alone, so `__len__` returns the total number of patches across all frames.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, ...] = 256,
        patch_overlap: float = 0.5,
        **kwargs,
    ):
        """Initialize dataset.

        Args:
            patch_size: Size of patches taken from single frames.
            patch_overlap: Overlap between neighboring patches.
            kwargs: Args passed to `GraphDataset`.
        """
        # maybe warn if `precompute_graph=False`, since this for many sequences will
        # produce empty batches (only an issue for multi-gpu training, so maybe warn
        # within `TrackingModel.setup()` if self.world_size>1).
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_indices = self._get_patches()

    def __len__(self):
        """Number of patches."""
        return len(self.patch_indices)

    def _get_patches(self) -> list[tuple[int, tuple[int, ...]]]:
        size = self.patch_size
        overlap = self.patch_overlap
        center_name = getattr(self.edge_finder, "center_name", "centroid")

        # patches only need to cover nodes that have an edge. both endpoints count: a
        # disk-backed store holds forward edges only, so last-frame nodes appear as 'dst'
        # exclusively and would otherwise never land in a patch.
        nodes = self.node_feats
        if (edge_full := getattr(self, "edge_data", None)) is not None:
            connected = collect(
                pl.concat(
                    [
                        edge_full.select(pl.col("src").alias("index")),
                        edge_full.select(pl.col("dst").alias("index")),
                    ]
                ).unique()
            )["index"]
            nodes = nodes.filter(
                pl.col("index").is_in(connected.implode())
                & ~pl.col("index").is_in(self.missing_trajectories)
            )

        frames_by_t = nodes.partition_by("t", as_dict=True)

        out = []
        for t in self.valid_frames:
            frame = frames_by_t.get((int(t),))
            if frame is None:
                continue
            coords = frame.select(rf"^{center_name}-\d$").to_numpy()
            if coords.size == 0:
                continue

            n_dim = coords.shape[1]
            sizes = list(size) if not isinstance(size, int) else [size] * n_dim

            starts = []
            for s, min_c, max_c in zip(
                sizes, coords.min(0).astype(int), coords.max(0).astype(int), strict=True
            ):
                if max_c - min_c <= s:
                    starts.append([min_c])
                else:
                    step = max(1, int(s * (1 - overlap)))
                    dim_starts = list(range(min_c, max_c - s + 1, step))
                    if dim_starts[-1] + s < max_c:
                        dim_starts.append(max_c - s)
                    starts.append(dim_starts)

            for origin in itertools.product(*starts):
                end = tuple(o + s for o, s in zip(origin, sizes, strict=True))
                if not np.all((coords >= origin) & (coords <= end), axis=1).any():
                    continue
                out.append((int(t), origin + end))

        return out

    def __getitem__(self, idx: int):
        """Get final data for single patch.

        Args:
            idx: Patch index.
        """
        t, patch_bbox = self.patch_indices[idx]
        edge_data = self._get_graph_for_frame(
            t, self.time_steps(return_all_combinations=False)
        )

        if edge_data is not None:
            center_name = getattr(self.edge_finder, "center_name", "centroid")

            # check if center is within nD bounding box
            n_dim = len(patch_bbox) // 2
            conditions = [
                pl.col(f"{center_name}-{i}").is_between(
                    patch_bbox[i], patch_bbox[i + n_dim]
                )
                for i in range(n_dim)
            ]
            valid_indices = (
                self.node_feats.filter(*conditions).select("index")["index"].implode()
            )
            edge_data = edge_data.filter(
                pl.col("src").is_in(valid_indices) | pl.col("dst").is_in(valid_indices)
            )

        edge_data = self._validate_and_drop_edges(edge_data, t)
        if edge_data is None:
            return Data(**self.template_graph)

        return self._construct_graph(edge_data)


class TrackingDataset(L.LightningDataModule):
    """Lightning DataModule that manages multiple `GraphDataset` instances.

    Orchestrates the full data lifecycle for training, validation, testing, and
    prediction:

    1. `prepare_data()`: Runs feature extraction (handcrafted and deep) for every
       sequence and writes the parquet and embedding caches to `feature_dir`. Validates
       the existing feature cache and either recomputes or skips for sequences where a
       valid cache already exists.
    2. `setup(stage)`: Loads cached data and instantiates `GraphDataset` per sequence per
       phase. For training, datasets are concatenated into a single `ConcatDataset`.
       Sequences are split into train, val, and test based on `fold` and a splits
       file stored alongside the data. For prediction, all requested sequences are used.
    3. `train_dataloader()`, `val_dataloader()`, and others: return configured
       `DataLoader` instances with appropriate worker counts, samplers, and
       multiprocessing context (`forkserver` on Unix, `spawn` on Windows).

    Most constructor arguments that vary per sequence (`data_dir`, `segmentation_name`,
    `data_format`, `img_name`) accept either a single value (broadcast to all
    sequences) or a list aligned with `sequence_ids`. This allows combining
    heterogeneous data sources in a single training run, e.g.,
    `data_dir=['toiam', 'spores']` and `sequences=['01', '01']` would load sequence
    '01' for the 'toiam' and 'spores' datasets.

    Graph construction parameters (`graph_search_radius`, `graph_time_step`, etc.) are
    forwarded to every `GraphDataset` and behave identically.
    """

    def __init__(
        self,
        data_dir: Path | str | list[Path | str],
        feature_dir: Path | str | list[Path | str],
        sequence_ids: list[str],
        edge_finder: EdgeFinder,
        handcrafted_feature_extractor: HandcraftedExtractor,
        deep_feature_extractor: CellLevelExtractor | None = None,
        segmentation_name: str | list[str] = "GT",
        data_format: str | list[str] = "ctc",
        img_name: str | None | list[str | None] = None,
        graph_search_radius: int | str | tuple[int, int] | tuple[int, int, int] = 200,
        graph_time_step: int | tuple[int, int] | tuple[int, int, int] = 1,
        graph_num_steps: int = 3,
        graph_connectivity: Literal["dense", "star", "sequential"] = "dense",
        graph_fov_size: int | float | None = None,
        graph_max_num_edges: int = 500_000,
        graph_edge_dropout: float = 0.0,
        graph_node_dropout: float = 0.0,
        graph_frame_dropout: float = 0.0,
        precompute_edges: bool = False,
        fold: int | None = 0,
        use_patches: bool = False,
        patch_size: int | tuple[int, ...] = 256,
        patch_overlap: float = 0.5,
        lazy: bool = True,
        spacing: SpacingLike = None,
        trust_cache: bool = False,
        **kwargs,
    ):
        """Initialize the data module.

        Args:
            data_dir: Root directory containing image and mask data.
            feature_dir: Base directory for cached features (node parquet, edge parquet,
                deep feature embeddings). Subdirectories are resolved automatically per
                sequence and segmentation.
            sequence_ids: Sequence IDs to operate on, e.g., ['01', '02'] or
                ['train', 'val'].
            edge_finder: See `GraphDataset`.
            handcrafted_feature_extractor: See `GraphDataset`.
            deep_feature_extractor: See `GraphDataset`.
            segmentation_name: See `GraphDataset`.
            data_format: See `GraphDataset`.
            img_name: See `GraphDataset`.
            graph_search_radius: See `GraphDataset`.
            graph_time_step: See `GraphDataset`.
            graph_num_steps: See `GraphDataset`.
            graph_connectivity: See `GraphDataset`.
            graph_fov_size: See `GraphDataset`.
            graph_max_num_edges: See `GraphDataset`.
            graph_edge_dropout: See `GraphDataset`.
            graph_node_dropout: See `GraphDataset`.
            graph_frame_dropout: See `GraphDataset`.
            precompute_edges: See `GraphDataset`.
            fold: Cross-validation fold index. Used to select the train, val, and test
                split from the splits file.
            use_patches: If `True`, split large spatial fields into overlapping patches
                before graph construction.
            patch_size: Spatial patch size (pixels) when `use_patches=True`.
            patch_overlap: Fractional overlap between adjacent patches.
            lazy: Use `pl.LazyFrame` for the edge data. Slower per query but much more
                memory-efficient for large sequences.
            spacing: Physical frame and voxel spacing forwarded to every `GraphDataset`
                (see `GraphDataset` for details).
            trust_cache: If `True`, skip hashing images/masks during `prepare_data()` and
                trust an existing cache whose name and structural metadata match,
                forwarded to every `GraphDataset` too (see `GraphDataset` for details).
            **kwargs: Dataloader and sequence-sampling settings, rejected if unknown.
                `batch_size`, `num_workers`, `shuffle`, `prefetch_factor`,
                `persistent_workers` and `pin_memory` configure the loaders,
                `sampling_alpha`, `sampling_floor` and `sampling_size_measure` how
                sequences are drawn. A `_val` suffix applies only to val, test, and
                predict (`batch_size_val`, `num_workers_val`, `num_workers_test`,
                `stride_val`), including for three graph params
                (`graph_search_radius_val`, `graph_time_step_val`,
                `graph_num_steps_val`). Unsuffixed keys apply to training.
        """
        super().__init__()

        self.fold = fold
        assert len(sequence_ids) > 0, "Please provide sequence ids."

        self.sequence_ids = sequence_ids
        self.data_dirs = self._resolve_list(data_dir, is_path=True)
        self.feature_dirs = self._resolve_list(feature_dir, is_path=True)
        self.segmentation_names = self._resolve_list(segmentation_name)
        self.data_formats = self._resolve_list(data_format)
        self.img_names = self._resolve_list(img_name)
        self.data_params = (
            self.sequence_ids,
            self.data_dirs,
            self.feature_dirs,
            self.segmentation_names,
            self.data_formats,
            self.img_names,
        )

        self.edge_finder = edge_finder
        self.hc_feat_extractor = handcrafted_feature_extractor
        self.deep_feat_extractor = deep_feature_extractor

        self.lazy = lazy
        self.precompute_edges = precompute_edges
        self.graph_search_radius = graph_search_radius
        self.graph_time_step = graph_time_step
        self.graph_num_steps = graph_num_steps
        self.graph_connectivity = graph_connectivity
        self.graph_max_num_edges = graph_max_num_edges
        self.graph_edge_dropout = graph_edge_dropout
        self.graph_node_dropout = graph_node_dropout
        self.graph_frame_dropout = graph_frame_dropout
        self.graph_fov_size = graph_fov_size

        self.spacing = spacing
        self.trust_cache = trust_cache

        self.use_patches = use_patches
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # number of positions, not frames per batch
        self.batch_size = kwargs.pop("batch_size", 1)
        self.batch_size_val = kwargs.pop("batch_size_val", self.batch_size)
        self.graph_val_overrides = {
            param: kwargs.pop(f"{param}_val")
            for param in ("graph_search_radius", "graph_time_step", "graph_num_steps")
            if f"{param}_val" in kwargs
        }
        self.num_workers = kwargs.pop("num_workers", 0)
        self.num_workers_val = kwargs.pop("num_workers_val", min(self.num_workers, 2))
        self.num_workers_test = kwargs.pop("num_workers_test", min(self.num_workers, 2))
        self.stride_val = kwargs.pop("stride_val", 1)
        self.shuffle = kwargs.pop("shuffle", None)  # if None will only shuffle train
        # sample sequences with probability proportional to size ** sampling_alpha
        # (alpha 0 = equal per sequence, 1 = proportional to size), floored at
        # sampling_floor. size is measured by sampling_size_measure.
        self.sampling_alpha = kwargs.pop("sampling_alpha", 0.5)
        self.sampling_floor = kwargs.pop("sampling_floor", 0.02)
        self.sampling_size_measure = kwargs.pop("sampling_size_measure", "frames")
        self.prefetch_factor = kwargs.pop("prefetch_factor", 2)
        self.persistent_workers = kwargs.pop("persistent_workers", self.num_workers > 0)
        self.pin_memory = kwargs.pop("pin_memory", torch.cuda.is_available())
        # a mistyped key in a dataset yaml would otherwise train at the default value
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        self.multiprocessing_context = default_multiprocessing_context()
        self.prepare_data_per_node = False

        # record info on cache (e.g., creation time  or if it was created from scratch)
        # this will be saved within `features.json` for debug and validation
        self.cache_records: list[dict] = []
        # per-sequence dataset_identity(), also saved within `features.json`
        self.dataset_metas: dict[str, dict] = {}

    def _resolve_list(self, it, is_path=False):
        if isinstance(it, list | tuple | ListConfig):
            if is_path:
                return [Path(fp) for fp in it]
            return list(it)

        n = len(self.sequence_ids)
        if is_path:
            return [Path(it)] * n
        return [it] * n

    def _requested_time_steps(self) -> list[int]:
        """Temporal strides the edge store must cover for the configured graph params.

        Independent of `graph_fov_size`: fov only adds anchors within the same strides,
        and boundary substitution stays on these strides, so the cache is fov-invariant.
        """
        step = expand_param_range(self.graph_time_step)

        nums = self.graph_num_steps
        if isinstance(nums, int):
            nums = [nums]
        elif isinstance(nums, list | tuple) and len(nums) in (2, 3):  # range
            nums = expand_param_range(nums)
        elif isinstance(nums, ListConfig):  # explicit list of num_steps values
            nums = list(nums)

        time_steps = set()
        for ti in step:
            for ni in nums:
                if ni > 0:
                    time_steps.update(list(range(ti, ti + ti * ni, ti)))
        return sorted(time_steps)

    def prepare_data(
        self,
    ) -> None:
        """Extract and save features."""
        # prepare_data impacts random states, so we restore rng after running. not
        # verified if this works, so benchmarks are either run without caching or all must
        # use the same cache.
        self.cache_records = []
        self.dataset_metas = {}
        with isolate_rng():
            total_seqs = len(self.sequence_ids)
            logger.debug(f"Preparing data for {total_seqs} sequences.")
            logger.debug(f"{self.sequence_ids} for {self.data_dirs}.")

            pbar = tqdm(
                zip(*self.data_params, strict=True),
                total=total_seqs,
                position=0,
                leave=True,
            )
            for seq_id, data_dir, feat_dir, seg, data_fmt, img_name in pbar:
                pbar.set_description(
                    f"Preparing data for {total_seqs} sequences ({seq_id})"
                )

                hc_dir = _get_seg_dir(feat_dir, seq_id, seg)
                hc_dir.mkdir(exist_ok=True, parents=True)

                require_lineage = seg in ["GT", "ST"]
                images, masks, percentiles = load_images_and_masks(
                    data_dir,
                    seq_id,
                    data_format=data_fmt,
                    lazy=True,
                    return_percentiles=True,
                    segmentation_name=seg,
                    img_name=img_name,
                    percentile_file=hc_dir / "percentiles.json",
                )

                if require_lineage:
                    lin_file, has_states = find_lineage_file(
                        data_dir=data_dir,
                        seq_id=seq_id,
                        data_format=data_fmt,
                        segmentation_name=seg,
                        with_states=True,
                    )
                    logger.debug(
                        f"Loading lineage file with{'' if has_states else 'out'} from: "
                        f"{lin_file}."
                    )
                else:
                    lin_file, has_states = None, False

                dataset_meta = (
                    None
                    if self.trust_cache
                    else dataset_identity(
                        images, masks, data_dir=data_dir, sequence_id=seq_id
                    )
                )
                self.dataset_metas[seq_id] = dataset_meta  # type: ignore

                hc_file = hc_dir / "nodes.parquet"
                logger.debug(f"Trying to load handcrafted features from {hc_file}.")
                hc_sig = cache_signature(hc_file)
                spatial = spatial_spacing(normalize_spacing(self.spacing), masks.ndim - 1)
                node_feats = self.hc_feat_extractor(
                    images,
                    masks,
                    lineage_file=lin_file,
                    filepath=hc_file,
                    sequence_id=seq_id if has_states else None,
                    validate=True,
                    image_percentiles=percentiles,
                    spacing=spatial,
                    dataset_meta=dataset_meta,
                )
                self.cache_records.append(
                    classify_cache("handcrafted", seq_id, hc_file, hc_sig)
                )

                # cache complete lineage GT graph for validation and debug
                if require_lineage and lin_file is not None:
                    gt_edges_file = hc_dir / "lineage_gt_edges.parquet"
                    build_lineage_gt_edges(node_feats).write_parquet(gt_edges_file)

                if self.deep_feat_extractor:
                    deep_dir = _get_deep_dir(
                        self.deep_feat_extractor, feat_dir, seq_id, seg
                    )
                    logger.debug(f"Trying to load deep features from {deep_dir}.")
                    deep_sig = cache_signature(deep_dir)
                    _run_deep_extractor(
                        self.deep_feat_extractor,
                        image=images,
                        masks=masks,
                        node_feats=node_feats,
                        output_dir=deep_dir,
                        image_percentiles=percentiles,
                        spacing=spatial,
                        dataset_meta=dataset_meta,
                    )
                    self.cache_records.append(
                        classify_cache("deep", seq_id, deep_dir, deep_sig)
                    )

                search_radius = max_search_radius(self.graph_search_radius, node_feats)
                time_steps = self._requested_time_steps()

                edge_file = (
                    _get_seg_dir(feat_dir, seq_id, seg) / self.edge_finder.cache_dirname()
                )
                logger.debug(f"Trying to load edges from {edge_file}.")
                edge_sig = cache_signature(edge_file)
                self.edge_finder(
                    node_feats=node_feats,
                    search_radius=search_radius,
                    time_steps=time_steps,
                    masks=masks,
                    filepath=edge_file,
                    overwrite=False,
                    connectivity="star",
                    validate=True,
                    dataset_meta=dataset_meta,
                )
                self.cache_records.append(
                    classify_cache("edges", seq_id, edge_file, edge_sig)
                )

            # unload encoder from GPU so it is not pickled with CUDA tensors when
            # DataLoader workers are spawned via forkserver
            if self.deep_feat_extractor and hasattr(
                self.deep_feat_extractor, "_unload_encoder"
            ):
                self.deep_feat_extractor._unload_encoder()

    def _load_splits(self) -> list[dict[Literal["train", "val", "test"], list[str]]]:
        """Load CV splits for each configured sequence instance.

        Returns:
            One split dictionary per entry in `self.sequence_ids`, in the same order.
        """
        splits = []
        # cache loaded yamls to prevent re-reading file for every sequence in the same dir
        loaded_yamls = {}

        for data_dir in self.data_dirs:
            # data_dir is a Path object from _resolve_list
            if data_dir not in loaded_yamls:
                split_file = data_dir / "splits.yaml"
                if split_file.exists():
                    with split_file.open("r") as fp:
                        # load the specific fold immediately
                        loaded_yamls[data_dir] = yaml.safe_load(fp)[self.fold]
                else:
                    raise FileNotFoundError(f"Splits file not found at {split_file}.")

            splits.append(loaded_yamls[data_dir])

        return splits

    def _load_dataset(
        self,
        splits: list[dict[Literal["train", "val", "test"], list[str]]] | None,
        phase: Literal["train", "val", "test", "predict"],
    ) -> list[Dataset] | Dataset | ConcatDataset:
        """Instantiate graph datasets for single phase.

        Args:
            splits: List of split dicts corresponding to self.data_params.
                If None (e.g. predict), all sequences are loaded.
            phase: Phase to create datasets for.
        """
        datasets = []
        splits_iter = splits if splits is not None else [None] * len(self.sequence_ids)
        DatasetClass = GraphPatchDataset if self.use_patches else GraphDataset

        # items: seq_id, data_dir, feat_dir, seg, data_fmt, img_name, split
        items = list(zip(*self.data_params, splits_iter, strict=True))
        filtered_items = [
            item for item in items if (phase == "predict") or (item[0] in item[-1][phase])
        ]

        if not filtered_items:
            raise ValueError(
                f"Could not find any valid datasets with the current data config:\n"
                f"{self.data_params}\n{splits}"
            )

        iterator = tqdm(filtered_items, desc=f"Loading Dataset ({phase})")
        for seq_id, data_dir, feat_dir, seg, data_fmt, img_name, _split in iterator:
            kwargs = {
                "data_dir": data_dir,
                "feature_dir": feat_dir,
                "sequence_id": seq_id,
                "edge_finder": self.edge_finder,
                "handcrafted_feature_extractor": self.hc_feat_extractor,
                "deep_feature_extractor": self.deep_feat_extractor,
                "segmentation_name": seg,
                "data_format": data_fmt,
                "img_name": img_name,
                "graph_search_radius": self.graph_search_radius,
                "graph_time_step": self.graph_time_step,
                "graph_num_steps": self.graph_num_steps,
                "graph_connectivity": self.graph_connectivity,
                "graph_fov_size": self.graph_fov_size,
                "graph_max_num_edges": self.graph_max_num_edges,
                "graph_edge_dropout": self.graph_edge_dropout,
                "graph_node_dropout": self.graph_node_dropout,
                "graph_frame_dropout": self.graph_frame_dropout,
                "training": phase == "train",
                "precompute_edges": self.precompute_edges,
                "lazy": self.lazy,
                "spacing": self.spacing,
                # validation already done in prepare_data
                "trust_cache": True,
            }
            if phase != "train":
                # graph parameters may either be explicitly set or automatically chosen
                # from sampling range (minimum value or first item of tuple)
                kwargs.update(self.graph_val_overrides)

            if self.use_patches:
                kwargs["patch_size"] = self.patch_size
                kwargs["patch_overlap"] = self.patch_overlap

            datasets.append(DatasetClass(**kwargs))  # type: ignore

        if len(datasets) == 0:
            raise ValueError(
                f"Could not find any valid datasets with the current data config:\n"
                f"{self.data_params}\n{splits}"
            )

        if len(datasets) > 1:
            return ConcatDataset(datasets) if phase == "train" else datasets

        return datasets[0]

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:  # type: ignore
        """Setup datasets and splits.

        Args:
            stage: Stage of lightning trainer that calls setup (e.g., trainer.fit,
                trainer.validate, etc.).
        """
        workers = {
            "fit": max(self.num_workers, self.num_workers_val),
            "validate": self.num_workers_val,
            "test": self.num_workers_test,
            "predict": self.num_workers,
        }.get(stage, 0)

        # multi-threading inside workers will deadlock, so to keep things safe we disable
        # it. should also work when enabled if features are precomputed (won't initialize)
        if workers > 0:
            self.hc_feat_extractor.n_jobs = 1
            self.edge_finder.n_jobs = 1
        if not self.precompute_edges:
            # on-the-fly-graph construction does not benefit from multi-threading in mast
            # cases. generally, we want to only use multi-threading in prepare_data or in
            # GraphDataset.__init__ (when not using workers)
            self.edge_finder.n_jobs = 1

        if stage in ("fit", "validate", "test"):
            splits = self._load_splits()

        if stage == "fit":
            self.dataset_train = self._load_dataset(splits, "train")
            self.dataset_val = self._load_dataset(splits, "val")
        elif stage == "validate":
            self.dataset_val = self._load_dataset(splits, "val")
        elif stage == "test":
            self.dataset_test = self._load_dataset(splits, "test")
        elif stage == "predict":
            # pass None for splits to ignore split logic and load everything
            self.dataset_pred = self._load_dataset(None, "predict")

        self._print_dataset_summary(stage)

    def _print_dataset_summary(self, stage: str) -> None:
        """Print key statistics for datasets configured in the current stage."""
        datasets_to_print = []
        if stage == "fit":
            datasets_to_print = [("train", self.dataset_train), ("val", self.dataset_val)]
        elif stage == "validate":
            datasets_to_print = [("val", self.dataset_val)]
        elif stage == "test":
            datasets_to_print = [("test", self.dataset_test)]
        elif stage == "predict":
            datasets_to_print = [("predict", self.dataset_pred)]

        logger.debug(f"Dataset config:\n{self}")

        output = ["Dataset Summary:"]

        for phase, ds in datasets_to_print:
            ds_list: list[GraphDataset] = (  # type: ignore
                ds.datasets
                if isinstance(ds, ConcatDataset)
                else ([ds] if not isinstance(ds, list) else ds)
            )

            num_seqs = len(ds_list)
            total_frames = sum(len(d) for d in ds_list)
            total_nodes = sum(d.node_feats.height for d in ds_list)

            edges_known = True
            total_edges = 0
            for d in ds_list:
                if (
                    getattr(d, "precompute_edges", False)
                    and getattr(d, "edge_data", None) is not None
                ):
                    try:
                        total_edges += collect(d.edge_data.select(pl.len())).item()  # type: ignore
                    except Exception:
                        edges_known = False
                else:
                    edges_known = False

            edges_str = f"{total_edges}" if edges_known else "?"
            output.append(
                f"- {phase.capitalize()} Dataset: {num_seqs} sequences, "
                f"{total_frames} frames, {total_nodes} nodes, {edges_str} edges"
            )

        output_str = "\n".join(output)
        logger.debug(output_str)
        print("\n", output_str, "\n")

    def _sequence_size(self, dataset: GraphDataset) -> float:
        """Sequence size under `sampling_size_measure`.

        'frames' uses the item count, 'cells' the node count, and 'divisions' the
        division edges, falling back to items when edges are not precomputed.
        """
        measure = self.sampling_size_measure
        if measure == "cells":
            return float(dataset.node_feats.height)
        if measure == "divisions":
            edge_data = getattr(dataset, "edge_data", None)
            if edge_data is None:
                return float(len(dataset))
            div_label = getattr(dataset.edge_finder, "div_label", 2)
            n_div = collect(edge_data.filter(pl.col("y") == div_label).select(pl.len()))
            count = float(n_div.item()) if n_div.height else 0.0
            # include sequences without divisions or very low number of samples
            return max(count, 1.0)
        return float(len(dataset))

    def _get_train_sampler(self, concat_dataset: ConcatDataset) -> WeightedRandomSampler:
        """Weighted sampler over sequences of different sizes.

        Sequences are sampled with probability proportional to `size ** sampling_alpha`
        (alpha 0 = equal per sequence, 1 = proportional to size), floored at
        `sampling_floor`.
        """
        subsets = cast("list[GraphDataset]", concat_dataset.datasets)
        lengths = np.array([len(d) for d in subsets], dtype=float)
        sizes = np.array([self._sequence_size(d) for d in subsets])

        probs = np.power(sizes, self.sampling_alpha)
        probs = probs / probs.sum()

        if self.sampling_floor > 0:
            probs = np.maximum(probs, self.sampling_floor)
            probs = probs / probs.sum()

        weights: list[float] = []
        for prob, length in zip(probs, lengths, strict=True):
            item_weight = prob / length if length > 0 else 0.0
            weights.extend([item_weight] * int(length))

        return WeightedRandomSampler(
            weights=weights, num_samples=len(concat_dataset), replacement=True
        )

    def _get_val_sampler(self, dataset: GraphDataset, step: int = 8):
        if step <= 1:
            return

        return SequentialSampler(list(range(0, len(dataset), step)))

    def _configure_dataloaders(
        self,
        dataset: Dataset | list[Dataset] | ConcatDataset[Dataset],
        phase: Literal["train", "val", "test", "predict"],
    ) -> DataLoader | list[DataLoader]:
        """Configure dataloaders for single phase.

        Args:
            dataset: Dataset passed to dataloader.
            phase: Phase of dataloader.

        Returns:
            A single dataloader when `dataset` is a single or concat dataset, or one
            dataloader per entry when `dataset` is a list.
        """
        return_multiple_loaders = isinstance(dataset, list)
        dataset_list = dataset if return_multiple_loaders else [dataset]

        workers = {
            "train": self.num_workers,
            "predict": self.num_workers,
            "val": self.num_workers_val,
            "test": self.num_workers_test,
        }
        num_workers = workers.get(phase, 0)

        sampler = None
        shuffle = phase == "train" if self.shuffle is None else self.shuffle
        if phase == "train" and isinstance(dataset, ConcatDataset):
            if shuffle:
                sampler = self._get_train_sampler(dataset)
                shuffle = False

        drop_last = phase == "train"
        dataloaders = [
            DataLoader(
                ds,  # type: ignore
                # all kwargs must be named for predict dataloader
                batch_size=self.batch_size if phase == "train" else self.batch_size_val,
                shuffle=shuffle,
                sampler=sampler
                if phase == "train"
                else self._get_val_sampler(ds, self.stride_val)  # type: ignore
                if phase == "val"
                else None,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
                persistent_workers=self.persistent_workers if num_workers > 0 else False,
                multiprocessing_context=self.multiprocessing_context
                if num_workers > 0
                else None,
            )
            for ds in dataset_list
        ]
        if return_multiple_loaders:
            return dataloaders

        return dataloaders[0]

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        dl = self._configure_dataloaders(self.dataset_train, phase="train")
        assert isinstance(dl, DataLoader)
        return dl

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create validation dataloader."""
        return self._configure_dataloaders(self.dataset_val, phase="val")

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create test dataloader."""
        return self._configure_dataloaders(self.dataset_test, phase="test")

    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create prediction dataloader."""
        return self._configure_dataloaders(self.dataset_pred, phase="predict")


def _resolve_feature_dir(
    feature_dir: Path | str | None,
) -> tuple[Path | None, tempfile.TemporaryDirectory | None]:
    """Resolve a feature directory, creating a temporary one for 'temp'.

    The caller has to keep the returned handle alive for as long as the cache is needed:
    the directory is removed once it is released. Placement follows `TMPDIR`.
    """
    if feature_dir is None:
        return None, None
    if isinstance(feature_dir, str) and feature_dir == "temp":
        handle = tempfile.TemporaryDirectory(prefix="baclct-features-")
        return Path(handle.name), handle
    return Path(feature_dir), None


def _run_deep_extractor(
    extractor: CellLevelExtractor,
    *,
    image,
    masks,
    node_feats: pl.DataFrame,
    output_dir: Path | None,
    image_percentiles: tuple[float, float] | None,
    spacing: tuple[float, ...] | None,
    timepoints: np.ndarray | None = None,
    validate: bool = False,
    dataset_meta: dict | None = None,
):
    """Call a deep feature extractor, assembling the kwargs it accepts."""
    kwargs: dict[str, Any] = {
        "image": image,
        "node_feats": node_feats,
        "output_dir": output_dir,
        "timepoints": timepoints,
        "image_percentiles": image_percentiles,
    }
    if isinstance(extractor, CellLevelExtractor):
        # validation is fully contained in dataset_meta
        kwargs["spacing"] = spacing
        kwargs["dataset_meta"] = dataset_meta
        if extractor.requires_masks:
            kwargs["masks"] = masks
    else:
        # other extractors might validate differently
        kwargs["validate"] = validate
    return extractor(**kwargs)
