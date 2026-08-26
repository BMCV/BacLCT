"""Utility class for training and using BacLCT.

Loads config, prepares datasets and models, and then can be used for:

    1. Training (`run_training()`). Requires `hydra` configs and uses caching.
    2. Tracking of a trained model on new data (`track()`). Loads a trained model, creates
       a graph and features (optionally cached), and tracks a single image sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import dask.array as da
import lightning as L
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from baclct.data.dataset import GraphDataset, TrackingDataset
from baclct.io import (
    clean_tracks_df,
    export_classification_results,
    export_combined_tracks,
    export_tracking_results_simple,
)
from baclct.models.lightning_model import TrackingModel
from baclct.tracking.postprocessing import (
    resolve_duplicate_predictions,
)
from baclct.tracking.tracker import BaseTracker, LAPTracker
from baclct.utils.config import (
    compose_package_config,
    instantiate_callbacks,
    instantiate_loggers,
    resolve_and_merge_configs,
    resolve_checkpoint,
    resolve_data_config,
)
from baclct.utils.data import (
    get_device,
    get_multiprocessing_context,
    resolve_accelerator,
)
from baclct.utils.feature_info import write_feature_info
from baclct.utils.logger import get_pylogger
from baclct.utils.pretrained import DEFAULT_MODEL, MODEL_SPECS, resolve_model_dir
from baclct.utils.progress import (
    LightningProgressCallback,
    ProgressCallback,
    report,
)
from baclct.utils.spacing import SpacingLike

logger = get_pylogger(__name__)

# pipeline is fully compatible with dask but decision is with user: warn above size
EDGE_MEMORY_HINT_BYTES = 1024**3


class BacLCT:
    """Unified entry point for model training and tracking."""

    # pre-trained models available by name. see `baclct.utils.pretrained` for the
    # registry and the download/cache logic.
    _MODEL_REGISTRY = MODEL_SPECS

    @staticmethod
    def _resolve_model_dir(model: str | Path, download: bool = True) -> Path:
        """Resolve a model name or path to an experiment directory.

        Accepts a path to an existing experiment directory or YAML config, or a registered
        model name. See `resolve_model_dir`.
        """
        return resolve_model_dir(model, download=download)

    @staticmethod
    def download_model(model: str, progress: ProgressCallback | None = None) -> Path:
        """Fetch a registered model, unless it is already available locally."""
        return resolve_model_dir(model, download=True, progress=progress)

    def __init__(
        self,
        config: DictConfig | str | Path | None = None,
        config_overrides: DictConfig | dict | list[str] | str | Path | None = None,
    ):
        """Initialize with a config source and any overrides.

        Nothing is resolved here: `track()` may name the model itself, and the packaged
        default config cannot be composed without a `dataset` and `task`.

        Args:
            config: Loaded config (OmegaConf or Hydra), a registered model name (e.g.
                'baclct_track'), path to an experiment directory, or path to a Hydra
                YAML config file. `None` composes the packaged default config, in which
                case `config_overrides` must supply `dataset` and `task`.
            config_overrides: Overrides applied on top of `config`, as a dotlist of
                'key=value' strings, a dict, a `DictConfig`, or a path to a YAML file.
                This is the single override channel for both training and tracking. Keys
                tied to the trained model (architecture, feature extractors, fold) are
                always restored from the original config, so a checkpoint cannot be
                paired with a feature setup it never saw.
        """
        self._config = config
        self._config_overrides = config_overrides
        self._experiment_dir: Path | None = None
        self._cfg: DictConfig | None = None
        self.dataset: GraphDataset | TrackingDataset | None = None
        self.model: TrackingModel | None = None
        self.edge_preds: pl.DataFrame | None = None
        self.node_preds: pl.DataFrame | None = None

    @property
    def cfg(self) -> DictConfig:
        """Config with overrides applied, resolved on first use."""
        if self._cfg is None:
            self._cfg = self._resolve_config(self._config)
        return self._cfg

    def _resolve_config(self, config: DictConfig | str | Path | None) -> DictConfig:
        """Resolve a config source, recording the experiment directory it came from.

        A model name or directory is a trained run, so the packaged inference defaults
        are applied to it. A `DictConfig` comes from a Hydra training job and is left as
        composed.
        """
        if config is None:
            overrides = self._config_overrides
            if overrides is not None and not isinstance(overrides, list):
                raise TypeError(
                    "Composing the packaged config needs `config_overrides` as a dotlist "
                    "(e.g. ['dataset=toiam', 'task=tracking']), so that config groups "
                    "can be selected."
                )
            dotlist = [str(item) for item in overrides] if overrides else []
            return compose_package_config(dotlist)

        if isinstance(config, DictConfig):
            return resolve_and_merge_configs(config, self._config_overrides)

        self._experiment_dir = BacLCT._resolve_model_dir(config)
        return resolve_and_merge_configs(
            self._experiment_dir, self._config_overrides, inference=True
        )

    def _instantiate_dataset(
        self, phase: Literal["train", "val", "test", "predict"] | None = None
    ):
        dataset_kwargs = resolve_data_config(self.cfg, phase=phase or "train")
        self.dataset: TrackingDataset = instantiate(self.cfg.data, **dataset_kwargs)
        assert isinstance(self.dataset, TrackingDataset), (
            "Dataset config should target `TrackingDataset`."
        )

    def _instantiate_model(self):
        self.model: TrackingModel = instantiate(self.cfg.model)

    def load_dataset_and_model(
        self, phase: Literal["train", "val", "test", "predict"] | None = None
    ):
        """Loads dataset and model from config."""
        self._instantiate_dataset(phase)
        self._instantiate_model()

    def run_training(self) -> tuple[str | None, dict[str, torch.Tensor] | None]:
        """Run the training based on a `hydra` config.

        All required things are defined within the config and overridable using the CLI.
        See `baclct.train`.
        """
        cfg = self.cfg
        self.load_dataset_and_model("train")

        loggers = instantiate_loggers(cfg.logger)
        callbacks = instantiate_callbacks(cfg.callbacks)
        ckpt_path = cfg.get("checkpoint")
        if ckpt_path is not None:
            logger.info(f"Continuing training from checkpoint {ckpt_path}.")

        trainer: L.Trainer = instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
        trainer.fit(
            cast(L.LightningModule, self.model), self.dataset, ckpt_path=ckpt_path
        )

        if cfg.get("run_test"):
            trainer.test(self.model, self.dataset, ckpt_path="best")

        metrics = (
            trainer.callback_metrics if hasattr(trainer, "callback_metrics") else None
        )

        # record the features this checkpoint was trained on (only for debug/validation)
        output_dir = OmegaConf.select(cfg, "paths.output_dir")
        if output_dir is not None and isinstance(self.dataset, TrackingDataset):
            write_feature_info(self.dataset, output_dir)

        return getattr(trainer.checkpoint_callback, "best_model_path", None), metrics

    def _resolve_classify_states(self, classify_states: bool | None) -> bool:
        """Decide whether states are predicted, given model support and user request."""
        supports_states = bool(self.cfg.get("num_node_classes"))
        if classify_states and not supports_states:
            logger.warning(
                "Model was not trained for node classification. Ignoring "
                "`classify_states` and predicting tracks only."
            )
        if classify_states is None:
            return supports_states
        return classify_states and supports_states

    def _track_and_export(
        self,
        dataset: GraphDataset,
        edge_preds: pl.DataFrame,
        node_preds: pl.DataFrame | None,
        output_dir: Path | str | None,
        sequence_id: str,
        export_format: Literal["ctc", "flat"],
        export_suffix: str | None,
        progress: ProgressCallback | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, BaseTracker]:
        """Reconstruct trajectories, clean them, and export in the requested format."""
        tracker_cfg = self.cfg.get("tracker")
        if tracker_cfg is not None:
            tracker: BaseTracker = instantiate(
                tracker_cfg, dataset=dataset, predictions=edge_preds
            )
        else:
            tracker = LAPTracker(dataset=dataset, predictions=edge_preds)
        tracker.progress = progress

        raw_tracks = tracker.track()
        if node_preds is not None:
            node_preds = resolve_duplicate_predictions(node_preds)
        tracks = clean_tracks_df(raw_tracks, node_preds)

        if output_dir is not None:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            self._export(
                tracker=tracker,
                raw_tracks=raw_tracks,
                tracks=tracks,
                node_preds=node_preds,
                output_dir=output_dir,
                name=sequence_id,
                export_format=export_format,
                export_suffix=export_suffix,
                progress=progress,
            )
        return tracks, node_preds, tracker

    def _export(
        self,
        tracker: BaseTracker,
        raw_tracks: pl.DataFrame,
        tracks: pl.DataFrame,
        node_preds: pl.DataFrame | None,
        output_dir: Path,
        name: str,
        export_format: Literal["ctc", "flat"],
        export_suffix: str | None,
        progress: ProgressCallback | None,
    ) -> None:
        """Write results to disk in the requested format.

        'flat' writes a single {name}{suffix}.csv/.tif straight into `output_dir` from the
        cleaned frame. 'ctc' writes a per-sequence directory (tracks, states, and a
        combined CSV) from the raw tracker frame, whose label_track/index columns the CTC
        exporters expect.
        """
        report(progress, "export", 0, 1, "Writing results")
        if export_format == "flat":
            export_tracking_results_simple(
                tracks,
                tracker.tracked_masks(),
                name,
                output_dir,
                suffix=export_suffix if export_suffix is not None else "_tracks",
            )
        else:
            target_dir = output_dir / f"{name}{export_suffix or ''}"
            tracker.export_results(raw_tracks, target_dir, format="ctc")
            if node_preds is not None:
                export_classification_results(
                    node_preds, output_dir=target_dir, tracks=raw_tracks
                )
            export_combined_tracks(raw_tracks, target_dir, node_preds)
        report(progress, "export", 1, 1, "Wrote results")

    def track(
        self,
        images: da.Array | np.ndarray,
        masks: da.Array | np.ndarray,
        model: str | Path | None = None,
        classify_states: bool | None = None,
        graph_search_radius: int | str | None = "2.5x",
        prune_edges_by: str | tuple[str, int | float | str] | None = None,
        patch_size: int | tuple[int, ...] | None = None,
        device: str | None = None,
        output_dir: Path | str | None = None,
        sequence_id: str = "pred",
        export_format: Literal["ctc", "flat"] = "ctc",
        export_suffix: str | None = None,
        cache_dir: Path | str | Literal["temp"] | None = None,
        progress: ProgressCallback | None = None,
        spacing: SpacingLike = None,
    ) -> tuple[np.ndarray, pl.DataFrame]:
        """Run tracking on a single sequence.

        Loads a trained model and applies it to images and segmentation masks, running
        feature extraction, GNN inference, and tracking. Unless `cache_dir` is given,
        features are not cached and nothing is written apart from the outputs.

        Throughput and the tracker are not arguments here but config keys, reachable
        through `config_overrides` at construction: `num_jobs_features`,
        `num_workers_encode`, `num_workers_predict`, `batch_size`, and `tracker.thr_corr`,
        `tracker.thr_div`, `tracker.segmentation_correction`.

        Args:
            images: Image sequence with shape `(T, H, W)`.
            masks: Instance-segmentation masks for `images`.
            model: Registered model name (e.g. 'baclct_track'), or path to an
                experiment directory. Falls back to the model given at construction, and
                to 'baclct_track' when neither names one.
            classify_states: Whether to predict life cycle states. `None` follows the
                model, which classifies states when it was trained with
                `num_node_classes`. `False` suppresses them even for a model that can,
                leaving no 'state' column, no node predictions, and no state export.
                `True` on a model that was not trained for it warns and is ignored.
            graph_search_radius: Maximum distance for an edge between two cells, as
                pixels or as a multiple of the expected cell size (e.g. `'2.5x'`, resolved
                against the median major axis length in the first frame). Relative by
                default, so the graph follows magnification and binning rather than
                assuming one pixel size. `None` uses the radius the model was trained
                with instead.
            prune_edges_by: How candidate edges are pruned, as a method name or a
                `(method, parameter)` pair, e.g. `('ellipse', 7)`. `None` takes the value
                the model was trained with, 'off' prunes nothing and leaves the search
                radius as the only limit. Deviating from the trained value moves the
                candidate edges away from the distribution the model saw during training.
            patch_size: Tile each frame into spatial patches of this size instead of
                building one whole-frame graph. Needed for large images where the
                full-frame graph exceeds memory.
            device: Where the encoder and the GNN run, e.g. 'cpu', 'cuda', 'cuda:1',
                'mps'. `None` picks cuda, then mps, then cpu.
            output_dir: Directory where tracking outputs are stored. `None` exports
                nothing and returns the results only.
            sequence_id: Name of the image sequence, used to key the feature cache and to
                name the exported results.
            export_format: Format used for storing tracking outputs. 'ctc' writes a
                per-sequence CTC directory. 'flat' writes '{sequence_id}_tracks.csv' and
                '{sequence_id}_tracks.tif' directly into `output_dir`.
            export_suffix: Appended to `sequence_id` during export.
            cache_dir: Where node, deep, and edge features are cached. `None` writes
                nothing and keeps the precomputed edges in memory, `'temp'` uses a
                temporary directory removed with the returned tracker and scans the edges
                lazily instead, and a path caches persistently for repeated runs over the
                same sequence. A persistent cache is keyed by `sequence_id`, so pass a
                distinct one per sequence.
            progress: Optional sink for progress events, called between frames, batches,
                and timepoints. Raise `TrackingCancelled` from it to abort the run.
            spacing: Physical frame and voxel spacing as a `{t, z, y, x}` mapping or a
                length-2/3/4 sequence. `t` rescales the temporal edge distance to match
                the frame rate the model trained on. When `masks` are every k-th frame of
                a denser acquisition, pass `spacing.t=k` to recover the training
                distribution. Unspecified axes default to `1.0`.

        Returns:
            Tracked masks (matching `masks`'s shape) and the cleaned tracks frame
            (`label`, `t`, `parent`, the single-cell features, and `state` when states
            are classified). Raw edge and node predictions are reachable afterward via
            `self.edge_preds`/`self.node_preds`, and the dataset and model via
            `self.dataset`/`self.model`. On-disk export still follows `export_format`.
        """
        # load config the model was trained on, then merge overrides on top
        if model is None and self._config is None:
            model = DEFAULT_MODEL
        if model is not None:
            self._cfg = self._resolve_config(model)
        cfg = self.cfg
        if self._experiment_dir is None:
            raise ValueError(
                "No checkpoint to track with: this pipeline was built from a composed "
                "config, which has no experiment directory. Pass `model` to track(), or "
                "construct BacLCT with a model name or experiment directory."
            )

        # locate the checkpoint before any feature extraction, so a missing one fails
        # fast. every shipped model holds exactly one, named 'best'
        ckpt_path = resolve_checkpoint(
            cfg.get("checkpoint") or "best", self._experiment_dir
        )
        classify_states = self._resolve_classify_states(classify_states)
        # the argument is the channel, a config override only fills in for an unset one
        device = device if device is not None else cfg.get("device")

        dataset = self._build_dataset(
            images,
            masks,
            graph_search_radius=graph_search_radius,
            prune_edges_by=prune_edges_by,
            patch_size=patch_size,
            device=device,
            sequence_id=sequence_id,
            cache_dir=cache_dir,
            spacing=spacing,
            progress=progress,
        )
        edge_preds, node_preds = self._predict(dataset, ckpt_path, device, progress)

        tracks, node_preds_df, tracker = self._track_and_export(
            dataset,
            edge_preds,
            node_preds if classify_states else None,
            output_dir=output_dir,
            sequence_id=sequence_id,
            export_format=export_format,
            export_suffix=export_suffix,
            progress=progress,
        )
        # keep the raw predictions reachable without widening the return tuple
        self.edge_preds = tracker.predictions
        self.node_preds = node_preds_df
        return tracker.tracked_masks(), tracks

    def _build_dataset(
        self,
        images: da.Array | np.ndarray,
        masks: da.Array | np.ndarray,
        graph_search_radius: int | str | None,
        prune_edges_by: str | tuple[str, int | float | str] | None,
        patch_size: int | tuple[int, ...] | None,
        device: str | None,
        sequence_id: str,
        cache_dir: Path | str | Literal["temp"] | None,
        spacing: SpacingLike,
        progress: ProgressCallback | None,
    ) -> GraphDataset:
        """Build the extractors and the single-sequence graph dataset for inference."""
        cfg = self.cfg
        if cache_dir is not None and cache_dir != "temp" and sequence_id == "pred":
            logger.warning(
                "Caching features under the default sequence_id 'pred'. Feature caches "
                "are keyed by sequence_id, so a second sequence tracked with the same "
                "cache_dir forces a rebuild instead of reusing anything. Pass a distinct "
                "`sequence_id` per sequence to actually benefit from the cache."
            )

        edge_finder_overrides = (
            {}
            if prune_edges_by is None
            else {"prune_edges_by": None if prune_edges_by == "off" else prune_edges_by}
        )
        edge_finder = instantiate(cfg.data.edge_finder, **edge_finder_overrides)
        handcrafted_extractor = instantiate(cfg.data.handcrafted_feature_extractor)
        deep_extractor = (
            instantiate(cfg.data.deep_feature_extractor, device=get_device(device))
            if cfg.data.get("deep_feature_extractor") is not None
            else None
        )
        # progress sinks are attributes so extractor configs stay unchanged
        for extractor in (edge_finder, handcrafted_extractor, deep_extractor):
            if extractor is not None:
                extractor.progress = progress

        self.dataset = GraphDataset.from_images_and_masks(
            images=images,
            masks=masks,
            edge_finder=edge_finder,
            handcrafted_feature_extractor=handcrafted_extractor,
            deep_feature_extractor=deep_extractor,
            sequence_id=sequence_id,
            feature_dir=_feature_dir_arg(cache_dir),
            graph_search_radius=(
                graph_search_radius
                if graph_search_radius is not None
                else cfg.data.graph_search_radius
            ),
            graph_time_step=cfg.data.graph_time_step,
            graph_num_steps=cfg.data.graph_num_steps,
            graph_connectivity=cfg.data.graph_connectivity,
            graph_fov_size=cfg.data.get("graph_fov_size", None),
            patch_size=patch_size,
            spacing=spacing,
            # not configurable at inference: the training cap samples edges at random
            # once exceeded, the per-window rebuild is far slower, and a stale cache is
            # never worth the hash it saves
            precompute_edges=True,
            graph_max_num_edges=-1,
            patch_overlap=0.0,
            trust_cache=False,
        )
        _warn_on_edge_memory(self.dataset, cache_dir, cfg.data.num_workers)
        return self.dataset

    def _predict(
        self,
        dataset: GraphDataset,
        ckpt_path: Path | str | None,
        device: str | None,
        progress: ProgressCallback | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Load the checkpoint and run GNN prediction, returning collected frames."""
        cfg = self.cfg
        num_workers = cfg.data.num_workers
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            shuffle=False,
            # `fork` (the default on linux) deadlocks with polars and the memory-mapped
            # deep feature caches. set per dataloader rather than process-wide, so we do
            # not change the start method of a host application (e.g. napari).
            multiprocessing_context=get_multiprocessing_context(num_workers),
        )

        self.model = instantiate(cfg.model)
        if ckpt_path:
            logger.info(f"Loading checkpoint: {ckpt_path}.")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"])

        accelerator, devices = resolve_accelerator(device)
        collector = _PredictionCollector()
        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=False,
            accelerator=accelerator,
            devices=devices,
            # "warn" keeps CPU/CUDA fully deterministic, while not raising on MPS, where
            # scatter_reduce has no deterministic kernel (warns once).
            deterministic="warn",
            # match the precision the model was trained with, for reproducible results
            precision=cfg.trainer.get("precision", "32-true"),
            callbacks=[LightningProgressCallback(progress), collector],
            # the tqdm bar writes to stderr, noise when a progress sink drives a front-end
            enable_progress_bar=progress is None,
        )
        trainer.predict(self.model, dataloaders=dataloader, return_predictions=False)
        return collector.collect()


def _feature_dir_arg(cache_dir: Path | str | None) -> Path | Literal["temp"] | None:
    """Normalize a `cache_dir` argument into a `feature_dir` for `GraphDataset`."""
    if cache_dir is None:
        return None
    return "temp" if cache_dir == "temp" else Path(cache_dir)


def _warn_on_edge_memory(
    dataset: GraphDataset, cache_dir: Path | str | None, num_workers: int
) -> None:
    """Point at the on-disk cache when the in-memory edge table is large."""
    edge_data = getattr(dataset, "edge_data", None)
    if cache_dir is not None or not isinstance(edge_data, pl.DataFrame):
        return

    size = edge_data.estimated_size()
    if size > EDGE_MEMORY_HINT_BYTES or num_workers > 0:
        logger.info(
            f"Holding {size / 1024**3:.1f} GiB of edges in memory and sharing them with "
            f"{num_workers} dataloader worker(s). Specify `cache_dir` ('temp' or "
            "directory) to read them lazily from temporary ('temp') or persistent (dir) "
            "cache."
        )


class _PredictionCollector(L.Callback):
    """Collect per-batch prediction frames as the predict loop produces them.

    Lightning's own `trainer.predict` return value holds every batch's raw tensors until
    the run ends, which dominates memory on a long sequence. With
    `return_predictions=False` this keeps only the frames.
    """

    def __init__(self) -> None:
        self.edges: list[pl.DataFrame] = []
        self.nodes: list[pl.DataFrame] = []

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Append one batch's edge and node frames."""
        if outputs is None:
            return
        edge_df, node_df = outputs
        if edge_df is not None:
            self.edges.append(edge_df)
        if node_df is not None:
            self.nodes.append(node_df)

    def collect(self) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Concatenate the collected frames, releasing the per-batch pieces."""
        if not self.edges:
            raise ValueError("No predictions were generated.")
        # rechunk=False avoids a full copy of a table that is about to be grouped anyway
        edges = pl.concat(self.edges, rechunk=False)
        self.edges.clear()
        nodes = pl.concat(self.nodes, rechunk=False) if self.nodes else None
        self.nodes.clear()
        return edges, nodes
