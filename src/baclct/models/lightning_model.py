"""Lightning module wrapping the message-passing GNN.

Includes training and inference logic, metric tracking and logging during training, as
well as loss computation and aggregation.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, cast

import lightning as L
import numpy as np
import polars as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data

if TYPE_CHECKING:
    from torchmetrics import MetricCollection

from baclct.data.dataset import GraphDataset, TrackingDataset
from baclct.io import node_preds_to_df
from baclct.models.loss import MultiTaskLoss
from baclct.models.model import MPModel
from baclct.tracking.postprocessing import (
    edge_preds_to_df,
    extract_prediction_stats,
    resolve_duplicate_predictions,
    select_extreme_samples,
)
from baclct.utils.data import collect
from baclct.utils.logger import get_pylogger
from baclct.utils.model import (
    build_metric_collections,
    class_counts_from_datasets,
    one_vs_rest_counts,
)

logger = get_pylogger(__name__)

# train-extra deps: one named flag per extra group (MPL_AVAILABLE covers seaborn too, same
# extra). heavy imports stay lazy in the methods that use them
TORCHMETRICS_AVAILABLE = importlib.util.find_spec("torchmetrics") is not None
MPL_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

warnings.filterwarnings(
    "ignore",
    message="Precision 16-mixed is not supported by the model summary",
    module="lightning",
)
warnings.filterwarnings("ignore", message=".*NaN values found in confusion matrix.*")
# sync_dist=True deadlocks on empty batches in multi-GPU runs
warnings.filterwarnings(
    "ignore",
    message=".*sync_dist=True.*when logging on epoch level in distributed setting.*",
    module="lightning",
)
# items are graph slices read from cached parquet, so more workers cost RAM and startup
# time without adding throughput. the suggested worker count would hurt performance.
warnings.filterwarnings(
    "ignore", message=".*does not have many workers.*", module="lightning"
)


class TrackingModel(L.LightningModule):
    """LightningModule wrapping `MPModel` for training and prediction.

    Coordinates the training loop on top of an `MPModel`:
      - Initialization of class-balanced loss weights from training class counts,
        supporting joint edge (`num_edge_classes`) and optional node (`num_node_classes`)
        classification.
      - Per-batch loss and metric tracking (F1 score and confusion matrices for edges and,
        when node classification is enabled, for nodes), with periodic logging of best-
        and worst-classified samples per class.
      - Aggregation of per-batch outputs across dataloaders into a per-sequence prediction
        dataframe consumed downstream by the tracker.
    """

    def __init__(
        self,
        graph_model: MPModel,
        criterion: MultiTaskLoss | None = None,
        optimizer: Callable[[Iterable[torch.nn.Parameter], float], Optimizer]
        | None = None,
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        lr: float = 0.001,
        num_edge_classes: int = 3,
        num_node_classes: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize lightning module.

        Args:
            graph_model: GNN for edge and optional node classification.
            criterion: Multi-task loss containing edge and optional node components.
            optimizer: Optimizer class or partial returning optimizer.
            lr_scheduler: LR scheduler class or partial returning LR scheduler.
            lr: Learning rate.
            num_edge_classes: Number of predicted edge classes, e.g., 3 for corr., div.,
                inactive.
            num_node_classes: Optional number of predicted node classes, e.g., number of
                life cycle states.
            **kwargs: Absorbs retired config keys (e.g. `feature_norm`) so checkpoints and
                saved `.hydra/config.yaml` files from earlier runs still instantiate.
        """
        super().__init__()

        self.save_hyperparameters(
            ignore=["graph_model", "criterion", "optimizer", "lr_scheduler", *kwargs]
        )

        self.model = graph_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.num_edge_classes = num_edge_classes
        self.num_node_classes = num_node_classes

        # initialize train/val/test_metrics
        self._setup_metrics()
        logger.debug(
            f"Computing metrics: Train {self.train_metrics}; "
            f"Val {self.val_metrics}; Test {self.test_metrics}"
        )

        # keep track of epoch outputs. val/test have multiple dataloaders.
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)
        self.debug_printed = defaultdict(bool)

        # (phase, dataloader) pairs whose pruned-out GT edges were already reported
        self._reported_pruned_edges: set[tuple[str, int | str]] = set()

    def _get_datamodule(self) -> TrackingDataset | None:
        try:
            datamodule = self.trainer.datamodule  # type: ignore
            assert datamodule is not None
            return datamodule
        except Exception as e:
            logger.warning(f"Could not find datamodule to initialize model dims: {e}")

    def setup(self, stage: str):
        """Hook called before the first train, validate, test, or predict loop.

        Rejects a training run configured without a loss (inference does not require
        loss), initializes the class-balanced loss weights from the class counts of the
        whole training set, and materializes lazy input layers, e.g.
        `pyg.nn.MLP(channel_list=[-1, ...])`, with a single forward pass. Without that
        pass the parameters would only appear on the first training batch, which multi-GPU
        training does not support.
        """
        if stage != "predict" and self.criterion is None:
            raise ValueError(
                "Please provide a loss function for training. "
                "It should be based on `models.loss.MultiTaskLoss`."
            )

        n_gpus = self.trainer.world_size
        if stage == "fit":
            datamodule = self._get_datamodule()
            if datamodule is None:
                logger.info(
                    "Could not access datamodule to instantiate loss weights. "
                    "Estimating based on batch stats."
                )
                if n_gpus > 1:
                    raise AttributeError(
                        "Could not access datamodule, but it is required to instantiate "
                        "model layers. Please provide them explicitly or restrict "
                        "training to a single GPU."
                    )
                return

            # initialize loss weights (only possible for precomputed graphs)
            if isinstance(self.criterion, MultiTaskLoss):
                if hasattr(datamodule, "dataset_train"):
                    edge_counts, node_counts = class_counts_from_datasets(
                        datamodule.dataset_train,
                        self.num_edge_classes,
                        self.num_node_classes,
                    )
                    self.criterion._initialize_weights(edge_counts, node_counts)
                else:
                    logger.warning(
                        "Could not find training dataset to compute loss weights."
                    )

            # initialize layers for multi-gpu training and model summary
            forward_keys = ["x_handcrafted", "x_deep", "edge_index", "edge_attr"]
            try:
                # try using dummy graph from dataset to avoid instantiating a
                # dataloader (which could disturb deterministic batching)
                dataset = datamodule.dataset_train
                if isinstance(dataset, ConcatDataset):
                    dataset = next(iter(dataset.datasets))
                if isinstance(dataset, list):  # only required for dataset_val/test
                    dataset = next(iter(dataset))
                dataset = cast("GraphDataset", dataset)

                template = dataset.template_graph
                assert template is not None
                forward_kwargs = {
                    # device is cpu, since called before moving model to self.device
                    k: template[k].to("cpu")
                    if isinstance(template[k], torch.Tensor)
                    else template[k]
                    for k in forward_keys
                    if k in template
                }
            except Exception as e:
                if n_gpus > 1:
                    # if sample graph is not provided by dataset, instantiate dataloader
                    # to ensure that layers are initialized for multi-gpu training
                    logger.warning(
                        f"Problem with init of model for multi-gpu training: {e}"
                    )
                else:
                    logger.debug(f"Could not build dummy graph for model init: {e}")
                    return

                dataloader: Iterable[Data] = datamodule.train_dataloader()
                batch: Data = next(iter(dataloader))
                forward_kwargs = {
                    # device is cpu, as hook is called before moving model to self.device
                    k: batch[k].to("cpu")
                    if isinstance(batch[k], torch.Tensor)
                    else batch[k]
                    for k in forward_keys
                    if k in batch
                }

            self.forward(**forward_kwargs)
            # set example input for model summary; supports kwargs
            self.example_input_array = forward_kwargs

    def _setup_metrics(self):
        if not TORCHMETRICS_AVAILABLE:
            logger.warning(
                "torchmetrics not available; training metrics will be disabled. "
                "Install with: pip install baclct[train]"
            )
            self.train_metrics = self.val_metrics = self.test_metrics = None
            return

        self.train_metrics, self.val_metrics, self.test_metrics = (
            build_metric_collections(self.num_edge_classes, self.num_node_classes)
        )

    def _preds_to_class(self, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
        if num_classes == 1:
            return (preds.squeeze(-1) > 0).long()
        return preds.argmax(-1)

    def forward(
        self,
        x_handcrafted: torch.Tensor | None,
        x_deep: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> dict[Literal["edge_predictions", "node_predictions"], list[torch.Tensor]]:
        """Run embedding, message passing, and classification."""
        return self.model(
            x_handcrafted=x_handcrafted,
            x_deep=x_deep,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def _log_loss(
        self,
        loss: torch.Tensor | float,
        stage: str,
        batch_size: int | None = None,
        add_dataloader_idx: bool = False,
    ):
        # don't `sync_dist`, since this deadlocks on empty batches on multi-gpu
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            add_dataloader_idx=add_dataloader_idx,  # required for val
        )

    def _should_log_epoch(self, nth: int = 1):
        if not self._trainer:
            return False

        return not self.trainer.sanity_checking and (self.current_epoch + 1) % nth == 0

    def _get_tb_logger(self):
        """Get tensorboard logger object for direct use."""
        if not self._trainer:
            return

        tb_logger = None
        for logi in self.trainer.loggers:
            if isinstance(logi, TensorBoardLogger):
                tb_logger = logi.experiment
                break

        return tb_logger

    def _get_csv_logger(self):
        """Get the CSVLogger object for direct, tensorboard-free scalar logging."""
        if not self._trainer:
            return None

        for logi in self.trainer.loggers:
            if isinstance(logi, CSVLogger):
                return logi
        return None

    def _dataset_for_dataloader(
        self, phase: Literal["train", "val", "test"], dataloader_idx: int | str | None
    ):
        """Resolve the single dataset backing a given dataloader."""
        dataset = getattr(self.trainer.datamodule, f"dataset_{phase}", None)  # type: ignore
        if dataset is None or dataloader_idx is None:
            return dataset
        if isinstance(dataset, list):
            return dataset[dataloader_idx]
        if isinstance(dataset, ConcatDataset):
            if isinstance(dataloader_idx, str):
                # string keys come from `data_source`, which namespaces the sequence id
                # with its dataset directory
                return next(
                    (
                        ds
                        for ds in dataset.datasets
                        if getattr(ds, "data_source", None) == dataloader_idx
                    ),
                    None,
                )
            return dataset.datasets[dataloader_idx]
        return dataset

    def _log_class_counts(
        self,
        epoch_edge_outputs: pl.DataFrame,
        outputs: dict[str | int, list[dict]],
        phase: str,
    ) -> None:
        """Log per-dataloader, per-class confusion counts to the CSV logger only.

        Tensorboard cannot show per-dataloader, per-class counts usefully, so they go to
        the CSV logger alone.
        """
        if phase not in ("val", "test"):
            return
        csv_logger = self._get_csv_logger()
        if csv_logger is None:
            return

        metrics: dict[str, float] = {}
        if len(epoch_edge_outputs) > 0 and "dataloader_idx" in epoch_edge_outputs.columns:
            stats = extract_prediction_stats(epoch_edge_outputs)
            for dl_idx in stats["dataloader_idx"].unique().to_list():
                grp = stats.filter(pl.col("dataloader_idx") == dl_idx)
                counts = one_vs_rest_counts(
                    grp["y"].to_numpy(),
                    grp["y_pred"].to_numpy(),
                    self.num_edge_classes,
                )
                for c, cnt in counts.items():
                    base = f"{phase}/edge_counts/dl{dl_idx}/c{c}"
                    for k, v in cnt.items():
                        metrics[f"{base}/{k}"] = float(v)

                metrics.update(self._pruned_edge_counts(phase, dl_idx))

        if self.num_node_classes:
            for dl_idx, output_list in outputs.items():
                node_df = self._aggregate_node_outputs_single_dataloader(output_list)
                if node_df is None or node_df.is_empty():
                    continue
                node_pred, node_y = _node_preds_and_labels(node_df)
                counts = one_vs_rest_counts(
                    node_y.numpy(), node_pred.numpy(), self.num_node_classes
                )
                for c, cnt in counts.items():
                    base = f"{phase}/node_counts/dl{dl_idx}/c{c}"
                    for k, v in cnt.items():
                        metrics[f"{base}/{k}"] = float(v)

        if metrics:
            csv_logger.log_metrics(metrics, step=self.global_step)
            csv_logger.save()

    def _pruned_edge_counts(self, phase: str, dl_idx: int | str) -> dict[str, float]:
        """GT correspondence and division edges not contained in the training graph."""
        if (phase, dl_idx) in self._reported_pruned_edges:
            return {}

        dataset = self._dataset_for_dataloader(phase, dl_idx)  # type: ignore
        gt_edges = getattr(dataset, "lineage_gt_edges", None)
        edge_data = getattr(dataset, "edge_data", None)
        if gt_edges is None or edge_data is None:
            return {}

        self._reported_pruned_edges.add((phase, dl_idx))
        gt_per_class = dict(gt_edges.group_by("y").len().iter_rows())

        # the store keeps one direction per pair, so both sides match on sorted endpoints
        def endpoints(frame: pl.LazyFrame) -> pl.LazyFrame:
            return frame.select(
                pl.min_horizontal("src", "dst").alias("lo"),
                pl.max_horizontal("src", "dst").alias("hi"),
                pl.exclude("src", "dst"),
            )

        missing = dict(
            collect(
                endpoints(gt_edges.lazy().select("src", "dst", "y"))
                .join(
                    endpoints(edge_data.lazy().select("src", "dst")),
                    on=["lo", "hi"],
                    how="anti",
                )
                .group_by("y")
                .len()
            ).iter_rows()
        )

        metrics: dict[str, float] = {}
        for c in (1, 2):
            gt_total = int(gt_per_class.get(c, 0))
            pruned_out = int(missing.get(c, 0))
            base = f"{phase}/edge_counts/dl{dl_idx}/c{c}"
            metrics[f"{base}/gt_total"] = float(gt_total)
            metrics[f"{base}/pruned_out"] = float(pruned_out)

            if gt_total > 0 and pruned_out > 0.1 * gt_total:
                logger.warning(
                    f"{phase} dataloader {dl_idx}: {pruned_out}/{gt_total} ground truth "
                    f"class-{c} edges are missing from the pruned graph. Consider a "
                    "larger graph_search_radius, more graph_num_steps, or less pruning."
                )

        return metrics

    def _log_confusion_matrix(
        self,
        conf_matrix: torch.Tensor,
        name: str,
        phase: str,
        raw_matrix: torch.Tensor | None = None,
    ) -> None:
        """Log confusion matrix to tensorboard."""
        tb = self._get_tb_logger()
        if tb is None or not MPL_AVAILABLE:
            return

        import matplotlib.pyplot as plt

        from baclct.viz.training import plot_confusion_matrix

        fig = plot_confusion_matrix(
            conf_matrix.cpu().numpy(),
            name,
            raw_matrix.cpu().numpy() if raw_matrix is not None else None,
        )
        try:
            tb.add_figure(f"{phase}/conf_{name}", fig, self.global_step)
        finally:
            plt.close(fig)

    def _sample_losses(
        self, stats: pl.DataFrame, task: Literal["edge", "node"]
    ) -> pl.DataFrame:
        """Add the per-sample loss of the configured component loss as a 'loss' column.

        The value is the loss of the final message passing step alone, neither scaled by
        the task weight nor summed over steps, so it is comparable across samples but not
        equal to the logged loss. Falls back to CE if loss cannot be evaluated per sample.
        """
        logits = torch.from_numpy(stats.select(r"^p\d$").to_numpy(writable=True))
        y = torch.from_numpy(stats["y"].to_numpy(writable=True)).long()

        loss_fn = getattr(self.criterion, f"{task}_loss_fn", None)
        loss_kwargs = getattr(self.criterion, f"{task}_loss_kwargs", {}) or {}
        n_classes = self.num_edge_classes if task == "edge" else self.num_node_classes

        losses = None
        # single-logit predictions (when no div. class) is expanded to probabilities by
        # extract_prediction_stats, so its cols are no longer the input the loss expects
        if loss_fn is not None and (n_classes or 0) > 1:
            try:
                losses = loss_fn(logits, y, reduction="none", **loss_kwargs)
            except Exception as e:
                # e.g. loss classes that fix their reduction at construction
                logger.debug(f"Falling back to cross entropy for {task} sample loss: {e}")

        if losses is None:
            losses = -torch.log(
                torch.from_numpy(stats["p_true"].to_numpy(writable=True)).clamp_min(1e-12)
            )

        return stats.with_columns(loss=pl.Series("loss", losses.flatten().numpy()))

    def _log_samples(
        self,
        edge_outputs: pl.DataFrame,
        node_outputs: pl.DataFrame | None,
        phase: Literal["train", "val", "test"],
    ) -> None:
        """Log the highest- and lowest-loss samples per class as image grids.

        With a focal loss the highest-loss samples are dominated by annotation noise.
        """
        tb = self._get_tb_logger()
        if tb is None or not MPL_AVAILABLE:
            return

        for task, outputs in (("edge", edge_outputs), ("node", node_outputs)):
            if outputs is None or outputs.is_empty():
                continue
            try:
                self._log_sample_grid(outputs, task, phase, tb)
            except Exception:
                logger.warning(f"Could not plot {task} samples ({phase})", exc_info=True)

    def _log_sample_grid(
        self,
        outputs: pl.DataFrame,
        task: Literal["edge", "node"],
        phase: Literal["train", "val", "test"],
        tb,
    ) -> None:
        import matplotlib.pyplot as plt

        from baclct.viz.training import crop_cells, plot_sample_grid

        n_worst = 4
        stats = self._sample_losses(extract_prediction_stats(outputs), task)
        rows = select_extreme_samples(stats, n_worst=n_worst, n_best=2)
        if not rows:
            return

        def crop_fn(sample: dict):
            indices = (
                [sample["index"]] if task == "node" else [sample["src"], sample["dst"]]
            )
            dataset = self._dataset_for_dataloader(phase, sample.get("dataloader_idx"))
            assert dataset is not None, "Could not find dataset for plotting callback."
            return crop_cells(dataset, indices)

        fig = plot_sample_grid(rows, crop_fn, n_worst=n_worst, title=f"{task}s")
        try:
            tb.add_figure(f"{phase}/{task}_samples", fig, self.global_step)
        finally:
            plt.close(fig)

    def _log_metrics(self, metrics: MetricCollection, phase: str) -> None:
        if phase == "train":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*NaN values found in confusion matrix.*",
                    category=UserWarning,
                )
                metrics_to_log = metrics.compute()
        else:
            metrics_to_log = metrics.compute()
        scalar_metrics = {}
        f1_scores = []

        conf_matrices = {}
        for key, val in metrics_to_log.items():
            if isinstance(key, str) and "confusion_matrix" in key:
                conf_matrices[key] = val
            else:
                scalar_metrics[key] = val
                if isinstance(key, str) and "f1_score" in key:
                    f1_scores.append(val)

        if self._should_log_epoch(10) or phase == "test":
            for key, val in conf_matrices.items():
                if "raw" in key:
                    continue
                name = key.split("/")[1].replace("_confusion_matrix", "")

                # try to find corresponding raw matrix
                prefix = key.split("confusion_matrix")[0]
                raw_key = f"{prefix}confusion_matrix_raw"
                raw_val = conf_matrices.get(raw_key)

                self._log_confusion_matrix(val, name, phase, raw_val)

        if f1_scores:
            scalar_metrics[f"{phase}/f1_score"] = torch.stack(f1_scores).mean()

        self.log_dict(scalar_metrics, on_step=False, on_epoch=True, sync_dist=True)

    def step(self, batch: Data, stage: str):
        """Basic logic for single step.

        Used for train/val/test/predict steps.

        When computing the graph on-the-fly, some batches might be empty. Since they are
        represented using empty tensors, loss computation is fully supported (empty loss)
        and should prevent deadlocks for multi-gpu training. Since returns are identical
        to non-empty batches, downstream aggregation handles empty outputs gracefully.

        Args:
            batch: Batch containing single or multiple disconnected trajectory graphs
                with node and edge features, as well as the corresponding indices and
                metadata.
            stage: Lightning training stage.
        """
        x_handcrafted = batch.get("x_handcrafted")
        x_deep = batch.get("x_deep")
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        assert isinstance(edge_index, torch.Tensor) and isinstance(
            edge_attr, torch.Tensor
        )
        output = self.forward(x_handcrafted, x_deep, edge_index, edge_attr)

        if stage == "predict":
            return output

        if self.criterion is None:
            raise ValueError("Loss was not configured for training.")

        targets = {"edge_labels": batch.y_edges, "node_labels": batch.y_nodes}
        ignore = batch.get("edge_ignore")

        if ignore is not None:
            targets["edge_labels"] = targets["edge_labels"][~ignore]
            output["edge_predictions"] = [p[~ignore] for p in output["edge_predictions"]]

        # loss computation is compatible with empty batches. it is critical to compute
        # the exact loss for multi-gpu training. we just won't return the outputs
        loss = self.criterion(output, targets)
        if next(iter(output["edge_predictions"])).numel() == 0:
            # dummy graph returns empty outputs which raise during metric computation
            # loss has to be returned for multi-gpu training, otherwise will deadlock
            return loss, None, None, None

        # return targets and ignore mask alongside loss and output
        return loss, output, targets, ignore

    def training_step(self, batch, batch_idx):
        """Single training step."""
        outputs = self.step(batch, "train")
        loss, preds, targets, ignore = outputs

        self._log_loss(loss, "train", max(1, batch.num_edges))

        if preds is None:
            return {"loss": loss}

        # update train metrics step-by-step
        edge_preds = preds["edge_predictions"][-1].detach()
        edge_y = targets["edge_labels"]

        if self.train_metrics is not None:
            for name, metric in self.train_metrics.items():
                if "edge" in name:
                    metric(
                        self._preds_to_class(edge_preds, self.num_edge_classes), edge_y
                    )

        if (
            self.num_node_classes
            and preds.get("node_predictions")
            and preds["node_predictions"][-1] is not None
        ):
            node_preds = preds["node_predictions"][-1].detach()
            node_y = targets["node_labels"]
            if self.train_metrics is not None:
                for name, metric in self.train_metrics.items():
                    if "node" in name:
                        metric(
                            self._preds_to_class(node_preds, self.num_node_classes),
                            node_y,
                        )

        if self._should_log_epoch(10):
            batch_outputs = self._get_batch_outputs(
                batch,
                loss,
                preds,
                targets,
                ignore,
                # sequence id is extracted per-edge in _get_batch_outputs
                dataloader_idx=None,
            )
            self.train_outputs["train"].append(batch_outputs)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Single validation step."""
        outputs = self.step(batch, "val")
        loss, preds, targets, ignore = outputs

        # always log loss to prevent metric sync deadlocks on empty batches
        self._log_loss(loss, "val", max(1, batch.num_edges), add_dataloader_idx=True)

        if (self._trainer and self.trainer.sanity_checking) or preds is None:
            return {"loss": loss}

        batch_outputs = self._get_batch_outputs(
            batch, loss, preds, targets, ignore, dataloader_idx
        )
        self.validation_outputs[dataloader_idx].append(batch_outputs)
        return batch_outputs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Single test step."""
        outputs = self.step(batch, "test")
        loss, preds, targets, ignore = outputs

        # always log loss to prevent metric sync deadlocks on empty batches
        self._log_loss(loss, "test", max(1, batch.num_edges), add_dataloader_idx=True)

        if preds is None:
            return {"loss": loss}

        batch_outputs = self._get_batch_outputs(
            batch, loss, preds, targets, ignore, dataloader_idx
        )
        self.test_outputs[dataloader_idx].append(batch_outputs)
        return batch_outputs

    def predict_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Edge and node predictions for one batch, already in their final frame form.

        Converting here rather than after the run keeps the predict loop's footprint
        proportional to the result: no batch is retained as raw tensors, and the node
        indices stay 32-bit instead of the int64 the graph carries.
        """
        output = self.step(batch, "predict")
        edge_preds = output.get("edge_predictions")

        # an empty template graph (a frame or patch without edges) still forwards, but
        # yields zero-length outputs the frame conversion cannot aggregate
        if not edge_preds or edge_preds[-1].numel() == 0:
            return None, None

        src, dst = self._map_index(batch)
        edge_df = edge_preds_to_df(
            src.cpu().numpy(),
            dst.cpu().numpy(),
            edge_preds[-1].detach().float().cpu().numpy(),
        )

        node_preds = output.get("node_predictions")
        node_df = None
        if node_preds and isinstance(node_preds[-1], torch.Tensor):
            node_df = node_preds_to_df(
                node_preds[-1].detach().float().cpu().numpy(),
                batch.node_mapping.detach().cpu().numpy(),
            )

        return edge_df, node_df

    @staticmethod
    def _map_index(batch: Data):
        """Remaps local node indices to global indices."""
        assert isinstance(batch.edge_index, torch.Tensor)
        src, dst = batch.edge_index
        return batch.node_mapping[src], batch.node_mapping[dst]

    def _get_batch_outputs(self, batch, loss, preds, targets, ignore, dataloader_idx):
        src, dst = self._map_index(batch)

        # filter the edge indices if an ignore mask is present
        if ignore is not None:
            src = src[~ignore]
            dst = dst[~ignore]

        # get sequence id per edge and per node
        if hasattr(batch, "data_source") and isinstance(batch.data_source, list):
            # mapping from local node to graph idx
            src_local = batch.edge_index[0]
            if ignore is not None:
                src_local = src_local[~ignore]
            graph_idx = batch.batch.cpu().numpy()
            sources = np.array(batch.data_source)
            data_source_per_edge = sources[graph_idx[src_local.cpu().numpy()]]
            data_source_per_node = sources[graph_idx]
        else:
            # e.g. single validation dataset
            data_source_per_edge = dataloader_idx
            data_source_per_node = dataloader_idx

        return {
            "edge_index": torch.vstack([src, dst]).detach().cpu(),
            "edge_preds": [p.detach().cpu() for p in preds["edge_predictions"]],
            "edge_y": targets["edge_labels"].detach().cpu(),
            "node_index": batch.node_mapping.detach().cpu()
            if batch.node_mapping is not None
            else None,
            "node_preds": [p.detach().cpu() for p in preds["node_predictions"]]
            if preds.get("node_predictions")
            else None,
            "node_y": targets["node_labels"].detach().cpu()
            if targets.get("node_labels") is not None
            else None,
            "loss": loss.detach().cpu(),
            # for training dataset, we use sequences to map within concat dataset
            # for validation datasets, we can just directly map to the dataloader
            "dataloader_idx": data_source_per_edge
            if dataloader_idx is None
            else dataloader_idx,
            "node_dataloader_idx": data_source_per_node
            if dataloader_idx is None
            else dataloader_idx,
        }

    def _aggregate_edge_outputs_single_dataloader(
        self, outputs: list[dict], dataloader_idx: int | str | None = None
    ) -> pl.DataFrame:
        if not outputs:
            return pl.DataFrame()

        src, dst = torch.cat([o["edge_index"] for o in outputs], 1).cpu().numpy()
        preds = (
            torch.cat([o["edge_preds"][-1] for o in outputs], 0)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        y = torch.cat([o["edge_y"] for o in outputs], 0).cpu().numpy()

        if dataloader_idx is not None:
            dl_idx_list = [dataloader_idx] * len(y)
        elif outputs[0].get("dataloader_idx") is not None:
            dl_idx_list = []
            for o in outputs:
                if isinstance(o["dataloader_idx"], (np.ndarray, list, torch.Tensor)):
                    dl_idx_list.extend(o["dataloader_idx"])
                else:
                    dl_idx_list.extend([o["dataloader_idx"]] * len(o["edge_y"]))

        return resolve_duplicate_predictions(
            edge_preds_to_df(src, dst, preds, y, dataloader_idx=dl_idx_list)
        )

    def _aggregate_edge_outputs(
        self, outputs: dict[str | int, list[dict]]
    ) -> pl.DataFrame:
        final_outputs = {}
        for dataloader_idx, output_list in outputs.items():
            if not output_list:
                continue

            final_outputs[dataloader_idx] = (
                self._aggregate_edge_outputs_single_dataloader(
                    output_list,
                    dataloader_idx=dataloader_idx if dataloader_idx != "train" else None,
                )
            )

        if not final_outputs:
            return pl.DataFrame()

        return pl.concat(list(final_outputs.values()))

    def _aggregate_node_outputs_single_dataloader(
        self, outputs: list[dict], dataloader_idx: int | str | None = None
    ) -> pl.DataFrame | None:
        """Node index, label, and deduplicated class logits of one dataloader."""
        if not self.num_node_classes:
            return None

        node_preds_list, node_y_list, node_idx_list, source_list = [], [], [], []
        for o in outputs:
            if (
                o.get("node_preds")
                and o["node_preds"][-1] is not None
                and o.get("node_y") is not None
            ):
                node_preds_list.append(o["node_preds"][-1].detach().float().cpu())
                node_y_list.append(o["node_y"].detach().cpu())
                node_idx_list.append(o["node_index"].detach().cpu())
                source = (
                    dataloader_idx
                    if dataloader_idx is not None
                    else o.get("node_dataloader_idx")
                )
                n_nodes = len(o["node_y"])
                if isinstance(source, (np.ndarray, list)):
                    source_list.extend(list(source))
                else:
                    source_list.extend([source] * n_nodes)

        if not node_preds_list:
            return None

        node_preds_tensor = torch.cat(node_preds_list, 0).numpy()
        node_y_tensor = torch.cat(node_y_list, 0).numpy()
        node_idx_tensor = torch.cat(node_idx_list, 0).numpy()

        n_preds = node_preds_tensor.shape[1]
        cols = [node_idx_tensor[..., None], node_y_tensor[..., None], node_preds_tensor]
        df_nodes = pl.DataFrame(
            np.concatenate(cols, axis=1),
            schema=["index", "y"] + [f"p{i}" for i in range(n_preds)],
        ).cast({"index": pl.Int64, "y": pl.Int64})

        if any(source is not None for source in source_list):
            df_nodes = df_nodes.with_columns(
                dataloader_idx=pl.Series("dataloader_idx", source_list)
            )

        return resolve_duplicate_predictions(df_nodes)

    def _aggregate_node_outputs(
        self, outputs: dict[str | int, list[dict]]
    ) -> pl.DataFrame | None:
        """Aggregate node predictions and labels from all dataloaders."""
        if not self.num_node_classes:
            return None

        frames = []
        for dataloader_idx, dataloader_outputs in outputs.items():
            df = self._aggregate_node_outputs_single_dataloader(
                dataloader_outputs,
                dataloader_idx=dataloader_idx if dataloader_idx != "train" else None,
            )
            if df is not None and not df.is_empty():
                frames.append(df)

        if not frames:
            return None

        return pl.concat(frames, how="diagonal")

    def _aggregate_and_log(
        self,
        outputs: dict[str | int, list[dict]],
        metrics: MetricCollection,
        phase: str,
        compute_metrics: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        # edge metrics
        epoch_edge_outputs = self._aggregate_edge_outputs(outputs)

        if compute_metrics and len(epoch_edge_outputs) > 0:
            edge_y_pred = self._preds_to_class(
                torch.from_numpy(
                    epoch_edge_outputs.select(r"^p\d$").to_numpy(writable=True)
                ),
                self.num_edge_classes,
            ).to(self.device)
            edge_y = torch.from_numpy(epoch_edge_outputs["y"].to_numpy(writable=True)).to(
                self.device
            )

            for name, metric in metrics.items():
                if "edge" in name:
                    metric(edge_y_pred, edge_y)

        # node metrics
        epoch_node_outputs = self._aggregate_node_outputs(outputs)
        if compute_metrics and epoch_node_outputs is not None:
            node_y_pred, node_y = _node_preds_and_labels(epoch_node_outputs)
            node_y_pred = node_y_pred.to(self.device)
            node_y = node_y.to(self.device)
            for name, metric in metrics.items():
                if "node" in name:
                    metric(node_y_pred, node_y)

        self._log_metrics(metrics, phase)

        try:
            self._log_class_counts(epoch_edge_outputs, outputs, phase)
        except Exception:
            logger.warning(f"{phase} per-class count logging failed", exc_info=True)

        return epoch_edge_outputs, epoch_node_outputs

    def _finalize_epoch(
        self,
        outputs: dict,
        metrics: MetricCollection | None,
        phase: Literal["train", "val", "test"],
        aggregate: bool = True,
        compute_metrics: bool = True,
        log_samples: bool = True,
    ) -> None:
        """Aggregate, log and reset metrics at the end of an epoch.

        To prevent DDP deadlocks, metrics have to be computed everytime. If outputs are
        empty (empty batch), compute has to be called on empty metric.

        Args:
            outputs: Accumulated per-batch outputs for the phase.
            metrics: Metric collection for the phase, or `None` when metrics are disabled.
            phase: One of 'train', 'val', 'test'.
            aggregate: Whether to build the per-epoch outputs and update metrics from
                them. Disabled for training on non-log epochs, where metrics are already
                updated per step.
            compute_metrics: Passed to `_aggregate_and_log`; `False` skips the metric
                update so per-step training metrics are not double-counted.
            log_samples: Whether to plot worst and best samples for this epoch.
        """
        if metrics is None:
            return

        computed = False
        if aggregate and outputs:
            try:
                epoch_edge_outputs, epoch_node_outputs = self._aggregate_and_log(
                    outputs, metrics, phase, compute_metrics=compute_metrics
                )
                computed = True
                if log_samples:
                    self._log_samples(epoch_edge_outputs, epoch_node_outputs, phase)
            except Exception:
                logger.warning(f"{phase} epoch-end aggregation failed", exc_info=True)

        if not computed:
            self._log_metrics(metrics, phase)

        outputs.clear()
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Hook to run aggregation and logging afer training epoch."""
        self._finalize_epoch(
            self.train_outputs,
            self.train_metrics,
            "train",
            aggregate=self._should_log_epoch(10),
            compute_metrics=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Hook to run aggregation and logging afer validation epoch."""
        # only short-circuit during a real sanity check; a trainer-less call (e.g. a
        # direct invocation in tests) must still drive the metric collective and reset
        if self._trainer is not None and self.trainer.sanity_checking:
            return

        self._finalize_epoch(
            self.validation_outputs,
            self.val_metrics,
            "val",
            log_samples=self._should_log_epoch(10),
        )

    def on_test_epoch_end(self) -> None:
        """Hook to run aggregation and logging afer test epoch."""
        self._finalize_epoch(self.test_outputs, self.test_metrics, "test")

    def on_before_optimizer_step(self, optimizer):
        """Log gradient distributions to identify dead neurons or exploding gradients."""
        if not self._trainer or self.trainer.sanity_checking:
            return

        if (
            self.global_step > 0
            and (self.global_step + 1) % (50 if self.current_epoch == 0 else 1250) == 0
        ):
            tb = self._get_tb_logger()
            if tb is None:
                return

            for name, param in self.model.named_parameters():
                if "weight" in name and param.grad is not None:
                    # focus on encoders and classifiers to keep log sizes manageable
                    if "classifier" in name or "encoder" in name:
                        tb.add_histogram(
                            f"gradients/{name}", param.grad, self.global_step
                        )

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers."""
        # optimizer is optional for inference
        if self.optimizer is None:
            raise ValueError("Please configure optimizer for training.")

        optimizer = self.optimizer(self.parameters(), lr=self.lr)  # type: ignore
        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def _node_preds_and_labels(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted classes and labels of an aggregated node prediction frame."""
    preds = torch.from_numpy(df.select(r"^p\d$").to_numpy(writable=True))
    labels = torch.from_numpy(df["y"].to_numpy(writable=True))
    return preds.argmax(-1), labels
