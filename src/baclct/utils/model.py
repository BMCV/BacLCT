"""Learning rate schedulers, metric collections, and other training utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import ConcatDataset, Dataset

from baclct.utils.data import collect

if TYPE_CHECKING:
    from torchmetrics import MetricCollection

    from baclct.data.dataset import GraphDataset


def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 10,
    start_factor: float = 0.01,
    max_epochs: int = 250,
    min_lr: float = 1e-6,
):
    """Util to get cosine annealing lr scheduler with warmup."""
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr
    )

    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def get_warmup_cosine_with_restarts_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 10,
    start_factor: float = 0.01,
    t_0: int = 50,
    t_mult: int = 2,
    min_lr: float = 1e-6,
):
    """Util to get cosine annealing lr scheduler with warmup."""
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=min_lr
    )

    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def _metric_collection(num_classes: int, prefix: str) -> dict[str, dict]:
    from torchmetrics.classification import ConfusionMatrix, F1Score

    multiclass = bool(num_classes and num_classes > 1)

    task = "multiclass" if multiclass else "binary"
    average = "macro" if multiclass else "micro"
    num_classes_kwarg = {"num_classes": num_classes} if multiclass else {}

    metrics = {
        f"{prefix}f1_score": F1Score(
            task=task,
            average=average,
            **num_classes_kwarg,  # type: ignore
        ),
        f"{prefix}confusion_matrix_true": ConfusionMatrix(
            task=task,
            normalize="true",
            **num_classes_kwarg,  # type: ignore
        ),
        f"{prefix}confusion_matrix_pred": ConfusionMatrix(
            task=task,
            normalize="pred",
            **num_classes_kwarg,  # type: ignore
        ),
        f"{prefix}confusion_matrix_raw": ConfusionMatrix(
            task=task,
            normalize="none",
            **num_classes_kwarg,  # type: ignore
        ),
    }
    train_metrics = {k: v.clone() for k, v in metrics.items()}
    return {"train": train_metrics, "val": metrics}


def build_metric_collections(
    num_edge_classes: int, num_node_classes: int | None = None
) -> tuple[MetricCollection, MetricCollection, MetricCollection]:
    """Build the train, val, and test metric collections for edges and optional nodes.

    Requires `torchmetrics`, so callers on the inference path must check availability
    first.

    Returns:
        F1 score plus raw, true-normalized, and pred-normalized confusion matrices, one
        independent copy per phase.
    """
    from torchmetrics import MetricCollection

    edge_metrics = _metric_collection(num_edge_classes, prefix="edge_")
    train_metrics = {**edge_metrics["train"]}
    val_metrics = {**edge_metrics["val"]}

    if num_node_classes:
        node_metrics = _metric_collection(num_node_classes, prefix="node_")
        train_metrics.update(node_metrics["train"])
        val_metrics.update(node_metrics["val"])

    return (
        MetricCollection(train_metrics, prefix="train/"),
        MetricCollection(val_metrics, prefix="val/"),
        MetricCollection(val_metrics, prefix="test/").clone(),
    )


def one_vs_rest_counts(
    y: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[int, dict[str, int]]:
    """One-vs-rest TP/FP/FN/TN per class from labels and predictions."""
    n_classes = num_classes if (num_classes and num_classes > 1) else 2
    total = int(y.shape[0])
    counts: dict[int, dict[str, int]] = {}
    for c in range(n_classes):
        tp = int(((y == c) & (y_pred == c)).sum())
        fp = int(((y != c) & (y_pred == c)).sum())
        fn = int(((y == c) & (y_pred != c)).sum())
        counts[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": total - tp - fp - fn}
    return counts


def class_counts_from_datasets(
    dataset_train: Dataset | ConcatDataset | list[Dataset],
    num_edge_classes: int,
    num_node_classes: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Edge and node class counts over the whole training set, for loss weighting.

    Edge counts require precomputed edges, since on-the-fly graphs only expose the edges
    of the current item. Either count is `None` when it cannot be determined.
    """
    if isinstance(dataset_train, ConcatDataset):
        datasets = dataset_train.datasets
    elif isinstance(dataset_train, list):
        datasets = dataset_train
    else:
        datasets = [dataset_train]

    edge_counts = []
    node_counts = []
    for ds in cast("list[GraphDataset]", datasets):
        if num_node_classes:  # >0 or not None
            # assumes every node is annotated. unannotated states are filled with 0 by
            # GraphDataset (see its missing_trajectories) and count towards class 0.
            node_counts.append(
                np.bincount(
                    np.clip(ds.node_feats["y"], 0, num_node_classes - 1),
                    minlength=num_node_classes,
                )
            )

        if num_edge_classes and ds.precompute_edges:
            edge_data = cast("pl.LazyFrame | pl.DataFrame", ds.edge_data)
            edge_y = collect(edge_data.select("y"))["y"]
            edge_counts.append(
                np.bincount(
                    np.clip(edge_y, 0, num_edge_classes - 1), minlength=num_edge_classes
                )
            )

    return (
        torch.from_numpy(np.sum(edge_counts, axis=0)) if edge_counts else None,
        torch.from_numpy(np.sum(node_counts, axis=0)) if node_counts else None,
    )
