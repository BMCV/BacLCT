"""Loss for simultaneous node and edge classification in a graph.

`focal_loss` implements Lin et al., ICCV 2017 (https://arxiv.org/abs/1708.02002) and
`class_balanced_loss` the effective-number reweighting of Cui et al., CVPR 2019
(https://arxiv.org/abs/1901.05555). Both follow the implementation of
https://github.com/wildoctopus/cbloss, extended here to take precomputed dataset class
counts instead of per-batch counts.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PRED_DICT: TypeAlias = dict[Literal["edge_predictions", "node_predictions"], list[Tensor]]
TARGET_DICT: TypeAlias = dict[Literal["edge_labels", "node_labels"], Tensor]


class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint node and edge classification.

    Returns a weighted sum of an optional edge loss and an optional node loss, so it can
    compute either component on its own or both together. Within-class weighting (e.g.,
    class-balanced focal loss for imbalance between correspondence, division, and inactive
    edges) is handled inside the component losses. This wrapper only applies the scalar
    weights `edge_loss_weight` and `node_loss_weight`.

    When predictions are a list of tensors (one per message passing step, as returned by
    `MPModel` with `aggregate_outputs=True`), per-step losses are reduced via
    `aggregate_losses` (`"sum"`, `"mean"`, or `None` to use only the final step) before
    scalar weighting.
    """

    def __init__(
        self,
        edge_loss_fn: Callable[..., Tensor] | None,
        node_loss_fn: Callable[..., Tensor] | None,
        edge_loss_weight: float | Tensor = 1.0,
        node_loss_weight: float | Tensor = 1.0,
        aggregate_losses: Literal["sum", "mean"] | None = "sum",
    ):
        """Initialize loss.

        Args:
            edge_loss_fn: Optional edge loss.
            node_loss_fn: Optional node loss.
            edge_loss_weight: Weight for edge component.
            node_loss_weight: Weight for node component.
            aggregate_losses: Loss aggregation for multi-step losses, e.g., with GNN.
        """
        super().__init__()
        self.edge_loss_fn = edge_loss_fn
        self.node_loss_fn = node_loss_fn
        self.edge_loss_weight = edge_loss_weight
        self.node_loss_weight = node_loss_weight

        self.aggregate_losses = aggregate_losses
        self.edge_loss_kwargs = {}
        self.node_loss_kwargs = {}

    @staticmethod
    def _get_loss_kwargs(loss_fn: Callable[..., Tensor] | None, counts: Tensor | None):
        if loss_fn is None or counts is None:
            return {}

        counts_tensor = counts.clone().detach().to(dtype=torch.float32)
        sig = inspect.signature(loss_fn)

        kwargs = {}
        if "counts" in sig.parameters:
            kwargs["counts"] = counts_tensor
        elif "weight" in sig.parameters:
            weights = _compute_frequency_weights(counts_tensor)
            kwargs["weight"] = weights
        return kwargs

    def _initialize_weights(
        self,
        edge_counts: torch.Tensor | None,
        node_counts: torch.Tensor | None,
    ):
        """Initialize global counts and corresponding kwargs for the loss functions."""
        self.edge_loss_kwargs = self._get_loss_kwargs(self.edge_loss_fn, edge_counts)
        self.node_loss_kwargs = self._get_loss_kwargs(self.node_loss_fn, node_counts)

    def _compute_single_loss(
        self,
        loss_fn: Callable[..., Tensor],
        pred: list[Tensor],
        target: Tensor,
        loss_kwargs: dict | None = None,
    ) -> Tensor:
        """Helper to handle list or standard outputs."""
        if loss_kwargs is None:
            loss_kwargs = {}

        # sanity check to prevent running on networks that return tensor which would cause
        # size mismatch
        if (pred is not None) and not isinstance(pred, list):
            raise ValueError("Expected predictions to be list of Tensors.")

        if not pred:  # empty list, None, or tensor
            return torch.tensor(0.0, device=target.device, requires_grad=True)

        losses = []
        for p in pred:
            if p.numel() == 0:
                losses.append(p.sum() * 0.0)
            else:
                losses.append(loss_fn(p, target, **loss_kwargs))

        if not self.aggregate_losses:
            return losses[-1]
        elif self.aggregate_losses == "mean":
            return torch.stack(losses).mean()
        elif self.aggregate_losses == "sum":
            return torch.stack(losses).sum()
        else:
            raise ValueError(
                f"{self.aggregate_losses=} unknown. Please use `sum`, `mean`, "
                "or handle aggregation in model."
            )

    def forward(
        self,
        predictions: PRED_DICT,
        targets: TARGET_DICT,
    ) -> Tensor:
        """Compute loss."""
        losses = []
        if self.edge_loss_fn is not None:
            edge_loss = self._compute_single_loss(
                self.edge_loss_fn,
                predictions["edge_predictions"],
                targets["edge_labels"],
                self.edge_loss_kwargs,
            )
            losses.append(self.edge_loss_weight * edge_loss)

        if self.node_loss_fn is not None:
            node_loss = self._compute_single_loss(
                self.node_loss_fn,
                predictions.get("node_predictions", []),
                targets["node_labels"],
                self.node_loss_kwargs,
            )
            losses.append(self.node_loss_weight * node_loss)

        return sum(losses)


def _compute_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Compute smoothed frequency weights.

    Weights are inversely proportional to the square root of class frequencies.
    """
    counts = counts.float()
    freq = counts / counts.sum()
    weights = 1.0 / (torch.sqrt(freq) + 1e-6)
    weights[counts == 0] = 0

    classes_present = (counts > 0).sum()
    if classes_present > 0:
        weights = weights / weights.sum() * classes_present

    return weights


def _compute_balanced_weights(
    target: torch.Tensor | None = None,
    num_classes: int | None = None,
    beta: float = 0.999,
    counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if counts is None:
        if target is None or num_classes is None:
            raise ValueError("Must provide either counts or both target and num_classes.")
        counts = torch.bincount(target, minlength=num_classes).float()
    else:
        counts = counts.float()

    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-6)
    weights[counts == 0] = 0

    classes_present = (counts > 0).sum()
    if classes_present > 0:
        weights = weights / weights.sum() * classes_present

    return weights, counts


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss.

    Args:
        input: Class logits.
        target: Class indices.
        alpha: Scalar weight applied to every sample.
        gamma: Focusing strength. 0 recovers cross entropy, larger values suppress
            easy samples further.
        reduction: 'mean' or 'sum'. Any other value returns the per-sample losses.
    """
    ce_loss = F.cross_entropy(input, target, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def class_balanced_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    loss_type: str = "focal",
    beta: float = 0.999,
    gamma: float = 2.0,
    reduction: str = "mean",
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-balanced loss.

    Per-class weights are `(1 - beta) / (1 - beta ** n_c)` for class counts `n_c`,
    normalized to sum to the number of classes present. Counts are taken from `counts`
    when given (e.g. over the full training set) and from `target` otherwise, in which
    case the are only computed for current batch.

    Args:
        input: Class logits.
        target: Class indices.
        num_classes: Number of predicted classes. 1 switches to sigmoid and binary cross
            entropy, balanced over inactive against active.
        loss_type: Base loss, either 'focal' or 'cross_entropy' ('ce').
        beta: Reweighting strength, between uniform weights at 0 and inverse frequency
            approaching 1.
        gamma: Focusing strength of the focal base loss.
        reduction: 'mean', 'sum', or 'weighted_mean' to divide by the number of samples
            not labelled class 0. Any other value returns the per-sample losses.
        counts: Class counts the weights are derived from.
    """
    if num_classes == 1:
        # binary classification: use BCE with sigmoid
        input = input.squeeze(-1)
        base_loss_fn = F.binary_cross_entropy_with_logits
        base_loss = base_loss_fn(input, target.float(), reduction="none")
        if loss_type == "focal":
            pt = torch.exp(-base_loss)
            base_loss = (1 - pt) ** gamma * base_loss
        # use 2 classes for balancing (inactive vs active); counts from the dataset
        # have shape (1,) when num_edge_classes=1, which is wrong for binary balancing
        _binary_counts = counts if (counts is not None and counts.numel() == 2) else None
        weights, _ = _compute_balanced_weights(
            target=target, num_classes=2, beta=beta, counts=_binary_counts
        )
    else:
        if loss_type == "focal":
            base_loss = focal_loss(
                input, target, alpha=1.0, gamma=gamma, reduction="none"
            )
        elif loss_type in ["cross_entropy", "ce"]:
            base_loss = F.cross_entropy(input, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'focal' or 'ce'.")
        weights, _ = _compute_balanced_weights(
            target=target, num_classes=num_classes, beta=beta, counts=counts
        )

    weights = weights.to(target.device)
    batch_weights = weights.gather(0, target)
    loss = batch_weights * base_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "weighted_mean":
        # always use batch counts for denominator to reflect current batch size properly
        # for binary (num_classes=1) use 2 as minlength so batch_counts[1] exists
        _minlength = max(num_classes, 2)
        batch_counts = torch.bincount(target, minlength=_minlength)
        return loss.sum() / torch.clamp_min(batch_counts[1:].sum(), 1)
    return loss
