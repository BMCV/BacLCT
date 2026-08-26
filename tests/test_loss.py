"""Test loss functions."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from baclct.models.loss import MultiTaskLoss, class_balanced_loss


@pytest.mark.parametrize("reduction", ["mean", "sum", "weighted_mean"])
@pytest.mark.parametrize(
    "loss_type, gamma", [("focal", 0.0), ("focal", 2.0), ("cross_entropy", 0.0)]
)
def test_class_balanced_loss(loss_type, gamma, reduction):
    """Each reduction of the per-sample losses, and gamma=0 recovering cross entropy."""
    num_classes = 3
    targets = torch.tensor([0] * 8 + [1, 2], dtype=torch.long)
    logits = torch.randn(10, num_classes)

    def cb_loss(reduction: str):
        return class_balanced_loss(
            logits,
            targets,
            num_classes=num_classes,
            loss_type=loss_type,
            gamma=gamma,
            reduction=reduction,
        )

    # an unnamed reduction returns the per-sample losses
    per_sample = cb_loss("none")
    loss = cb_loss(reduction)

    expected = {
        "mean": per_sample.mean(),
        "sum": per_sample.sum(),
        # divides by the samples not labelled class 0
        "weighted_mean": per_sample.sum() / 2,
    }[reduction]
    assert loss.ndim == 0
    assert torch.allclose(loss, expected)

    if loss_type == "focal" and gamma == 0:
        ce_loss = class_balanced_loss(
            logits,
            targets,
            num_classes=num_classes,
            loss_type="cross_entropy",
            reduction=reduction,
        )
        assert torch.allclose(loss, ce_loss)


def test_multitask_loss_global_vs_batch_weights():
    """Dataset-wide class counts reach the loss and change it against batch counts."""
    torch.manual_seed(1510)
    num_classes = 3

    target = torch.tensor([0] * 10 + [1] * 50 + [2] * 5)
    pred = [torch.randn(len(target), num_classes, requires_grad=True)]

    edge_loss_fn = partial(
        class_balanced_loss,
        num_classes=num_classes,
        loss_type="focal",
        beta=0.999,
        gamma=2.0,
        reduction="mean",
    )

    criterion_batch = MultiTaskLoss(edge_loss_fn=edge_loss_fn, node_loss_fn=None)
    loss_batch = criterion_batch({"edge_predictions": pred}, {"edge_labels": target})

    criterion_global = MultiTaskLoss(edge_loss_fn=edge_loss_fn, node_loss_fn=None)
    global_counts = torch.tensor([10000.0, 50000.0, 5000.0])
    criterion_global._initialize_weights(edge_counts=global_counts, node_counts=None)

    loss_global = criterion_global({"edge_predictions": pred}, {"edge_labels": target})
    assert not torch.allclose(loss_batch, loss_global)

    assert "counts" in criterion_global.edge_loss_kwargs
    assert torch.allclose(criterion_global.edge_loss_kwargs["counts"], global_counts)
    assert criterion_global.node_loss_kwargs == {}
