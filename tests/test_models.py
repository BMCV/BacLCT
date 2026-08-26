"""GNN forward pass, DDP-safe empty batches, and the training-log helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from baclct.models.lightning_model import TrackingModel
from baclct.models.loss import MultiTaskLoss
from baclct.models.model import (
    NodeModel,
    TimeAwareNodeModel,
)
from baclct.tracking.postprocessing import select_extreme_samples
from baclct.utils.model import one_vs_rest_counts

HANDCRAFTED_FEATS, DEEP_FEATS, EDGE_FEATS = 8, 16, 4
NUM_EDGE_CLASSES, NUM_NODE_CLASSES = 3, 4


@pytest.fixture
def mock_graph_model():
    """Stand-in for `MPModel` that classifies edges and nodes with one linear layer."""
    node_feats = HANDCRAFTED_FEATS + DEEP_FEATS

    class SimpleGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_classifier = nn.Linear(
                node_feats * 2 + EDGE_FEATS, NUM_EDGE_CLASSES
            )
            self.node_classifier = nn.Linear(node_feats, NUM_NODE_CLASSES)

        def forward(self, x_handcrafted, x_deep, edge_index, edge_attr):
            x = torch.cat([x_handcrafted, x_deep], dim=1)
            src, dst = edge_index
            edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
            return {
                "edge_predictions": [self.edge_classifier(edge_input)],
                "node_predictions": [self.node_classifier(x)],
            }

    return SimpleGNN()


@pytest.fixture
def lit_model(mock_graph_model):
    """`TrackingModel` around the stand-in, with both classification heads active."""
    return TrackingModel(
        graph_model=mock_graph_model,
        criterion=MultiTaskLoss(nn.CrossEntropyLoss(), nn.CrossEntropyLoss()),
        optimizer=torch.optim.AdamW,
        num_edge_classes=NUM_EDGE_CLASSES,
        num_node_classes=NUM_NODE_CLASSES,
    )


@pytest.fixture
def empty_batch():
    """An empty batch, as frame dropout produces."""
    return Data(
        x_handcrafted=torch.empty((0, HANDCRAFTED_FEATS)),
        x_deep=torch.empty((0, DEEP_FEATS)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, EDGE_FEATS)),
        y_edges=torch.empty((0,), dtype=torch.long),
        y_nodes=torch.empty((0,), dtype=torch.long),
        num_nodes=0,
        node_mapping=torch.empty((0,), dtype=torch.long),
    )


@pytest.mark.parametrize(
    "node_model_class",
    [NodeModel, TimeAwareNodeModel],
)
@pytest.mark.parametrize(
    "handcrafted_features, deep_features, edge_features, hidden_dim",
    [
        (4, 8, 8, 8),
        # distinct dims, so a layer wired to the wrong one cannot fit
        (16, 32, 4, 8),
    ],
)
def test_mpn_model_forward_pass(
    node_model_class,
    handcrafted_features,
    deep_features,
    edge_features,
    hidden_dim,
    mpn_model_factory,
):
    """Both node models wire up end to end and score every edge."""
    num_nodes = 10
    num_edges = 20
    num_classes = 3

    model = mpn_model_factory(
        node_model_class=node_model_class,
        handcrafted_features=handcrafted_features,
        deep_features=deep_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
    )

    x_handcrafted = torch.randn(num_nodes, handcrafted_features)
    x_deep = torch.randn(num_nodes, deep_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_features)
    output = model(
        x_handcrafted=x_handcrafted,
        x_deep=x_deep,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    assert "edge_predictions" in output
    assert output["edge_predictions"][-1].shape == (num_edges, num_classes)


def test_empty_batch_training_step_no_deadlock(
    monkeypatch, mock_graph_model, lit_model, empty_batch
):
    """An empty batch keeps the DDP backward and loss-logging collectives symmetric.

    The loss has to require grad and `_log_loss` has to run before the early return, or
    this rank skips an all-reduce the others are waiting on.
    """
    mock_log_loss = MagicMock()
    monkeypatch.setattr(lit_model, "_log_loss", mock_log_loss)

    mock_trainer = MagicMock()
    mock_trainer.is_sanity_checking = False
    mock_trainer.current_epoch = 5
    lit_model.trainer = mock_trainer

    outputs = lit_model.training_step(empty_batch, batch_idx=0)
    loss = outputs["loss"]

    assert loss.requires_grad is True, (
        "Loss on empty batch must require_grad to trigger DDP hooks."
    )

    loss.backward()
    for name, param in mock_graph_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient."
        assert (param.grad == 0).all(), (
            f"Parameter {name} gradient should be exactly zero."
        )

    assert mock_log_loss.called, "_log_loss was not called on empty batch."


def test_empty_batch_predict_step_returns_no_frames(lit_model, empty_batch):
    """A frame or patch without edges yields the empty template, which predict must skip.

    The model still forwards it, so the guard has to look at the tensor, not at the list
    holding it.
    """
    edge_df, node_df = lit_model.predict_step(empty_batch, batch_idx=0)

    assert edge_df is None
    assert node_df is None


@pytest.mark.parametrize("phase", ["val", "test"])
def test_epoch_end_metrics_symmetric_on_empty_outputs(monkeypatch, lit_model, phase):
    """Epoch end drives the metric collective even when a rank saw only empty batches.

    `metrics.compute()` inside `_log_metrics` is a cross-rank all-reduce, so a rank that
    accumulated nothing must still call it and reset.
    """
    mock_log_metrics = MagicMock()
    monkeypatch.setattr(lit_model, "_log_metrics", mock_log_metrics)

    metrics = getattr(lit_model, f"{phase}_metrics")
    mock_reset = MagicMock()
    monkeypatch.setattr(metrics, "reset", mock_reset)

    # outputs default to empty defaultdicts, i.e. this rank accumulated nothing
    hook_name = "on_validation_epoch_end" if phase == "val" else "on_test_epoch_end"
    getattr(lit_model, hook_name)()

    assert mock_log_metrics.called, (
        f"_log_metrics was not called on empty {phase} outputs; ranks would desync."
    )
    assert mock_reset.called, f"{phase}_metrics.reset was not called."


def test_class_counts_reconstructs_confusion():
    """Per-class one-vs-rest TP/FP/FN/TN match a hand-built 3-class confusion."""
    y = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 1])

    counts = one_vs_rest_counts(y, y_pred, num_classes=3)

    total = len(y)
    for cnt in counts.values():
        assert cnt["tp"] + cnt["fp"] + cnt["fn"] + cnt["tn"] == total

    # class 2 (rare division-like): one correct (idx5), one missed as class 1 (idx6)
    assert counts[2] == {"tp": 1, "fp": 0, "fn": 1, "tn": 5}
    # class 1: tp at idx2,3, fn idx4 (->0), fp idx1 (0->1) and idx6 (2->1)
    assert counts[1]["tp"] == 2
    assert counts[1]["fn"] == 1
    assert counts[1]["fp"] == 2


def test_extreme_samples_rank_by_loss_and_dedupe():
    """Log figures show the highest-loss samples once per cell, mirrors collapsed."""
    stats = pl.DataFrame(
        {
            # (1, 2) and (2, 1) are the same pair in a bidirectional graph
            "src": [1, 2, 3, 3, 5],
            "dst": [2, 1, 4, 6, 7],
            "y": [0, 0, 1, 1, 1],
            "y_pred": [1, 1, 0, 1, 1],
            "p_true": [0.1, 0.1, 0.2, 0.5, 0.9],
            "loss": [8.0, 8.0, 4.0, 2.0, 0.5],
        }
    )

    # n_worst=2 leaves room for src 3 twice, so only the dedup keeps it out
    rows = dict(select_extreme_samples(stats, n_worst=2, n_best=1))

    assert rows[0].height == 1, "Mirrored edges must collapse to a single sample."
    assert rows[1]["src"].to_list() == [3, 5]
    assert rows[1]["loss"].to_list() == [4.0, 0.5]
