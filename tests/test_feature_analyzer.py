"""Feature and prediction extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import torch

from baclct.data.dataset import GraphDataset
from baclct.features.extractors import HandcraftedExtractor
from baclct.features.graph import EdgeFinder
from baclct.models.lightning_model import TrackingModel
from baclct.models.model import NodeModel
from baclct.viz.features import FeatureAnalyzer


@pytest.fixture
def analyzer_and_dataset(
    single_sequence_data, toy_handcrafted_features, mpn_model_factory
):
    """Analyzer over one prepared sequence, with an untrained model behind it."""
    data_dir, feature_dir, seq_id = single_sequence_data

    edge_finder = EdgeFinder(
        prune_edges_by=("radius", 1.5),
        feature_names=("dist_spat", "dist_temp"),
        n_jobs=1,
    )
    handcrafted_extractor = HandcraftedExtractor(
        n_jobs=1, feature_names=["area", "axis_major_length"]
    )

    mock_deep_extractor = MagicMock()
    mock_deep_extractor.name = "mock_deep"
    mock_deep_extractor.return_value = torch.randn(len(toy_handcrafted_features), 5)
    dataset = GraphDataset(
        data_dir=data_dir,
        feature_dir=feature_dir,
        sequence_id=seq_id,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=handcrafted_extractor,
        deep_feature_extractor=mock_deep_extractor,
        precompute_edges=True,
    )
    dataset.deep_feats = torch.randn(len(dataset.node_feats), 5)

    x_handcrafted, x_deep = dataset._get_node_features(dataset.node_feats)
    assert x_handcrafted is not None and x_deep is not None

    graph_model = mpn_model_factory(
        node_model_class=NodeModel,
        handcrafted_features=x_handcrafted.shape[-1],
        deep_features=x_deep.shape[-1],
        edge_features=2,
        hidden_dim=8,
    )

    def dummy_criterion(preds, targets):
        return torch.nn.functional.cross_entropy(
            preds["edge_predictions"][-1], targets["edge_labels"]
        )

    model = TrackingModel(
        graph_model=graph_model,
        criterion=dummy_criterion,  # type: ignore
        optimizer=torch.optim.AdamW,
        num_edge_classes=3,
        num_node_classes=0,
    )

    analyzer = FeatureAnalyzer(dataset=dataset, model=model)
    return analyzer, dataset, seq_id


@pytest.mark.parametrize(
    "feature_type, feature_state",
    [
        ("edge", "initial"),
        ("handcrafted", "initial"),
        ("deep", "initial"),
        ("combined", "initial"),
        ("edge", "learned"),
        # the learned branch treats the three node feature types identically
        ("handcrafted", "learned"),
    ],
)
def test_feature_analyzer_features(analyzer_and_dataset, feature_type, feature_state):
    """Initial and learned features keep their columns and one row per edge or node."""
    analyzer, dataset, seq_id = analyzer_and_dataset
    edge_data = dataset.edge_data
    if isinstance(edge_data, pl.LazyFrame):
        edge_data = edge_data.collect()

    features = analyzer.get_features(
        feature_type=feature_type, feature_state=feature_state
    )

    assert (features["sequence_id"] == seq_id).all()
    if feature_type == "edge":
        assert {"src", "dst", "y"} <= set(features.columns)
        assert features.height == edge_data.height
    else:
        assert {"index", "label"} <= set(features.columns)
        assert features.height == dataset.node_feats.height

    if feature_state == "learned" and feature_type == "edge":
        assert features["learned_features"].null_count() == 0
    elif feature_state == "learned":
        # only a node that no surviving edge touches is left without an embedding
        in_graph = pl.concat([edge_data["src"], edge_data["dst"]]).unique()
        missing = features.filter(pl.col("learned_features").is_null())["index"]
        assert not missing.is_in(in_graph.implode()).any()
    elif feature_type == "edge":
        # dist_temp is log-normalized in place, so a shifted schema lands elsewhere
        dist_temp = edge_data["dist_temp"].to_numpy() * dataset.spacing["t"]
        assert np.allclose(
            features["dist_temp"].to_numpy(),
            np.sign(dist_temp) * np.log1p(np.abs(dist_temp)),
        )
        assert "dist_spat" in features.columns
    else:
        if feature_type in ("handcrafted", "combined"):
            assert "area_norm" in features.columns
        if feature_type in ("deep", "combined"):
            assert "deep_features" in features.columns


def test_feature_analyzer_misclassifications(analyzer_and_dataset):
    """Only the edges whose predicted class differs from the label are returned."""
    analyzer, _, seq_id = analyzer_and_dataset

    misclassifications = analyzer.get_misclassifications()

    assert seq_id in misclassifications
    wrong = misclassifications[seq_id]
    assert {"y", "y_pred", "p0"} <= set(wrong.columns)
    assert wrong.height > 0
    assert (wrong["y"] != wrong["y_pred"]).all()
