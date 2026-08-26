"""End-to-end runs over the toy sequence: training, prediction, tracking, and export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import hydra
import lightning as L
import numpy as np
import polars as pl
import pytest
import tifffile
import torch
import torch.nn as nn
from conftest import prepare_dataset_structure
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

from baclct.api import BacLCT
from baclct.data.dataset import GraphDataset, TrackingDataset
from baclct.features.extractors import HandcraftedExtractor
from baclct.features.graph import EdgeFinder
from baclct.io import (
    create_trajectory_masks,
    export_tracking_results_ctc,
)
from baclct.models.lightning_model import TrackingModel
from baclct.models.loss import MultiTaskLoss
from baclct.models.model import TimeAwareNodeModel
from baclct.tracking.metrics import compute_ctcmetrics
from baclct.tracking.tracker import LAPTracker
from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def test_full_pipeline_on_perfect_predictions(
    tmp_path: Path,
    toy_handcrafted_features: pl.DataFrame,
    perfect_predictions_factory,
    toy_masks: np.ndarray,
    toy_data_dir: Path,
    toy_lineage: pl.DataFrame,
    basic_extractors,
):
    """Perfect edge predictions track to a CTC-valid export scoring TRA 1.0."""
    sequence_data = {
        "01": {
            "node_features": toy_handcrafted_features,
            "masks": toy_masks,
            "lineage": toy_lineage,
        }
    }
    data_dir, feature_dir = prepare_dataset_structure(tmp_path, sequence_data)

    _, hc_extractor = basic_extractors
    dataset = GraphDataset(
        data_dir=data_dir,
        feature_dir=feature_dir,
        sequence_id="01",
        edge_finder=None,  # type: ignore - not used
        handcrafted_feature_extractor=hc_extractor,
        deep_feature_extractor=None,
        data_format="ctc",
        segmentation_name="GT",
        precompute_edges=False,
    )

    tracker = LAPTracker(
        dataset=dataset,
        predictions=perfect_predictions_factory(toy_handcrafted_features, [1]),
        thr_corr=0.5,
        norm_fn="softmax",
        segmentation_correction=None,
    )
    predicted_tracks = tracker.track()

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    pred_masks = create_trajectory_masks(
        tracks=predicted_tracks,
        masks=dataset.masks,
        label_old="label",
        label_new="label_track",
    )
    export_tracking_results_ctc(
        tracks=predicted_tracks,
        masks_tracked=pred_masks,
        res_dir=pred_dir,
    )

    # py_ctcmetrics expects the GT lineage and masks under a TRA subdirectory
    gt_dir = tmp_path / "gt"
    tra_dir = gt_dir / "TRA"
    tra_dir.mkdir(parents=True)
    for i, mask_frame in enumerate(toy_masks):
        tifffile.imwrite(tra_dir / f"t{i:03d}.tif", mask_frame)
    shutil.copy(toy_data_dir / "man_track.txt", tra_dir / "man_track.txt")

    assert compute_ctcmetrics(gt_dir=gt_dir, pred_dir=pred_dir, validation_only=True)
    results = compute_ctcmetrics(gt_dir=gt_dir, pred_dir=pred_dir, metrics=["TRA"])
    assert isinstance(results, dict)  # only for ty

    tra_metric = results.get("TRA")
    assert tra_metric is not None, f"Could not find TRA metric in results: {results}"
    assert tra_metric == 1.0, f"Expected TRA score of 1.0, but got {tra_metric}"


def _run_training_loop(
    data_dir: Path,
    feature_dir: Path,
    graph_model: torch.nn.Module,
    criterion: nn.Module,
    num_node_classes: int = 0,
    **dataset_kwargs,
):
    """Run a two-epoch training loop, returning its losses, batches, outputs and model."""
    datamodule = TrackingDataset(
        data_dir=data_dir,
        feature_dir=feature_dir,
        sequence_ids=["01"],
        # reproducibility_data_dir pre-seeds nodes.parquet with more columns than this
        # extractor's own props would recompute, so trust it rather than rebuild it away
        trust_cache=True,
        edge_finder=EdgeFinder(
            # multiplier on half the major axis
            prune_edges_by=("radius", 1.5),
            feature_names=("dist_spat", "dist_temp", "iou", "overlap"),
            n_jobs=1,
            center_name="center",
        ),
        handcrafted_feature_extractor=HandcraftedExtractor(
            feature_names=[
                "area",
                "orientation",
                "axis_major_length",
                "axis_minor_length",
            ],
            extra_props=["center_local"],
            n_jobs=1,
        ),
        data_format="ctc",
        segmentation_name="GT",
        fold=0,
        batch_size=1,
        precompute_edges=True,
        **dataset_kwargs,
    )

    model = TrackingModel(
        graph_model=graph_model,  # type: ignore
        criterion=criterion,  # type: ignore
        lr=1e-3,
        num_node_classes=num_node_classes,
        optimizer=torch.optim.AdamW,
    )

    class LossCallback(L.Callback):
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.batches = []
            self.outputs = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.batches.append(batch)
            self.outputs.append(outputs)
            self.train_losses.append(outputs["loss"].item())

        def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx=0,
        ):
            self.val_losses.append(outputs["loss"].item())

    loss_callback = LossCallback()
    trainer = L.Trainer(
        limit_train_batches=5,
        limit_val_batches=5,
        max_epochs=2,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        # losses are compared exactly; "warn" so MPS does not raise
        deterministic="warn",
        callbacks=[loss_callback],
    )
    trainer.fit(model, datamodule=datamodule)

    return (
        loss_callback.train_losses,
        loss_callback.val_losses,
        loss_callback.batches,
        loss_callback.outputs,
        model,
    )


@pytest.mark.slow
@pytest.mark.parametrize("multitask", (False, True), ids=["edges_only", "multitask"])
def test_training_reproducibility(reproducibility_data_dir, mpn_model_factory, multitask):
    """Two runs at one seed give the same losses, and the loss falls over the run."""
    data_dir, feature_dir = reproducibility_data_dir
    seed = 1510
    handcrafted_features = 4
    num_node_classes = 4 if multitask else 0

    def build_model():
        return mpn_model_factory(
            node_model_class=TimeAwareNodeModel,
            handcrafted_features=handcrafted_features,
            deep_features=0,
            edge_features=4,
            hidden_dim=16,
            node_classifier=(
                nn.Linear(handcrafted_features, num_node_classes) if multitask else None
            ),
        )

    if multitask:
        criterion: nn.Module = MultiTaskLoss(nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    else:

        def edge_loss(p, t):
            return nn.CrossEntropyLoss()(p["edge_predictions"][-1], t["edge_labels"])

        criterion = cast(nn.Module, edge_loss)

    L.seed_everything(seed)
    losses1, losses1_val, batches1, outputs1, _ = _run_training_loop(
        data_dir, feature_dir, build_model(), criterion, num_node_classes
    )
    assert losses1[0] > losses1[-1], (
        f"Training loss did not decrease during training.\n"
        f"Initial loss: {losses1[0]}, Final loss: {losses1[-1]}\n"
        f"Batches: {batches1}\n"
        f"Run 1 outputs: {outputs1}"
    )
    assert losses1_val[0] > losses1_val[-1], (
        f"Validation loss did not decrease during training.\n"
        f"Initial loss: {losses1_val[0]}, Final loss: {losses1_val[-1]}"
    )

    if torch.backends.mps.is_available():
        pytest.skip(
            "MPS+AMP is not deterministic across runs. Training and validation loss "
            "properly decrease. Reproducibility only tested on CUDA."
        )

    L.seed_everything(seed)
    losses2, _, batches2, outputs2, _ = _run_training_loop(
        data_dir, feature_dir, build_model(), criterion, num_node_classes
    )

    assert losses1 == losses2, (
        f"Training was not reproducible with seed {seed}.\n"
        f"Run 1 losses: {losses1}\n"
        f"Run 2 losses: {losses2}\n"
        f"Run 1 batches: {batches1}\n"
        f"Run 2 batches: {batches2}\n"
        f"Run 1 outputs: {outputs1}\n"
        f"Run 2 outputs: {outputs2}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("dropout", (0.5, 1.0), ids=["partial_dropout", "total_dropout"])
def test_training_with_dropout(reproducibility_data_dir, mpn_model_factory, dropout):
    """Graph dropout never trains on an empty graph, however much it removes.

    At 1.0 every item loses all nodes and edges and falls back to the unaugmented graph.
    One rate covers all three, since any of them at 1.0 empties the graph on its own.
    """
    data_dir, feature_dir = reproducibility_data_dir
    L.seed_everything(1510)

    graph_model = mpn_model_factory(
        node_model_class=TimeAwareNodeModel,
        handcrafted_features=4,
        deep_features=0,
        edge_features=4,
        hidden_dim=16,
    )

    def criterion(p, t):
        return nn.CrossEntropyLoss()(p["edge_predictions"][-1], t["edge_labels"])

    _, _, batches, _, _ = _run_training_loop(
        data_dir,
        feature_dir,
        graph_model,
        cast(nn.Module, criterion),
        graph_frame_dropout=dropout,
        graph_node_dropout=dropout,
        graph_edge_dropout=dropout,
    )

    assert batches
    assert not any(bool(batch.empty_flag.any()) for batch in batches)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("task", "precompute_edges"),
    (("tracking", True), ("tracking_with_states", False)),
)
def test_train_and_predict_flow(
    reproducibility_data_dir, task, precompute_edges, toy_images, toy_masks
):
    """A trained checkpoint records its features and tracks reproducibly from disk.

    Task and edge source are paired rather than combined, since they do not interact.
    """
    if torch.backends.mps.is_available():
        pytest.skip(
            "MPS+AMP is not deterministic across runs. Reproducibility only "
            "tested on CUDA."
        )

    data_dir, feature_dir = reproducibility_data_dir
    run_dir = data_dir.parent / "run"
    # don't create directory, as Hydra will do it.

    num_node_classes = 3 if task == "tracking_with_states" else "null"

    overrides = [
        "dataset=spores",
        f"task={task}",
        f"data.precompute_edges={precompute_edges}",
        f"paths.output_dir={run_dir!s}",
        f"paths.data_dir={data_dir!s}",
        f"paths.feature_dir={feature_dir!s}",
        f"num_node_classes={num_node_classes}",
        "debug=tests",
    ]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # train
    with hydra.initialize_config_module(
        config_module="baclct.config",
        job_name="train_test",
        version_base="1.3",
    ):
        cfg_train = hydra.compose(
            config_name="default", overrides=overrides + ["callbacks=tests"]
        )

        # mimic @hydra.main by saving the config to disk for the prediction step
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg_train, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        pipeline_train = BacLCT(cfg_train)
        checkpoint_path, _ = pipeline_train.run_training()

    # if dataloader is empty lightning does not raise but returns "." as checkpoint
    assert checkpoint_path is not None and checkpoint_path != ".", (
        "Training did not produce a checkpoint path."
    )
    assert Path(checkpoint_path).exists(), (
        f"Checkpoint file not found at {checkpoint_path}"
    )

    # training records the features the checkpoint was trained on
    info_path = run_dir / "features.json"
    assert info_path.exists(), "Training did not write features.json."
    info = json.loads(info_path.read_text())
    train = info["features"]["train"]
    assert train["edge_features"]["names"], "Edge feature names missing in info."
    node = train["node_features"]
    assert node["n"] > 0
    assert set(node["mean"]) == set(node["names"])
    # split assignment is recorded for reproducibility, test is listed but not loaded
    assert info["splits"]["train"], "Train split missing in info."

    # predict
    with hydra.initialize_config_module(
        config_module="baclct.config",
        job_name="predict_test",
        version_base="1.3",
    ):
        # the model and checkpoint come from the run dir; inference defaults to its own
        # graph geometry, so pin it to what the model was trained with ('debug=tests'
        # uses graph_num_steps=1)
        pipeline_pred = BacLCT(
            run_dir,
            config_overrides={
                "checkpoint": str(checkpoint_path),
                "data": {"graph_num_steps": 1},
            },
        )

        L.seed_everything(1510, workers=True)
        _, tracks1 = pipeline_pred.track(
            images=toy_images,
            masks=toy_masks,
            sequence_id="01",
        )
        dataset1 = pipeline_pred.dataset
        edge_preds1, node_preds1 = pipeline_pred.edge_preds, pipeline_pred.node_preds

        L.seed_everything(1510, workers=True)
        _, tracks2 = pipeline_pred.track(
            images=toy_images,
            masks=toy_masks,
            sequence_id="01",
        )
        dataset2 = pipeline_pred.dataset
        edge_preds2, node_preds2 = pipeline_pred.edge_preds, pipeline_pred.node_preds

        assert isinstance(pipeline_pred.dataset, GraphDataset)
        assert not pipeline_pred.dataset.training

    assert isinstance(dataset1, GraphDataset)
    assert isinstance(dataset2, GraphDataset)
    assert dataset1.node_feats.equals(dataset2.node_feats), "Features are not identical"
    if dataset1.edge_data is not None:
        edges1 = dataset1.edge_data
        edges2 = dataset2.edge_data
        if hasattr(edges1, "collect"):
            edges1 = edges1.collect()  # type: ignore
            edges2 = edges2.collect()  # type: ignore

        assert_frame_equal(edges1, edges2, check_row_order=True)  # type: ignore

    assert edge_preds1 is not None and edge_preds2 is not None
    assert_frame_equal(edge_preds1, edge_preds2, rel_tol=1e-5, abs_tol=1e-5)
    assert tracks1.equals(tracks2), "Tracking results are not identical"

    assert (node_preds1 is None) == (task == "tracking")
    if node_preds1 is not None:
        assert node_preds2 is not None
        assert_frame_equal(node_preds1, node_preds2, rel_tol=1e-5, abs_tol=1e-5)


def test_feature_scaling(reproducibility_data_dir):
    """Normalized node and edge features stay finite, non-zero and under the cutoff."""
    data_dir, feature_dir = reproducibility_data_dir
    cutoff = 20.0

    edge_finder = EdgeFinder(
        prune_edges_by=("radius", 1.5),  # multiplier on half the major axis
        feature_names=(
            "dist_spat",
            "dist_temp",
            "iou",
            "overlap",
            "cosine_similarity",
        ),
        extra_features=["relative_size"],
        edge_normalization="cell_size",
        n_jobs=1,
    )
    handcrafted_feature_extractor = HandcraftedExtractor(
        props=[
            "area",
            "axis_major_length",
            "axis_minor_length",
            "intensity_mean",
            "intensity_min",
            "intensity_max",
            "orientation",
            "eccentricity",
        ],
        extra_props=["center_local", "thickness"],
        feature_names=[
            "area",
            "intensity",
            "eccentricity",
            "thickness",
        ],
        feature_norm_fn="scale_relative_size",
        n_jobs=1,
    )
    datamodule = TrackingDataset(
        data_dir=data_dir,
        feature_dir=feature_dir,
        sequence_ids=["01"],
        edge_finder=edge_finder,
        handcrafted_feature_extractor=handcrafted_feature_extractor,
        data_format="ctc",
        segmentation_name="GT",
        fold=0,
        batch_size=1,
        precompute_edges=True,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    logger.info(f"Handcrafted: {handcrafted_feature_extractor.extracted_features}")
    assert handcrafted_feature_extractor.extracted_features

    assert hasattr(batch, "x_handcrafted") and isinstance(
        batch.x_handcrafted, torch.Tensor
    )
    node_features = batch.x_handcrafted
    assert node_features.numel() > 0, "Node features are empty."
    assert not torch.all(node_features == 0), "All node features are zero."
    assert not torch.isnan(node_features).any(), "Node features contain NaNs."

    max_val_node = torch.abs(node_features).max()
    try:
        assert max_val_node < cutoff, (
            f"Node features are too large: max value is {max_val_node}"
        )
    except AssertionError as err:
        df = pl.DataFrame(
            node_features,
            schema=dict.fromkeys(
                handcrafted_feature_extractor.extracted_features, pl.Float32
            ),
        )
        logger.error(df.describe())
        raise err

    assert hasattr(batch, "edge_attr") and isinstance(batch.edge_attr, torch.Tensor)
    edge_features = batch.edge_attr
    assert edge_features.numel() > 0, "Edge features are empty."
    assert not torch.all(edge_features == 0), "All edge features are zero."
    assert not torch.isnan(edge_features).any(), "Edge features contain NaNs."

    max_val_edge = torch.abs(edge_features).max()
    assert max_val_edge < cutoff, (
        f"Edge features are too large: max value is {max_val_edge}"
    )


@pytest.mark.parametrize("with_images", [True, False], ids=["with_images", "masks_only"])
def test_pipeline_handcrafted_features_images_optional(
    toy_images: np.ndarray,
    toy_masks: np.ndarray,
    basic_extractors,
    with_images: bool,
):
    """Masks alone carry the handcrafted pipeline from features to an assembled graph."""
    edge_finder, hc_extractor = basic_extractors
    images = toy_images if with_images else None

    dataset = GraphDataset(
        data_dir=None,
        feature_dir=None,
        sequence_id="test",
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        deep_feature_extractor=None,
        images=images,
        masks=toy_masks,
    )

    assert dataset.images is (toy_images if with_images else None)
    assert dataset.node_feats.height > 0
    assert "axis_minor_length" in dataset.node_feats.columns
    # percentiles are the one thing that needs intensities
    assert (dataset.image_percentiles is not None) == with_images

    item = dataset[0]
    assert item.num_nodes > 0  # type: ignore
    assert item.num_edges > 0
