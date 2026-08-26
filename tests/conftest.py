"""Fixtures shared across the full test suite, all derived from one toy sequence.

`tests/assets/toy_data/` holds four frames of real microscopy data with a CTC lineage and
life cycle states. A test that needs cells starts from `toy_handcrafted_features`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import tifffile
import torch
import torch.nn as nn
import yaml

from baclct.features.extractors import HandcraftedExtractor
from baclct.features.graph import EdgeFinder
from baclct.io import load_lineage
from baclct.models.model import (
    EdgeModel,
    MPModel,
    NodeModel,
    TimeAwareNodeModel,
)
from baclct.models.node_encoder import NodeEncoder
from baclct.utils.data import set_multiprocessing_context

set_multiprocessing_context()

# the napari tests need a Qt platform, offscreen by default for a headless run
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# MPS has no deterministic scatter kernel, so strict-deterministic training and tracking
# raise there. only the tests comparing runs numerically carry this, so plain training and
# inference are exercised on MPS as everywhere else.
xfail_on_mps = pytest.mark.xfail(
    torch.backends.mps.is_available(),
    reason="MPS lacks a deterministic scatter kernel, so deterministic GNN "
    "training and tracking are unsupported.",
    run=False,
    strict=False,
)

EDGE_PRED_SCHEMA = {
    "src": pl.UInt32,
    "dst": pl.UInt32,
    "p0": pl.Float32,
    "p1": pl.Float32,
    "p2": pl.Float32,
}


def _get_mock_edges(
    node_feats: pl.DataFrame,
    search_radius: int = 50,
    time_steps: list[int] | None = None,
    masks: np.ndarray | None = None,
    connectivity: Literal["dense", "sequential", "star"] = "sequential",
) -> pl.DataFrame:
    """Candidate edges over `node_feats`, built by a real `EdgeFinder`."""
    if time_steps is None:
        time_steps = [1]

    edge_finder = EdgeFinder(bidirectional=False, n_jobs=1)

    all_edges = []
    max_time = max(node_feats["t"]) if not node_feats.is_empty() else 0

    for t in range(max_time + 1):
        if t >= max_time:
            continue

        edges = edge_finder.get_edges_for_frame(
            node_feats=node_feats,
            t=t,
            search_radius=search_radius,
            time_steps=time_steps,
            masks=masks,
            connectivity=connectivity,
        )
        if edges is not None and not edges.is_empty():
            all_edges.append(edges)

    if not all_edges:
        return pl.DataFrame()

    return pl.concat(all_edges)


def _get_perfect_predictions(edge_df: pl.DataFrame) -> pl.DataFrame:
    """Edge scores that are very high for the ground-truth class `y`."""
    if edge_df.is_empty():
        return pl.DataFrame(schema=EDGE_PRED_SCHEMA)

    # expected to be logits, so perfect preds are >> 1
    return (
        edge_df.with_columns(
            p0=pl.when(pl.col("y") == 0).then(10.0).otherwise(-10.0),
            p1=pl.when(pl.col("y") == 1).then(10.0).otherwise(-10.0),
            p2=pl.when(pl.col("y") == 2).then(10.0).otherwise(-10.0),
        )
        .select("src", "dst", "p0", "p1", "p2")
        .cast(EDGE_PRED_SCHEMA)  # type: ignore
    )


def _find_division(node_feats: pl.DataFrame) -> tuple[int, list[int]]:
    """Label of a dividing cell and the labels of its two daughters."""
    # parent 0 means "no parent", so its group is every root cell, not a division
    counts = (
        node_feats.filter(pl.col("parent") != 0)
        .group_by("parent", maintain_order=True)
        .len("count")
    )
    parent_label = counts.filter(pl.col("count") > 1)["parent"][0]
    daughters = node_feats.filter(parent=parent_label)["label"].unique().sort()

    return parent_label, daughters.to_list()


@pytest.fixture(scope="session")
def toy_data_dir() -> Path:
    """Directory holding the toy images, masks, lineage, and states."""
    return Path(__file__).parent / "assets" / "toy_data"


@pytest.fixture(scope="session")
def toy_masks(toy_data_dir: Path) -> np.ndarray:
    """Load toy masks."""
    return tifffile.imread(toy_data_dir / "sample_masks.tif")


@pytest.fixture(scope="session")
def toy_images(toy_data_dir: Path) -> np.ndarray:
    """Load toy images."""
    return tifffile.imread(toy_data_dir / "sample_images.tif")


@pytest.fixture(scope="session")
def toy_lineage(toy_data_dir: Path) -> pl.DataFrame:
    """The toy CTC lineage."""
    lineage = load_lineage(toy_data_dir / "man_track.txt")
    assert isinstance(lineage, pl.DataFrame)  # narrows the `as_numpy` return type

    return lineage


@pytest.fixture(scope="session")
def toy_lineage_with_states(toy_data_dir: Path) -> pl.DataFrame:
    """The toy lineage with a life cycle state per cell and frame."""
    lineage = load_lineage(toy_data_dir / "states.txt", with_states=True, seq_id=None)
    assert isinstance(lineage, pl.DataFrame)  # narrows the `as_numpy` return type

    return lineage


@pytest.fixture(scope="session")
def toy_handcrafted_features(
    toy_images: np.ndarray, toy_masks: np.ndarray, toy_data_dir: Path
) -> pl.DataFrame:
    """Real `HandcraftedExtractor` output over the toy sequence."""
    extractor = HandcraftedExtractor(
        props=[
            "area",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
            "intensity_mean",
        ],
        extra_props=["center_local", "thickness"],
        feature_norm_fn="scale_relative_size",
        n_jobs=1,
    )

    # __call__ computes the properties and joins the lineage. filepath=None skips caching.
    features_df = extractor(
        image=toy_images,
        masks=toy_masks,
        lineage_file=toy_data_dir / "man_track.txt",
        filepath=None,
    )
    # without the lineage join the failure surfaces in an unrelated fixture
    required = ["parent", "state", "len_init"]
    assert all(it in features_df.columns for it in required), (
        f"Missing {required} in the extractor output: {features_df.columns}"
    )

    return features_df


@pytest.fixture(scope="session")
def toy_features_with_empty_frame_factory(toy_handcrafted_features: pl.DataFrame):
    """Factory for node features with whole frames removed."""

    def _create_features(remove_frames: list[int]):
        return toy_handcrafted_features.filter(~pl.col("t").is_in(remove_frames))

    return _create_features


@pytest.fixture(scope="session")
def perfect_predictions_factory():
    """Factory for ground-truth edge scores over a node set and a set of strides."""

    def _create_predictions(node_feats: pl.DataFrame, time_steps: list[int]):
        edge_df = _get_mock_edges(
            node_feats,
            search_radius=50,
            time_steps=time_steps,
            connectivity="dense",
        )
        return _get_perfect_predictions(edge_df)

    return _create_predictions


@pytest.fixture(scope="session")
def toy_features_with_spurious_detections(
    toy_handcrafted_features: pl.DataFrame,
) -> tuple[pl.DataFrame, list[int]]:
    """Three false-positive detections, each replacing a real cell at t=1, 2, 3.

    Each spurious node sits at the position of the cell it replaces, so it is the only
    remaining candidate for that cell's predecessor. Only the prediction keeps it out of
    the trajectory.

    Returns:
        Node features and the labels of the three spurious detections.
    """
    next_label = int(toy_handcrafted_features["index"].to_numpy().max()) + 1
    # only cells that continue a trajectory, so the predecessor has a candidate to lose,
    # and a different one per frame so the three do not form a trajectory of their own
    continuing = [
        toy_handcrafted_features.filter(
            pl.col("t") == t,
            pl.col("label").is_in(
                toy_handcrafted_features.filter(t=t - 1)["label"].to_list()
            ),
        )
        for t in (1, 2, 3)
    ]
    replaced = pl.concat(frame.slice(i, 1) for i, frame in enumerate(continuing))
    spurious_labels = [next_label + i for i in range(replaced.height)]
    # the replaced cell's `index` is reused: `index` has to stay ordered by frame, or
    # `merge_edge_predictions` sees backward edges it cannot pair and drops everything
    spurious_nodes = replaced.with_columns(
        label=pl.Series(spurious_labels, dtype=replaced.schema["label"]),
        parent=pl.lit(0, dtype=replaced.schema["parent"]),
    )
    return (
        pl.concat(
            [
                toy_handcrafted_features.filter(
                    ~pl.col("index").is_in(replaced["index"].implode())
                ),
                spurious_nodes,
            ]
        ).sort("index"),
        spurious_labels,
    )


@pytest.fixture(scope="session")
def predictions_for_spurious_detections(
    toy_features_with_spurious_detections: tuple[pl.DataFrame, list[int]],
) -> pl.DataFrame:
    """Ground-truth edge scores for the graph carrying spurious nodes.

    A false positive carries a label no other frame has, so every edge touching one is
    inactive in the ground truth and needs no separate suppression.
    """
    node_feats, _ = toy_features_with_spurious_detections
    return _get_perfect_predictions(
        _get_mock_edges(node_feats, search_radius=100, time_steps=[1])
    )


@pytest.fixture(scope="session")
def toy_features_with_missing_daughter(
    toy_handcrafted_features: pl.DataFrame,
) -> tuple[pl.DataFrame, int, int]:
    """A division in the toy data with one daughter deleted from every frame.

    Returns:
        Node features, the parent label, and the label of the surviving daughter.
    """
    parent_label, daughters = _find_division(toy_handcrafted_features)
    node_feats = toy_handcrafted_features.filter(pl.col("label") != daughters[0])

    return node_feats, parent_label, daughters[1]


@pytest.fixture(scope="session")
def predictions_for_missing_daughter(
    toy_features_with_missing_daughter: tuple[pl.DataFrame, int, int],
    toy_masks: np.ndarray,
) -> pl.DataFrame:
    """Ground-truth edge scores for the graph with one daughter missing."""
    node_feats, *_ = toy_features_with_missing_daughter
    edge_df = _get_mock_edges(
        node_feats,
        search_radius=50,
        time_steps=[1],
        masks=toy_masks,
    )
    return _get_perfect_predictions(edge_df)


@pytest.fixture
def mpn_model_factory() -> Callable[..., MPModel]:
    """Factory for an `MPModel` with a chosen node model and layer dimensions."""

    def _create_mpn_model(
        node_model_class,
        handcrafted_features,
        deep_features,
        edge_features,
        hidden_dim,
        num_layers=2,
        node_classifier=None,
    ) -> MPModel:
        embed_dim = hidden_dim

        node_encoder = NodeEncoder(fusion_mode="concat")
        edge_encoder = nn.Linear(edge_features, embed_dim)

        if node_model_class == NodeModel:
            node_model_kwargs = {
                "node_mlp": nn.Sequential(
                    nn.Linear(
                        handcrafted_features + deep_features + hidden_dim, hidden_dim
                    ),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, handcrafted_features + deep_features),
                ),
                "flow_mlp": nn.Linear(
                    handcrafted_features + deep_features + hidden_dim, hidden_dim
                ),
            }
        elif node_model_class == TimeAwareNodeModel:
            node_model_kwargs = {
                "node_mlp": nn.Sequential(
                    nn.Linear(
                        handcrafted_features + deep_features + 2 * hidden_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, handcrafted_features + deep_features),
                ),
                "flow_in_mlp": nn.Linear(
                    handcrafted_features + deep_features + hidden_dim, hidden_dim
                ),
                "flow_out_mlp": nn.Linear(
                    handcrafted_features + deep_features + hidden_dim, hidden_dim
                ),
            }
        else:
            raise ValueError(f"Unsupported node model class: {node_model_class}")

        node_model = node_model_class(**node_model_kwargs)

        edge_mlp = nn.Sequential(
            nn.Linear(
                2 * (handcrafted_features + deep_features) + embed_dim, hidden_dim
            ),  # 2*node + edge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        edge_model = EdgeModel(edge_mlp=edge_mlp)
        edge_classifier = nn.Sequential(nn.Linear(hidden_dim, 3))

        model = MPModel(
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            node_model=node_model,
            edge_model=edge_model,
            edge_classifier=edge_classifier,
            node_classifier=node_classifier,
            num_layers=num_layers,
        )
        return model

    return _create_mpn_model


@pytest.fixture(scope="session")
def toy_tracks_df(toy_handcrafted_features: pl.DataFrame) -> pl.DataFrame:
    """`toy_handcrafted_features` promoted to `tracks`, one trajectory per cell."""
    return toy_handcrafted_features.with_columns(
        label_track=pl.col("label"), parent_track=pl.col("parent")
    )


@pytest.fixture(scope="session")
def basic_extractors():
    """A standard `EdgeFinder` and `HandcraftedExtractor` pair."""
    edge_finder = EdgeFinder(
        # multiplier on half the major axis, so ~20 px on the toy cells
        prune_edges_by=("radius", 1.5),
        feature_names=("dist_spat", "dist_temp"),
        bidirectional=False,
        center_name="center",
        n_jobs=1,
    )
    hc_extractor = HandcraftedExtractor(
        feature_names=["axis_minor_length", "axis_major_length"],
        extra_props=["center_local"],
        feature_norm_fn="scale_relative_size",
        n_jobs=1,
    )
    return edge_finder, hc_extractor


@pytest.fixture
def mock_dataset_factory():
    """Factory for the minimal dataset interface `LAPTracker` reads."""

    def _create_mock_dataset(node_feats, masks=None):
        dataset = Mock()
        dataset.node_feats = node_feats
        dataset.masks = masks
        return dataset

    return _create_mock_dataset


@pytest.fixture(scope="session")
def single_sequence_data(
    tmp_path_factory: pytest.TempPathFactory,
    toy_handcrafted_features,
    toy_masks,
    toy_images,
):
    """The toy sequence written to disk in CTC layout."""
    tmp_path = tmp_path_factory.mktemp("single_seq")
    seq_id = "toy_seq"
    sequences = {
        seq_id: {
            "node_features": toy_handcrafted_features,
            "masks": toy_masks,
            "images": toy_images,
        }
    }
    data_dir, feature_dir = prepare_dataset_structure(tmp_path, sequences)
    return data_dir, feature_dir, seq_id


def prepare_dataset_structure(
    tmp_path: Path,
    sequences: dict[str, dict],
    splits: dict | None = None,
) -> tuple[Path, Path]:
    """Write sequences to a CTC directory layout under `tmp_path`.

    Each entry of `sequences` may carry `node_features`, `masks`, `images`, `lineage`
    and `lineage_with_states`. Anything absent is skipped.

    Returns:
        The data and feature directories.
    """
    data_dir = tmp_path / "data"
    feature_dir = tmp_path / "features"
    data_dir.mkdir(exist_ok=True, parents=True)
    feature_dir.mkdir(exist_ok=True, parents=True)

    if splits:
        with open(data_dir / "splits.yaml", "w") as f:
            yaml.dump(splits, f)

    for seq_id, data in sequences.items():
        node_feats = data.get("node_features")
        if node_feats is not None:
            node_feats_dir = feature_dir / seq_id / "GT"
            node_feats_dir.mkdir(parents=True, exist_ok=True)
            node_feats.write_parquet(node_feats_dir / "nodes.parquet")

        masks = data.get("masks")
        if masks is not None:
            mask_dir = data_dir / f"{seq_id}_GT" / "TRA"
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i, m_frame in enumerate(masks):
                tifffile.imwrite(mask_dir / f"man_track{i:03d}.tif", m_frame)

        images = data.get("images")
        img_dir = data_dir / seq_id
        img_dir.mkdir(parents=True, exist_ok=True)
        if images is not None:
            for i, img_frame in enumerate(images):
                tifffile.imwrite(img_dir / f"t{i:03d}.tif", img_frame)
        elif masks is not None:
            # the loaders always expect images, so write zeros when none are given
            for i, m_frame in enumerate(masks):
                tifffile.imwrite(
                    img_dir / f"t{i:03d}.tif", np.zeros_like(m_frame, dtype=np.uint8)
                )

        lineage = data.get("lineage")
        if lineage is not None:
            gt_dir = data_dir / f"{seq_id}_GT" / "TRA"
            gt_dir.mkdir(parents=True, exist_ok=True)
            lineage.write_csv(
                gt_dir / "man_track.txt", separator=" ", include_header=False
            )

        lineage_with_states = data.get("lineage_with_states")
        if lineage_with_states is not None:
            lineage_with_states.write_csv(data_dir / "states.txt")

    return data_dir, feature_dir


@pytest.fixture(scope="module")
def reproducibility_data_dir(
    tmp_path_factory: pytest.TempPathFactory,
    toy_handcrafted_features: pl.DataFrame,
    toy_masks: np.ndarray,
    toy_images: np.ndarray,
    toy_lineage: pl.DataFrame,
    toy_lineage_with_states: pl.DataFrame,
):
    """A training-ready directory holding the toy sequence as '01'."""
    tmp_path = tmp_path_factory.mktemp("reproducibility_data")

    # fold 100 is what 'debug=testing' composes, fold 0 what the other tests do
    splits = {
        0: {"train": ["01"], "val": ["01"]},
        100: {"train": ["01"], "val": ["01"], "test": ["01"]},
    }

    sequence_data = {
        "01": {
            "node_features": toy_handcrafted_features,
            "masks": toy_masks,
            "images": toy_images,
            "lineage": toy_lineage,
            "lineage_with_states": toy_lineage_with_states.with_columns(
                sequence_id=pl.lit(1)
            ),
        }
    }

    data_dir, feature_dir = prepare_dataset_structure(tmp_path, sequence_data, splits)

    return data_dir, feature_dir
