"""Feature and prediction extraction for analysis figures.

`FeatureAnalyzer` is a utility class to extract input or output features of a model.
`get_features` returns either the features that go into the model (handcrafted, deep,
combined, or edge) or the embeddings the GNN turns them into, `get_misclassifications`
returns the edges the model got wrong, and `get_predictions` returns every edge prediction
next to its label. The class can be instantiated from a checkpoint's experiment directory
with `from_config`, or using a `GraphDataset` and a `TrackingModel` directly. Setting
`cache_dir` writes one parquet per sequence and reuses it on the next call.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from baclct import BacLCT
from baclct.data.dataset import GraphDataset, TrackingDataset
from baclct.models.lightning_model import TrackingModel
from baclct.tracking.postprocessing import (
    edge_preds_to_df,
    extract_prediction_stats,
    merge_edge_predictions,
    resolve_duplicate_predictions,
)
from baclct.utils.config import resolve_checkpoint
from baclct.utils.data import get_multiprocessing_context
from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


class FeatureAnalyzer:
    """Class for extracting initial/learned features, embeds, or misclassifications."""

    def __init__(
        self,
        dataset: GraphDataset | list[GraphDataset],
        model: TrackingModel | None = None,
        device: str | None = None,
        cache_dir: Path | None = None,
        verbose: bool = True,
    ):
        """Initialize analyzer.

        Args:
            dataset: Graph dataset containing handcrafted and optional deep feature
                extractors.
            model: Optional TrackingModel to compute learned features and
                misclassifications.
            device: Model device.
            cache_dir: Optional directory for caching per-sequence feature parquets.
                Cache files are written to cache_dir/{sequence_id}/{type}_{state}.parquet.
            verbose: Show tqdm progress bars at sequence and batch level.
        """
        if not isinstance(dataset, list):
            dataset = [dataset]
        self.datasets = dataset

        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.verbose = verbose
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def from_config(
        cls,
        config: DictConfig | str | Path,
        config_overrides: DictConfig | dict | None = None,
        phase: Literal["train", "val", "test", "predict"] = "predict",
        cache_dir: Path | None = None,
        verbose: bool = True,
    ) -> FeatureAnalyzer:
        """Instantiate the FeatureAnalyzer from a hydra config or experiment dir.

        Args:
            config: Config or path to config or experiment dir.
            config_overrides: Overrides merged on top of `config`.
            phase: Cross-validation split that should be loaded. Predict loads all files.
            cache_dir: Optional directory for caching per-sequence feature parquets.
                Cache files are written to cache_dir/{sequence_id}/{type}_{state}.parquet.
            verbose: Show tqdm progress bars at sequence and batch level.
        """
        pipeline = BacLCT(config, config_overrides)
        pipeline.load_dataset_and_model(phase)
        assert isinstance(pipeline.dataset, TrackingDataset) and isinstance(
            pipeline.model, TrackingModel
        )
        cfg = pipeline.cfg

        if isinstance(config, Path | str):
            experiment_dir: Path | None = Path(config)
        elif cfg.get("experiment_dir"):
            experiment_dir = Path(cfg.experiment_dir)
        else:
            experiment_dir = None

        if experiment_dir:
            logger.info(f"Loading model from experiment dir: {experiment_dir}")
            try:
                ckpt_path = resolve_checkpoint(cfg.get("checkpoint"), experiment_dir)
            except FileNotFoundError as err:
                raise ValueError("Please provide model checkpoint for tracking.") from err

            logger.info(f"Loading checkpoint: {ckpt_path}.")
            pipeline.model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=False)[
                    "state_dict"
                ]
            )

        stages = {"train": "fit", "val": "validate", "test": "test", "predict": "predict"}
        pipeline.dataset.setup(stages[phase])  # type: ignore

        dataset_name = f"dataset_{phase if phase != 'predict' else 'pred'}"
        if not hasattr(pipeline.dataset, dataset_name):
            raise AttributeError(
                f"Could not find {dataset_name}. Did you call `dataset.setup()`?"
            )

        dataset = getattr(pipeline.dataset, dataset_name)
        if isinstance(dataset, ConcatDataset):
            dataset = dataset.datasets

        return cls(
            dataset=dataset,
            model=pipeline.model,
            cache_dir=cache_dir,
            verbose=verbose,
        )

    def _get_raw_tensors(self, dataset: GraphDataset):
        """Extract un-remapped tensors for the entire graph."""
        assert dataset.precompute_edges, "GraphDataset must have precompute_edges=True."

        edge_data = dataset.edge_data
        if isinstance(edge_data, pl.LazyFrame):
            edge_data = edge_data.collect()

        node_data = dataset.node_feats
        x_handcrafted, x_deep = dataset._get_node_features(node_data)

        assert isinstance(edge_data, pl.DataFrame)
        edge_index = edge_data.select("src", "dst").to_numpy().T
        edge_index_tensor = torch.from_numpy(edge_index).long()

        edge_attr = dataset._get_edge_features(edge_data, node_data)
        return x_handcrafted, x_deep, edge_index_tensor, edge_attr, node_data, edge_data

    @torch.no_grad()
    def _step(self, model: nn.Module, batch: Data):
        # adapted from model.lightning_model.TrackingModel
        x_hc = batch.get("x_handcrafted")
        x_deep = batch.get("x_deep")
        x_handcrafted = x_hc.to(self.device) if x_hc is not None else None
        x_deep = x_deep.to(self.device) if x_deep is not None else None

        return model(
            x_handcrafted,
            x_deep,
            batch.edge_index.to(self.device),  # type: ignore
            batch.edge_attr.to(self.device),  # type: ignore
        )

    def _predict(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        return_embeds: bool,
        edge_direction: Literal["both", "future"] = "both",
    ):
        preds = defaultdict(list)
        for batch in tqdm(dataloader, desc="Predicting edges"):
            src, dst = batch.edge_index
            if src.numel() == 0:
                continue
            src = batch.node_mapping[src]
            dst = batch.node_mapping[dst]

            preds["predictions"].append(self._step(model, batch))
            preds["index"].append(
                {
                    "edge_index": np.stack(
                        [[u, v] for u, v in zip(src, dst, strict=True)], 1
                    ),
                    "node_index": batch.node_mapping,
                }
            )

        return self._store_predictions(
            preds, return_embeds=return_embeds, edge_direction=edge_direction
        )

    def _store_predictions(
        self,
        predictions,
        return_embeds: bool = False,
        edge_direction: Literal["both", "future"] = "both",
    ):
        edge_preds, edge_index, node_preds, node_index = [], [], [], []
        preds_list, indices_list = predictions["predictions"], predictions["index"]

        for preds, indices in zip(preds_list, indices_list, strict=True):
            if preds is not None:
                if preds.get("edge_predictions"):
                    edge_preds.append(preds["edge_predictions"][-1].cpu().numpy())
                    edge_index.append(indices["edge_index"])
                if preds.get("node_predictions"):
                    node_preds.append(preds["node_predictions"][-1].cpu().numpy())
                    node_index.append(indices["node_index"].cpu().numpy())

        edge_preds = np.concat(edge_preds)
        src, dst = np.concat(edge_index, 1)

        # one item per window
        edge_preds_df = resolve_duplicate_predictions(
            edge_preds_to_df(src, dst, edge_preds, y=None)
        )
        if return_embeds:
            # without the classifier the p columns are one embedding, not class scores
            edge_preds_df = edge_preds_df.select(
                "src", "dst", pl.concat_arr(r"^p\d+$").alias("p0")
            )
            node_df = None
            if node_preds:
                node_preds_arr = np.concat(node_preds)
                node_index_arr = np.concat(node_index)
                node_df = resolve_duplicate_predictions(
                    pl.DataFrame(
                        node_preds_arr,
                        schema=[f"p{i}" for i in range(node_preds_arr.shape[-1])],
                    ).with_columns(index=pl.Series(node_index_arr).cast(pl.UInt32))
                ).select("index", pl.concat_arr(r"^p\d+$").alias("p0"))
            return {"edges": edge_preds_df, "nodes": node_df}

        return merge_edge_predictions(edge_preds_df, edge_direction)

    def run_model(
        self,
        dataset: GraphDataset,
        return_embeddings: bool = False,
        batch_size=1,
        num_workers=0,
        edge_direction: Literal["both", "future"] = "both",
    ):
        """Run the model on the full graph, optionally returning inner embeddings."""
        if self.model is None:
            raise ValueError("Model must be provided.")

        # adapted from dataset.TrackingDataset._configure_dataloaders (without samplers)
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            multiprocessing_context=get_multiprocessing_context(num_workers),
        )
        if not return_embeddings:
            return self._predict(
                self.model, dataloader, return_embeds=False, edge_direction=edge_direction
            )

        # replace classification layer with identity, run prediction to extract feats,
        # store outputs (as edge/node_predictions). afterward cleanup to restore model.
        inner_model = self.model.model
        orig_edge_cls = inner_model.edge_classifier
        orig_node_cls = inner_model.node_classifier

        try:
            inner_model.edge_classifier = torch.nn.Identity()
            inner_model.node_classifier = torch.nn.Identity()
            return self._predict(
                inner_model, dataloader, return_embeds=True, edge_direction=edge_direction
            )
        finally:
            inner_model.edge_classifier = orig_edge_cls
            inner_model.node_classifier = orig_node_cls

    def get_features(
        self,
        feature_type: Literal["handcrafted", "deep", "combined", "edge"] = "handcrafted",
        feature_state: Literal["initial", "learned"] = "initial",
        indices: list[int] | None = None,
    ) -> pl.DataFrame:
        """Helper to extract features from all datasets.

        Args:
            feature_type: The type of features to extract.
            feature_state: The state of the features to extract.
            indices: Global node indices to retrieve features for. If `None`, returns
                all features.

        Returns:
            Features and metadata for the requested nodes.
        """
        all_features = []
        desc = f"{feature_type} ({feature_state}) features"
        for ds in tqdm(self.datasets, desc=desc, disable=not self.verbose):
            assert isinstance(ds, GraphDataset)

            _cache_path = None
            if self.cache_dir is not None and indices is None:
                assert isinstance(self.cache_dir, Path) and (ds.sequence_id is not None)
                _cache_path = (
                    self.cache_dir
                    / ds.sequence_id
                    / f"{feature_type}_{feature_state}.parquet"
                )
                if _cache_path.exists():
                    all_features.append(pl.read_parquet(_cache_path))
                    continue

            if feature_state == "initial":
                x_handcrafted, x_deep, _, edge_attr, node_data, edge_data = (
                    self._get_raw_tensors(ds)
                )

                if feature_type == "edge":
                    # cosine_similarity is excluded from feature_cols (computed at
                    # runtime from deep features) but _get_edge_features appends it
                    edge_feat_cols = list(ds.edge_finder.feature_cols)
                    if (
                        "cosine_similarity" in ds.edge_finder.feature_names
                        and x_deep is not None
                    ):
                        edge_feat_cols = edge_feat_cols + ["cosine_similarity"]

                    feats = pl.DataFrame(
                        np.concat(
                            [
                                edge_data.select("src", "dst", "y").to_numpy(),
                                edge_attr.numpy(),
                            ],
                            1,
                        ),
                        schema=["src", "dst", "y"] + edge_feat_cols,
                    )
                    if indices is not None:
                        feats = feats.filter(
                            pl.col("src").is_in(indices) | pl.col("dst").is_in(indices)
                        )
                    features = feats
                elif feature_type in ["handcrafted", "deep", "combined"]:
                    selection = ["index", "label"] + (
                        ["state"] if "state" in node_data.columns else []
                    )
                    features = node_data.select(selection)

                    if (
                        feature_type in ["handcrafted", "combined"]
                        and x_handcrafted is not None
                    ):
                        hc_names = ds.handcrafted_feature_extractor.extracted_features
                        features = features.hstack(
                            pl.DataFrame(x_handcrafted.numpy(), schema=hc_names)
                        )

                    if feature_type in ["deep", "combined"] and x_deep is not None:
                        features = features.with_columns(
                            deep_features=pl.Series(x_deep.numpy())
                        )

                    if indices is not None:
                        features = features.filter(pl.col("index").is_in(indices))
                else:
                    raise ValueError(f"Unknown node feature {feature_type=}.")

            elif feature_state == "learned":
                preds = self.run_model(ds, return_embeddings=True)
                # preds = {"edges": (src, dst, y, p0), "nodes": (index, p0) or None}

                if feature_type == "edge":
                    edge_data = ds.edge_data
                    if isinstance(edge_data, pl.LazyFrame):
                        edge_data = edge_data.collect()
                    assert isinstance(edge_data, pl.DataFrame)

                    features = edge_data.select("src", "dst", "y").join(
                        preds["edges"].select(
                            "src", "dst", learned_features=pl.col("p0")
                        ),
                        on=["src", "dst"],
                        how="left",
                    )
                    if indices is not None:
                        features = features.filter(
                            pl.col("src").is_in(indices) | pl.col("dst").is_in(indices)
                        )
                elif feature_type in ["handcrafted", "deep", "combined"]:
                    if preds["nodes"] is None:
                        raise ValueError(
                            f"Model produces no node predictions for {feature_type=}. "
                            "Ensure the model has a node classifier."
                        )
                    selection = ["index", "label"] + (
                        ["state"] if "state" in ds.node_feats.columns else []
                    )
                    features = ds.node_feats.select(selection).join(
                        preds["nodes"].rename({"p0": "learned_features"}),
                        on="index",
                        how="left",
                    )
                    if indices is not None:
                        features = features.filter(pl.col("index").is_in(indices))
                else:
                    raise ValueError(f"Unknown node feature {feature_type=}.")
            else:
                raise ValueError(f"Unknown feature state {feature_state=}.")

            result = features.with_columns(sequence_id=pl.lit(ds.sequence_id))
            if _cache_path is not None:
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                result.write_parquet(_cache_path)

            all_features.append(result)

        return pl.concat(all_features, how="diagonal_relaxed")

    @torch.no_grad()
    def get_misclassifications(self) -> dict[str, pl.DataFrame]:
        """Runs model on datasets and returns edge misclassifications."""
        if self.model is None:
            raise ValueError("Model must be provided to compute misclassifications.")

        all_outputs = {}
        for ds in tqdm(
            self.datasets, desc="Misclassifications", disable=not self.verbose
        ):
            assert isinstance(ds, GraphDataset)

            preds = self.run_model(ds, return_embeddings=False)
            # preds is pl.DataFrame (src, dst, p0...) without y -> join from edge_data
            edge_data = ds.edge_data
            if isinstance(edge_data, pl.LazyFrame):
                edge_data = edge_data.collect()
            assert isinstance(edge_data, pl.DataFrame)

            preds_with_y = preds.join(
                edge_data.select("src", "dst", "y"), on=["src", "dst"], how="left"
            )
            misclassifications = extract_prediction_stats(preds_with_y).filter(
                pl.col("y") != pl.col("y_pred")
            )
            all_outputs[ds.sequence_id] = misclassifications

        return all_outputs

    @torch.no_grad()
    def get_predictions(self) -> pl.DataFrame:
        """Run model on all datasets and return all edge predictions with labels.

        Like get_misclassifications() but unfiltered. Returns a combined DataFrame
        with a sequence_id column, suitable for joining y_pred onto edge UMAP metadata.
        """
        if self.model is None:
            raise ValueError("Model must be provided to compute predictions.")

        all_outputs = []
        for ds in tqdm(self.datasets, desc="Predictions", disable=not self.verbose):
            assert isinstance(ds, GraphDataset)

            _cache_path = None
            if self.cache_dir is not None:
                assert isinstance(self.cache_dir, Path) and (ds.sequence_id is not None)
                _cache_path = self.cache_dir / ds.sequence_id / "edge_predictions.parquet"
                if _cache_path.exists():
                    all_outputs.append(pl.read_parquet(_cache_path))
                    continue

            preds = self.run_model(ds, return_embeddings=False)
            # preds is pl.DataFrame (src, dst, p0...) without y -> join from edge_data
            edge_data = ds.edge_data
            if isinstance(edge_data, pl.LazyFrame):
                edge_data = edge_data.collect()
            assert isinstance(edge_data, pl.DataFrame)

            preds_with_y = preds.join(
                edge_data.select("src", "dst", "y"), on=["src", "dst"], how="left"
            )
            result = extract_prediction_stats(preds_with_y).with_columns(
                sequence_id=pl.lit(ds.sequence_id)
            )
            if _cache_path is not None:
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                result.write_parquet(_cache_path)
            all_outputs.append(result)

        return pl.concat(all_outputs, how="diagonal_relaxed")
