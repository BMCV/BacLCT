"""Record the features and splits a checkpoint was trained on.

While the Hydra config and logs stored next to the model contain everything required to
reproduce a training run, it provides no information on the exact composition of the
features. While there generally should not be any randomness involved during feature
extraction and loading, features might change through corruption of the dataset, random
ordering of sequences and columns, or through temporary (uncommited) changes to the code.

The written feature info contains the version, the train/val/test split, and the exact
order of the expected feature columns and their mean values (per-split). Feature
statistics are recorded for the train and val splits only.

Information on test split cache creation is included since all caches are created within
`prepare_data()` (before training). During `setup()` and `fit()`, only the training and
validation dataset are used.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import torch
from torch.utils.data import ConcatDataset

import baclct
from baclct.utils.data import collect
from baclct.utils.logger import get_pylogger

if TYPE_CHECKING:
    from baclct.data.dataset import GraphDataset, TrackingDataset

logger = get_pylogger(__name__)

FEATURE_INFO_FILENAME = "features.json"


def cache_signature(path: Path | None) -> float | None:
    """Latest mtime under `path` (file or directory tree), or None if absent.

    For a cache held as a directory rather than a single file, the maximum mtime of any
    contained file is used, so writing a new entry changes the signature.
    """
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def classify_cache(
    kind: str, sequence: str, path: Path | None, signature_before: float | None
) -> dict:
    """Classify a cache action by comparing pre- and post-call mtime signatures.

    `signature_before` is the `cache_signature` captured before the extractor ran. Status
    is 'used' (reused unchanged), 'invalidated' (existed but recomputed), 'created'
    (written fresh), 'absent' (no cache, e.g. caching disabled), or 'removed'.
    """
    after = cache_signature(path)
    if signature_before is None and after is None:
        status = "absent"
    elif signature_before is None:
        status = "created"
    elif after is None:
        status = "removed"
    elif after == signature_before:
        status = "used"
    else:
        status = "invalidated"

    record: dict[str, Any] = {
        "kind": kind,
        "sequence": sequence,
        "status": status,
        "path": str(path) if path is not None else None,
    }
    if path is not None and path.exists():
        st = path.stat()
        birth = getattr(st, "st_birthtime", None)
        created = birth if birth else st.st_ctime
        modified = after if after is not None else st.st_mtime
        record["created_at"] = datetime.fromtimestamp(created, tz=UTC).isoformat()
        record["modified_at"] = datetime.fromtimestamp(modified, tz=UTC).isoformat()
    return record


def _iter_graph_datasets(split: Any) -> list[GraphDataset]:
    """Flatten a phase dataset (single, list, or ConcatDataset) into GraphDatasets."""
    if split is None:
        return []
    if isinstance(split, ConcatDataset):
        return list(split.datasets)  # type: ignore
    if isinstance(split, (list, tuple)):
        return list(split)
    return [split]


def _node_feature_stats(
    graph_datasets: list[GraphDataset], extractor: Any
) -> dict | None:
    """Names, count, and per-column mean of the handcrafted node features."""
    if extractor is None or getattr(extractor, "feature_names", None) is None:
        return None

    node_feats = pl.concat(
        [gd.node_feats for gd in graph_datasets], how="diagonal_relaxed"
    )
    x = extractor._transform(node_feats)
    if x is None or extractor.extracted_features is None:
        return None

    names = list(extractor.extracted_features)
    means = x.mean(dim=0).tolist()
    return {
        "names": names,
        "n": int(x.shape[0]),
        "mean": dict(zip(names, means, strict=True)),
    }


def _edge_feature_stats(
    graph_datasets: list[GraphDataset], edge_finder: Any
) -> dict | None:
    """Model edge feature names, count, and per-column mean of relational features.

    Means cover only the relational columns the edge finder computes (`feature_cols`);
    deep-derived edge features such as cosine similarity are listed under `names` but not
    averaged here.
    """
    if edge_finder is None or not getattr(edge_finder, "feature_names", None):
        return None

    names = list(edge_finder.feature_names)
    feature_cols = list(getattr(edge_finder, "feature_cols", []) or [])

    feature_sum: torch.Tensor | None = None
    n_total = 0
    for gd in graph_datasets:
        edge_data = getattr(gd, "edge_data", None)
        if edge_data is None or not feature_cols:
            continue

        # don't materialize full dataframe, but only requested columns
        df = collect(edge_data.lazy().select(feature_cols))

        _EDGE_STATS_SAMPLE_CAP = 1_000_000
        if df.height > _EDGE_STATS_SAMPLE_CAP:
            df = df.sample(_EDGE_STATS_SAMPLE_CAP, seed=0)

        attr = edge_finder._transform(df, gd.node_feats, spacing_t=gd.spacing["t"])
        if attr is not None and attr.numel():
            # accumulate sum and count rather than concatenating every dataset's edges
            col_sum = attr.sum(dim=0)
            feature_sum = col_sum if feature_sum is None else feature_sum + col_sum
            n_total += attr.shape[0]

    if feature_sum is None or not feature_cols:
        return {"names": names, "n": 0, "mean": None}

    means = (feature_sum / n_total).tolist()
    return {
        "names": names,
        "n": int(n_total),
        "mean": dict(zip(feature_cols, means, strict=True)),
    }


def _deep_feature_info(graph_datasets: list[GraphDataset], extractor: Any) -> dict | None:
    """Deep feature extractor identity and embedding dimensionality."""
    if extractor is None:
        return None

    info: dict[str, Any] = {"extractor": type(extractor).__name__}
    for attr in ("model_name", "model_type", "name"):
        if getattr(extractor, attr, None) is not None:
            info[attr] = getattr(extractor, attr)
            break

    for gd in graph_datasets:
        deep_feats = getattr(gd, "deep_feats", None)
        if deep_feats is not None and getattr(deep_feats, "ndim", 0) == 2:
            info["dim"] = int(deep_feats.shape[1])
            break

    return info


def _split_sequence_ids(graph_datasets: list[GraphDataset]) -> list[str]:
    """Sequence IDs of the loaded graph datasets, order preserved, duplicates removed."""
    ids = [
        cast(str, gd.sequence_id)
        for gd in graph_datasets
        if getattr(gd, "sequence_id", None) is not None
    ]
    return list(dict.fromkeys(ids))


def _test_sequence_ids(dataset: TrackingDataset) -> list[str]:
    """Test sequence IDs from the splits file for the configured fold.

    Test data is never loaded during training, so only the sequence names are recorded.
    """
    try:
        splits = dataset._load_splits()
    except Exception as err:
        logger.debug(f"Could not read test split assignment: {err}")
        return []
    test = [seq for split in splits for seq in split.get("test", [])]
    return list(dict.fromkeys(test))


def _safe(label: str, fn, *args):
    """Run a stats collector, returning None on any failure."""
    try:
        return fn(*args)
    except Exception as err:
        logger.warning(f"Could not collect {label}: {err}")
        return None


def _phase_feature_info(
    dataset: TrackingDataset, graph_datasets: list[GraphDataset]
) -> dict | None:
    """Node, edge, and deep feature description for a single split's graph datasets."""
    if not graph_datasets:
        return None
    return {
        "node_features": _safe(
            "node features",
            _node_feature_stats,
            graph_datasets,
            dataset.hc_feat_extractor,
        ),
        "edge_features": _safe(
            "edge features", _edge_feature_stats, graph_datasets, dataset.edge_finder
        ),
        "deep_features": _safe(
            "deep features",
            _deep_feature_info,
            graph_datasets,
            dataset.deep_feat_extractor,
        ),
    }


def build_feature_info(dataset: TrackingDataset) -> dict:
    """Collect the feature description and split assignment from the train and val splits.

    Feature statistics are computed per split for train and val only.
    """
    train = _iter_graph_datasets(getattr(dataset, "dataset_train", None))
    val = _iter_graph_datasets(getattr(dataset, "dataset_val", None))

    return {
        "baclct_version": baclct.__version__,
        "splits": {
            "fold": getattr(dataset, "fold", None),
            "train": _split_sequence_ids(train),
            "val": _split_sequence_ids(val),
            "test": _test_sequence_ids(dataset),
        },
        "features": {
            "train": _phase_feature_info(dataset, train),
            "val": _phase_feature_info(dataset, val),
        },
        "caches": list(getattr(dataset, "cache_records", []) or []),
        "dataset_identity": dict(getattr(dataset, "dataset_metas", {}) or {}),
    }


def write_feature_info(dataset: TrackingDataset, output_dir: Path | str) -> Path | None:
    """Write `features.json` to `output_dir`. Never raises, returns path on success."""
    try:
        info = build_feature_info(dataset)
        path = Path(output_dir) / FEATURE_INFO_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2))
        logger.info(f"Wrote feature info to {path}.")
        return path
    except Exception as err:
        logger.warning(f"Could not write feature info: {err}")
        return None
