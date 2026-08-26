"""Normalize and deduplicate GNN predictions."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy.special import softmax

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def edge_preds_to_df(
    src: np.ndarray,
    dst: np.ndarray,
    preds: np.ndarray,
    y: np.ndarray | None = None,
    dataloader_idx: np.ndarray | list | None = None,
) -> pl.DataFrame:
    """Convert edge predictions to dataframe.

    Args:
        src: Source node index of each edge.
        dst: Destination node index of each edge.
        preds: Edge predictions.
        y: Edge labels. The column is omitted when `None`, as it is at inference.
        dataloader_idx: Dataloader index or other ids.
    """
    src, dst, preds = np.asarray(src), np.asarray(dst), np.asarray(preds)
    assert len(src) > 0, "Could not aggregate zero-length edge predictions."

    # built column by column: stacking the integer indices with the float32 scores first
    # would promote the whole block to float64, several times the size of the result
    columns: dict[str, pl.Series] = {
        "src": pl.Series("src", src, dtype=pl.UInt32),
        "dst": pl.Series("dst", dst, dtype=pl.UInt32),
    }
    if y is not None:
        columns["y"] = pl.Series("y", np.asarray(y), dtype=pl.Int64)
    for i in range(preds.shape[-1]):
        columns[f"p{i}"] = pl.Series(f"p{i}", preds[:, i], dtype=pl.Float32)
    if dataloader_idx is not None:
        columns["dataloader_idx"] = pl.Series("dataloader_idx", dataloader_idx)

    return pl.DataFrame(columns)


def merge_edge_predictions(
    edge_predictions: pl.DataFrame, how: Literal["future", "both"]
) -> pl.DataFrame:
    """Reduce a bidirectional graph to one forward edge per pair of nodes.

    Both modes return forward edges (`src < dst`), since the tracker reads `src` to `dst`
    as the direction of time. With 'both', each pair's backward prediction is averaged
    into its forward one. Divisions survive that average because `EdgeFinder` labels a
    division in both directions, so `dau` to `par` is class 2 just like `par` to `dau`.
    """
    fut = edge_predictions.filter(pl.col("src") < pl.col("dst"))
    if how == "future":
        return fut

    past = edge_predictions.filter(pl.col("src") > pl.col("dst"))
    if fut.height == 0 or past.height == 0:
        if fut.height != past.height:
            logger.warning(
                "Found edges in only one temporal direction. Returning the forward "
                "edges without aggregation."
            )
        return fut

    num_preds = len(fut.select(r"^p\d$").columns)

    select_cols = [
        "src",
        "dst",
        *([pl.col("y")] if "y" in fut.columns else []),
        *[pl.mean_horizontal(f"p{i}", f"p{i}_past") for i in range(num_preds)],
    ]
    if "dataloader_idx" in fut.columns:
        select_cols.append("dataloader_idx")

    return fut.join(
        past.rename({"src": "dst", "dst": "src"}), on=("src", "dst"), suffix="_past"
    ).select(*select_cols)


def edge_preds_to_softmax(edge_preds: pl.DataFrame) -> pl.DataFrame:
    """Apply softmax to edge predictions."""
    n = len(edge_preds.select(r"^p\d$").columns)

    return edge_preds.select(
        pl.exclude(r"^p\d$"),
        p_norm=pl.concat_arr(r"^p\d$")
        .map_batches(lambda x: softmax(x, 1))
        .arr.to_struct([f"p{i}" for i in range(n)]),
    ).unnest("p_norm")


def edge_preds_to_sigmoid(edge_preds: pl.DataFrame) -> pl.DataFrame:
    """Apply sigmoid to edge predictions."""
    n = len(edge_preds.select(r"^p\d$").columns)

    return edge_preds.select(
        pl.exclude(r"^p\d$"),
        p_norm=pl.concat_arr(r"^p\d$")
        .map_batches(
            lambda x: 1 / (1 + np.exp(-np.array(x))),
            is_elementwise=True,
            return_dtype=pl.Array(pl.Float64, n),
        )
        .arr.to_struct([f"p{i}" for i in range(n)]),
    ).unnest("p_norm")


def expand_binary_predictions(edge_preds: pl.DataFrame) -> pl.DataFrame:
    """Expand single-class binary predictions to 3-class predictions.

    Applied to make predictions from model without learned division detection compatible
    with the LAP tracker. In the LAP tracker, divisions are resolved greedy and are
    converted into correspondences in the case of a single-daughter division. This also
    keeps the multi-frame tracking logic in-place.
    """
    p_logit = edge_preds["p0"].to_numpy()
    normalized = (max(p_logit) <= 1) and (min(p_logit) >= 0)
    p_active = (
        (1 / (1 + np.exp(-p_logit))).astype(np.float32)
        if not normalized
        else p_logit.astype(np.float32)
    )
    if not normalized:
        logger.debug("Found unnormalized logits. Normalized using sigmoid.")

    return edge_preds.with_columns(
        p0=pl.Series("p0", 1 - p_active),
        p1=pl.lit(0.0).cast(pl.Float32),
        p2=pl.Series("p2", p_active),
    )


def extract_prediction_stats(edge_preds: pl.DataFrame) -> pl.DataFrame:
    """Computes true class probability and predicted classes from edge predictions.

    Used for training logs.
    """
    if "y" not in edge_preds.columns:
        raise ValueError("DataFrame must contain 'y' column with ground truth.")

    pred_cols = edge_preds.select(r"^p\d$").columns
    if len(pred_cols) == 1:
        # binary (1-logit) case: expand to 2-class probabilities via sigmoid so that
        # p_list[y] and argmax work correctly for y in {0, 1}
        p_logit = edge_preds["p0"].to_numpy()
        p_active = (1 / (1 + np.exp(-p_logit))).astype(np.float32)
        df = edge_preds.with_columns(
            p0=pl.Series("p0", 1 - p_active),
            p1=pl.Series("p1", p_active),
        )
    else:
        df = edge_preds_to_softmax(edge_preds)

    return (
        df.with_columns(p_list=pl.concat_list(r"^p\d$"))
        .with_columns(
            p_true=pl.col("p_list").list.get("y"), y_pred=pl.col("p_list").list.arg_max()
        )
        .drop("p_list")
    )


def select_extreme_samples(
    stats: pl.DataFrame, n_worst: int = 4, n_best: int = 2
) -> list[tuple[int, pl.DataFrame]]:
    """Per true class, the highest- and lowest-loss samples for the training log figures.

    Mirrored edges and repeated source cells are collapsed first, so one cell cannot fill
    a row.

    Args:
        stats: Output of `extract_prediction_stats` with an added 'loss' column.
        n_worst: Highest-loss samples per class.
        n_best: Lowest-loss samples per class.

    Returns:
        True class and its samples, worst first, for every class that has samples.
    """
    if "src" in stats.columns and "dst" in stats.columns:
        stats = stats.with_columns(
            _lo=pl.min_horizontal("src", "dst"), _hi=pl.max_horizontal("src", "dst")
        ).unique(subset=["_lo", "_hi"], keep="first")
        dedup_on = "src"
    else:
        dedup_on = "index"

    rows = []
    for y_class in sorted(stats["y"].unique().to_list()):
        samples = (
            stats.filter(pl.col("y") == y_class)
            .sort("loss", descending=True)
            .unique(subset=dedup_on, keep="first", maintain_order=True)
        )
        if samples.is_empty():
            continue
        if samples.height > n_worst + n_best:
            samples = pl.concat([samples.head(n_worst), samples.tail(n_best)])
        rows.append((int(y_class), samples))

    return rows


def resolve_duplicate_predictions(df: pl.DataFrame) -> pl.DataFrame:
    """Average duplicate predictions for the same edge or node.

    Useful when evaluating on overlapping patches or frames.
    """
    if "src" in df.columns and "dst" in df.columns:
        group_cols = ["src", "dst"]
    elif "index" in df.columns:
        group_cols = ["index"]
    else:
        raise ValueError("DataFrame must contain ('src', 'dst') or 'index' columns.")

    agg_exprs = [
        pl.col(c).mean() for c in df.columns if c.startswith("p") and c[1:].isdigit()
    ]
    if "y" in df.columns:
        agg_exprs.append(pl.col("y").first())
    if "dataloader_idx" in df.columns:
        agg_exprs.append(pl.col("dataloader_idx").first())

    original_cols = df.columns
    return df.group_by(group_cols).agg(agg_exprs).sort(group_cols).select(original_cols)
