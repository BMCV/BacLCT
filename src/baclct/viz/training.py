"""Figures logged during training: confusion matrices and per-class sample grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from skimage.transform import resize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from baclct.data.dataset import GraphDataset

# spine colors marking the two sample groups of a row
_WORST_COLOR = "#b2182b"
_BEST_COLOR = "#2166ac"


def _resize_and_letterbox(img: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """Resizes image and pads short axis."""
    target_h, target_w = target_size
    h, w = img.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = resize(img, (new_h, new_w), anti_aliasing=True)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))

    return np.pad(resized_img, pad_width, mode="constant", constant_values=0)


def plot_confusion_matrix(
    conf_matrix: np.ndarray, name: str, raw_matrix: np.ndarray | None = None
) -> Figure:
    """Heatmap of a normalized confusion matrix, annotated with raw counts if given."""
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        if raw_matrix is not None:
            annot = np.empty_like(conf_matrix, dtype=object)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    annot[i, j] = f"{conf_matrix[i, j]:.2g}\n({int(raw_matrix[i, j])})"
            fmt = ""
        else:
            annot = True
            fmt = ".2g"

        sns.heatmap(conf_matrix, annot=annot, cbar=False, fmt=fmt, ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {name}")
    except Exception:
        plt.close(fig)
        raise

    return fig


def crop_cells(
    dataset: GraphDataset, indices: Sequence[int]
) -> tuple[np.ndarray, list[int]]:
    """Stack the letterboxed image crops of the given nodes, in the order requested.

    Returns:
        The horizontally stacked crops, separated by a dark seam, and the frame index of
        each crop.
    """
    assert dataset.images is not None, "Cell crops need images."
    feats = dataset.node_feats.filter(pl.col("index").is_in(list(indices))).select(
        "index", "t", r"^bbox-\d$"
    )
    # crops are 2D, so a 3D bounding box fails here and the caller falls back to a warning
    rows = {int(row[0]): row[1:] for row in feats.rows()}

    images, frames = [], []
    for index in indices:
        t, ymin, xmin, ymax, xmax = rows[int(index)]
        img = dataset.images[t]
        if isinstance(img, da.Array):
            img = img.compute()
        images.append(_resize_and_letterbox(img[ymin:ymax, xmin:xmax]))
        frames.append(int(t))

    combined = np.hstack(images)
    width = images[0].shape[1]
    for k in range(1, len(images)):
        combined[:, k * width - 3 : k * width + 3] = 0

    return combined, frames


def plot_sample_grid(
    rows: list[tuple[int, pl.DataFrame]],
    crop_fn: Callable[[dict], tuple[np.ndarray, list[int]]],
    n_worst: int,
    title: str,
) -> Figure:
    """Grid of image crops, one row per true class, ordered from worst to best loss.

    Args:
        rows: True class and its samples, already ordered with the `n_worst` highest-loss
            samples first.
        crop_fn: Maps a sample to its stacked crops and their frame indices.
        n_worst: Number of leading samples per row that belong to the worst group.
        title: Figure title.
    """
    # crop before laying out, so the panel aspect is known and a failure costs no figure
    crops = [[crop_fn(df.row(i, named=True)) for i in range(df.height)] for _, df in rows]
    n_cols = max(len(row_crops) for row_crops in crops)
    height, width = crops[0][0][0].shape[:2]

    panel_w = 1.9
    panel_h = panel_w * height / width + 0.55  # room for the two-line panel title
    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=n_cols,
        figsize=(panel_w * n_cols, panel_h * len(rows) + 0.4),
        squeeze=False,
        layout="constrained",
    )
    try:
        fig.suptitle(
            f"{title}: worst (red) and best (blue) per true class, ranked by loss",
            fontsize=11,
            fontweight="bold",
        )

        for row_idx, ((y_class, df), row_crops) in enumerate(
            zip(rows, crops, strict=True)
        ):
            for col_idx, ax in enumerate(axs[row_idx]):
                ax.set_xticks([])
                ax.set_yticks([])

                if col_idx >= len(row_crops):
                    ax.set_visible(False)
                    continue

                sample = df.row(col_idx, named=True)
                image, frames = row_crops[col_idx]
                ax.imshow(image, cmap="gray")

                color = _WORST_COLOR if col_idx < n_worst else _BEST_COLOR
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(1.5)

                if col_idx == 0:
                    ax.set_ylabel(f"True: {y_class}", fontsize=9, fontweight="bold")

                label = f"{y_class} -> {sample['y_pred']}"
                if len(frames) > 1:
                    label += f", dt {frames[-1] - frames[0]:+d}"
                ax.set_title(
                    f"{label}\np {sample['p_true']:.3f}, loss {sample['loss']:.2f}",
                    fontsize=8,
                    color=color,
                    pad=4,
                )
    except Exception:
        plt.close(fig)
        raise

    return fig
