"""Data conversion and processing helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

import numpy as np
import polars as pl
import torch

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# lets a run be pinned to another start method, e.g. to exercise the windows path on linux
MP_CONTEXT_ENV_VAR = "BACLCT_MP_CONTEXT"


def default_multiprocessing_context() -> str:
    """Start method for worker processes: `spawn` on Windows, `forkserver` elsewhere.

    `$BACLCT_MP_CONTEXT` overrides the platform default. `fork` is unusable either way:
    polars and the memory-mapped feature caches deadlock in forked children.
    """
    override = os.environ.get(MP_CONTEXT_ENV_VAR)
    if override:
        return override
    return "spawn" if sys.platform.startswith("win") else "forkserver"


def set_multiprocessing_context(context: str | None = None):
    """Set the process-wide multiprocessing start method."""
    context = context or default_multiprocessing_context()
    mp.set_start_method(context, force=True)

    return context


def get_device(device: str | None):
    """Get torch device."""
    if device and device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_accelerator(device: str | None) -> tuple[str, int | list[int]]:
    """Map a torch device string onto a lightning `(accelerator, devices)` pair.

    Accepts `None`/`'auto'` (let lightning pick), a bare accelerator (`'cpu'`, `'cuda'`,
    `'mps'`), or a device with an index (`'cuda:1'`), which selects that single device.
    """
    if not device or device == "auto":
        return "auto", 1

    accelerator, _, index = device.partition(":")
    if index:
        return accelerator, [int(index)]
    return accelerator, 1


def get_multiprocessing_context(num_workers: int) -> str | None:
    """Start method for dataloader workers, or `None` when running single-process.

    Returns the context so it can be passed to a single dataloader instead of being set
    process-wide, which would be hostile inside a host application (e.g. napari).
    """
    if num_workers <= 0:
        return None
    return default_multiprocessing_context()


def col_to_tensor(
    df: pl.DataFrame,
    col: str | list[str],
    dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    """Convert `polars` dataframe column to tensor."""
    cols = df.select(col) if not isinstance(col, str) else df[col]
    cols_np = cols.to_numpy(writable=True)
    if transpose:
        cols_np = np.transpose(cols_np)

    return torch.as_tensor(cols_np, dtype=dtype)


def collect(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialize a lazy dataframe and/or return an eager dataframe."""
    if isinstance(df, pl.LazyFrame):
        return df.collect()  # type: ignore

    return df
