"""Single-sequence graph datasets and multi-sequence DataModule."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import GraphDataset

__all__ = ["GraphDataset"]


def __getattr__(name: str):
    if name == "GraphDataset":
        from .dataset import GraphDataset

        return GraphDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
