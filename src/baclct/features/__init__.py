"""Single-cell feature extractors, graph construction, and feature definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extractors import CellLevelExtractor, HandcraftedExtractor

__all__ = ["CellLevelExtractor", "HandcraftedExtractor"]


def __getattr__(name: str):
    if name in ("CellLevelExtractor", "HandcraftedExtractor"):
        from .extractors import CellLevelExtractor, HandcraftedExtractor

        mapping = {
            "CellLevelExtractor": CellLevelExtractor,
            "HandcraftedExtractor": HandcraftedExtractor,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
