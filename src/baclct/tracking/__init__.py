"""Tracker and segmentation error correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .segmentation_correction import SegmentationCorrector
    from .tracker import BaseTracker, LAPTracker

__all__ = [
    "BaseTracker",
    "LAPTracker",
    "SegmentationCorrector",
]


def __getattr__(name: str):
    if name in ("BaseTracker", "LAPTracker"):
        from .tracker import BaseTracker, LAPTracker

        mapping = {
            "BaseTracker": BaseTracker,
            "LAPTracker": LAPTracker,
        }
        return mapping[name]
    if name == "SegmentationCorrector":
        from .segmentation_correction import SegmentationCorrector

        return SegmentationCorrector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
