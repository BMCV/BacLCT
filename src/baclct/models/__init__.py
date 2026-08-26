"""Graph model components and training logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lightning_model import TrackingModel
    from .model import MPModel

__all__ = ["MPModel", "TrackingModel"]


def __getattr__(name: str):
    if name == "TrackingModel":
        from .lightning_model import TrackingModel

        return TrackingModel
    if name == "MPModel":
        from .model import MPModel

        return MPModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
