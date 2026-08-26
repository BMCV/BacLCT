"""Trajectory and feature visualization.

`TrajectoryVisualizer` draws tracks over the images as frames, videos, or napari layers.
`FeatureAnalyzer` extracts the features, embeddings, and predictions behind the paper
figures. Both are kept for convenience and downstream integrations rather than advertised
as features, so their interfaces may change between releases. See the module docstrings of
`baclct.viz.trajectories` and `baclct.viz.features` for usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .features import FeatureAnalyzer
    from .trajectories import TrajectoryVisualizer

__all__ = ["FeatureAnalyzer", "TrajectoryVisualizer"]


def __getattr__(name: str):
    if name == "TrajectoryVisualizer":
        from .trajectories import TrajectoryVisualizer

        return TrajectoryVisualizer
    if name == "FeatureAnalyzer":
        from .features import FeatureAnalyzer

        return FeatureAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
