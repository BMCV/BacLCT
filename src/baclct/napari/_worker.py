"""Background execution of a tracking run for the napari plugin.

`BacLCT.track()` is a single blocking call, so it runs on a nested thread while a
`thread_worker` generator drains the progress events it emits. That keeps the viewer
responsive, gives napari something to yield, and turns the progress callback into a Qt
signal the widget can bind to.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from napari.qt.threading import thread_worker

from baclct.api import BacLCT
from baclct.utils.progress import TrackingCancelled

if TYPE_CHECKING:
    import polars as pl

# fraction of the progress bar each stage occupies, in execution order. only used to keep
# the bar moving forwards at a somewhat accurate rate. heavily dependent on graph
# construction, features, system (CPU is slow on encode), etc.
STAGE_WEIGHTS: dict[str, float] = {
    "model": 0.05,
    "features": 0.10,
    "encode": 0.55,
    "edges": 0.05,
    "predict": 0.15,
    "tracking": 0.05,
    "export": 0.05,
}


@dataclass
class ProgressEvent:
    """A single progress report from the pipeline."""

    stage: str
    current: int = 0
    total: int | None = None
    message: str = ""

    @property
    def fraction(self) -> float:
        """Overall completion in `[0, 1]`, from the stage and its position within it."""
        offset = 0.0
        for stage, weight in STAGE_WEIGHTS.items():
            if stage == self.stage:
                within = (self.current / self.total) if self.total else 0.0
                return min(1.0, offset + weight * within)
            offset += weight
        return offset

    @property
    def label(self) -> str:
        """Human-readable status line."""
        text = self.message or self.stage
        if self.total:
            return f"{text} ({self.current}/{self.total})"
        return text


@dataclass
class TrackParams:
    """Everything a tracking run needs, collected from the widget."""

    images: np.ndarray
    masks: np.ndarray
    model: str | Path
    sequence_id: str = "pred"
    export_suffix: str | None = None
    classify_states: bool | None = None
    graph_search_radius: int | str | None = None
    prune_edges_by: str | tuple[str, float | str] | None = None
    tracker_overrides: dict[str, Any] = field(default_factory=dict)
    cache_dir: Path | Literal["temp"] | None = None
    output_dir: Path | None = None
    export_format: Literal["ctc", "flat"] = "flat"
    device: str | None = None
    num_jobs: int = 1
    num_workers_encode: int = 2
    num_workers: int = 0
    batch_size: int = 1
    patch_size: int | None = None


@dataclass
class TrackResult:
    """Outputs of a tracking run, ready to be turned into layers."""

    tracks: pl.DataFrame
    masks_tracked: np.ndarray


class CancelToken:
    """Cooperative cancellation flag, checked whenever the pipeline reports progress."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()


def _config_overrides(params: TrackParams) -> dict[str, Any]:
    """Widget settings that reach the pipeline as config keys rather than arguments."""
    overrides: dict[str, Any] = {
        "num_workers_predict": params.num_workers,
        "num_workers_encode": params.num_workers_encode,
        "num_jobs_features": params.num_jobs,
        "batch_size": params.batch_size,
    }
    if params.tracker_overrides:
        overrides["tracker"] = dict(params.tracker_overrides)
    return overrides


def _run_track(params: TrackParams, progress, cancel: CancelToken) -> TrackResult:
    """Download the model if needed, then run tracking. Executes off the Qt thread."""
    model_dir = BacLCT.download_model(str(params.model), progress=progress)
    if cancel.cancelled:
        raise TrackingCancelled

    pipeline = BacLCT(model_dir, config_overrides=_config_overrides(params))
    masks_tracked, tracks = pipeline.track(
        images=params.images,
        masks=params.masks,
        classify_states=params.classify_states,
        graph_search_radius=params.graph_search_radius,
        prune_edges_by=params.prune_edges_by,
        patch_size=params.patch_size,
        device=params.device,
        output_dir=params.output_dir,
        sequence_id=params.sequence_id,
        export_format=params.export_format,
        export_suffix=params.export_suffix,
        cache_dir=params.cache_dir,
        progress=progress,
    )
    return TrackResult(tracks=tracks, masks_tracked=masks_tracked)


# ignore_errors keeps superqt from re-raising failures into the Qt event loop: the widget
# connects to `errored` and reports them itself.
@thread_worker(ignore_errors=True)
def tracking_worker(
    params: TrackParams, cancel: CancelToken
) -> Generator[ProgressEvent | None, None, TrackResult | None]:
    """Run tracking in the background, yielding `ProgressEvent`s as it goes.

    Returns the result, or `None` when the run was cancelled. Cancellation is not an
    error: superqt re-raises whatever reaches the `errored` signal into the Qt event
    loop, and a user pressing cancel should not produce a traceback.
    """
    events: queue.Queue[ProgressEvent] = queue.Queue()

    def on_progress(*, stage, current=0, total=None, message=""):
        if cancel.cancelled:
            raise TrackingCancelled
        events.put(ProgressEvent(stage, current, total, message))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_track, params, on_progress, cancel)
        while True:
            try:
                yield events.get(timeout=0.1)
            except queue.Empty:
                if future.done():
                    break
                # keepalive, so the generator keeps ticking through a long silent step
                yield None

    try:
        return future.result()
    except TrackingCancelled:
        return None
