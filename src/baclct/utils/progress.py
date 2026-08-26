"""Progress reporting and cooperative cancellation for long-running tracking runs.

`BacLCT.track()` is a single blocking call that spends minutes in feature extraction,
prediction, and tracking. GUI front-ends (e.g. the napari plugin) need to show progress
and offer a cancel button, so the pipeline accepts an optional `ProgressCallback`.

Cancellation is cooperative: a callback raises `TrackingCancelled` and the pipeline lets
it propagate. Callbacks are therefore invoked between units of work (a frame, a batch, a
timepoint), which is also the granularity at which a run can be aborted.

Producers attach a callback as a plain `progress` attribute rather than a constructor
argument, so hydra configs and existing call sites are unaffected. Use `report()` to emit
from a producer that may or may not have one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import lightning as L

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class TrackingCancelled(Exception):
    """Raised by a `ProgressCallback` to abort a tracking run.

    Deliberately not a `RuntimeError`: superqt's worker treats a `RuntimeError` as a
    deleted C++ object, warns, and returns without emitting `errored` or `finished`, which
    would leave a cancelling GUI waiting forever.
    """


@runtime_checkable
class ProgressCallback(Protocol):
    """Sink for progress events.

    Implementations may raise `TrackingCancelled` to abort the run.
    """

    def __call__(
        self,
        *,
        stage: str,
        current: int = 0,
        total: int | None = None,
        message: str = "",
    ) -> None:
        """Report progress within `stage`."""
        ...


def report(
    progress: ProgressCallback | None,
    stage: str,
    current: int = 0,
    total: int | None = None,
    message: str = "",
) -> None:
    """Emit a progress event when a callback is attached."""
    if progress is None:
        return
    progress(stage=stage, current=current, total=total, message=message)


def track_iter(
    iterable: Iterable[Any],
    progress: ProgressCallback | None,
    stage: str,
    total: int | None = None,
    message: str = "",
) -> Iterator[Any]:
    """Yield from `iterable`, reporting progress after each item.

    Reports once with `current=0` up front so a GUI can show the stage before the first
    (potentially slow) item completes.
    """
    if progress is None:
        yield from iterable
        return

    report(progress, stage, 0, total, message)
    for i, item in enumerate(iterable, start=1):
        yield item
        report(progress, stage, i, total, message)


class LightningProgressCallback(L.Callback):
    """Bridges Lightning's predict loop to a `ProgressCallback`.

    Checks for cancellation before each batch and reports after each one, so a run can be
    aborted mid-prediction.
    """

    def __init__(self, progress: ProgressCallback | None, stage: str = "predict"):
        """Initialize with the sink to forward batch progress to."""
        self.progress = progress
        self.stage = stage

    def _total(self, trainer: L.Trainer) -> int | None:
        batches = trainer.num_predict_batches
        if isinstance(batches, list):
            batches = sum(batches)
        # lightning reports float('inf') for iterable datasets without a length
        return int(batches) if batches and batches != float("inf") else None

    def on_predict_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Give the callback a chance to raise `TrackingCancelled`."""
        report(self.progress, self.stage, batch_idx, self._total(trainer), "Predicting")

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Report completion of a predict batch."""
        report(
            self.progress, self.stage, batch_idx + 1, self._total(trainer), "Predicting"
        )
