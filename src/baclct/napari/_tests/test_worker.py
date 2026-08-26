"""Tests for the background tracking worker.

`BacLCT.track` is faked so these exercise the threading, progress, and cancellation wiring
without loading a model.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from baclct.napari import _worker
from baclct.napari._worker import (
    CancelToken,
    ProgressEvent,
    TrackParams,
    tracking_worker,
)


def _fake_tracks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label": [1],
            "t": [0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "parent": [0],
        }
    )


@pytest.fixture
def params() -> TrackParams:
    masks = np.zeros((2, 8, 8), dtype=np.uint16)
    masks[:, 2:4, 2:4] = 1
    return TrackParams(
        images=masks.astype(np.uint16),
        masks=masks,
        model="baclct_track",
        sequence_id="seq",
    )


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Replace model resolution and tracking with fast fakes that report progress."""
    monkeypatch.setattr(
        _worker.BacLCT, "download_model", staticmethod(lambda *a, **k: tmp_path)
    )
    monkeypatch.setattr(_worker.BacLCT, "__init__", lambda self, *a, **k: None)

    def fake_track(self, *, masks, progress, **kwargs):
        for i in range(1, 4):
            progress(stage="features", current=i, total=3, message="Extracting")
            time.sleep(0.01)
        progress(stage="tracking", current=1, total=1, message="Linking")
        return masks, _fake_tracks()

    monkeypatch.setattr(_worker.BacLCT, "track", fake_track)


def test_worker_reports_progress_and_returns(qtbot, params, fake_pipeline):
    worker = tracking_worker(params, CancelToken())

    events: list[ProgressEvent] = []
    results = []
    worker.yielded.connect(lambda e: events.append(e))
    worker.returned.connect(results.append)

    with qtbot.waitSignal(worker.finished, timeout=10_000):
        worker.start()

    assert len(results) == 1
    assert results[0].tracks.equals(_fake_tracks())
    assert results[0].masks_tracked.shape == params.masks.shape

    reported = [e for e in events if e is not None]
    assert {e.stage for e in reported} == {"features", "tracking"}
    # progress must never run backwards
    fractions = [e.fraction for e in reported]
    assert fractions == sorted(fractions)


def test_worker_cancellation_returns_none_without_erroring(qtbot, params, fake_pipeline):
    """Cancelling must not raise.

    superqt re-raises anything reaching `errored` into the Qt event loop, so a user
    pressing cancel would otherwise be shown a traceback.
    """
    cancel = CancelToken()
    worker = tracking_worker(params, cancel)

    errors = []
    results = []
    worker.errored.connect(errors.append)
    worker.returned.connect(results.append)
    # cancel as soon as the pipeline reports anything
    worker.yielded.connect(lambda _: cancel.cancel())

    with qtbot.waitSignal(worker.finished, timeout=10_000):
        worker.start()

    assert not errors
    assert results == [None]


def test_worker_failure_surfaces_as_an_error(qtbot, params, monkeypatch, tmp_path):
    """A genuine failure still reaches `errored` rather than being swallowed."""
    monkeypatch.setattr(
        _worker.BacLCT, "download_model", staticmethod(lambda *a, **k: tmp_path)
    )
    monkeypatch.setattr(_worker.BacLCT, "__init__", lambda self, *a, **k: None)

    def boom(self, **kwargs):
        raise ValueError("no checkpoint")

    monkeypatch.setattr(_worker.BacLCT, "track", boom)

    worker = tracking_worker(params, CancelToken())
    errors = []
    worker.errored.connect(errors.append)

    with qtbot.waitSignal(worker.finished, timeout=10_000):
        worker.start()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_progress_event_fraction_is_bounded():
    assert ProgressEvent("features", 0, 10).fraction == pytest.approx(0.05)
    assert ProgressEvent("export", 1, 1).fraction == pytest.approx(1.0)
    # deep encoding is its own stage between handcrafted features and edges
    assert ProgressEvent("encode", 0, 4).fraction == pytest.approx(0.15)
    # a stage with no total contributes only the stages before it
    assert ProgressEvent("predict").fraction == pytest.approx(0.75)
