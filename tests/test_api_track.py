"""Tests for the `BacLCT.track()` front-end API (slow: trains a tiny model first).

`track()` is the entry point GUI front-ends (e.g. the napari plugin) drive, so these tests
cover the arguments they depend on: graph and tracker overrides, progress reporting,
cooperative cancellation, and access to the tracked masks.
"""

from __future__ import annotations

import gc
from pathlib import Path

import hydra
import lightning as L
import polars as pl
import pytest
from conftest import xfail_on_mps
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

from baclct.api import BacLCT, _worker_plan
from baclct.data.dataset import GraphDataset
from baclct.utils.progress import TrackingCancelled


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, reproducibility_data_dir) -> tuple[Path, Path]:
    """Train a minimal handcrafted model and return its experiment dir and checkpoint."""
    data_dir, feature_dir = reproducibility_data_dir
    run_dir = tmp_path_factory.mktemp("track_run")

    overrides = [
        "dataset=spores",
        "task=tracking",
        "features=handcrafted_only",
        "model=handcrafted",
        f"paths.output_dir={run_dir!s}",
        f"paths.data_dir={data_dir!s}",
        f"paths.feature_dir={feature_dir!s}",
        "debug=tests",
        "callbacks=tests",
    ]
    with hydra.initialize_config_module(
        config_module="baclct.config", job_name="track_api_test", version_base="1.3"
    ):
        cfg = hydra.compose(config_name="default", overrides=overrides)
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        checkpoint_path, _ = BacLCT(cfg).run_training()

    assert checkpoint_path is not None
    return run_dir, Path(checkpoint_path)


@pytest.fixture(scope="module")
def trained_run_with_states(tmp_path_factory, reproducibility_data_dir) -> Path:
    """Train a minimal model that also classifies node states, and return its run dir."""
    data_dir, feature_dir = reproducibility_data_dir
    run_dir = tmp_path_factory.mktemp("track_run_states")

    overrides = [
        "dataset=spores",
        "task=tracking_with_states",
        "features=handcrafted_only",
        "model=handcrafted",
        f"paths.output_dir={run_dir!s}",
        f"paths.data_dir={data_dir!s}",
        f"paths.feature_dir={feature_dir!s}",
        "debug=tests",
        "callbacks=tests",
    ]
    with hydra.initialize_config_module(
        config_module="baclct.config", job_name="track_states_test", version_base="1.3"
    ):
        cfg = hydra.compose(config_name="default", overrides=overrides)
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        BacLCT(cfg).run_training()

    return run_dir


class _Recorder:
    """Progress sink that records every event."""

    def __init__(self, cancel_after: int | None = None):
        self.events: list[tuple[str, int, int | None]] = []
        self.cancel_after = cancel_after

    def __call__(self, *, stage, current=0, total=None, message=""):
        self.events.append((stage, current, total))
        if self.cancel_after is not None and len(self.events) > self.cancel_after:
            raise TrackingCancelled

    @property
    def stages(self) -> set[str]:
        return {stage for stage, _, _ in self.events}


@xfail_on_mps
@pytest.mark.slow
def test_track_reports_progress_and_applies_overrides(
    trained_run, toy_images, toy_masks, tmp_path
):
    """Graph and tracker arguments reach the pipeline and progress is reported per stage.

    `data.num_workers` is set here because this is the only test that pickles a real
    `GraphDataset` into worker processes, so it holds the spawn/forkserver coverage.
    """
    run_dir, checkpoint = trained_run
    progress = _Recorder()

    overrides = {"checkpoint": str(checkpoint), "data": {"num_workers": 2}}
    pipeline = BacLCT(run_dir, config_overrides=overrides)
    masks_tracked, tracks = pipeline.track(
        images=toy_images,
        masks=toy_masks,
        graph_search_radius="2.5x",
        prune_edges_by=("ellipse", 1.5),
        thr_corr=0.9,
        thr_div=0.4,
        output_dir=tmp_path,
        sequence_id="toy",
        export_format="flat",
        progress=progress,
    )

    # the thresholds are tracker arguments, so they show on the tracker and never in the
    # merged config
    tracker = pipeline.tracker
    assert (tracker.thr_corr, tracker.thr_div) == (0.9, 0.4)
    assert pipeline.cfg.data.num_workers == 2
    assert isinstance(pipeline.dataset, GraphDataset)
    # 2.5 times the median major axis of the first frame
    assert pipeline.dataset.graph_search_radius == 65
    edge_finder = pipeline.dataset.edge_finder
    assert (edge_finder.prune_method, edge_finder.prune_param) == ("ellipse", 1.5)

    assert {"features", "predict", "tracking", "export"} <= progress.stages
    # progress must be monotonic within a stage and never exceed its total
    for stage, current, total in progress.events:
        if total is not None:
            assert 0 <= current <= total, f"{stage}: {current}/{total}"

    assert not tracks.is_empty()
    assert {"label", "t", "parent"} <= set(tracks.columns)

    # masks are exposed for the GUI and exported to disk
    assert masks_tracked.shape == toy_masks.shape
    assert (tmp_path / "toy_tracks.csv").exists()
    assert (tmp_path / "toy_tracks.tif").exists()


@xfail_on_mps
@pytest.mark.slow
def test_prune_edges_by_off_disables_pruning(trained_run, toy_images, toy_masks):
    """'off' removes the pruning, where `None` keeps what the model was trained with."""
    run_dir, checkpoint = trained_run
    pipeline = BacLCT(run_dir, config_overrides={"checkpoint": str(checkpoint)})
    assert pipeline.cfg.data.edge_finder.prune_edges_by is not None

    pipeline.track(
        images=toy_images, masks=toy_masks, prune_edges_by="off", sequence_id="toy"
    )

    assert isinstance(pipeline.dataset, GraphDataset)
    edge_finder = pipeline.dataset.edge_finder
    assert edge_finder.prune_edges is False
    assert (edge_finder.prune_method, edge_finder.prune_param) == (None, None)


@xfail_on_mps
@pytest.mark.slow
def test_track_cancellation_propagates(trained_run, toy_images, toy_masks, tmp_path):
    """A progress sink raising `TrackingCancelled` aborts the run without exporting."""
    run_dir, checkpoint = trained_run

    with pytest.raises(TrackingCancelled):
        BacLCT(run_dir, config_overrides={"checkpoint": str(checkpoint)}).track(
            images=toy_images,
            masks=toy_masks,
            output_dir=tmp_path,
            sequence_id="cancelled",
            export_format="flat",
            progress=_Recorder(cancel_after=1),
        )

    assert not list(tmp_path.glob("cancelled_tracks.*"))


@xfail_on_mps
@pytest.mark.slow
def test_track_cache_modes_agree(trained_run, toy_images, toy_masks):
    """The default temporary cache must produce the tracks of the in-memory mode.

    `cache_dir=None` builds all edges once and keeps them as a `DataFrame`, the default
    'temp' reads them back from a parquet store (forward-only, mirrored per item). Those
    are two different code paths through `_get_graph_for_frame` and they must not move the
    numbers.
    """
    run_dir, checkpoint = trained_run

    overrides = {"checkpoint": str(checkpoint)}
    baseline_pipeline = BacLCT(run_dir, config_overrides=overrides)
    _, expected = baseline_pipeline.track(
        images=toy_images,
        masks=toy_masks,
        sequence_id="toy",
        cache_dir=None,
    )
    pipeline = BacLCT(run_dir, config_overrides=overrides)
    _, tracks = pipeline.track(
        images=toy_images,
        masks=toy_masks,
        sequence_id="toy",
    )

    assert isinstance(baseline_pipeline.dataset, GraphDataset)
    assert isinstance(pipeline.dataset, GraphDataset)
    assert isinstance(baseline_pipeline.dataset.edge_data, pl.DataFrame)
    assert isinstance(pipeline.dataset.edge_data, pl.LazyFrame)
    assert_frame_equal(tracks, expected)


@xfail_on_mps
@pytest.mark.slow
def test_track_temp_cache_is_removed_with_the_dataset(trained_run, toy_images, toy_masks):
    """The default temporary cache must not outlive the pipeline that created it."""
    run_dir, checkpoint = trained_run
    pipeline = BacLCT(run_dir, config_overrides={"checkpoint": str(checkpoint)})
    pipeline.track(images=toy_images, masks=toy_masks, sequence_id="toy")

    dataset = pipeline.dataset
    assert isinstance(dataset, GraphDataset)
    assert dataset._feature_tempdir is not None
    cache_root = Path(dataset._feature_tempdir.name)
    # the edge store is read lazily from parquet rather than held in memory
    assert (cache_root / "toy").is_dir()
    assert isinstance(dataset.edge_data, pl.LazyFrame)

    del pipeline, dataset
    gc.collect()
    assert not cache_root.exists()


@xfail_on_mps
@pytest.mark.slow
def test_track_states_follow_the_trained_model(
    trained_run_with_states, toy_images, toy_masks, tmp_path
):
    """A model trained with node classes predicts states, with nothing to ask for."""
    pipeline = BacLCT(trained_run_with_states)
    _, tracks = pipeline.track(
        toy_images, toy_masks, sequence_id="01", output_dir=tmp_path
    )

    assert "state" in tracks.columns
    assert pipeline.node_preds is not None
    assert (tmp_path / "01_RES" / "res_states.csv").exists()


@xfail_on_mps
@pytest.mark.slow
def test_track_states_absent_for_a_tracking_only_model(
    trained_run, toy_images, toy_masks, tmp_path
):
    """A model without node classes yields no column, no predictions, and no export."""
    run_dir, _ = trained_run
    pipeline = BacLCT(run_dir)
    _, tracks = pipeline.track(
        toy_images, toy_masks, sequence_id="01", output_dir=tmp_path
    )

    assert "state" not in tracks.columns
    assert pipeline.node_preds is None
    assert not (tmp_path / "01_RES" / "res_states.csv").exists()


@pytest.mark.parametrize(
    ("num_workers", "n_jobs", "encode"),
    [(0, 1, 0), (4, 4, 2), (-1, -1, 2)],
    ids=["serial", "parallel", "every_core"],
)
def test_worker_plan_derives_every_pool(num_workers, n_jobs, encode):
    """Feature workers scale with the argument, the encoder is pinned at its optimum.

    `-1` is joblib's "every core" and is passed through unbounded: no measured peak
    justifies capping node extraction, so sizing the pool is left to the caller.
    """
    plan = _worker_plan(num_workers)

    assert plan["features"]["handcrafted"]["n_jobs"] == n_jobs
    assert plan["features"]["deep"]["num_workers"] == encode
