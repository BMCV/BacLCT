"""End-to-end test of the napari plugin against the real pipeline (slow).

Trains a minimal model, points the widget at it, and presses "Start tracking", so the
whole chain runs for real: widget -> background worker -> `BacLCT.track()` -> result
layers. The plugin's own unit tests fake the pipeline; this one does not.

Skipped unless napari is installed (`pixi run -e napari pytest tests/test_napari_e2e.py`).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import hydra
import lightning as L
import pytest
from omegaconf import OmegaConf

from baclct.api import BacLCT

pytest.importorskip("napari")

from baclct.napari._widget import RADIUS_LAYER_NAME, BacLCTWidget  # noqa: E402


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, reproducibility_data_dir) -> Path:
    """Train a minimal handcrafted model and return its experiment directory."""
    data_dir, feature_dir = reproducibility_data_dir
    run_dir = tmp_path_factory.mktemp("napari_run")

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
        config_module="baclct.config", job_name="napari_e2e", version_base="1.3"
    ):
        cfg = hydra.compose(config_name="default", overrides=overrides)
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        BacLCT(cfg).run_training()

    return run_dir


@pytest.mark.slow
def test_tracking_from_the_widget_adds_layers(
    qtbot, monkeypatch, make_napari_viewer, trained_run, toy_images, toy_masks, tmp_path
):
    """Pressing 'Start tracking' runs the real pipeline and adds the result layers."""
    viewer = make_napari_viewer()
    viewer.add_image(toy_images, name="raw")
    viewer.add_labels(toy_masks, name="masks")

    widget = BacLCTWidget(viewer)
    # a model from disk, since the pretrained weights are not fetched in tests
    widget._model_source.value = "From path"
    widget._model_path.value = trained_run
    widget._use_output.value = True
    widget._output_dir.value = tmp_path
    widget._device.value = "cpu"

    # the widget itself never asks for dataloader workers, but this is the only test that
    # pickles a real GraphDataset into worker processes, so it holds the spawn coverage
    build = widget._build_params
    monkeypatch.setattr(widget, "_build_params", lambda: replace(build(), num_workers=2))

    assert widget._run.enabled is True

    widget._on_run()
    assert widget._worker is not None
    with qtbot.waitSignal(widget._worker.finished, timeout=600_000):
        pass

    assert widget._status.value.startswith("Done"), widget._status.value

    # both results are named after the segmentation they were tracked from
    tracked = viewer.layers["masks (tracked)"]
    lineage = viewer.layers["masks (lineage)"]
    assert tracked.data.shape == toy_masks.shape
    assert len(lineage.data) > 0
    # the tracked masks would otherwise sit under the segmentation they replace
    assert viewer.layers["masks"].visible is False

    # the run also exported to the chosen output directory
    assert list(tmp_path.glob("*_tracks.csv"))
    assert list(tmp_path.glob("*_tracks.tif"))

    # the session cache outlives the run, so tracking again reuses these features
    cache_dir = widget._build_params().cache_dir
    assert list(cache_dir.glob("*/*/nodes.parquet"))  # type: ignore
    assert list(cache_dir.glob("*/*/edges_prune-*"))  # type: ignore

    # the widget is usable again afterwards
    assert widget._run.enabled is True
    assert widget._cancel_button.enabled is False


@pytest.mark.slow
def test_radius_preview_on_real_masks(make_napari_viewer, toy_images, toy_masks):
    """The radius preview draws shapes for the frame currently shown."""
    viewer = make_napari_viewer()
    viewer.add_image(toy_images, name="raw")
    viewer.add_labels(toy_masks, name="masks")

    widget = BacLCTWidget(viewer)
    widget._prune_method.value = "ellipse"
    widget._prune_param.value = 7.0
    widget._on_show_radius()

    shapes = viewer.layers[RADIUS_LAYER_NAME]
    # one search-radius circle and one pruning ellipse per previewed cell
    assert len(shapes.data) > 0
    assert len(shapes.data) % 2 == 0
