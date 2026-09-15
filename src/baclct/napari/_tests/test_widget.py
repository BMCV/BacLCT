"""Tests for the BacLCT widget.

The tracking pipeline is monkeypatched throughout: these cover the widget's own logic
(layer defaults, validation, model selection, cancellation wiring) and none of the model,
so they run without torch weights. Running it for real, and the result layers that come
back, are covered by `tests/test_napari_e2e.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from baclct.io import DEFAULT_EXPORT_SUFFIX
from baclct.napari._widget import (
    PRETRAINED_MODELS,
    RADIUS_LAYER_NAME,
    BacLCTWidget,
)
from baclct.napari._worker import TrackResult
from baclct.utils.pretrained import MODEL_SPECS


@pytest.fixture
def widget(make_napari_viewer, images, masks) -> BacLCTWidget:
    """A widget over one image layer and one segmentation layer, ready to run."""
    viewer = make_napari_viewer()
    viewer.add_image(images, name="raw")
    viewer.add_labels(masks, name="masks")
    return BacLCTWidget(viewer)


def _write_model_config(path, body: str):
    """Write a minimal `.hydra/config.yaml`, so a directory reads as a model."""
    (path / ".hydra").mkdir(parents=True, exist_ok=True)
    (path / ".hydra" / "config.yaml").write_text(body)
    return path


def test_choices_match_the_registries():
    """Neither picker may drift from the thing it offers."""
    from baclct.features.graph import EdgePruneArg
    from baclct.napari._widget import PRUNE_CHOICES, PRUNE_HELP

    assert set(PRETRAINED_MODELS) == set(MODEL_SPECS)
    values = {value for _, value in PRUNE_CHOICES}
    # every non-training pipeline method is offered, and each choice has a description
    assert {m for m in EdgePruneArg.__args__ if m != "gt"} <= values
    assert set(PRUNE_HELP) == values


# --------------------------------------------------------------------------- selection


def test_widget_selects_default_layers(make_napari_viewer, images, masks):
    viewer = make_napari_viewer()
    viewer.add_image(images, name="raw")
    viewer.add_labels(np.zeros_like(masks), name="other")
    viewer.add_labels(masks, name="masks")

    widget = BacLCTWidget(viewer)

    assert widget._image_layer.value.name == "raw"
    # a layer named like a mask wins over the first labels layer
    assert widget._labels_layer.value.name == "masks"
    assert widget._run.enabled is True
    assert widget._status.value == ""


def test_channel_axis_is_sliced(make_napari_viewer, masks):
    viewer = make_napari_viewer()
    multichannel = np.zeros((3, 2, 60, 60), dtype=np.uint16)
    multichannel[:, 1] = 7
    viewer.add_image(multichannel)
    viewer.add_labels(masks)

    widget = BacLCTWidget(viewer)
    assert widget.is_multichannel() is True
    assert widget._run.enabled is True

    widget._channel.value = 1
    assert widget._image_data().shape == masks.shape
    assert np.all(widget._image_data() == 7)


def test_lazy_layers_are_not_materialized(make_napari_viewer, images, masks):
    """`_refresh` runs on every layer change, so it must not compute a dask stack."""
    dask = pytest.importorskip("dask.array")
    computed = []

    class _Tracked(dask.Array):
        def __array__(self, *args, **kwargs):
            computed.append(True)
            return super().__array__(*args, **kwargs)

    def track(array):
        return _Tracked(array.dask, array.name, array.chunks, array.dtype)

    viewer = make_napari_viewer()
    viewer.add_image(track(dask.from_array(images, chunks=(1, *images.shape[1:]))))
    viewer.add_labels(track(dask.from_array(masks, chunks=(1, *masks.shape[1:]))))

    widget = BacLCTWidget(viewer)
    widget._refresh()

    assert widget._run.enabled is True
    assert not computed


# -------------------------------------------------------------------------- validation


def test_run_is_blocked_without_usable_layers(make_napari_viewer, masks):
    """Every reason that comes from the layers themselves."""
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((3, 10, 10), dtype=np.uint16))
    labels = viewer.add_labels(masks)
    widget = BacLCTWidget(viewer)

    assert widget._run.enabled is False
    assert "differ" in widget._status.value

    # both shipped models consume image features, so masks alone are not enough
    viewer.layers.remove(image)
    widget._image_layer.reset_choices()
    widget._refresh()
    assert widget._run.enabled is False
    assert "image layer" in widget._status.value.lower()

    viewer.layers.remove(labels)
    widget._labels_layer.reset_choices()
    widget._refresh()
    assert widget._run.enabled is False
    assert "segmentation" in widget._status.value.lower()


def test_search_radius_is_sanitized(widget):
    """A free-text radius is parsed to pixels before it reaches the pipeline."""
    cases = [("150", 150), (" 150 ", 150), ("42.7", 43), ("2.5x", "2.5x"), ("", None)]
    for text, expected in cases:
        widget._radius.value = text
        assert widget._search_radius() == expected, text
        assert widget._build_params().graph_search_radius == expected, text

    # a typo is caught before the run starts, well ahead of feature extraction
    widget._radius.value = "2.5y"
    widget._refresh()
    assert widget._run.enabled is False
    assert "radius" in widget._status.value.lower()


# ------------------------------------------------------------------------------- model


def test_model_config_seeds_the_radius_and_the_preview(widget, tmp_path):
    """The fields take the model's values, and 'Default' previews its pruning."""
    widget._model_source.value = "From path"
    widget._model_path.value = _write_model_config(
        tmp_path / "trained",
        "data:\n  graph_search_radius: 150\n"
        "features:\n  edges:\n    prune_edges_by:\n    - ellipse\n    - 7\n",
    )
    widget._refresh()

    assert widget._radius.value == "150"
    assert widget._build_params().graph_search_radius == 150
    assert widget._default_prune_edges_by() == ("ellipse", 7.0)
    # the run still defers to the model, so the override stays None
    assert widget._build_params().prune_edges_by is None

    # a hand-edited radius survives, since only a model change reseeds the field
    widget._radius.value = "80"
    widget._refresh()
    assert widget._radius.value == "80"

    # a per-cell column is passed on as-is, since the preview resolves it per cell
    widget._model_path.value = _write_model_config(
        tmp_path / "by_column",
        "features:\n  edges:\n    prune_edges_by:\n    - dilated_overlap\n"
        "    - thickness\n",
    )
    widget._refresh()
    assert widget._default_prune_edges_by() == ("dilated_overlap", "thickness")
    widget._on_show_radius()
    assert len(widget._viewer.layers[RADIUS_LAYER_NAME].data) == 2

    # a directory that exists but carries no .hydra config falls back to the defaults
    empty = tmp_path / "empty"
    empty.mkdir()
    widget._model_path.value = empty
    widget._refresh()
    assert widget._default_prune_edges_by() is None
    assert widget._radius.value == "2.5x"


# ----------------------------------------------------------------------------- pruning


def test_pruning_method_drives_the_parameter_and_the_preview(widget):
    """Each method's parameter, what reaches `track()`, and the region it previews."""
    viewer = widget._viewer

    # 'off' prunes nothing, so the search region is the only shape drawn
    widget._prune_method.value = "off"
    assert widget._build_params().prune_edges_by == "off"
    widget._on_show_radius()
    assert len(viewer.layers[RADIUS_LAYER_NAME].data) == 1

    # 'radius' filters by a scaled cell radius, which is a second shape
    widget._prune_method.value = "radius"
    assert widget._prune_edges_by() == ("radius", widget._prune_param.value)
    widget._on_show_radius()
    assert len(viewer.layers[RADIUS_LAYER_NAME].data) == 2
    # pressing again replaces the layer instead of stacking a second one
    widget._on_show_radius()
    assert sum(ly.name == RADIUS_LAYER_NAME for ly in viewer.layers) == 1

    # `EdgeFinder` filters 'overlap' on intersection / area, so >1 prunes every edge
    widget._prune_method.value = "overlap"
    assert widget._prune_param.max == 1.0
    assert widget._prune_edges_by() == ("overlap", widget._prune_param.value)

    # switching back restores the axis-scaling range and factor the other methods use
    widget._prune_method.value = "ellipse"
    assert widget._prune_param.max == 100.0
    assert widget._prune_param.value == 3.0

    # the dilation is sized by the cell's own thickness and ignores the spinbox
    widget._prune_method.value = "dilated_overlap"
    assert widget._prune_edges_by() == ("dilated_overlap", "thickness")


# ------------------------------------------------------------------------------ params


def test_controls_are_collected_into_track_params(widget):
    """The controls reach `track()`, and default to leaving the model's values alone."""
    # the extra pools are a net loss with the default features, so nothing is spun up
    assert widget._num_workers.value == 0
    # tiling is opt-in, so the patch size is not collected until the box is ticked
    widget._patch_size.value = 128
    assert widget._build_params().patch_size is None

    widget._num_workers.value = 6
    widget._thr_corr.value = 0.8
    widget._thr_div.value = 0.5
    widget._seg_correction.value = False
    widget._prune_method.value = "ellipse"
    widget._prune_param.value = 6.0
    widget._use_patches.value = True

    params = widget._build_params()

    assert params.num_workers == 6
    assert (params.thr_corr, params.thr_div) == (0.8, 0.5)
    assert params.segmentation_correction is False
    assert params.prune_edges_by == ("ellipse", 6.0)
    assert params.patch_size == 128
    # the cache key is the layer name, which also names the export
    assert params.sequence_id == widget._output_name()


def test_export_suffix_follows_the_format_and_survives_being_cleared(widget):
    """The suffix defaults per format, and an emptied field means no suffix at all.

    '' has to reach `track()` as '', not as `None`: `None` is what asks for the format
    default back.
    """
    assert widget._export_format.value == "flat"
    assert widget._build_params().export_suffix == DEFAULT_EXPORT_SUFFIX["flat"]

    widget._export_format.value = "ctc"
    assert widget._build_params().export_suffix == DEFAULT_EXPORT_SUFFIX["ctc"]

    widget._export_suffix.value = ""
    assert widget._build_params().export_suffix == ""


def test_a_second_run_reuses_the_cache_and_replaces_the_layers(
    widget, masks, tracks, tmp_path
):
    """What the widget carries between runs: the session cache and the result layers."""
    viewer = widget._viewer
    assert widget._cache_mode.value == "Session"
    cache_dir = widget._build_params().cache_dir
    assert cache_dir.is_dir()
    assert widget._build_params().cache_dir == cache_dir

    widget._cache_mode.value = "Directory"
    # a directory mode without a selection blocks the run rather than caching nowhere
    assert widget._run.enabled is False
    widget._cache_dir.value = tmp_path
    widget._refresh()
    assert widget._build_params().cache_dir == tmp_path
    assert widget._run.enabled is True

    widget._cache_mode.value = "In memory"
    assert widget._build_params().cache_dir is None

    # the layers of a previous run are dropped instead of uniquified alongside it
    other = viewer.add_labels(masks, name="other_seg")
    result = TrackResult(tracks=tracks, masks_tracked=masks)
    widget._run_layer = viewer.layers["masks"]
    widget._on_finished(result)
    widget._run_layer = other
    widget._on_finished(result)

    names = [ly.name for ly in viewer.layers]
    assert "other_seg (tracked)" in names and "other_seg (lineage)" in names
    assert not any(name.startswith("masks (") for name in names)


def test_cancel_reports_the_outcome(qtbot, widget, monkeypatch):
    """Cancelling must end on 'Cancelled.', which needs the `returned` signal to fire."""
    import time

    from baclct.napari import _worker

    monkeypatch.setattr(
        _worker.BacLCT, "download_model", staticmethod(lambda *a, **k: ".")
    )
    monkeypatch.setattr(_worker.BacLCT, "__init__", lambda self, *a, **k: None)

    def slow_track(self, *, masks, progress, **kwargs):
        for i in range(200):
            progress(stage="features", current=i, total=200, message="Extracting")
            time.sleep(0.01)
        raise AssertionError("cancellation should have stopped this")

    monkeypatch.setattr(_worker.BacLCT, "track", slow_track)

    widget._on_run()
    worker = widget._worker
    worker.yielded.connect(lambda _: widget._on_cancel())

    with qtbot.waitSignal(worker.finished, timeout=10_000):
        pass

    assert widget._status.value == "Cancelled."
    assert widget._worker is None
