"""Tests for the BacLCT widget.

The tracking pipeline is monkeypatched throughout: these cover the widget's own logic
(layer defaults, validation, model selection, progress and cancellation wiring), not the
model, so they run without torch weights.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from baclct.napari._layers import add_result_layers
from baclct.napari._widget import (
    PRETRAINED_MODELS,
    RADIUS_LAYER_NAME,
    BacLCTWidget,
)
from baclct.napari._worker import TrackResult
from baclct.utils.pretrained import MODEL_SPECS


@pytest.fixture
def masks() -> np.ndarray:
    masks = np.zeros((3, 60, 60), dtype=np.uint16)
    for t in range(3):
        masks[t, 10:20, 10:18] = 1
        masks[t, 30:40, 30:38] = 2
    return masks


@pytest.fixture
def images(masks) -> np.ndarray:
    return (masks > 0).astype(np.uint16) * 100


@pytest.fixture
def widget(make_napari_viewer, images, masks) -> BacLCTWidget:
    """A widget over one image layer and one segmentation layer, ready to run."""
    viewer = make_napari_viewer()
    viewer.add_image(images, name="raw")
    viewer.add_labels(masks, name="masks")
    return BacLCTWidget(viewer)


def _tracks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label": [1, 1],
            "t": [0, 1],
            "centroid-0": [15.0, 15.0],
            "centroid-1": [14.0, 14.0],
            "parent": [0, 0],
        }
    )


def _write_model_config(path, body: str) -> None:
    """Write a minimal `.hydra/config.yaml`, so a directory reads as a model."""
    (path / ".hydra").mkdir(exist_ok=True)
    (path / ".hydra" / "config.yaml").write_text(body)


def test_widget_constructs_without_layers(make_napari_viewer):
    widget = BacLCTWidget(make_napari_viewer())

    assert widget._run.enabled is False
    assert "segmentation" in widget._status.value.lower()


def test_widget_opens_from_the_plugin_menu(make_napari_viewer):
    # goes through napari's own viewer injection, which the direct calls above bypass
    viewer = make_napari_viewer()

    _, widget = viewer.window.add_plugin_dock_widget("baclct", "BacLCT")

    assert isinstance(widget, BacLCTWidget)


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


@pytest.mark.parametrize("model", PRETRAINED_MODELS)
def test_every_registered_model_is_selectable(widget, model):
    widget._pretrained.value = model

    assert widget.model_name == model
    assert model in widget._resolved_model.value
    # the description comes from the registry, so the picker cannot drift from it
    assert widget._model_help.value == MODEL_SPECS[model].description


def test_pretrained_choices_match_the_registry():
    assert set(PRETRAINED_MODELS) == set(MODEL_SPECS)


def test_prune_choices_cover_the_pipeline_methods():
    from baclct.features.graph import EdgePruneArg
    from baclct.napari._widget import PRUNE_CHOICES, PRUNE_HELP

    values = {value for _, value in PRUNE_CHOICES}
    # every non-training pipeline method is offered, and each choice has a description
    assert {m for m in EdgePruneArg.__args__ if m != "gt"} <= values
    assert set(PRUNE_HELP) == values


def test_prune_description_follows_the_selection(widget):
    widget._prune_method.value = "ellipse"
    assert "elliptical" in widget._prune_help.value

    widget._prune_method.value = "radius"
    assert "circular" in widget._prune_help.value

    widget._prune_method.value = "off"
    assert "No pruning" in widget._prune_help.value


def test_pruning_off_leaves_the_search_radius_as_the_only_limit(widget):
    """'Off' prunes nothing, unlike 'Circle', which filters by a scaled cell radius."""
    widget._prune_method.value = "off"

    assert widget._build_params().prune_edges_by == "off"
    # nothing is pruned, so the preview draws the search region and no second shape
    widget._on_show_radius()
    assert len(widget._viewer.layers[RADIUS_LAYER_NAME].data) == 1

    widget._prune_method.value = "radius"
    assert widget._prune_edges_by() == ("radius", widget._prune_param.value)
    widget._on_show_radius()
    assert len(widget._viewer.layers[RADIUS_LAYER_NAME].data) == 2


@pytest.mark.parametrize(
    ("method", "visible"),
    [("ellipse", True), ("overlap", True), ("dilated_overlap", False), ("off", False)],
)
def test_pruning_parameter_is_shown_only_where_it_applies(qtbot, widget, method, visible):
    """The parameter is hidden for the methods that take none."""
    # Qt reports every child of an unshown container as hidden, so show and expand first
    qtbot.addWidget(widget.native)
    widget._additional_section.expand(animate=False)
    widget.show()

    widget._prune_method.value = method

    assert widget._prune_param.visible is visible


def test_overlap_pruning_is_a_fraction(widget):
    """`EdgeFinder` filters `overlap` on intersection / area, so >1 prunes every edge."""
    widget._prune_method.value = "overlap"

    assert widget._prune_param.max == 1.0
    assert 0.0 <= widget._prune_param.value <= 1.0
    assert widget._prune_edges_by() == ("overlap", widget._prune_param.value)

    # switching back restores the axis-scaling range and factor the other methods use
    widget._prune_method.value = "ellipse"
    assert widget._prune_param.max == 100.0
    assert widget._prune_param.value == 3.0


def test_expanded_overlap_is_sized_by_cell_thickness(widget):
    """The dilation is sized by the cell's own thickness, not by the spinbox."""
    widget._prune_method.value = "dilated_overlap"

    assert widget._prune_edges_by() == ("dilated_overlap", "thickness")


def test_default_preview_resolves_a_per_cell_prune_parameter(widget, tmp_path):
    """The spore classification model prunes by a column name, resolved per cell."""
    _write_model_config(
        tmp_path,
        "features:\n  edges:\n    prune_edges_by:\n"
        "    - dilated_overlap\n    - thickness\n",
    )
    widget._model_source.value = "From path"
    widget._model_path.value = tmp_path

    assert widget._default_prune_edges_by() == ("dilated_overlap", "thickness")
    widget._on_show_radius()
    # the pruned region is drawn, not silently dropped as an unknown parameter
    assert RADIUS_LAYER_NAME in widget._viewer.layers
    assert len(widget._viewer.layers[RADIUS_LAYER_NAME].data) == 2


def test_default_preview_resolves_the_model_pruning(widget, tmp_path):
    """'Default' has no region of its own, so the preview reads the model's config."""
    _write_model_config(
        tmp_path, "features:\n  edges:\n    prune_edges_by:\n    - ellipse\n    - 7\n"
    )
    widget._model_source.value = "From path"
    widget._model_path.value = tmp_path

    assert widget._default_prune_edges_by() == ("ellipse", 7.0)
    # the actual run still defers to the model, so the override stays None
    assert widget._prune_edges_by() is None
    assert widget._build_params().prune_edges_by is None


def test_default_preview_falls_back_when_model_unavailable(widget, tmp_path):
    widget._model_source.value = "From path"
    widget._model_path.value = tmp_path  # exists but carries no .hydra config

    assert widget._default_prune_edges_by() is None


def test_search_radius_is_seeded_from_the_model(widget, tmp_path):
    """The field takes the model's radius, rather than overriding it with its own."""
    _write_model_config(tmp_path, "data:\n  graph_search_radius: 150\n")
    widget._model_source.value = "From path"
    widget._model_path.value = tmp_path
    widget._refresh()

    assert widget._radius.value == "150"
    assert widget._build_params().graph_search_radius == 150

    # a hand-edited radius survives, since only a model change reseeds the field
    widget._radius.value = "80"
    widget._refresh()
    assert widget._radius.value == "80"


def test_search_radius_keeps_its_default_without_a_model_config(widget, tmp_path):
    widget._model_source.value = "From path"
    widget._model_path.value = tmp_path
    widget._refresh()

    assert widget._radius.value == "2.5x"


def test_advanced_parameters_fold_into_a_collapsed_section(make_napari_viewer):
    from superqt import QCollapsible

    widget = BacLCTWidget(make_napari_viewer())

    # tracking and runtime controls sit in their titled groups, not the flat layout
    assert widget._thr_corr in widget._tracking_group
    assert widget._device in widget._runtime_group
    assert widget._thr_corr not in widget

    # tiling, cache and output stay flat and always visible
    assert widget._use_patches in widget
    assert widget._export_format in widget

    # a single 'Additional parameters' section, collapsed by default
    layout = widget._content.layout()
    top_level = [layout.itemAt(i).widget() for i in range(layout.count())]
    sections = [w for w in top_level if isinstance(w, QCollapsible)]
    assert [s.text() for s in sections] == ["Additional parameters"]
    assert not sections[0].isExpanded()


def test_widget_is_docked_as_a_scroll_area(make_napari_viewer):
    """The docked `native` must be the scroll area, not the inner content."""
    from qtpy.QtWidgets import QScrollArea

    widget = BacLCTWidget(make_napari_viewer())

    assert isinstance(widget.native, QScrollArea)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("150", 150), (" 150 ", 150), ("2.5x", "2.5x"), ("", None), ("42.7", 43)],
)
def test_search_radius_is_sanitized(widget, text, expected):
    """A free-text radius must reach the pipeline as pixels, not an unparsed string."""
    widget._radius.value = text
    assert widget._search_radius() == expected
    assert widget._build_params().graph_search_radius == expected


@pytest.mark.parametrize("text", ["abc", "2.5y", "xx", "1.2.3"])
def test_unparseable_search_radius_blocks_the_run(widget, text):
    """A typo must be caught before the run, not minutes into feature extraction."""
    widget._radius.value = text
    widget._refresh()

    assert widget._run.enabled is False
    assert "radius" in widget._status.value.lower()


def test_image_layer_is_required(make_napari_viewer, masks):
    """Both shipped models consume image features, so masks alone are not enough."""
    viewer = make_napari_viewer()
    viewer.add_labels(masks)

    widget = BacLCTWidget(viewer)

    assert widget._run.enabled is False
    assert "image layer" in widget._status.value.lower()


def test_mismatched_shapes_are_rejected(make_napari_viewer, masks):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((3, 10, 10), dtype=np.uint16))
    viewer.add_labels(masks)

    widget = BacLCTWidget(viewer)

    assert widget._run.enabled is False
    assert "differ" in widget._status.value


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


def test_build_params_sequence_id_is_the_output_name(widget):
    """`sequence_id` no longer hashes mask content.

    The feature cache guards content identity itself (see `dataset_identity`), so the
    widget's cache key is just the layer name, which also names the export.
    """
    params = widget._build_params()
    assert params.sequence_id == widget._output_name()


def test_session_cache_outlives_a_single_run(widget, tmp_path):
    """The cache belongs to the widget, so a second run reads the first run's features."""
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


def test_parallel_jobs_never_reach_the_predict_dataloader(widget):
    """The runtime controls raise joblib jobs and encoder workers, never predict ones.

    They point in opposite directions: a joblib pool and the patch croppers pay off on a
    long movie, while dataloader workers are a net loss on the cached prediction path on
    every platform.
    """
    assert widget._num_jobs.value == 1
    assert widget._num_workers_encode.value == 2
    assert widget._build_params().num_workers == 0

    widget._num_jobs.value = 6
    widget._num_workers_encode.value = 4
    params = widget._build_params()
    assert params.num_jobs == 6
    assert params.num_workers_encode == 4
    assert params.num_workers == 0


def test_tracker_overrides_are_collected(widget):
    widget._thr_corr.value = 0.8
    widget._thr_div.value = 0.5
    widget._seg_correction.value = False
    widget._prune_method.value = "ellipse"
    widget._prune_param.value = 6.0

    params = widget._build_params()

    assert params.tracker_overrides == {
        "thr_corr": 0.8,
        "thr_div": 0.5,
        "segmentation_correction": None,
    }
    assert params.prune_edges_by == ("ellipse", 6.0)


@pytest.mark.parametrize(("use_patches", "expected"), [(False, None), (True, 128)])
def test_patch_size_is_collected_only_when_enabled(widget, use_patches, expected):
    widget._use_patches.value = use_patches
    widget._patch_size.value = 128

    assert widget._build_params().patch_size == expected


def test_show_radius_adds_a_shapes_layer(widget):
    viewer = widget._viewer

    widget._on_show_radius()
    assert RADIUS_LAYER_NAME in viewer.layers
    n_shapes = len(viewer.layers[RADIUS_LAYER_NAME].data)

    # pressing again replaces the layer instead of stacking a second one
    widget._on_show_radius()
    assert len(viewer.layers[RADIUS_LAYER_NAME].data) == n_shapes
    assert sum(ly.name == RADIUS_LAYER_NAME for ly in viewer.layers) == 1


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


def test_results_are_named_after_the_segmentation_and_replace_the_last_run(widget, masks):
    """Both layers take the segmentation's name, and a later run replaces them."""
    viewer = widget._viewer
    other = viewer.add_labels(masks, name="other_seg")
    result = TrackResult(tracks=_tracks(), masks_tracked=masks)

    widget._run_layer = viewer.layers["masks"]
    widget._on_finished(result)

    assert [ly.name for ly in widget._result_layers] == [
        "masks (tracked)",
        "masks (lineage)",
    ]
    # the tracked masks would otherwise sit under the segmentation they replace
    assert viewer.layers["masks"].visible is False

    widget._run_layer = other
    widget._on_finished(result)

    names = [ly.name for ly in viewer.layers]
    assert "other_seg (tracked)" in names and "other_seg (lineage)" in names
    assert not any(name.startswith("masks (") for name in names)


def test_add_result_layers_replaces_previous_run(make_napari_viewer, masks):
    viewer = make_napari_viewer()

    add_result_layers(viewer, _tracks(), masks, "seq")
    add_result_layers(viewer, _tracks(), masks, "seq")

    # napari uniquifies duplicate names, so only the layer count shows the replacement
    assert len(viewer.layers) == 2
    assert {ly.name for ly in viewer.layers} == {"seq (tracked)", "seq (lineage)"}
