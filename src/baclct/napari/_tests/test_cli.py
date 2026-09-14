"""Tests for the `baclct napari` command.

These build the viewer directly instead of through the `make_napari_viewer` fixture,
which segfaults on some OpenGL stacks, and they never start a Qt
event loop: `napari.run` is patched out and `_open_viewer` is called on its own.
"""

from __future__ import annotations

from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from typer.main import get_command

from baclct import cli


@pytest.fixture
def open_viewer(sequence):
    """Run `_open_viewer` over the fixture sequence, closing the viewer afterwards."""
    viewers = []

    def _open(lazy: bool = True, **settings):
        images, masks = sequence
        viewer, widget = cli._open_viewer(images, masks, lazy, settings)
        viewers.append(viewer)
        return viewer, widget

    yield _open
    for viewer in viewers:
        viewer.close()


def _invoke(*args: str):
    """Invoke `baclct napari` with the viewer and the event loop stubbed out."""
    runner = CliRunner()
    with (
        patch.object(cli, "_open_viewer") as open_viewer,
        patch.dict("sys.modules"),
    ):
        import napari

        with patch.object(napari, "run"):
            result = runner.invoke(
                get_command(cli.app),
                ["napari", *args],
                catch_exceptions=True,
            )
    return open_viewer, result


# --------------------------------------------------------------------- argument handling


@pytest.mark.parametrize(
    ("args", "message", "have_napari"),
    [
        (("{images}",), "both IMAGES and MASKS", True),
        (
            ("{images}", "{masks}", "--prune-edges-by", "x"),
            "Unknown --prune-edges-by",
            True,
        ),
        (("{images}", "{masks}"), "baclct[napari]", False),
    ],
    ids=["lone_path", "unknown_choice", "napari_missing"],
)
def test_usage_errors(sequence, args, message, have_napari):
    """Unusable invocations give a usage error instead of a traceback out of napari."""
    paths = {"images": sequence[0], "masks": sequence[1]}
    with patch.object(cli, "NAPARI_AVAILABLE", have_napari):
        _, result = _invoke(*(str(arg).format(**paths) for arg in args))

    assert result.exit_code != 0
    assert message in result.output


def test_unset_flags_are_not_forwarded(sequence):
    """A flag that was not passed must not reach the widget as an override."""
    images, masks = sequence
    open_viewer, result = _invoke(str(images), str(masks), "--thr-div", "0.4")

    assert result.exit_code == 0, result.output
    settings = open_viewer.call_args.args[3]
    assert settings["thr_div"] == 0.4
    assert "thr_corr" not in settings
    assert "patch_size" not in settings
    # --no-lazy is opt-in, so the plain form loads on demand
    assert open_viewer.call_args.args[2] is True

    open_viewer, _ = _invoke(str(images), str(masks), "--no-lazy")
    assert open_viewer.call_args.args[2] is False


def test_no_paths_opens_an_empty_viewer():
    """With neither path the command still opens napari with the widget docked."""
    open_viewer, result = _invoke()

    assert result.exit_code == 0, result.output
    images, masks, _lazy, _settings = open_viewer.call_args.args
    assert images is None and masks is None


@pytest.mark.parametrize(
    ("images", "masks", "expected"),
    [
        # 'ctc': the sequence is what the image directory and the mask branch share
        ("01", "01_GT/TRA", "01"),
        # 'dirs': both sit in the sequence directory
        ("seq/01/BF", "seq/01/Segmentation", "01"),
        # 'flat': the sequence is the shared filename prefix
        ("01_images.tif", "01_masks.tif", "01"),
        # nothing shared, and a single letter is not an id
        ("images.tif", "masks.tif", None),
    ],
)
def test_sequence_id_is_what_both_paths_agree_on(tmp_path, images, masks, expected):
    paths = []
    for name in (images, masks):
        path = tmp_path / name
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True)
        paths.append(path)

    assert cli._sequence_id(*paths) == expected


def test_float_masks_are_cast_for_the_labels_layer(tmp_path, masks):
    """A float label image still opens: a Labels layer only accepts integers."""
    path = tmp_path / "float_masks.tif"
    tifffile.imwrite(path, masks.astype(np.float32), photometric="minisblack")

    cast = cli._as_labels(cli._read_frames(path, lazy=True))

    assert np.issubdtype(cast.dtype, np.integer)
    assert isinstance(cast, da.Array)


# ---------------------------------------------------------------------- opening a viewer


def test_lazy_layers_reach_the_widget_and_the_preview(open_viewer):
    """The layers hold frame-chunked dask arrays, and stay lazy on the way out."""
    viewer, widget = open_viewer(lazy=True)

    image_layer, mask_layer = viewer.layers[0], viewer.layers[1]
    assert isinstance(image_layer.data, da.Array)
    assert image_layer.data.chunks[0] == (1, 1, 1)
    assert widget._image_layer.value is image_layer
    assert widget._labels_layer.value is mask_layer
    assert widget._run.enabled is True

    params = widget._build_params()
    assert isinstance(params.images, da.Array)
    assert isinstance(params.masks, da.Array)

    # the preview materializes the masks itself, so it survives a lazy layer
    widget._on_show_radius()
    assert "BacLCT search radius" in [layer.name for layer in viewer.layers]


def test_a_ctc_pair_opens_named_and_preset(tmp_path, masks):
    """A CTC pair opens as '01 images' and '01 masks' instead of '01' and 'TRA'."""
    images_dir, masks_dir = tmp_path / "01", tmp_path / "01_GT" / "TRA"
    images_dir.mkdir()
    masks_dir.mkdir(parents=True)
    for t, frame in enumerate(masks):
        tifffile.imwrite(images_dir / f"t{t:03d}.tif", frame * 100)
        tifffile.imwrite(masks_dir / f"t{t:03d}.tif", frame)

    settings = {"graph_search_radius": "trained", "cache_dir": "temp"}
    viewer, widget = cli._open_viewer(images_dir, masks_dir, False, settings)
    try:
        assert [layer.name for layer in viewer.layers] == ["01 images", "01 masks"]
        # --no-lazy reads the stack up front instead of holding dask
        assert isinstance(viewer.layers[0].data, np.ndarray)

        # 'trained' is the CLI spelling of the blank field, 'temp' of the session cache
        assert widget._radius.value == ""
        assert widget._build_params().graph_search_radius is None
        assert widget._cache_mode.value == "Session"
        # and what was not passed keeps the widget's own defaults
        assert (widget._thr_corr.value, widget._thr_div.value) == (0.50, 0.30)
        assert widget._seg_correction.value is True
    finally:
        viewer.close()


def test_settings_round_trip_into_track_params(open_viewer, tmp_path):
    """Every applied flag comes back out of the params the widget builds.

    Passed together, since the model reseeds the radius, the pruning method resets its
    parameter, and the export format resets the suffix.
    """
    cache, out = tmp_path / "cache", tmp_path / "results"
    _, widget = open_viewer(
        model="baclct_toiam_pc",
        graph_search_radius="42",
        prune_edges_by="overlap",
        prune_param="0.55",
        thr_corr=0.7,
        thr_div=0.4,
        segmentation_correction=False,
        device="cpu",
        num_workers=2,
        batch_size=4,
        patch_size=128,
        cache_dir=cache,
        output_dir=out,
        export_format="ctc",
        export_suffix="_mine",
    )

    # --output-dir fills the path in and also ticks the export box
    assert widget._use_output.value is True
    params = widget._build_params()
    assert params.graph_search_radius == 42
    assert params.prune_edges_by == ("overlap", 0.55)
    assert (params.thr_corr, params.thr_div) == (0.7, 0.4)
    assert params.segmentation_correction is False
    assert params.device == "cpu"
    assert (params.num_workers, params.batch_size, params.patch_size) == (2, 4, 128)
    assert params.cache_dir == cache
    assert params.output_dir == out
    assert (params.export_format, params.export_suffix) == ("ctc", "_mine")
