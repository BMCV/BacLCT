"""Test the argument handling of the `baclct-train` and `baclct` CLIs.

The success paths are covered end-to-end by `test_e2e_configs.py` and `test_api_track.py`.
These tests only cover the argument validation, sequence selection, and error propagation
that the real runs do not exercise.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf
from typer.testing import CliRunner

from baclct.cli import app as baclct_app
from baclct.train import main as train_main


def test_train_main_missing_task():
    """Error is raised if task is not configured."""
    cfg = OmegaConf.create({"some": "config"})
    with pytest.raises(ValueError, match="Task was not configured"):
        train_main(cfg)


@patch("baclct.train.BacLCT")
def test_train_main_propagates_training_errors(mock_baclct):
    """Training is bare-bone: `run_training` failures propagate, not swallowed."""
    cfg = OmegaConf.create({"task_configured": True})
    mock_pipeline = MagicMock()
    mock_pipeline.run_training.side_effect = RuntimeError("err")
    mock_baclct.return_value = mock_pipeline

    with pytest.raises(RuntimeError, match="err"):
        train_main(cfg)


@pytest.fixture
def ctc_data_dir(tmp_path):
    """A CTC dataset directory with two sequences, '01' and '02'."""
    data_dir = tmp_path / "data"
    for seq in ("01", "02"):
        (data_dir / seq).mkdir(parents=True)
        (data_dir / f"{seq}_GT" / "TRA").mkdir(parents=True)
    return data_dir


@pytest.fixture
def track_cli(tmp_path, ctc_data_dir):
    """Run `baclct-track` against a mocked pipeline and image loader.

    Returns the `BacLCT.track` mock and the click result. `track_raises` makes the first
    sequence fail, `loaded` is what the loader hands back.
    """
    runner = CliRunner()

    def _run(
        *args: str,
        track_raises: bool = False,
        default_flags: bool = True,
        loaded: tuple = (None, None),
    ):
        argv = list(args)
        if default_flags:
            argv = [
                "--data-dir",
                str(ctc_data_dir),
                "--output-dir",
                str(tmp_path / "out"),
                *argv,
            ]
        with (
            patch("baclct.cli.BacLCT") as mock_baclct,
            patch("baclct.cli.load_images_and_masks", return_value=loaded),
        ):
            track = mock_baclct.return_value.track
            if track_raises:
                track.side_effect = [RuntimeError("boom"), None]
            # `track` is a subcommand of the `baclct` group now
            return track, runner.invoke(
                baclct_app, ["track", *argv], catch_exceptions=True
            )

    return _run


def _tracked(track) -> list[str]:
    """Sequence ids the CLI passed to `BacLCT.track`, in call order."""
    return [call.kwargs["sequence_id"] for call in track.call_args_list]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--sequences", "02", "--sequences", "01"), ["02", "01"]),
        (("--sequences", "02,01"), ["02", "01"]),
        ((), ["01", "02"]),
    ],
    ids=["repeated", "comma", "detected"],
)
def test_track_main_selects_sequences(track_cli, args, expected):
    """Sequences come from `--sequences` in the given order, else from the data dir."""
    track, result = track_cli(*args)
    assert result.exit_code == 0, result.output
    assert _tracked(track) == expected


def test_track_main_selects_sequences_from_a_split(track_cli, tmp_path):
    """`--split-file` reads the ids of the requested fold and phase."""
    split_file = tmp_path / "splits.yaml"
    split_file.write_text(yaml.dump({0: {"train": ["01"], "val": ["02"]}}))

    track, result = track_cli(
        "--split-file", str(split_file), "--fold", "0", "--phase", "val"
    )
    assert result.exit_code == 0, result.output
    assert _tracked(track) == ["02"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("{data}/i.tif",), "both IMAGES"),
        (("{data}/i.tif", "{data}/m.tif", "--data-dir", "{data}"), "not both"),
        (
            ("{data}/i.tif", "{data}/m.tif", "--sequences", "a", "--sequences", "b"),
            "--sequences",
        ),
        (("--data-dir", "{data}", "--data-format", "zarr"), "zarr"),
        (("--data-dir", "{data}", "--sequences", "01", "--sequences", "99"), "99"),
        (("--data-dir", "{empty}"), "No 'ctc' sequences"),
        (("--data-dir", "{data}", "--split-file", "{split}"), "--fold"),
    ],
    ids=[
        "lone_path",
        "paths_and_data_dir",
        "several_ids_for_one_sequence",
        "unknown_format",
        "unknown_sequence",
        "empty_data_dir",
        "split_without_fold",
    ],
)
def test_track_main_rejects_bad_invocations(
    track_cli, ctc_data_dir, tmp_path, args, message
):
    """Unusable invocations are usage errors, raised before any sequence is loaded."""
    empty = tmp_path / "empty"
    empty.mkdir()
    split = tmp_path / "splits.yaml"
    split.write_text(yaml.dump({0: {"test": ["01"]}}))
    paths = {"data": ctc_data_dir, "empty": empty, "split": split}

    track, result = track_cli(*(arg.format(**paths) for arg in args), default_flags=False)

    assert result.exit_code != 0
    assert message in result.output.replace('"', "'")
    assert not track.called


def test_track_main_tracks_a_flat_dataset(track_cli, tmp_path):
    """Paired tifs are found by their filenames, once --data-format names the layout."""
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    for seq in ("a", "b"):
        (flat_dir / f"{seq}_images.tif").touch()
        (flat_dir / f"{seq}_masks.tif").touch()

    track, result = track_cli(
        "--data-dir", str(flat_dir), "--data-format", "flat", default_flags=False
    )
    assert result.exit_code == 0, result.output
    assert _tracked(track) == ["a", "b"]
    # a flat input exports flat, without the user asking
    assert track.call_args.kwargs["export_format"] == "flat"


def test_track_main_takes_two_explicit_paths(track_cli, tmp_path):
    """The two-path form needs no layout, and names the results after the sequence.

    The id is what the two paths agree on ('s1' here), since the leaf of a mask path
    rarely names the sequence: '01_GT/TRA' is the segmentation of '01'.
    """
    images, masks = tmp_path / "s1_img.tif", tmp_path / "s1_seg.tif"
    for path in (images, masks):
        path.touch()

    with patch("baclct.cli._read_frames", return_value="array") as read:
        track, result = track_cli(str(images), str(masks), default_flags=False)

    assert result.exit_code == 0, result.output
    assert _tracked(track) == ["s1"]
    assert [call.args[0] for call in read.call_args_list] == [masks, images]
    assert track.call_args.kwargs["output_dir"] == tmp_path
    assert track.call_args.kwargs["export_format"] == "flat"


@pytest.mark.parametrize(
    # a dry run never reads a frame, so it stays lazy whatever the flag says. the plain
    # forms are covered below, on what actually reaches `track()`
    ("args", "expected_lazy"),
    [(("--dry-run",), True), (("--dry-run", "--no-lazy"), True)],
    ids=["dry-run", "dry-run-no-lazy"],
)
def test_track_main_lazy_flag_reaches_the_loader(
    track_cli, tmp_path, args, expected_lazy
):
    """--dry-run forces lazy reads on, whatever --lazy/--no-lazy says."""
    images, masks = tmp_path / "s1_img.tif", tmp_path / "s1_seg.tif"
    for path in (images, masks):
        path.touch()

    with patch("baclct.cli._read_frames", return_value=np.zeros((2, 4, 4))) as read:
        _, result = track_cli(str(images), str(masks), *args, default_flags=False)

    assert result.exit_code == 0, result.output
    assert {call.kwargs["lazy"] for call in read.call_args_list} == {expected_lazy}


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_track_main_hands_the_pipeline_what_lazy_asked_for(track_cli, tmp_path, lazy):
    """The --lazy flag decides whether the frames reach `BacLCT.track` as dask."""
    import dask.array as da
    import tifffile

    frames = np.zeros((3, 8, 8), np.uint16)
    images, masks = tmp_path / "s1_img.tif", tmp_path / "s1_seg.tif"
    for path in (images, masks):
        tifffile.imwrite(path, frames)

    track, result = track_cli(
        str(images), str(masks), *([] if lazy else ["--no-lazy"]), default_flags=False
    )

    assert result.exit_code == 0, result.output
    passed = track.call_args.args[0]
    assert isinstance(passed, da.Array) is lazy
    if lazy:
        # one chunk per frame is what lets the encoder use workers
        assert passed.chunks[0] == (1, 1, 1)


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize(
    ("args", "existing"),
    [
        ((), ("01_RES/res_track.txt",)),
        (("--export-format", "flat"), ("01_tracks.csv", "01_tracks.tif")),
        (("--export-suffix", "_mine"), ("01_mine/res_track.txt",)),
    ],
    ids=["ctc", "flat", "suffix"],
)
def test_track_main_skips_existing_results(
    track_cli, tmp_path, args, existing, overwrite
):
    """Results already on disk are skipped unless `--overwrite` is given.

    Both formats count, under whichever suffix the run would write: 'ctc' as a non-empty
    result directory, 'flat' as the CSV/TIF pair.
    """
    for relative in existing:
        path = tmp_path / "out" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    track, result = track_cli(*args, *(["--overwrite"] if overwrite else []))
    assert result.exit_code == 0, result.output
    assert _tracked(track) == (["01", "02"] if overwrite else ["02"])


def test_track_main_redoes_a_half_written_flat_export(track_cli, tmp_path):
    """One file of the flat pair is an interrupted export, so it is not skipped."""
    (tmp_path / "out").mkdir(parents=True)
    (tmp_path / "out" / "01_tracks.csv").touch()

    track, result = track_cli("--export-format", "flat")
    assert result.exit_code == 0, result.output
    assert _tracked(track) == ["01", "02"]


def test_track_main_dry_run_reports_the_inputs_without_tracking(track_cli, ctc_data_dir):
    """`--dry-run` reports what it resolved, flags a mismatch, and writes nothing."""
    track, result = track_cli(
        "--data-dir",
        str(ctc_data_dir),
        "--dry-run",
        default_flags=False,
        loaded=(np.zeros((3, 4, 5), np.uint16), np.zeros((2, 4, 5), np.uint8)),
    )

    assert result.exit_code == 0, result.output
    assert not track.called
    assert "images (3, 4, 5) uint16, masks (2, 4, 5) uint8" in result.output
    assert "frame counts differ" in result.output
    assert not (ctc_data_dir / "baclct_results").exists()


def test_track_main_fails_on_a_failed_sequence(track_cli):
    """One sequence raising must not read as a successful run."""
    _, result = track_cli(track_raises=True)
    assert result.exit_code != 0
    assert "Failed on 1 of 2: ['01']" in str(result.exception)


def test_track_main_forwards_flags_to_track(track_cli):
    """Flags reach `track()` as arguments, and unset ones stay at the trained values."""
    track, result = track_cli(
        "--num-workers",
        "4",
        "--batch-size",
        "8",
        "--thr-div",
        "0.4",
        "--no-segmentation-correction",
        "--device",
        "cpu",
    )
    assert result.exit_code == 0, result.output
    kwargs = track.call_args.kwargs
    assert kwargs["num_workers"] == 4
    assert kwargs["batch_size"] == 8
    assert kwargs["thr_div"] == 0.4
    assert kwargs["segmentation_correction"] is False
    assert kwargs["device"] == "cpu"

    track, _ = track_cli()
    kwargs = track.call_args.kwargs
    assert (kwargs["thr_corr"], kwargs["thr_div"]) == (None, None)
    assert kwargs["segmentation_correction"] is None


@pytest.mark.parametrize(
    ("args", "kwarg", "expected"),
    [
        ((), "prune_edges_by", None),
        (("--prune-edges-by", "off"), "prune_edges_by", "off"),
        (
            ("--prune-edges-by", "ellipse", "--prune-param", "7"),
            "prune_edges_by",
            ("ellipse", 7.0),
        ),
        (
            ("--prune-edges-by", "dilated_overlap"),
            "prune_edges_by",
            ("dilated_overlap", "thickness"),
        ),
        (("--graph-search-radius", "trained"), "graph_search_radius", None),
        (("--graph-search-radius", "120"), "graph_search_radius", 120),
        (("--graph-search-radius", "2.5x"), "graph_search_radius", "2.5x"),
        ((), "cache_dir", "temp"),
        (("--cache-dir", "memory"), "cache_dir", None),
        (("--cache-dir", "/data/cache"), "cache_dir", Path("/data/cache")),
    ],
    ids=[
        "prune_default",
        "prune_off",
        "prune_param",
        "prune_default_param",
        "radius_trained",
        "radius_pixels",
        "radius_multiple",
        "cache_temp",
        "cache_memory",
        "cache_path",
    ],
)
def test_track_main_normalizes_flag_values(track_cli, args, kwarg, expected):
    """String flags are converted to the values `track()` expects before it sees them."""
    track, result = track_cli(*args)
    assert result.exit_code == 0, result.output
    assert track.call_args.kwargs[kwarg] == expected
