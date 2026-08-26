"""Test the argument handling of the `baclct-train` and `baclct-track` CLIs.

The success paths are covered end-to-end by `test_e2e_configs.py` and `test_api_track.py`.
These tests only cover the argument validation, sequence selection, and error propagation
that the real runs do not exercise.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml
from omegaconf import OmegaConf

from baclct.track import main as track_main
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
def track_cli(tmp_path, monkeypatch):
    """Run `baclct-track` against a mocked pipeline and image loader.

    The returned callable takes extra CLI flags and yields the `BacLCT` and
    `BacLCT.track` mocks, so a test can read the sequences and arguments the CLI decided
    on. `track_raises` makes the first sequence fail.
    """
    data_dir = tmp_path / "data"
    for seq in ("01", "02"):
        (data_dir / seq).mkdir(parents=True)
    # resolve_model_dir only returns a path unchanged when it exists
    (tmp_path / "run").mkdir()

    def _run(*args: str, track_raises: bool = False):
        argv = [
            "baclct-track",
            "--model",
            str(tmp_path / "run"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            *args,
        ]
        monkeypatch.setattr("sys.argv", argv)
        with (
            patch("baclct.track.BacLCT") as mock_baclct,
            patch("baclct.track.load_images_and_masks", return_value=(None, None)),
        ):
            track = mock_baclct.return_value.track
            if track_raises:
                track.side_effect = [RuntimeError("boom"), None]
            track_main()
            return mock_baclct, track

    return _run


def _tracked(track) -> list[str]:
    """Sequence ids the CLI passed to `BacLCT.track`, in call order."""
    return [call.kwargs["sequence_id"] for call in track.call_args_list]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--sequences", "02", "01"), ["02", "01"]),
        ((), ["01", "02"]),
    ],
    ids=["explicit", "discovered"],
)
def test_track_main_selects_sequences(track_cli, args, expected):
    """Sequences come from `--sequences` in the given order, else from the data dir."""
    _, track = track_cli(*args)
    assert _tracked(track) == expected


def test_track_main_selects_sequences_from_a_split(track_cli, tmp_path):
    """`--split-file` reads the ids of the requested fold and phase, and needs a fold."""
    split_file = tmp_path / "splits.yaml"
    split_file.write_text(yaml.dump({0: {"train": ["01"], "val": ["02"]}}))

    _, track = track_cli("--split-file", str(split_file), "--fold", "0", "--phase", "val")
    assert _tracked(track) == ["02"]

    with pytest.raises(ValueError, match="--fold"):
        track_cli("--split-file", str(split_file))


def test_track_main_discovers_ctc_layouts_only(track_cli):
    """Other layouts cannot be walked for sequences, so they need an explicit id."""
    with pytest.raises(ValueError, match="auto-discovery"):
        track_cli("--data-format", "flat")


@pytest.mark.parametrize("overwrite", [False, True])
def test_track_main_skips_existing_results(track_cli, tmp_path, overwrite):
    """A non-empty result directory is skipped unless `--overwrite` is given."""
    done = tmp_path / "out" / "GT" / "01"
    done.mkdir(parents=True)
    (done / "res_track.txt").touch()

    _, track = track_cli(*(["--overwrite"] if overwrite else []))
    assert _tracked(track) == (["01", "02"] if overwrite else ["02"])


def test_track_main_fails_on_a_failed_sequence(track_cli):
    """One sequence raising must not read as a successful run."""
    with pytest.raises(SystemExit, match=r"Failed on 1 of 2: \['01'\]"):
        track_cli(track_raises=True)


def test_track_main_translates_flags_to_config_overrides(track_cli):
    """Throughput flags become flat runtime keys, 'trained' keeps the trained radius.

    Placement is a `track()` argument rather than a config key, so `--device` has to
    reach the call and not the overrides.
    """
    mock_baclct, track = track_cli(
        "--graph-search-radius",
        "trained",
        "--num-jobs-features",
        "4",
        "--thr-div",
        "0.4",
        "--no-segmentation-correction",
        "--device",
        "cpu",
    )

    overrides = mock_baclct.call_args.kwargs["config_overrides"]
    assert overrides["num_jobs_features"] == 4
    assert "device" not in overrides
    assert overrides["tracker"] == {"thr_div": 0.4, "segmentation_correction": None}
    assert track.call_args.kwargs["graph_search_radius"] is None
    assert track.call_args.kwargs["device"] == "cpu"


def test_track_main_leaves_tracker_untouched_without_flags(track_cli):
    """Thresholds and correction stay at the model's own values unless asked."""
    mock_baclct, _ = track_cli()
    assert "tracker" not in mock_baclct.call_args.kwargs["config_overrides"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), None),
        (("--prune-edges-by", "off"), "off"),
        (("--prune-edges-by", "ellipse", "--prune-param", "7"), ("ellipse", 7.0)),
        (("--prune-edges-by", "dilated_overlap"), ("dilated_overlap", "thickness")),
    ],
    ids=["default", "off", "ellipse", "dilated_overlap_default_param"],
)
def test_track_main_parses_pruning(track_cli, args, expected):
    """Pruning flags become the method name or the (method, parameter) pair."""
    _, track = track_cli(*args)
    assert track.call_args.kwargs["prune_edges_by"] == expected


def test_track_main_forwards_export_and_state_flags(track_cli):
    """`--export-format` and `--classify-states` reach `track()` unchanged."""
    _, track = track_cli("--export-format", "flat", "--classify-states")
    assert track.call_args.kwargs["export_format"] == "flat"
    assert track.call_args.kwargs["classify_states"] is True


def test_track_main_parses_a_pixel_radius_as_an_int(track_cli):
    """A plain pixel count must not arrive as a string, unlike a '2.5x' multiple."""
    _, track = track_cli("--graph-search-radius", "120")
    assert track.call_args.kwargs["graph_search_radius"] == 120

    _, track = track_cli("--graph-search-radius", "2.5x")
    assert track.call_args.kwargs["graph_search_radius"] == "2.5x"
