"""Test data loading, saving, and caching logic."""

from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
import numpy as np
import polars as pl
import pytest
import tifffile
import yaml

from baclct.io import (
    cached_percentiles,
    coordinate_columns,
    create_trajectory_masks,
    export_combined_tracks,
    export_tracking_results_ctc,
    find_lineage_file,
    get_percentiles,
    get_sequences_from_split,
    load_images_and_masks,
    load_lineage,
    node_preds_to_df,
)


def test_get_sequences_from_split(tmp_path: Path):
    """Sequences come from the requested fold and phase, unknown ones raise."""
    split_content = {
        0: {"train": ["seq1", "seq2"], "val": ["seq3"]},
        1: {"train": ["seq4"], "val": ["seq5", "seq6"]},
    }
    split_file = tmp_path / "splits.yaml"
    with open(split_file, "w") as f:
        yaml.dump(split_content, f)

    assert get_sequences_from_split(split_file, fold=0, phase="train") == [
        "seq1",
        "seq2",
    ]
    assert get_sequences_from_split(split_file, fold=1, phase="val") == [
        "seq5",
        "seq6",
    ]
    assert sorted(get_sequences_from_split(split_file, fold=0)) == [
        "seq1",
        "seq2",
        "seq3",
    ]

    with pytest.raises(ValueError):
        get_sequences_from_split(split_file, fold=99)

    with pytest.raises(ValueError):
        get_sequences_from_split(split_file, fold=1, phase="test")


def test_ctc_export_import_roundtrip(
    tmp_path: Path, toy_masks: np.ndarray, toy_tracks_df: pl.DataFrame
):
    """A CTC export carries the trajectory labels and the lineage of the tracks."""
    res_dir = tmp_path / "results"
    # toy_tracks_df labels each object by its mask label, which would make the
    # relabeling the identity. offset it, and drop one label so untracked masks exist.
    untracked = toy_tracks_df["label"].to_numpy().max()
    tracks = toy_tracks_df.filter(pl.col("label") != untracked).with_columns(
        label_track=pl.col("label") + 1000
    )

    tracked_masks = create_trajectory_masks(
        tracks=tracks,
        masks=toy_masks,
        label_old="label",
        label_new="label_track",
    )
    exported_lbep, _ = export_tracking_results_ctc(
        tracks=tracks,
        masks_tracked=tracked_masks,
        res_dir=res_dir,
        fill_gaps=False,
    )

    loaded_mask_paths = sorted(res_dir.glob("*.tif"))
    loaded_masks = np.stack([tifffile.imread(p) for p in loaded_mask_paths])
    loaded_lbep = np.loadtxt(res_dir / "res_track.txt", dtype=np.int64)

    assert len(loaded_mask_paths) == len(toy_masks)
    np.testing.assert_array_equal(tracked_masks, loaded_masks)
    np.testing.assert_array_equal(exported_lbep, loaded_lbep)

    assert set(np.unique(loaded_masks)) - {0} == set(tracks["label_track"].to_list())

    expected_lbep = (
        tracks.group_by("label_track")
        .agg(
            pl.col("t").min(), pl.col("t").max().alias("t_max"), pl.first("parent_track")
        )
        .sort("t", "label_track")
        .to_numpy()
    )
    np.testing.assert_array_equal(loaded_lbep, expected_lbep)


def _write_layout(
    root: Path, data_format: str, images: np.ndarray, masks: np.ndarray
) -> Path:
    """Write `images` and `masks` for sequence '01' in the given on-disk layout."""
    data_dir = root / data_format
    if data_format == "flat":
        data_dir.mkdir(parents=True)
        tifffile.imwrite(data_dir / "01_images.tif", images, photometric="minisblack")
        tifffile.imwrite(data_dir / "01_masks.tif", masks, photometric="minisblack")
        return data_dir

    if data_format == "ctc":
        img_dir, seg_dir = data_dir / "01", data_dir / "01_GT" / "TRA"
    else:
        img_dir, seg_dir = data_dir / "01" / "BF", data_dir / "01" / "Segmentation"

    img_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    for t, (img, msk) in enumerate(zip(images, masks, strict=True)):
        tifffile.imwrite(img_dir / f"t{t:03d}.tif", img)
        tifffile.imwrite(seg_dir / f"t{t:03d}.tif", msk)
    return data_dir


@pytest.mark.parametrize("data_format", ["ctc", "flat", "dirs"])
def test_load_images_and_masks_layouts(
    tmp_path: Path, toy_images: np.ndarray, toy_masks: np.ndarray, data_format: str
):
    """Every layout yields `(T, H, W)`, and a lazy read matches an eager one."""
    data_dir = _write_layout(tmp_path, data_format, toy_images, toy_masks)

    lazy_imgs, lazy_masks = load_images_and_masks(
        data_dir, "01", data_format=data_format, lazy=True
    )
    eager_imgs, eager_masks = load_images_and_masks(
        data_dir, "01", data_format=data_format, lazy=False
    )

    assert isinstance(lazy_imgs, da.Array)
    np.testing.assert_array_equal(np.asarray(lazy_imgs), toy_images)
    np.testing.assert_array_equal(np.asarray(lazy_masks), toy_masks)
    np.testing.assert_array_equal(np.asarray(eager_imgs), toy_images)
    np.testing.assert_array_equal(np.asarray(eager_masks), toy_masks)


def test_load_images_and_masks_strict(
    tmp_path: Path, toy_images: np.ndarray, toy_masks: np.ndarray
):
    """Missing images are fatal only under `strict`, missing masks always are."""
    data_dir = _write_layout(tmp_path, "ctc", toy_images, toy_masks)
    for fp in (data_dir / "01").iterdir():
        fp.unlink()

    with pytest.raises(ValueError, match="Could not find images"):
        load_images_and_masks(data_dir, "01", segmentation_name="GT")

    images, masks = load_images_and_masks(
        data_dir, "01", segmentation_name="GT", strict=False
    )
    assert images is None
    assert len(masks) == len(toy_masks)

    for fp in (data_dir / "01_GT" / "TRA").iterdir():
        fp.unlink()
    with pytest.raises(ValueError, match="Could not find masks"):
        load_images_and_masks(data_dir, "01", segmentation_name="GT", strict=False)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16"])
@pytest.mark.parametrize("lazy", [False, True])
def test_percentiles_match_numpy(dtype: str, lazy: bool):
    """`get_percentiles` returns exactly what `np.percentile` returns."""
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    stack = rng.integers(max(info.min, -500), 500, size=(5, 16, 16)).astype(dtype)
    images = da.from_array(stack, chunks=(1, 16, 16)) if lazy else stack

    for quantiles in [(0.05, 99.95), (25.0, 75.0), (0.0, 100.0)]:
        expected = tuple(float(v) for v in np.percentile(stack, list(quantiles)))
        assert get_percentiles(images, quantiles) == expected

    # a small n puts the interpolation between two order statistics in play, where a
    # one-sided lerp would differ
    tiny = np.array([2, 5, 10, 10, 14, 34, 48, 52], dtype=dtype)
    for q in np.linspace(0, 100, 41):
        quantiles = (float(q), float(q))
        expected = tuple(float(v) for v in np.percentile(tiny, list(quantiles)))
        assert get_percentiles(tiny, quantiles) == expected


def test_percentiles_do_not_materialize(monkeypatch: pytest.MonkeyPatch):
    """Integer sequences are reduced without falling back to a full materialization."""
    from baclct.io import load as load_module

    stack = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    images = da.from_array(stack, chunks=(1, 8, 8))
    expected = tuple(float(v) for v in np.percentile(stack, [0.05, 99.95]))

    def _fail(*args, **kwargs):
        raise AssertionError("materialized the sequence")

    monkeypatch.setattr(load_module.np, "percentile", _fail)
    # only the histogram counts are computed, never the sequence itself
    monkeypatch.setattr(da.Array, "__array__", _fail)
    assert get_percentiles(images) == expected


def test_percentiles_float_falls_back():
    """Float input still matches `np.percentile`, lazy or not."""
    rng = np.random.default_rng(1)
    stack = rng.random((4, 8, 8)).astype(np.float32)
    expected = tuple(float(v) for v in np.percentile(stack, [0.05, 99.95]))

    assert get_percentiles(stack) == expected
    assert get_percentiles(da.from_array(stack, chunks=(1, 8, 8))) == expected


def test_cached_percentiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The cache file is written once and then wins over recomputation."""
    from baclct.io import load as load_module

    cache_file = tmp_path / "percentiles.json"
    stack = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)

    percentiles = cached_percentiles(stack, cache_file)
    assert percentiles == get_percentiles(stack)
    assert all(isinstance(p, float) for p in percentiles)
    assert json.loads(cache_file.read_text()) == list(percentiles)

    def _fail(*args, **kwargs):
        raise AssertionError("recomputed despite a cache file")

    monkeypatch.setattr(load_module, "get_percentiles", _fail)
    assert cached_percentiles(stack, cache_file) == percentiles
    assert cached_percentiles(None, tmp_path / "absent.json") is None


def test_find_lineage_file(tmp_path: Path):
    """The CTC search walks TRA, then the segmentation dir, then the dataset root."""
    tra_dir = tmp_path / "01_GT" / "TRA"
    tra_dir.mkdir(parents=True)

    assert find_lineage_file(tmp_path, "01", segmentation_name="GT") == (None, False)

    (tmp_path / "man_track.txt").touch()
    assert find_lineage_file(tmp_path, "01", segmentation_name="GT") == (
        tmp_path / "man_track.txt",
        False,
    )

    (tra_dir / "man_track.txt").touch()
    assert find_lineage_file(tmp_path, "01", segmentation_name="GT") == (
        tra_dir / "man_track.txt",
        False,
    )

    # states are preferred over a plain CTC lineage when asked for
    (tra_dir / "states.txt").touch()
    assert find_lineage_file(tmp_path, "01", segmentation_name="GT") == (
        tra_dir / "states.txt",
        True,
    )
    assert find_lineage_file(
        tmp_path, "01", segmentation_name="GT", with_states=False
    ) == (tra_dir / "man_track.txt", False)


def test_load_lineage_with_states(toy_data_dir: Path, toy_masks: np.ndarray):
    """A states file carries one annotated row per mask label per frame."""
    states_file = toy_data_dir / "states.txt"

    lineage = load_lineage(states_file, with_states=True, seq_id="9")
    assert isinstance(lineage, pl.DataFrame)
    assert lineage.columns == ["label", "t", "state", "parent", "sequence_id"]
    for t, frame in enumerate(toy_masks):
        labels = np.unique(frame)
        annotated = lineage.filter(t=t)["label"].sort().to_list()
        assert annotated == labels[labels != 0].tolist()

    # a sequence id of `None` reads the file whole and drops the column
    whole_file = load_lineage(states_file, with_states=True, seq_id=None)
    assert isinstance(whole_file, pl.DataFrame)
    assert "sequence_id" not in whole_file.columns
    assert whole_file.height == lineage.height

    with pytest.raises(AssertionError):
        load_lineage(states_file, with_states=True, seq_id="1")


@pytest.mark.parametrize("with_preds", [False, True])
def test_export_combined_tracks_schema(
    tmp_path: Path, toy_tracks_df: pl.DataFrame, with_preds: bool
):
    """`res_tracks.csv` carries lineage and state only, never single-cell features."""
    node_preds_df = None
    if with_preds:
        rng = np.random.default_rng(0)
        node_preds_df = node_preds_to_df(
            rng.random((toy_tracks_df.height, 3)).astype(np.float32),
            toy_tracks_df["index"].to_numpy(),
        )

    export_combined_tracks(toy_tracks_df, tmp_path, node_preds_df=node_preds_df)
    out = pl.read_csv(tmp_path / "res_tracks.csv")

    expected = ["label", "t", "cy", "cx", "parent"]
    assert out.columns == (expected + ["state"] if with_preds else expected)
    assert out.height == toy_tracks_df.height
    assert out.sort("label", "t").to_dicts() == out.to_dicts()

    if with_preds:
        probabilities = node_preds_df.select(r"^p\d+$").to_numpy()
        expected_state = pl.DataFrame(
            {
                "label": toy_tracks_df["label_track"],
                "t": toy_tracks_df["t"],
                "state": probabilities.argmax(axis=1),
            }
        ).sort("label", "t")
        assert out.select("label", "t", "state").to_dicts() == expected_state.to_dicts()


def test_coordinate_columns_uses_pixel_space_under_spacing(toy_tracks_df: pl.DataFrame):
    """Export reads pixel indices, so `_px` wins over the physical-unit columns."""
    assert coordinate_columns(toy_tracks_df) == ["center-0", "center-1"]

    # a non-unit spacing leaves `center-*` in physical units and adds a `_px` family
    scaled = toy_tracks_df.with_columns(
        (pl.col("center-0") * 0.5).alias("center-0"),
        (pl.col("center-1") * 0.5).alias("center-1"),
        pl.col("center-0").alias("center-0_px"),
        pl.col("center-1").alias("center-1_px"),
    )
    assert coordinate_columns(scaled) == ["center-0_px", "center-1_px"]
