"""Test node features and graph construction."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn
from conftest import _get_mock_edges
from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation, disk
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader

from baclct.features.extractors import (
    CellLevelExtractor,
    HandcraftedExtractor,
)
from baclct.features.graph import EdgeFinder, build_lineage_gt_edges
from baclct.features.normalization import scale_relative_size
from baclct.features.patches import PatchWeighting
from baclct.io import dataset_identity
from baclct.models.encoder import compute_patch_weights, normalize_imagenet
from baclct.utils.data import collect, get_multiprocessing_context
from baclct.utils.spacing import needs_spacing


def check_dilated_overlap_from_masks(
    radius: str | int | list[int],
    node_feats: pl.DataFrame,
    edge_data: pl.DataFrame,
    masks: np.ndarray,
):
    """Dilate every mask and keep the edges whose dilations intersect.

    The paper's original formulation used actual dilation. This is extremely slow, since
    it requires images and used overlap checks. This was refactored to use a distance
    check which is much faster (i.e. can use kdtree, sampling of boundary pixels, early
    stopping, etc.).
    """
    h, w = masks[0].shape

    # super inefficient but only used for testing with small dataset
    dilated = np.zeros((node_feats.height, h, w), dtype=np.bool)
    for (t,), grouped in node_feats.group_by("t"):
        radii = (
            np.repeat(radius, grouped.height)
            if isinstance(radius, int)
            else grouped[radius].cast(pl.Int64)
            if isinstance(radius, str)
            else np.asarray(radius, dtype=int)
        )
        labels = grouped["label"]
        index = grouped["index"]
        for r, label, idx in zip(radii, labels, index, strict=True):
            dilated[idx] = dilation(masks[t] == label, disk(r, decomposition="sequence"))

    edge_index = edge_data.select("src", "dst")
    overlap = np.zeros(edge_index.height, dtype=bool)
    for i, (src, dst) in enumerate(edge_index.to_numpy()):
        overlap[i] = np.logical_and(dilated[src], dilated[dst]).sum() > 0

    return edge_index.with_columns(pl.Series("keep", overlap))


def _single_cell_frame() -> tuple[np.ndarray, np.ndarray]:
    """A flat image and a single 2x2 object at (2, 2), both 10x10."""
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[2:4, 2:4] = 1
    return np.ones((10, 10)), mask


def test_handcrafted_extractor_computes_props():
    """Regionprops output carries the mandatory props at their exact values."""
    image, mask = _single_cell_frame()
    extractor = HandcraftedExtractor(n_jobs=1)
    features_df = extractor._compute(images=[image], masks=[mask])  # type: ignore

    assert "area" in features_df.columns
    assert "centroid-0" in features_df.columns
    assert "centroid-1" in features_df.columns
    assert features_df["area"][0] == 4
    assert features_df["centroid-0"][0] == 2.5
    assert features_df["centroid-1"][0] == 2.5
    assert features_df.height == 1


def test_handcrafted_extractor_spacing_scales_sizes():
    """Physical voxel spacing scales regionprops sizes and centroids."""
    image, mask = _single_cell_frame()
    extractor = HandcraftedExtractor(n_jobs=1)
    unit = extractor._compute(images=[image], masks=[mask])  # type: ignore
    scaled = extractor._compute(images=[image], masks=[mask], spacing=(2.0, 2.0))  # type: ignore

    # area is a volume, so it scales with the product of the spacing (2 * 2)
    assert scaled["area"][0] == unit["area"][0] * 4
    # centroids and axis lengths are linear, so they scale with the spacing
    assert scaled["centroid-0"][0] == unit["centroid-0"][0] * 2
    assert scaled["centroid-1"][0] == unit["centroid-1"][0] * 2
    assert scaled["axis_major_length"][0] == pytest.approx(
        unit["axis_major_length"][0] * 2
    )
    # bbox stays in voxel-index units for cropping and overlap
    assert scaled["bbox-0"][0] == unit["bbox-0"][0]


def test_handcrafted_extractor_spacing_cache(tmp_path):
    """The spacing recorded in `meta.json` decides whether a cache is reused."""
    assert not needs_spacing(None) and not needs_spacing((1.0, 1.0))
    assert needs_spacing((4.95, 1.0, 1.0))

    image, mask = _single_cell_frame()
    fp = tmp_path / "nodes.parquet"
    meta = tmp_path / "meta.json"
    extractor = HandcraftedExtractor(n_jobs=1)

    # default spacing writes the cache without a spacing entry,
    # so existing caches stay valid
    extractor([image], [mask], filepath=fp)  # type: ignore
    assert fp.exists() and not meta.exists()

    # a non-unit spacing rebuilds and records the spacing
    feats = extractor([image], [mask], filepath=fp, spacing=(2.0, 2.0))  # type: ignore
    assert meta.exists() and json.loads(meta.read_text())["spacing"] == [2.0, 2.0]
    assert feats["area"][0] == 16

    # mismatching spacing without overwrite refuses to return the stale cache
    with pytest.raises(ValueError, match="different"):
        extractor([image], [mask], filepath=fp, spacing=(3.0, 4.0), overwrite=False)  # type: ignore

    # spacing=None loads the cache as built (its recorded spacing), leaving meta intact
    feats = extractor([image], [mask], filepath=fp, spacing=None)  # type: ignore
    assert meta.exists() and feats["area"][0] == 16

    # requesting unit spacing again rebuilds and clears the spacing entry
    extractor([image], [mask], filepath=fp, spacing=(1.0, 1.0))  # type: ignore
    assert not meta.exists()


def test_handcrafted_extractor_dataset_identity_cache(tmp_path):
    """A mismatched `dataset_meta` forces a rebuild rather than reusing the cache."""
    image, mask = _single_cell_frame()
    images, masks = np.stack([image]), np.stack([mask])
    fp = tmp_path / "nodes.parquet"
    extractor = HandcraftedExtractor(n_jobs=1)
    identity = dataset_identity(images, masks)

    extractor(images, masks, filepath=fp, dataset_meta=identity)
    meta = tmp_path / "meta.json"
    assert json.loads(meta.read_text())["dataset"] == identity

    # a different mask changes the identity, so the cache is rejected and rebuilt
    other_masks = masks.copy()
    other_masks[0, 5:7, 5:7] = 2
    other_identity = dataset_identity(images, other_masks)
    feats = extractor(images, other_masks, filepath=fp, dataset_meta=other_identity)
    assert json.loads(meta.read_text())["dataset"] == other_identity
    assert feats["label"].to_list() == [1, 2]

    # matching identity without overwrite reuses the cache
    reused = extractor(
        images, other_masks, filepath=fp, overwrite=False, dataset_meta=other_identity
    )
    assert reused["label"].to_list() == [1, 2]

    # mismatching identity always rebuilds and persists, even with overwrite=False
    feats = extractor(images, masks, filepath=fp, overwrite=False, dataset_meta=identity)
    assert feats["label"].to_list() == [1]
    assert json.loads(meta.read_text())["dataset"] == identity


def _synthetic_3d_sequence():
    """Two cubes over two frames, each shifted by one voxel in y between frames."""
    masks = np.zeros((2, 6, 24, 24), dtype=np.uint16)
    for t in range(2):
        masks[t, 1:4, 2 + t : 7 + t, 2:7] = 1
        masks[t, 1:4, 14 + t : 19 + t, 14:19] = 2
    images = np.ones_like(masks, dtype=np.uint8)
    return images, masks


def _synthetic_3d_node_feats() -> tuple[np.ndarray, pl.DataFrame]:
    """Masks of `_synthetic_3d_sequence` and node features ready for edge finding."""
    images, masks = _synthetic_3d_sequence()
    node_feats = scale_relative_size(
        HandcraftedExtractor(n_jobs=1, verbose=False)._compute(images, masks)
    ).with_columns(parent=pl.lit(0))
    return masks, node_feats


def test_handcrafted_extractor_3d():
    """Handcrafted extraction runs on volumetric masks with the default prop set."""
    images, masks = _synthetic_3d_sequence()
    extractor = HandcraftedExtractor(n_jobs=1, verbose=False)
    feats = extractor._compute(images, masks)

    # 3D centroids and bboxes are present
    # regression/reminder for later: orientation is not a default (would be 2D-only)
    for col in ("centroid-0", "centroid-1", "centroid-2", "bbox-0", "bbox-5"):
        assert col in feats.columns
    assert "orientation" not in feats.columns
    assert feats.height == 4  # two cells over two frames


def test_handcrafted_extractor_2d_only_prop_raises_on_3d():
    """A 2D-only prop requested on volumetric masks fails fast (no silent drop)."""
    images, masks = _synthetic_3d_sequence()
    extractor = HandcraftedExtractor(props=["orientation"], n_jobs=1, verbose=False)
    with pytest.raises(Exception):  # noqa: B017 - skimage raises NotImplementedError/ValueError
        extractor._compute(images, masks)


def test_handcrafted_extractor_px_columns():
    """Spacing adds pixel-space `_px` copies of centroids; unit spacing adds none."""
    images, masks = _synthetic_3d_sequence()
    extractor = HandcraftedExtractor(n_jobs=1, verbose=False)

    spacing = (4.95, 1.0, 1.0)
    feats = extractor._compute(images, masks, spacing=spacing)
    assert {"centroid-0_px", "centroid-1_px", "centroid-2_px"} <= set(feats.columns)
    # pixel coordinate is the physical coordinate divided by the axis spacing
    for px, phys in zip(feats["centroid-0_px"], feats["centroid-0"], strict=True):
        assert px == pytest.approx(phys / 4.95)
    # in-plane axes are unit-spaced, so their _px equals the physical value
    for px, phys in zip(feats["centroid-1_px"], feats["centroid-1"], strict=True):
        assert px == pytest.approx(phys)

    # unit spacing leaves no _px columns (existing 2D caches stay byte-identical)
    plain = extractor._compute(images, masks)
    assert not any(c.endswith("_px") for c in plain.columns)


def test_edge_finder_3d_overlap_and_distance():
    """Edge finding on 3D data uses nD distance and the nD overlap ROI."""
    masks, node_feats = _synthetic_3d_node_feats()
    edge_finder = EdgeFinder(
        prune_edges_by=("radius", 50),
        feature_names=("dist_spat", "dist_temp", "overlap"),
        bidirectional=False,
        center_name="centroid",
        n_jobs=1,
    )
    edges = collect(
        edge_finder(node_feats, search_radius=50, time_steps=[1], masks=masks)
    )

    # each cube overlaps its shifted self across frames
    matches = edges.filter(pl.col("y") == 1)
    assert matches.height == 2
    assert matches["overlap"].to_numpy().min() > 0.5
    # the only displacement is one voxel in y, so the 3D distance is exactly 1
    assert matches["dist_spat"].to_numpy() == pytest.approx(1.0)
    # the other pairs are 14 voxels apart in y and x, well beyond a 2D-only distance
    assert edges.filter(pl.col("y") == 0)["dist_spat"].to_numpy().min() > 10


def test_dilated_overlap_pruning_rejects_3d():
    """dilated_overlap pruning is 2D-only and fails clearly on volumetric masks."""
    masks, node_feats = _synthetic_3d_node_feats()
    edge_finder = EdgeFinder(
        prune_edges_by=("dilated_overlap", 5),
        feature_names=("dist_spat", "dist_temp"),
        bidirectional=False,
        center_name="centroid",
        n_jobs=1,
    )
    with pytest.raises(NotImplementedError, match="2D only"):
        collect(edge_finder(node_feats, search_radius=50, time_steps=[1], masks=masks))


def test_graph_constructor_finds_neighbors():
    """A KDTree query links only the pair within the radius, in the queried direction."""
    node_features = pl.DataFrame(
        {
            "index": [0, 1, 2],
            "t": [0, 0, 1],
            "label": [10, 20, 10],
            "centroid-0": [10, 50, 51],
            "centroid-1": [10, 50, 51],
            "parent": [0, 0, 0],
        }
    )
    edge_finder = EdgeFinder(prune_edges_by=None, bidirectional=False, n_jobs=1)

    # node 2 sits sqrt(2) from node 1 and ~58 from node 0
    edges = edge_finder.find_edge_pairs(node_features, t_src=0, t_dst=1, radius=5)

    assert edges is not None
    assert edges.height == 1
    assert edges["src"][0] == 1
    assert edges["dst"][0] == 2


@pytest.mark.parametrize(
    "treat_divs_as, div_label", [("division", 2), ("correspondence", 1), ("negative", 0)]
)
def test_graph_constructor_labels_division(treat_divs_as, div_label):
    """Divisions are labeled 2 whatever `treat_divs_as` is, and remapped afterwards."""
    # parent (label 10) at t=0 divides into two children at t=1
    edge_data = pl.DataFrame(
        {
            "src": [0, 0],
            "dst": [1, 2],
            "label_src": [10, 10],
            "label_dst": [20, 30],
            "parent_src": [0, 0],
            "parent_dst": [10, 10],
        }
    )
    edge_finder = EdgeFinder(treat_divs_as=treat_divs_as, n_jobs=1)

    labeled_edges = edge_finder._label_edges(edge_data)
    assert labeled_edges["y"].to_list() == [2, 2]  # type: ignore

    remapped = edge_finder._remap_div_label(labeled_edges)
    assert remapped["y"].to_list() == [div_label, div_label]  # type: ignore


def test_edge_finder_prune_by_radius():
    """Edges beyond the source's scaled major axis are dropped."""
    # edge 1: dist_spat (2.82) is within radius (10/2 * 1.0 = 5)
    # edge 2: dist_spat (20) is outside radius (5)
    edge_data = pl.DataFrame(
        {
            "dist_spat": [2.82, 20.0],
            "axis_major_length_src": [10.0, 10.0],
        }
    )

    edge_finder = EdgeFinder(prune_edges_by=("radius", 1.0), n_jobs=1)
    pruned = edge_finder._prune_edges(edge_data)
    assert isinstance(pruned, pl.DataFrame)
    assert pruned.height == 1
    assert pruned["dist_spat"][0] == 2.82

    # test with a larger factor to include the second edge
    edge_finder_2 = EdgeFinder(prune_edges_by=("radius", 4.0), n_jobs=1)
    pruned_2 = edge_finder_2._prune_edges(edge_data)
    assert isinstance(pruned_2, pl.DataFrame)
    assert pruned_2.height == 2


def test_edge_finder_prune_by_ellipse():
    """Only the destination inside the source's ellipse, rotated by 45 degrees, survives.

    Un-rotated, dst 1 lands at (7.07, 0) and solves the ellipse to 0.5, dst 2 at
    (10.6, 10.6) and to above 1.
    """
    edge_data = pl.DataFrame(
        {
            "centroid-0_src": [0.0, 0.0],
            "centroid-1_src": [0.0, 0.0],
            "axis_major_length_src": [20.0, 20.0],
            "axis_minor_length_src": [10.0, 10.0],
            "orientation_src": [np.pi / 4, np.pi / 4],
            "centroid-0_dst": [5.0, 15.0],
            "centroid-1_dst": [5.0, 0.0],
        }
    )

    edge_finder = EdgeFinder(prune_edges_by=("ellipse", 1.0), n_jobs=1)
    pruned = edge_finder._prune_edges(edge_data)
    assert isinstance(pruned, pl.DataFrame)
    assert pruned.height == 1
    assert pruned["centroid-0_dst"][0] == 5.0


def test_edge_finder_prune_by_gt():
    """GT pruning keeps labeled edges that point forward in time, nothing else."""
    edge_data = pl.DataFrame(
        {
            "y": [1, 2, 0, 1, 1],
            "dist_temp": [1, 1, 1, 0, -1],
        }
    )

    edge_finder = EdgeFinder(prune_edges_by="gt", n_jobs=1)
    pruned = edge_finder._prune_edges(edge_data)
    assert isinstance(pruned, pl.DataFrame)
    assert pruned.height == 2
    assert 0 not in pruned["y"].to_list()
    assert all(d > 0 for d in pruned["dist_temp"])


def test_edge_finder_prune_by_dilated_overlap(toy_masks, toy_handcrafted_features):
    """Boundary-distance pruning keeps the edges the paper's dilation keeps."""
    radius = 3
    candidates = collect(
        EdgeFinder(prune_edges_by=None, n_jobs=1)(
            toy_handcrafted_features, 50, [1], masks=toy_masks
        )
    ).select("src", "dst")

    edge_finder = EdgeFinder(prune_edges_by=("dilated_overlap", radius), n_jobs=1)
    kept = edge_finder._prune_edges_dilated_overlap(
        edge_data=candidates.lazy(),
        node_feats=toy_handcrafted_features,
        masks=toy_masks,
        sampling_rate=1,
        radius_multiplier=1.0,
    ).collect()
    reference = check_dilated_overlap_from_masks(
        radius=radius,
        node_feats=toy_handcrafted_features,
        edge_data=candidates,
        masks=toy_masks,
    ).filter(pl.col("keep"))

    assert kept.height < candidates.height / 2
    # boundary distance is a approximation of the dilation,
    # so it may keep fewer edges, never other ones
    assert set(kept.select("src", "dst").iter_rows()) <= set(
        reference.select("src", "dst").iter_rows()
    )
    assert kept.height == pytest.approx(reference.height, rel=0.05)


def test_scale_relative_node_features():
    """`scale_relative_size` divides sizes by their median in the first frame."""
    features = pl.DataFrame(
        {
            "t": [0, 0, 1, 1],
            "axis_major_length": [10.0, 20.0, 15.0, 25.0],
            "area": [50.0, 70.0, 60.0, 80.0],
            "intensity_mean": [100, 100, 100, 100],
        }
    )

    # median at t=0: axis_major_length=15.0, area=60.0
    normalized_features = scale_relative_size(features)

    for col in ("axis_major_length_norm", "area_norm", "len_init", "area_init"):
        assert col in normalized_features.columns
    assert "intensity_mean" in normalized_features.columns  # untouched columns survive

    assert np.isclose(normalized_features["len_init"][0], 15.0)
    assert np.isclose(normalized_features["area_init"][0], 60.0)
    assert np.isclose(
        normalized_features["axis_major_length_norm"][0], np.log(10.0 / 15.0)
    )
    assert np.isclose(
        normalized_features["axis_major_length_norm"][1], np.log(20.0 / 15.0)
    )
    assert np.isclose(normalized_features["area_norm"][0], np.log(50.0 / 60.0))


def _edge_norm_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two edges plus the `len_init` cell-size normalization scales them by."""
    return (
        pl.DataFrame({"dist_spat": [5.0, 15.0], "dist_temp": [1, 2]}),
        pl.DataFrame({"len_init": [10.0]}),
    )


@pytest.mark.parametrize("edge_normalization, scale", [("cell_size", 10.0), (5.0, 5.0)])
def test_edge_finder_normalization(edge_normalization, scale):
    """dist_spat becomes a log ratio of the scaling length, dist_temp a symlog."""
    edge_data, node_feats = _edge_norm_inputs()
    edge_finder = EdgeFinder(
        feature_names=["dist_spat", "dist_temp"],
        edge_normalization=edge_normalization,
        n_jobs=1,
    )

    transformed = edge_finder._transform(edge_data, node_feats)
    assert transformed is not None
    x = transformed.numpy()

    assert not np.any(np.isnan(x))
    assert x[:, 0] == pytest.approx(np.log(np.array([5.0, 15.0]) / scale), abs=1e-6)
    assert x[:, 1] == pytest.approx(np.log([2, 3]))


def test_edge_finder_normalization_min_max():
    """min_max scales every feature column jointly, in `feature_cols` order."""
    edge_data, node_feats = _edge_norm_inputs()
    edge_finder = EdgeFinder(
        feature_names=["dist_spat", "dist_temp"],
        edge_normalization="min_max",
        n_jobs=1,
    )

    transformed = edge_finder._transform(edge_data, node_feats)
    assert transformed is not None
    assert np.allclose(transformed.numpy(), minmax_scale(edge_data.to_numpy()))


def test_edge_finder_temporal_spacing():
    """spacing_t rescales dist_temp before the symlog, leaving dist_spat untouched."""
    edge_data, node_feats = _edge_norm_inputs()
    edge_finder = EdgeFinder(
        feature_names=["dist_spat", "dist_temp"],
        edge_normalization="cell_size",
        n_jobs=1,
    )

    # spacing_t=1 reproduces the default behavior (dist_temp -> log(|t| + 1))
    base = edge_finder._transform(edge_data, node_feats, spacing_t=1.0)
    assert base is not None
    base_np = base.numpy()
    assert np.isclose(base_np[0, 1], np.log(2))
    assert np.isclose(base_np[1, 1], np.log(3))

    # a sparser acquisition (every 3rd frame) scales dist_temp by k=3
    scaled = edge_finder._transform(edge_data, node_feats, spacing_t=3.0)
    assert scaled is not None
    scaled_np = scaled.numpy()
    assert np.isclose(scaled_np[0, 1], np.log(1 * 3 + 1))
    assert np.isclose(scaled_np[1, 1], np.log(2 * 3 + 1))

    # spatial distance is independent of temporal spacing
    assert np.allclose(scaled_np[:, 0], base_np[:, 0])


class _MockEncoder(nn.Module):
    def __init__(self, arch="MockEncoder"):
        super().__init__()
        self.arch = arch


def _make_cell_extractor():
    return CellLevelExtractor(encoder=_MockEncoder())


@pytest.mark.parametrize(
    "padding, pretrained", [(None, True), ("constant", True), ("constant", False)]
)
def test_cell_level_extractor_padding(
    toy_images, toy_handcrafted_features, padding, pretrained
):
    """A crop reaching past the border is either slid back in or padded with zeros."""
    # the extractor takes (T, C, H, W)
    images_for_extractor = [np.stack([frame] * 3) for frame in toy_images]
    # node 1 sits at y=22.6, so its 64 px box runs past the top of the 300x530 image
    node_to_test = toy_handcrafted_features.filter(pl.col("index") == 1)

    extractor = CellLevelExtractor(
        encoder=_MockEncoder(),
        input_size_img=64,
        padding=padding,
        normalize_for_pretrained=pretrained,
    )
    coords = extractor._get_boxes_and_padding(node_to_test, toy_images[0].shape)
    patches, _, indices = extractor.get_single_cell_images(
        images_for_extractor,  # type: ignore
        coords,
    )

    assert patches.shape == (1, 3, 224, 224)
    assert indices == node_to_test["index"].to_list()

    # for a pretrained encoder, zero maps to the imagenet-normalized value
    norm_zero = torch.tensor(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    ).view(3, 1, 1)
    if not pretrained:
        norm_zero *= 0.0

    patch = patches[0]
    if padding is None:
        # the box slid back inside, so even its top-left corner samples image content
        assert not torch.allclose(patch[:, 0:1, 0:1], norm_zero, atol=1e-4)
    else:
        assert torch.allclose(
            patch[:, 0:1, :], norm_zero.expand_as(patch[:, 0:1, :]), atol=1e-4
        )
        # the bottom of the box lies inside the image
        assert not torch.allclose(
            patch[:, -1:, :], norm_zero.expand_as(patch[:, -1:, :]), atol=1e-4
        )


def test_cell_level_extractor_3d_stopgap():
    """The 3D stopgap crops an in-plane patch at each cell's mid-plane z slice."""
    images, masks = _synthetic_3d_sequence()
    # use a random image so foreground/background actually differ per slice
    images = np.random.default_rng(0).integers(0, 255, size=masks.shape, dtype=np.uint8)
    node_feats = HandcraftedExtractor(n_jobs=1, verbose=False)._compute(
        images, masks, spacing=(4.95, 1.0, 1.0)
    )
    extractor = CellLevelExtractor(
        encoder=_MockEncoder(), input_size_img="bbox", input_size_enc=32, verbose=False
    )

    coords = extractor._get_boxes_and_padding(node_feats, image_size=images.shape[-2:])
    # a z mid-plane is chosen per cell, here the cells span z 1..3
    assert coords["z"] is not None
    assert set(np.unique(coords["z"])).issubset({1, 2, 3})

    patches, mask_patches, _ = extractor.get_single_cell_images(images, coords, masks)
    assert patches.shape == (node_feats.height, 3, 32, 32)
    assert mask_patches is not None
    assert mask_patches.shape == (node_feats.height, 1, 32, 32)
    # the chosen z slice actually intersects each cell, so every mask patch has foreground
    assert (mask_patches.sum(dim=(1, 2, 3)) > 0).all()


@pytest.mark.parametrize(
    "strategy, unit_size", [("axis_major_length", 20), ("median_size", 25)]
)
def test_cell_level_feature_crop_size(strategy, unit_size):
    """A feature-based crop size is divided back to pixels, matching the bbox space."""
    node_feats = pl.DataFrame(
        {
            "index": [0, 1],
            "t": [0, 0],
            "label": [1, 2],
            # physical centroid (x2) and its pixel-space copy under spacing (2, 2)
            "centroid-0": [40.0, 40.0],
            "centroid-1": [40.0, 40.0],
            "centroid-0_px": [20.0, 20.0],
            "centroid-1_px": [20.0, 20.0],
            "bbox-0": [10, 10],
            "bbox-1": [10, 10],
            "bbox-2": [30, 30],
            "bbox-3": [30, 30],
            # physical lengths under spacing (2, 2), median 25
            "axis_major_length": [20.0, 30.0],
        }
    )
    extractor = CellLevelExtractor(
        encoder=_MockEncoder(), input_size_img=(strategy, 1.0), verbose=False
    )

    unit = extractor._get_boxes_and_padding(node_feats, image_size=(100, 100))
    spaced = extractor._get_boxes_and_padding(
        node_feats, image_size=(100, 100), spacing=(2.0, 2.0)
    )

    # without spacing the feature counts as pixels, with (2, 2) it is halved
    assert int(unit["s_full"][0]) == unit_size
    assert int(spaced["s_full"][0]) == unit_size // 2


def test_cell_level_cache_roundtrip_scatter(tmp_path):
    """`_save` scatters embeddings by global index; `_load` gathers requested rows."""
    extractor = _make_cell_extractor()
    cache_dir = tmp_path / extractor.name

    n, dim = 6, 8
    rng = np.random.default_rng(0)
    feats = torch.from_numpy(rng.standard_normal((n, dim)).astype(np.float32))
    # non-identity permutation: compute order differs from global node index order
    indices = [3, 0, 5, 1, 4, 2]
    extractor._save(feats, cache_dir, indices, dataset_meta=None)

    assert (cache_dir / "embeddings.npy").exists()
    assert (cache_dir / "meta.json").exists()
    assert not (cache_dir / "indices.npy").exists()
    assert np.load(cache_dir / "embeddings.npy", mmap_mode="r").shape == (n, dim)

    requested = [4, 1, 5]
    loaded = extractor._load(cache_dir, requested, dataset_meta=None)

    # each returned row must equal the original embedding for that global index
    idx_to_feat = {idx: feats[row] for row, idx in enumerate(indices)}
    for k, idx in enumerate(requested):
        assert torch.allclose(loaded[k], idx_to_feat[idx], atol=1e-2)


def test_cell_level_cache_out_of_range(tmp_path):
    """Requesting an index beyond the stored node count names the incomplete cache."""
    extractor = _make_cell_extractor()
    cache_dir = tmp_path / extractor.name
    feats = torch.zeros((3, 4))
    extractor._save(feats, cache_dir, [0, 1, 2], dataset_meta=None)

    with pytest.raises(IndexError, match="covers 3 nodes"):
        extractor._load(cache_dir, [5], dataset_meta=None)


def test_cell_level_save_sizes_by_requested_nodes(tmp_path):
    """`n_nodes` sizes the array, so cells without a patch cannot truncate the cache."""
    extractor = _make_cell_extractor()
    cache_dir = tmp_path / extractor.name
    extractor._save(torch.zeros((3, 4)), cache_dir, [0, 1, 2], None, n_nodes=10)

    assert np.load(cache_dir / "embeddings.npy", mmap_mode="r").shape == (10, 4)
    assert extractor._load(cache_dir, [9], dataset_meta=None) is not None


def test_cell_level_cache_covers(tmp_path):
    """`cache_covers` distinguishes a complete cache from a partial or foreign one."""
    extractor = _make_cell_extractor()
    extractor._save(
        torch.zeros((3, 4)), tmp_path / extractor.name, [0, 1, 2], dataset_meta=None
    )

    assert extractor.cache_covers(tmp_path, 3)
    assert not extractor.cache_covers(tmp_path, 4)  # partial: written by one window only
    assert not extractor.cache_covers(None, 3)
    assert not extractor.cache_covers(tmp_path / "elsewhere", 3)

    other = CellLevelExtractor(encoder=_MockEncoder(arch="DifferentEncoder"))
    assert not other.cache_covers(tmp_path, 3)

    # a dataset_meta mismatch also invalidates, even with matching name/metadata
    identity = dataset_identity(np.zeros((1, 4, 4)), np.zeros((1, 4, 4), dtype=np.uint16))
    other_identity = dataset_identity(
        np.ones((1, 4, 4)), np.zeros((1, 4, 4), dtype=np.uint16)
    )
    extractor._save(
        torch.zeros((3, 4)), tmp_path / extractor.name, [0, 1, 2], dataset_meta=identity
    )
    assert extractor.cache_covers(tmp_path, 3, identity)
    assert not extractor.cache_covers(tmp_path, 3, other_identity)


def test_cell_level_windowed_call_never_writes_cache(
    tmp_path, toy_images, toy_handcrafted_features
):
    """A call restricted to `timepoints` must not replace a full cache with its window."""

    class _MeanEncoder(nn.Module):
        arch = "MeanEncoder"

        def forward(self, imgs):
            return imgs.mean(dim=(2, 3))

    extractor = CellLevelExtractor(
        encoder=_MeanEncoder(), input_size_enc=32, verbose=False
    )
    cache_dir = tmp_path / extractor.name

    extractor(
        image=toy_images,
        node_feats=toy_handcrafted_features,
        output_dir=tmp_path,
        timepoints=np.asarray([0]),
    )
    assert not cache_dir.exists()

    # the full-sequence call does write, and covers every node
    extractor(image=toy_images, node_feats=toy_handcrafted_features, output_dir=tmp_path)
    n_nodes = int(toy_handcrafted_features["index"].to_numpy().max()) + 1
    assert extractor.cache_covers(tmp_path, n_nodes)

    # a later windowed call reads it back rather than clobbering it
    extractor(
        image=toy_images,
        node_feats=toy_handcrafted_features,
        output_dir=tmp_path,
        timepoints=np.asarray([toy_handcrafted_features["t"].to_numpy().max()]),
    )
    assert extractor.cache_covers(tmp_path, n_nodes)


def test_cell_level_cache_validation(tmp_path):
    """`_load` reuses the cache on matching meta/dataset identity and rejects mismatches.

    A fresh extractor per check: `_load` only compares `meta.json` on a cache_dir's
    first access (see `_open_embedding_caches`).
    """
    cache_dir = tmp_path / _make_cell_extractor().name
    feats = torch.zeros((3, 4))
    identity = dataset_identity(np.zeros((1, 4, 4)), np.zeros((1, 4, 4), dtype=np.uint16))
    _make_cell_extractor()._save(feats, cache_dir, [0, 1, 2], dataset_meta=identity)

    # matching identity -> reuse
    assert _make_cell_extractor()._load(cache_dir, [0, 1], identity) is not None
    # identity mismatch -> reject
    other_identity = dataset_identity(
        np.ones((1, 4, 4)), np.zeros((1, 4, 4), dtype=np.uint16)
    )
    assert _make_cell_extractor()._load(cache_dir, [0, 1], other_identity) is None
    # metadata mismatch -> reject
    other = CellLevelExtractor(encoder=_MockEncoder(arch="DifferentEncoder"))
    assert other._load(cache_dir, [0, 1], None) is None


def test_cell_level_cache_reuses_memmap(tmp_path):
    """Repeated loads reuse a single memmap keyed by cache dir."""
    extractor = _make_cell_extractor()
    cache_dir = tmp_path / extractor.name
    extractor._save(torch.zeros((3, 4)), cache_dir, [0, 1, 2], dataset_meta=None)

    extractor._load(cache_dir, [0], dataset_meta=None)
    first = extractor._open_embedding_caches[str(cache_dir)]
    extractor._load(cache_dir, [1], dataset_meta=None)
    assert len(extractor._open_embedding_caches) == 1
    assert extractor._open_embedding_caches[str(cache_dir)] is first


@pytest.mark.parametrize("num_workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_cell_patch_dataset_matches_serial_crops(
    toy_images, toy_masks, toy_handcrafted_features, num_workers
):
    """Chunked patches equal the serial crops, single-channel and expanded alike.

    Resizing one channel and broadcasting afterwards has to match resizing three
    identical channels.
    """
    extractor = CellLevelExtractor(
        encoder=_MockEncoder(), input_size_enc=64, cells_per_item=7, verbose=False
    )
    coords = extractor._get_boxes_and_padding(
        toy_handcrafted_features, toy_images[0].shape
    )
    expected, expected_masks, _ = extractor.get_single_cell_images(
        toy_images, coords, toy_masks
    )

    dataset = extractor._build_patch_dataset(
        toy_images, coords, toy_masks, None, for_encoder=True
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        multiprocessing_context=get_multiprocessing_context(num_workers),
    )

    assert expected_masks is not None
    for chunk in loader:
        rows = chunk["rows"]
        # the encoder path keeps one channel and normalizes on the device
        patches = normalize_imagenet(chunk["patches"].repeat(1, 3, 1, 1))
        assert torch.allclose(patches, expected[rows], atol=1e-5)
        assert torch.equal(chunk["masks"], expected_masks[rows])


def test_cell_patch_dataset_emits_weights_for_masked_encoder(
    toy_images, toy_masks, toy_handcrafted_features
):
    """A masked encoder gets ready-made patch weights instead of mask patches."""
    extractor = CellLevelExtractor(
        encoder=_MockEncoder(), input_size_enc=64, cells_per_item=32, verbose=False
    )
    extractor.weighting = PatchWeighting(gamma=2.0, patchsize=16)
    coords = extractor._get_boxes_and_padding(
        toy_handcrafted_features.head(8), toy_images[0].shape
    )
    dataset = extractor._build_patch_dataset(
        toy_images, coords, toy_masks, None, for_encoder=True
    )

    chunk = dataset[0]
    assert chunk["masks"] is None
    assert chunk["weights"].shape == (len(chunk["rows"]), (64 // 16) ** 2)


def test_cell_patches_skip_zero_sized_crops(toy_images, toy_handcrafted_features):
    """A cell whose crop collapses to zero pixels drops out with its index."""
    node_feats = toy_handcrafted_features.head(4)
    extractor = CellLevelExtractor(
        encoder=_MockEncoder(), input_size_enc=32, cells_per_item=8, verbose=False
    )
    coords = extractor._get_boxes_and_padding(node_feats, toy_images[0].shape)
    coords["s_full"] = np.asarray(coords["s_full"]).copy()
    coords["s_full"][1] = 0

    dataset = extractor._build_patch_dataset(
        toy_images, coords, None, None, for_encoder=True
    )
    rows = torch.cat([chunk["rows"] for chunk in dataset])
    patches = torch.cat([chunk["patches"] for chunk in dataset])

    assert len(rows) == len(patches) == len(node_feats) - 1
    assert 1 not in rows.tolist()


def test_patch_weights_match_reference():
    """Patch weights average the per-crop normalized distance field over each patch."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 12:20] = 1

    weights = compute_patch_weights(mask, gamma=2.0, patchsize=16)

    dist = distance_transform_edt(mask == 0)
    reference = ((1.0 - dist / dist.max()) ** 2.0).reshape(2, 16, 2, 16).mean(axis=(1, 3))
    assert np.allclose(weights, reference.ravel())

    # degenerate crops carry no distance gradient and weight every patch equally
    assert np.allclose(compute_patch_weights(np.ones((32, 32), np.uint8), 2.0, 16), 1.0)
    assert np.allclose(compute_patch_weights(np.zeros((32, 32), np.uint8), 2.0, 16), 1.0)


@pytest.fixture
def edge_finder_and_cache(tmp_path):
    """A directed EdgeFinder and a path for its edge store."""
    edge_finder = EdgeFinder(
        prune_edges_by=None,
        feature_names=("dist_spat", "dist_temp"),
        n_jobs=1,
        bidirectional=False,
    )
    cache_file = tmp_path / "edges.parquet"
    return edge_finder, cache_file


def test_edge_finder_save_load_cycle(edge_finder_and_cache, toy_handcrafted_features):
    """A store round-trips: the loaded graph equals the one that was computed."""
    edge_finder, cache_file = edge_finder_and_cache
    radius = 50
    time_steps = [1]

    edges_computed = edge_finder(
        node_feats=toy_handcrafted_features,
        search_radius=radius,
        time_steps=time_steps,
        filepath=cache_file,
    )
    assert cache_file.exists()

    edges_loaded = edge_finder(
        node_feats=toy_handcrafted_features,
        search_radius=radius,
        time_steps=time_steps,
        filepath=cache_file,
    )

    edges_computed = collect(edges_computed.sort(["src", "dst"]))
    edges_loaded = collect(edges_loaded.sort(["src", "dst"]))
    assert edges_computed.equals(edges_loaded)


def test_edge_finder_load_filtering(edge_finder_and_cache, toy_handcrafted_features):
    """A smaller radius and fewer strides filter the store instead of rebuilding it."""
    edge_finder, cache_file = edge_finder_and_cache

    radius_large = 100
    steps_all = [1, 2]
    edge_finder(
        node_feats=toy_handcrafted_features,
        search_radius=radius_large,
        time_steps=steps_all,
        filepath=cache_file,
    )

    radius_small = 20
    steps_subset = [1]
    edges_filtered = edge_finder(
        node_feats=toy_handcrafted_features,
        search_radius=radius_small,
        time_steps=steps_subset,
        filepath=cache_file,
    )
    edges_filtered = collect(edges_filtered)

    assert max(edges_filtered["dist_spat"]) <= radius_small
    assert edges_filtered["dist_temp"].unique().to_list() == steps_subset

    # the store keeps a partition per stride at the full radius
    assert {p.name for p in cache_file.glob("dt=*")} == {"dt=1", "dt=2"}
    edges_cached = pl.read_parquet(str(cache_file / "dt=*" / "part.parquet"))
    assert max(edges_cached["dist_spat"]) > radius_small
    assert 2 in edges_cached["dist_temp"].unique()


def test_edge_finder_recomputes_on_invalid_cache(
    edge_finder_and_cache, toy_handcrafted_features
):
    """A directed store cannot serve a bidirectional finder, so the mismatch rebuilds.

    The rebuilt store keeps forward edges only and mirrors the backward ones on load.
    """
    edge_finder_directed, cache_file = edge_finder_and_cache
    steps = [1]
    radius = 50

    edges_directed = collect(
        edge_finder_directed(
            node_feats=toy_handcrafted_features,
            search_radius=radius,
            time_steps=steps,
            filepath=cache_file,
        )
    )
    cached = pl.read_parquet(str(cache_file / "dt=*" / "part.parquet"))
    assert (cached["dist_temp"] > 0).all()

    edge_finder_bidir = EdgeFinder(
        prune_edges_by=None,
        feature_names=("dist_spat", "dist_temp"),
        n_jobs=1,
        bidirectional=True,
    )
    edges_recomputed = collect(
        edge_finder_bidir(
            node_feats=toy_handcrafted_features,
            search_radius=radius,
            time_steps=steps,
            filepath=cache_file,
        )
    )

    backward = edges_recomputed.filter(pl.col("dist_temp") < 0)
    assert backward.height >= edges_directed.height


def test_edge_finder_persists_unnormalized(
    edge_finder_and_cache, toy_handcrafted_features
):
    """Saved graph features must not be normalized."""
    edge_finder, cache_file = edge_finder_and_cache
    radius = 50
    steps = [1]

    edges_out = collect(
        edge_finder(
            node_feats=toy_handcrafted_features,
            search_radius=radius,
            time_steps=steps,
            filepath=cache_file,
        )
    )
    edges_on_disk = pl.read_parquet(str(cache_file / "dt=*" / "part.parquet"))

    max_dist = max(edges_on_disk["dist_spat"])
    # a normalized dist_spat would be log(50 / len_init), well below 5
    assert max_dist > 5
    assert np.isclose(max_dist, max(edges_out["dist_spat"]))

    transformed = edge_finder._transform(edges_out, toy_handcrafted_features)
    assert transformed is not None
    assert transformed[:, 0].max() < 5  # dist_spat is the first feature column


def test_edge_finder_incremental_extension(tmp_path, toy_handcrafted_features):
    """Requesting a new stride extends the store without recomputing existing ones."""
    finder = EdgeFinder(
        feature_names=("dist_spat", "dist_temp"), n_jobs=1, bidirectional=True
    )
    store = tmp_path / "edges_store"

    finder(
        node_feats=toy_handcrafted_features,
        search_radius=60,
        time_steps=[1, 2],
        filepath=store,
    )
    assert {p.name for p in store.glob("dt=*")} == {"dt=1", "dt=2"}
    dt1 = store / "dt=1" / "part.parquet"
    mtime_before = dt1.stat().st_mtime_ns

    extended = collect(
        finder(
            node_feats=toy_handcrafted_features,
            search_radius=60,
            time_steps=[1, 2, 3],
            filepath=store,
        )
    )
    # the new stride is added, the existing partition is untouched
    assert {p.name for p in store.glob("dt=*")} == {"dt=1", "dt=2", "dt=3"}
    assert dt1.stat().st_mtime_ns == mtime_before
    assert set(extended["dist_temp"].abs().unique().to_list()) == {1, 2, 3}


def test_handcrafted_transform_invariant_to_column_order(toy_images, toy_masks):
    """`_transform` returns identical features regardless of input column order.

    `intensity` expands to `intensity_min/max/mean`, the multi-column case whose layout
    can shift across the joins `GraphDataset` does before calling `_transform`.
    """
    extractor = HandcraftedExtractor(
        props=["area", "intensity_min", "intensity_max", "intensity_mean"],
        feature_names=["area", "intensity"],
        feature_norm_fn="scale_relative_size",
        n_jobs=1,
    )
    node_feats = extractor(image=toy_images, masks=toy_masks, filepath=None)

    # sanity: the multi-column intensity expansion is actually exercised
    intensity_cols = [c for c in node_feats.columns if c.startswith("intensity_")]
    assert len(intensity_cols) >= 3

    transformed = extractor._transform(node_feats)
    assert transformed is not None
    selected = list(extractor.extracted_features)  # type: ignore
    transformed_again = extractor._transform(node_feats)
    assert torch.equal(transformed, transformed_again)  # type: ignore
    assert list(extractor.extracted_features) == selected  # type: ignore

    # shuffling the input columns must not change the result
    rng = np.random.default_rng(0)
    for _ in range(2):
        shuffled_cols = list(node_feats.columns)
        rng.shuffle(shuffled_cols)
        assert shuffled_cols != list(node_feats.columns)  # sanity: order actually changed
        shuffled = node_feats.select(shuffled_cols)

        transformed_shuffled = extractor._transform(shuffled)
        assert torch.equal(transformed, transformed_shuffled)  # type: ignore
        assert list(extractor.extracted_features) == selected  # type: ignore


def test_handcrafted_reuses_cache(toy_images, toy_masks, tmp_path):
    """A valid cache is reused instead of recomputed under the default config.

    A transform emits a column under a name of its own, and `min_node_area` makes the
    cache a legitimate subset of the mask labels. Neither may invalidate it.
    """
    extractor = HandcraftedExtractor(
        props=[
            "area",
            "axis_major_length",
            "axis_minor_length",
            "intensity_mean",
            "intensity_min",
            "intensity_max",
            "eccentricity",
            "solidity",
            "orientation",
        ],
        extra_props=["center_local", "thickness"],
        extra_transforms=["aspect_ratio"],
        feature_names=[
            "area",
            "aspect_ratio",
            "intensity",
            "eccentricity",
            "solidity",
            "thickness",
        ],
        feature_norm_fn="scale_relative_size",
        min_node_area=32,
        n_jobs=1,
    )
    cache_file = tmp_path / "nodes.parquet"
    extractor(image=toy_images, masks=toy_masks, filepath=cache_file, validate=True)
    assert cache_file.exists()

    # reuse the cache; transforms are re-derived on load
    reused = extractor(
        image=toy_images, masks=toy_masks, filepath=cache_file, validate=True
    )
    assert "aspect_ratio" in reused.columns

    # an area-filtered cache (missing a sub-threshold cell) still validates: `_validate`
    # only checks that the requested columns are present, not row-level content
    loaded = pl.read_parquet(cache_file)
    dropped = int(loaded.filter(t=0)["label"][0])
    filtered = loaded.filter(~((pl.col("t") == 0) & (pl.col("label") == dropped)))
    assert extractor._validate(filtered, toy_masks) is not None


def test_build_lineage_gt_edges_counts():
    """Lineage GT edges enumerate correspondences and divisions from labels alone."""
    # track 1 spans t0..t2 and divides at t2 into daughters 2 and 3 (both parent 1)
    node_feats = pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4, 5, 6],
            "t": [0, 1, 2, 2, 3, 2, 3],
            "label": [1, 1, 1, 2, 2, 3, 3],
            "parent": [0, 0, 0, 1, 1, 1, 1],
        }
    )
    edges = build_lineage_gt_edges(node_feats)

    corr = set(edges.filter(pl.col("y") == 1).select("src", "dst").iter_rows())
    div = set(edges.filter(pl.col("y") == 2).select("src", "dst").iter_rows())

    # consecutive same-label appearances
    assert corr == {(0, 1), (1, 2), (3, 4), (5, 6)}
    # parent's last node (index 2) to each daughter's first node
    assert div == {(2, 3), (2, 5)}


def test_build_lineage_gt_edges_superset_of_candidate_gt(toy_handcrafted_features):
    """Lineage GT edges cover every GT-labeled edge the candidate graph can produce."""
    candidate = _get_mock_edges(
        toy_handcrafted_features, search_radius=50, time_steps=[1]
    )
    candidate_gt = set(candidate.filter(pl.col("y") > 0).select("src", "dst").iter_rows())

    lineage = build_lineage_gt_edges(toy_handcrafted_features)
    lineage_edges = set(lineage.select("src", "dst").iter_rows())

    assert lineage.filter(pl.col("y") == 2).height > 0  # toy data has divisions
    # the lineage graph is unpruned, so it must contain all candidate GT links
    assert candidate_gt <= lineage_edges
