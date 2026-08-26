"""Test datasets and feature composition."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from conftest import prepare_dataset_structure
from polars.testing import assert_frame_equal
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data

from baclct.data.dataset import GraphDataset, GraphPatchDataset, TrackingDataset
from baclct.features.extractors import CellLevelExtractor, HandcraftedExtractor
from baclct.features.graph import EdgeFinder
from baclct.utils import feature_info as fi
from baclct.utils.data import collect
from baclct.utils.feature_info import cache_signature, classify_cache
from baclct.utils.graph_params import _expected_cell_size, resolve_search_radius
from baclct.utils.spacing import (
    normalize_spacing,
    position_columns,
    spatial_spacing,
)


def _edge_finder(
    *,
    bidirectional: bool = False,
    prune_edges_by=("radius", 1.5),  # multiplier on half the major axis
    feature_names=("dist_spat", "dist_temp"),
    extra_features=None,
) -> EdgeFinder:
    """`EdgeFinder` on the toy medial-axis centers. Arguments are what a test varies."""
    return EdgeFinder(
        prune_edges_by=prune_edges_by,
        feature_names=feature_names,
        extra_features=extra_features,
        bidirectional=bidirectional,
        center_name="center",
        n_jobs=1,
    )


def _graph_dataset(
    single_sequence_data,
    hc_extractor,
    edge_finder,
    *,
    cached: bool = False,
    cls: type[GraphDataset] = GraphDataset,
    **overrides,
) -> GraphDataset:
    """`GraphDataset` on the prepared toy sequence. `overrides` are what a test varies.

    `cached` points the features at the prepared directory, so precomputed edges are
    scanned from the parquet store instead of being held in memory. `trust_cache=True`
    since the fixture's cache has no `dataset_identity()` sidecar to compare against.
    """
    data_dir, feature_dir, seq_id = single_sequence_data
    kwargs = {
        "data_dir": data_dir,
        "feature_dir": feature_dir if cached else None,
        "sequence_id": seq_id,
        "edge_finder": edge_finder,
        "handcrafted_feature_extractor": hc_extractor,
        "deep_feature_extractor": None,
        "data_format": "ctc",
        "segmentation_name": "GT",
        "graph_search_radius": 50,
        "graph_time_step": 1,
        "graph_num_steps": 1,
        "trust_cache": True,
    }
    return cls(**(kwargs | overrides))  # type: ignore


def create_mock_datamodule(
    tmp_path, sequences, node_features, toy_masks, batch_size=1, **extra_kwargs
):
    """Create a `TrackingDataset` over `sequences` written into `tmp_path`."""
    split_content = {0: sequences}
    all_seq_ids = [s for phase in sequences.values() for s in phase]
    sequence_data = {
        seq_id: {"node_features": node_features, "masks": toy_masks}
        for seq_id in all_seq_ids
    }

    data_dir, feature_dir = prepare_dataset_structure(
        tmp_path, sequence_data, split_content
    )

    kwargs = {
        "data_dir": data_dir,
        "feature_dir": feature_dir,
        "sequence_ids": all_seq_ids,
        "edge_finder": _edge_finder(bidirectional=True, prune_edges_by=None),
        "handcrafted_feature_extractor": HandcraftedExtractor(
            extra_props=["center_local"], feature_norm_fn="scale_relative_size", n_jobs=1
        ),
        "data_format": "ctc",
        "segmentation_name": "GT",
        "graph_time_step": 1,
        "graph_num_steps": 3,
        "fold": 0,
        "batch_size": batch_size,
        "shuffle": False,
    }
    kwargs.update(extra_kwargs)
    return TrackingDataset(**kwargs)  # type: ignore


def test_graph_dataset_assembles_item(single_sequence_data, basic_extractors):
    """An item carries handcrafted features, edges, and labels as finite typed tensors."""
    edge_finder, hc_extractor = basic_extractors
    dataset = _graph_dataset(single_sequence_data, hc_extractor, edge_finder, cached=True)

    graph_item = dataset[0]
    assert graph_item is not None
    assert isinstance(graph_item, Data)
    for k in ["x_handcrafted", "edge_index", "y_edges", "y_nodes", "num_nodes"]:
        assert k in graph_item
        v = graph_item[k]
        if isinstance(v, torch.Tensor):
            assert torch.isnan(v).sum() == 0, f"Found NaN in {k}: {v[:5]}"

    assert graph_item.x_handcrafted.dtype == torch.float32
    assert graph_item.edge_index.dtype == torch.long  # type: ignore
    assert graph_item.y_edges.dtype == torch.long
    assert graph_item.y_nodes.dtype == torch.long
    assert graph_item.edge_index.shape[0] == 2  # type: ignore
    assert graph_item.x_handcrafted.shape[0] == graph_item.num_nodes
    assert graph_item.edge_index.shape[1] == len(graph_item.y_edges)  # type: ignore
    assert graph_item.x_handcrafted.shape[1] == len(hc_extractor.feature_names)


def test_feature_info_records_trained_features(
    single_sequence_data, basic_extractors, tmp_path
):
    """Feature info captures the trained node/edge feature names and stats."""
    _, _, seq_id = single_sequence_data
    edge_finder, hc_extractor = basic_extractors
    dataset = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        cached=True,
        precompute_edges=True,
    )

    node = fi._node_feature_stats([dataset], hc_extractor)
    assert node
    assert node["n"] == dataset.node_feats.height
    assert node["names"] == list(hc_extractor.extracted_features)
    assert set(node["mean"]) == set(node["names"])

    edge = fi._edge_feature_stats([dataset], edge_finder)
    assert edge
    assert edge["names"] == list(edge_finder.feature_names)
    assert edge["n"] > 0
    assert set(edge["mean"]) == set(edge_finder.feature_cols)

    # round-trip through a TrackingDataset-like object, including cache provenance
    cache_records = [{"kind": "handcrafted", "sequence": seq_id, "status": "used"}]
    fake_datamodule = SimpleNamespace(
        dataset_train=dataset,
        hc_feat_extractor=hc_extractor,
        edge_finder=edge_finder,
        deep_feat_extractor=None,
        fold=0,
        cache_records=cache_records,
    )
    path = fi.write_feature_info(fake_datamodule, tmp_path)  # type: ignore
    assert path is not None and path.exists()

    info = json.loads(path.read_text())
    train = info["features"]["train"]
    assert train["node_features"]["names"] == node["names"]
    assert train["edge_features"]["names"] == edge["names"]
    assert train["deep_features"] is None
    # val split was not loaded, so it carries no feature stats
    assert info["features"]["val"] is None
    assert info["splits"]["train"] == [seq_id]
    assert info["splits"]["fold"] == 0
    assert info["caches"] == cache_records


def test_classify_cache(tmp_path):
    """`classify_cache` reports used/invalidated/created/absent with paths and times."""
    cache = tmp_path / "nodes.parquet"

    # no cache file at all
    sig = cache_signature(cache)
    assert classify_cache("handcrafted", "01", cache, sig)["status"] == "absent"

    # first write is a fresh creation
    sig = cache_signature(cache)
    cache.write_text("x")
    created = classify_cache("handcrafted", "01", cache, sig)
    assert created["status"] == "created"
    assert created["path"] == str(cache)
    assert "created_at" in created and "modified_at" in created

    # untouched cache is reused
    sig = cache_signature(cache)
    assert classify_cache("handcrafted", "01", cache, sig)["status"] == "used"

    # a rewrite (newer mtime) is an invalidation
    sig = cache_signature(cache)
    os.utime(cache, (time.time() + 10, time.time() + 10))
    assert classify_cache("handcrafted", "01", cache, sig)["status"] == "invalidated"

    # caches held as a directory are tracked by their newest contained file
    store = tmp_path / "feats"
    store.mkdir()
    (store / "0.0").write_text("a")
    sig = cache_signature(store)
    assert classify_cache("deep", "01", store, sig)["status"] == "used"


def _get_frames(graph, node_feats) -> np.ndarray:
    """Frames the nodes of `graph` came from."""
    return np.unique(
        node_feats.filter(pl.col("index").is_in(graph.node_mapping.numpy()))["t"]
    )


def test_precompute_radius_augmentation(single_sequence_data, basic_extractors):
    """A range graph_search_radius samples a per-item spatial cap under precompute_edges.

    Regression: the precompute path filtered only by (t, dist_temp), so a range radius was
    a silent no-op (all edges up to the store's max radius were always used).
    """
    edge_finder, hc_extractor = basic_extractors
    kwargs = {
        "precompute_edges": True,
        "training": True,
    }

    ranged = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        graph_search_radius=[10, 50, 20],
        **kwargs,
    )
    assert ranged._radius_aug
    t = int(ranged.valid_frames[0])
    ts = ranged.time_steps(return_all_combinations=False)

    heights, maxima = set(), set()
    for _ in range(50):
        ed = ranged._get_graph_for_frame(t, ts)
        if ed is not None and ed.height:
            heights.add(ed.height)
            maxima.add(float(ed["dist_spat"].max()))
    # sampling the radius per item varies the edge set and never exceeds the store max
    assert len(heights) > 1
    assert max(maxima) <= 50.0

    fixed = _graph_dataset(single_sequence_data, hc_extractor, edge_finder, **kwargs)
    assert not fixed._radius_aug
    fixed_heights = {fixed._get_graph_for_frame(t, ts).height for _ in range(10)}
    assert len(fixed_heights) == 1  # scalar radius: identical graph every item


def test_position_columns_prefers_px_and_generalizes_to_nd():
    """Axis-ordered position names, preferring pixel space over physical units."""
    # 2D, no spacing: center over centroid, no _px
    assert position_columns(["center-0", "center-1", "centroid-0"]) == [
        "center-0",
        "center-1",
    ]
    # 3D physical + pixel columns: the _px family wins, all three axes in order
    cols_3d = [
        "centroid-0",
        "centroid-1",
        "centroid-2",
        "centroid-0_px",
        "centroid-1_px",
        "centroid-2_px",
    ]
    assert position_columns(cols_3d) == [
        "centroid-0_px",
        "centroid-1_px",
        "centroid-2_px",
    ]
    # non-contiguous axis set (missing axis 1) falls through
    assert position_columns(["center-0", "center-2"]) == []


def test_normalize_spacing():
    """Spacing specs normalize to a {t, z, y, x} float mapping, default 1.0."""
    assert normalize_spacing() == {"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}
    assert normalize_spacing({"t": 3}) == {"t": 3.0, "z": 1.0, "y": 1.0, "x": 1.0}
    assert normalize_spacing({"z": 4.95}) == {"t": 1.0, "z": 4.95, "y": 1.0, "x": 1.0}
    # partial sequences are ambiguous, so a sequence must give the full (t, z, y, x)
    assert normalize_spacing([1, 5, 2, 4]) == {"t": 1.0, "z": 5.0, "y": 2.0, "x": 4.0}
    # spatial_spacing returns the (z, y, x)-ordered trailing axes for regionprops
    sp = normalize_spacing([1, 5, 2, 4])
    assert spatial_spacing(sp, ndim=2) == (2.0, 4.0)
    assert spatial_spacing(sp, ndim=3) == (5.0, 2.0, 4.0)

    with pytest.raises(ValueError):
        normalize_spacing({"q": 1.0})
    with pytest.raises(ValueError):
        normalize_spacing([2, 4])
    with pytest.raises(ValueError):
        normalize_spacing([5, 2, 4])
    with pytest.raises(ValueError):
        normalize_spacing([1, 2, 3, 4, 5])


def test_dataset_spacing_scales_dist_temp(single_sequence_data, basic_extractors):
    """spacing.t passed to GraphDataset rescales the dist_temp edge feature."""
    _, hc_extractor = basic_extractors
    edge_finder = _edge_finder(prune_edges_by=None)
    kwargs = {"graph_num_steps": 3, "precompute_edges": True}

    ds1 = _graph_dataset(single_sequence_data, hc_extractor, edge_finder, **kwargs)
    dsk = _graph_dataset(
        single_sequence_data, hc_extractor, edge_finder, spacing={"t": 3.0}, **kwargs
    )
    assert ds1.spacing["t"] == 1.0 and dsk.spacing["t"] == 3.0

    t = int(ds1.valid_frames[0])
    ts = ds1.time_steps(return_all_combinations=False)
    edge_data = ds1._get_graph_for_frame(t, ts)
    assert edge_data is not None and edge_data.height

    dt_idx = edge_finder.feature_cols.index("dist_temp")
    raw = edge_data["dist_temp"].to_numpy()
    feat_1 = ds1._get_edge_features(edge_data, ds1.node_feats).numpy()[:, dt_idx]
    feat_k = dsk._get_edge_features(edge_data, dsk.node_feats).numpy()[:, dt_idx]

    assert np.allclose(feat_1, np.sign(raw) * np.log(np.abs(raw) + 1))
    assert np.allclose(feat_k, np.sign(raw) * np.log(np.abs(raw * 3) + 1))


def test_relative_search_radius_resolves_to_pixels(
    single_sequence_data, basic_extractors
):
    """A 'Nx' search radius resolves to round(N * expected cell size) in pixels."""
    _, hc_extractor = basic_extractors
    edge_finder = _edge_finder(prune_edges_by=None)
    kwargs = {"precompute_edges": True}

    ds = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        graph_search_radius="2.5x",
        **kwargs,
    )
    expected = max(1, round(2.5 * _expected_cell_size(ds.node_feats)))
    # the string radius is resolved to absolute pixels during init
    assert ds.graph_search_radius == expected
    assert isinstance(ds.graph_search_radius, int)

    # an int radius passes through unchanged
    ds_int = _graph_dataset(
        single_sequence_data, hc_extractor, edge_finder, graph_search_radius=123, **kwargs
    )
    assert ds_int.graph_search_radius == 123

    # malformed relative radii fail clearly
    with pytest.raises(ValueError):
        resolve_search_radius("2.5", ds.node_feats)
    with pytest.raises(ValueError):
        resolve_search_radius("abcx", ds.node_feats)
    with pytest.raises(ValueError):
        resolve_search_radius("-1x", ds.node_feats)


@pytest.mark.parametrize(
    "graph_connectivity, bidirectional, graph_fov_size, cached",
    [
        ("dense", True, None, False),
        ("dense", False, None, False),
        ("star", True, None, False),
        ("star", False, None, False),
        ("sequential", True, None, False),
        ("sequential", False, None, False),
        ("dense", True, None, True),
        ("dense", True, 2.0, True),
    ],
)
def test_precompute_matches_on_the_fly(
    single_sequence_data,
    basic_extractors,
    graph_connectivity,
    bidirectional,
    graph_fov_size,
    cached,
):
    """Both edge paths return the same edges for every frame, boundaries included.

    A cached store is scanned and filtered per item, an in-memory one is sliced, and
    either mirrors the backward half of the window itself, so all of them have to agree
    with the on-the-fly builder that constructs the window from scratch. `relative_size`
    is included because mirroring negates it instead of copying it.
    """
    _, hc_extractor = basic_extractors
    edge_finder = _edge_finder(
        bidirectional=bidirectional,
        feature_names=("dist_spat", "dist_temp", "overlap", "iou", "relative_size"),
        extra_features=["relative_size"],
    )
    kwargs = {
        "cached": cached,
        "graph_num_steps": 2,
        "graph_connectivity": graph_connectivity,
        "graph_fov_size": graph_fov_size,
    }
    ds_otf = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        precompute_edges=False,
        **kwargs,
    )
    ds_pre = _graph_dataset(
        single_sequence_data, hc_extractor, edge_finder, precompute_edges=True, **kwargs
    )
    assert ds_otf.node_feats.equals(ds_pre.node_feats)

    steps = ds_pre.time_steps(return_all_combinations=False)
    cols = ["src", "dst", "dist_temp", "dist_spat", "overlap", "iou", "relative_size"]
    for t in range(ds_pre.num_frames):
        e_otf = ds_otf._get_graph_for_frame(t, steps)
        e_pre = ds_pre._get_graph_for_frame(t, steps)
        assert (e_otf is None) == (e_pre is None), f"Only one path has edges at {t}."
        if e_pre is None:
            continue
        assert_frame_equal(
            e_otf.select(cols).sort(cols),  # type: ignore
            e_pre.select(cols).sort(cols),
            check_row_order=False,
        )


@pytest.mark.parametrize("fov_size", (1.5, 2))
def test_graph_fov_size_extends_temporal_window(
    single_sequence_data,
    basic_extractors,
    fov_size,
):
    """graph_fov_size adds anchor frames but keeps the max edge distance unchanged.

    Both an int and a float are accepted. The toy sequence is short enough that either
    anchor count saturates it.
    """
    _, hc_extractor = basic_extractors
    edge_finder = _edge_finder(bidirectional=True)
    kwargs = {"graph_connectivity": "dense", "precompute_edges": True}

    ds_base = _graph_dataset(single_sequence_data, hc_extractor, edge_finder, **kwargs)
    ds_fov = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        graph_fov_size=fov_size,
        **kwargs,
    )

    # pick a center frame that has room to expand in both directions
    mid = int(max(ds_base.node_feats["t"])) // 2
    g_base = ds_base[mid]
    g_fov = ds_fov[mid]

    assert _get_frames(g_base, ds_base.node_feats).tolist() == [mid, mid + 1]
    all_frames = list(range(ds_fov.num_frames))
    assert _get_frames(g_fov, ds_fov.node_feats).tolist() == all_frames

    # widening the window must not lengthen the individual edges
    frame_of_node = (
        ds_fov.node_feats.filter(pl.col("index").is_in(g_fov.node_mapping.numpy()))["t"]
        .to_numpy()
        .astype(int)
    )
    src, dst = g_fov.edge_index.numpy()  # type: ignore
    assert np.abs(frame_of_node[dst] - frame_of_node[src]).max() <= ds_fov.graph_num_steps


@pytest.mark.parametrize(
    "graph_fov_size, time_steps, expected_anchors",
    [
        (None, [1, 2, 3], [0]),
        (0.5, [1], [0]),  # half an anchor rounds down to none
        (0.5, [1, 2, 3], [-1, 0, 1]),
        (1.0, [1, 2, 3], [-3, -2, -1, 0, 1, 2, 3]),
        (1.0, [2, 4], [-4, -2, 0, 2, 4]),  # anchors sit one stride apart, not one frame
    ],
)
def test_fov_anchor_count(graph_fov_size, time_steps, expected_anchors):
    """Each side gets `ceil(fov * num_steps - 0.5)` anchors, spaced by the sampled stride.

    The toy sequence saturates at four frames, too short to tell anchor counts apart, so
    the resolution is checked on its own.
    """
    # bare instance: _resolve_fov_combinations only reads the attributes set below
    dataset = GraphDataset.__new__(GraphDataset)
    dataset.graph_fov_size = graph_fov_size
    dataset.edge_finder = _edge_finder()
    # under 'star' every pair starts at its anchor, so the sources are the anchors
    dataset.graph_connectivity = "star"

    pairs = dataset._resolve_fov_combinations(0, time_steps)
    assert sorted({t_src for t_src, _ in pairs}) == expected_anchors


def test_resolve_boundary_pairs_preserve_stride():
    """Out-of-bounds edges are substituted by same-stride in-bounds edges, never off-grid.

    Reflecting a single endpoint across the boundary would invent strides outside the
    configured set (e.g. dt=6 from a step-5 config), which the base-stride cache cannot
    serve.
    """
    t_max = 40

    # in-bounds pair kept as is
    assert EdgeFinder.resolve_boundary_pairs([(10, 20)], t_max, True) == [(10, 20)]
    # t -> t+n out of bounds becomes t-n -> t (same stride)
    assert EdgeFinder.resolve_boundary_pairs([(38, 48)], t_max, True) == [(28, 38)]
    # start boundary: t-n -> t out of bounds becomes t -> t+n
    assert EdgeFinder.resolve_boundary_pairs([(-5, 5)], t_max, True) == [(5, 15)]
    # too short to substitute (t-n < 0) is dropped
    assert EdgeFinder.resolve_boundary_pairs([(5, 20)], 8, True) == []
    # no backward edges available: out-of-bounds is simply dropped
    assert EdgeFinder.resolve_boundary_pairs([(38, 48)], t_max, False) == []

    # sweep the non-contiguous strides that triggered the original off-grid bug
    strides = [5, 10, 15]
    candidates = [(a, a + s) for s in strides for a in range(-15, t_max + 15)]
    for src, dst in EdgeFinder.resolve_boundary_pairs(candidates, t_max, True):
        assert 0 <= src < t_max and 0 <= dst < t_max
        assert abs(dst - src) in strides


@pytest.mark.parametrize("graph_num_steps", (1, 3))
def test_edge_finder_connectivity_equivalence(
    single_sequence_data,
    basic_extractors,
    graph_num_steps,
):
    """Star and dense produce the same global edge set, sequential a subset of it.

    `_compute` silently substitutes 'star' for 'dense', so dense is reached here through
    `get_edges_for_frame` instead.
    """
    _, hc_extractor = basic_extractors
    edge_finder = _edge_finder(bidirectional=True, prune_edges_by=None)
    ds = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        graph_num_steps=graph_num_steps,
    )

    node_feats = ds.node_feats
    masks = ds.masks
    search_radius = 50
    time_steps = ds.time_steps(return_all_combinations=True)

    edges_star = edge_finder._compute(
        node_feats, search_radius, time_steps, masks, connectivity="star"
    ).collect()
    assert isinstance(edges_star, pl.DataFrame)
    edges_star = edges_star.sort("src", "dst")

    # mirrors the loop structure of _compute, minus the override
    dense_frames = []
    for t in range(max(node_feats["t"]) + 1):
        frame_edges = edge_finder.get_edges_for_frame(
            node_feats,
            t,
            search_radius=search_radius,
            time_steps=time_steps,
            masks=masks,
            connectivity="dense",
            mirror_at_last_frames=True,
        )
        if frame_edges is not None:
            dense_frames.append(frame_edges.lazy())
    edges_dense = (
        pl.concat(dense_frames).unique(subset=["src", "dst"]).collect().sort("src", "dst")
    )

    edges_seq = edge_finder._compute(
        node_feats, search_radius, time_steps, masks, connectivity="sequential"
    ).collect()
    assert isinstance(edges_seq, pl.DataFrame)
    edges_seq = edges_seq.sort("src", "dst")

    assert_frame_equal(
        edges_star.select("src", "dst"),
        edges_dense.select("src", "dst"),
    )

    if graph_num_steps == 1:
        # a single stride collapses all three connectivities onto the same edges
        assert_frame_equal(
            edges_star.select("src", "dst"),
            edges_seq.select("src", "dst"),
        )
    else:
        assert edges_seq.height < edges_star.height, (
            "Sequential should produce fewer edges than star for multi-step graphs."
        )
        star_pairs = set(
            zip(edges_star["src"].to_list(), edges_star["dst"].to_list(), strict=True)
        )
        seq_pairs = set(
            zip(edges_seq["src"].to_list(), edges_seq["dst"].to_list(), strict=True)
        )
        assert seq_pairs.issubset(star_pairs), (
            "Sequential edges must be a subset of star edges."
        )


def test_tracking_datamodule_rejects_unknown_kwargs(
    tmp_path: Path, toy_handcrafted_features, toy_masks
):
    """A mistyped key in a dataset yaml fails loudly instead of using the default."""
    sequences = {"train": ["seq1"], "val": ["seq2"]}
    with pytest.raises(TypeError, match="graph_search_radus"):
        create_mock_datamodule(
            tmp_path,
            sequences,
            toy_handcrafted_features,
            toy_masks,
            graph_search_radus=999,
        )


def test_tracking_datamodule_setup(tmp_path: Path, toy_handcrafted_features, toy_masks):
    """Each stage instantiates the datasets of its split, with training only on 'fit'."""
    sequences = {"train": ["seq1", "seq2"], "val": ["seq3"], "test": ["seq4", "seq5"]}
    datamodule = create_mock_datamodule(
        tmp_path, sequences, toy_handcrafted_features, toy_masks
    )

    datamodule.setup("fit")
    assert hasattr(datamodule, "dataset_train")
    assert isinstance(datamodule.dataset_train, ConcatDataset)
    assert len(datamodule.dataset_train.datasets) == 2
    train_seq_ids = sorted([d.sequence_id for d in datamodule.dataset_train.datasets])
    assert train_seq_ids == ["seq1", "seq2"]
    assert isinstance(datamodule.dataset_train.datasets[0], GraphDataset)
    assert datamodule.dataset_train.datasets[0].training

    assert hasattr(datamodule, "dataset_val")
    assert isinstance(datamodule.dataset_val, GraphDataset)
    assert len(datamodule.dataset_val) > 0
    assert datamodule.dataset_val.sequence_id == "seq3"
    assert not datamodule.dataset_val.training

    datamodule.setup("test")
    assert hasattr(datamodule, "dataset_test")
    assert isinstance(datamodule.dataset_test, list)
    assert len(datamodule.dataset_test) == 2
    test_seq_ids = sorted([d.sequence_id for d in datamodule.dataset_test])
    assert test_seq_ids == ["seq4", "seq5"]
    assert isinstance(datamodule.dataset_test[0], GraphDataset)
    assert not datamodule.dataset_test[0].training

    datamodule.setup("predict")
    assert hasattr(datamodule, "dataset_pred")
    assert isinstance(datamodule.dataset_pred, list)
    assert len(datamodule.dataset_pred) > 0
    assert isinstance(datamodule.dataset_pred[0], GraphDataset)
    assert not datamodule.dataset_pred[0].training


@pytest.mark.parametrize(
    "time_step, expected_partitions",
    [
        ((1, 2), {"dt=1", "dt=2"}),  # 2-tuple inclusive range
        ((1, 3, 2), {"dt=1", "dt=3"}),  # 3-tuple range with step
    ],
)
def test_prepare_data_with_dynamic_graph_properties(
    tmp_path, toy_handcrafted_features, toy_masks, time_step, expected_partitions
):
    """prepare_data builds the edge store when graph properties are given as ranges.

    The store name uses the max radius and gets a partition per stride in the union over
    the sampled (time_step, num_steps) ranges, including stepped (min, max, step) ranges.
    """
    sequences = {"train": ["seq1"], "val": ["seq2"], "test": ["seq3"]}
    datamodule = create_mock_datamodule(
        tmp_path,
        sequences,
        toy_handcrafted_features,
        toy_masks,
        graph_search_radius=(40, 60),
        graph_time_step=time_step,
        graph_num_steps=1,
    )

    # this previously raised with parametrized (range) graph properties
    datamodule.prepare_data()

    stores = list(Path(tmp_path).glob("**/edges_prune-none"))
    assert len(stores) == len(["seq1", "seq2", "seq3"])
    for store in stores:
        assert {p.name for p in store.glob("dt=*")} == expected_partitions

    # the precomputed store loads and assembles a graph
    datamodule.setup("fit")
    assert len(datamodule.dataset_val) > 0
    assert datamodule.dataset_val[0].num_edges > 0


def test_resolve_param_and_time_steps():
    """_resolve_param expands int, (min, max), and (min, max, step) specs."""
    # bare instance: these methods only read the attributes set below
    dataset = GraphDataset.__new__(GraphDataset)

    dataset.training = True
    assert dataset._resolve_param(5) == 5
    assert dataset._resolve_param(5, return_all_possible_values=True) == [5]

    result = dataset._resolve_param((2, 4))
    assert result in [2, 3, 4]
    assert np.array_equal(
        dataset._resolve_param((2, 4), return_all_possible_values=True),
        np.array([2, 3, 4]),
    )

    # 3-tuple adds an inclusive step
    assert dataset._resolve_param((1, 7, 2), return_all_possible_values=True) == [
        1,
        3,
        5,
        7,
    ]
    assert dataset._resolve_param((2, 8, 2)) in [2, 4, 6, 8]

    with pytest.raises(TypeError):
        dataset._resolve_param((1, 2, 3, 4))

    dataset.training = False
    assert dataset._resolve_param((2, 4)) == 2
    assert dataset._resolve_param((2, 8, 2)) == 2  # lower bound, deterministic

    dataset.graph_time_step = 2
    dataset.graph_num_steps = 3
    dataset.training = True
    assert dataset.time_steps(return_all_combinations=True) == [2, 4, 6]
    assert dataset.time_steps(return_all_combinations=False) == [2, 4, 6]

    dataset.graph_time_step = (1, 3)
    dataset.graph_num_steps = 2
    expected_all = [1, 2, 3, 4, 6]
    assert dataset.time_steps(return_all_combinations=True) == expected_all
    possible_single = [[1, 2], [2, 4], [3, 6]]
    assert dataset.time_steps(return_all_combinations=False) in possible_single


def test_maybe_sample_edges_caps_only_during_training():
    """The edge cap applies during training only, eval keeps the full graph."""
    # bare instance: _maybe_sample_edges only reads the attributes set below
    dataset = GraphDataset.__new__(GraphDataset)
    dataset.graph_edge_dropout = 0.0
    dataset.graph_max_num_edges = 10

    edges = pl.DataFrame({"src": np.arange(100), "dst": np.arange(100)})

    dataset.training = True
    assert dataset._maybe_sample_edges(edges).height == 10

    dataset.training = False
    assert dataset._maybe_sample_edges(edges).height == 100


def test_datamodule_val_graph_overrides_and_batch_size(
    tmp_path: Path, toy_handcrafted_features, toy_masks
):
    """val/test/predict use `_val` fallbacks for sampled graph params and batch size."""
    sequences = {"train": ["seq1"], "val": ["seq2"]}
    datamodule = create_mock_datamodule(
        tmp_path,
        sequences,
        toy_handcrafted_features,
        toy_masks,
        batch_size=4,
        batch_size_val=1,
        graph_search_radius=200,
        graph_search_radius_val=80,
        graph_time_step=1,
        graph_time_step_val=2,
        graph_num_steps=3,
        graph_num_steps_val=1,
    )
    datamodule.setup("fit")

    assert datamodule.dataset_train.graph_search_radius == 200
    assert datamodule.dataset_train.graph_time_step == 1
    assert datamodule.dataset_train.graph_num_steps == 3

    assert datamodule.dataset_val.graph_search_radius == 80
    assert datamodule.dataset_val.graph_time_step == 2
    assert datamodule.dataset_val.graph_num_steps == 1

    assert datamodule.train_dataloader().batch_size == 4
    assert datamodule.val_dataloader().batch_size == 1


@pytest.mark.parametrize(
    "missing_frames, expect_empty_items",
    (([1], False), ([0, 1], False), ([1, 2], True)),
)
@pytest.mark.parametrize("batch_size", (1, 3))
def test_tracking_datamodule_empty_batch(
    tmp_path: Path,
    toy_features_with_empty_frame_factory,
    toy_masks,
    missing_frames,
    expect_empty_items,
    batch_size,
):
    """Frames left without a graph are batched as empty template graphs, not dropped."""
    sequences = {"train": ["seq1"], "val": ["seq2"]}
    handcrafted_features = toy_features_with_empty_frame_factory(missing_frames)

    datamodule = create_mock_datamodule(
        tmp_path, sequences, handcrafted_features, toy_masks, batch_size=batch_size
    )
    datamodule.setup("fit")

    empty_flags = [
        bool(flag)
        for batch in datamodule.val_dataloader()
        for flag in np.atleast_1d(batch.empty_flag)
    ]
    assert len(empty_flags) == len(datamodule.dataset_val)
    assert any(empty_flags) == expect_empty_items


@pytest.mark.parametrize(
    "missing_frames, time_steps_to_check, expected_edges",
    [
        ([2], [1, 2], [(0, 1), (1, 3)]),  # one missing, skip
        ([3], [1], [(0, 1), (1, 2)]),  # one missing, no skip
        ([1, 2], [1], []),  # two missing, no skip
        ([1, 2], [1, 2, 3], [(0, 3)]),  # two missing, skip
    ],
)
def test_dataset_with_missing_frames(
    tmp_path: Path,
    toy_features_with_empty_frame_factory,
    toy_masks: np.ndarray,
    missing_frames,
    time_steps_to_check,
    expected_edges,
):
    """Missing frames are bridged by a larger stride, or raise when none can reach."""
    node_feats = toy_features_with_empty_frame_factory(remove_frames=missing_frames)
    seq_id = f"missing_{'_'.join(map(str, missing_frames))}"

    existing_frames = node_feats["t"].unique().to_list()
    masks_with_empty = np.array(
        [
            toy_masks[t] if t in existing_frames else np.zeros_like(toy_masks[t])
            for t in range(len(toy_masks))
        ]
    )

    sequence_data = {seq_id: {"node_features": node_feats, "masks": masks_with_empty}}
    data_dir, feature_dir = prepare_dataset_structure(tmp_path, sequence_data)

    def _create_dataset():
        return GraphDataset(
            data_dir=data_dir,
            feature_dir=feature_dir,
            sequence_id=seq_id,
            edge_finder=_edge_finder(prune_edges_by=None),
            handcrafted_feature_extractor=HandcraftedExtractor(
                extra_props=["center_local"],
                feature_norm_fn="scale_relative_size",
                n_jobs=1,
            ),
            deep_feature_extractor=None,
            data_format="ctc",
            segmentation_name="GT",
            graph_search_radius=100,
            graph_time_step=min(time_steps_to_check),
            graph_num_steps=len(time_steps_to_check),
            precompute_edges=True,
        )

    if len(expected_edges) == 0:
        with pytest.raises(ValueError):
            _create_dataset()
        return

    dataset = _create_dataset()
    graph = collect(dataset.edge_data)
    assert graph is not None

    node_times = dataset.node_feats.select(["index", "t"])
    edges_with_time = graph.join(node_times, left_on="src", right_on="index").join(
        node_times, left_on="dst", right_on="index", suffix="_dst"
    )

    actual_edges = set(edges_with_time.select("t", "t_dst").rows())
    assert actual_edges == set(expected_edges)


def test_graph_patch_dataset(single_sequence_data, basic_extractors):
    """Indexing runs over (frame, patch) pairs and each patch assembles its own graph."""
    edge_finder, hc_extractor = basic_extractors
    dataset = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        cached=True,
        cls=GraphPatchDataset,
        patch_size=64,
        patch_overlap=0.0,
    )

    assert len(dataset) > 0
    assert len(dataset.patch_indices) == len(dataset)  # type: ignore

    graph_item = dataset[0]
    assert isinstance(graph_item, Data)
    assert not graph_item.empty_flag
    for k in ["x_handcrafted", "edge_index", "y_edges", "y_nodes", "num_nodes"]:
        assert k in graph_item
    assert graph_item.x_handcrafted.shape[0] == graph_item.num_nodes


def test_patch_dataset_covers_last_frame_with_edge_store(
    single_sequence_data, basic_extractors
):
    """Patches must cover the last frame even when the edge store is forward-only."""
    _, hc_extractor = basic_extractors
    dataset = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        _edge_finder(bidirectional=True),
        cached=True,
        cls=GraphPatchDataset,
        precompute_edges=True,
        patch_size=64,
        patch_overlap=0.0,
    )

    # a forward-only store is the precondition for the regression
    assert dataset._mirror_per_item
    last_frame = int(dataset.valid_frames.max())
    assert last_frame in {t for t, _ in dataset.patch_indices}  # type: ignore


def test_patch_dataset_filters_each_frame_once(
    single_sequence_data, basic_extractors, monkeypatch
):
    """All patches of a frame share one edge lookup instead of refiltering per patch."""
    edge_finder, hc_extractor = basic_extractors
    dataset = _graph_dataset(
        single_sequence_data,
        hc_extractor,
        edge_finder,
        cached=True,
        cls=GraphPatchDataset,
        precompute_edges=True,
        patch_size=64,
        patch_overlap=0.0,
    )

    first_frame = dataset.patch_indices[0][0]
    patches = [i for i, (t, _) in enumerate(dataset.patch_indices) if t == first_frame]
    assert len(patches) > 1, "need several patches per frame to test the memo"

    calls = []
    original = dataset._build_edge_filter

    def counting_filter(t, time_steps):
        calls.append(t)
        return original(t, time_steps)

    monkeypatch.setattr(dataset, "_build_edge_filter", counting_filter)
    dataset._graph_filter_cache.clear()
    dataset._frame_edge_memo = None  # warmed by the template graph during __init__

    for idx in patches:
        dataset[idx]

    assert calls == [first_frame]

    # the filtered edges themselves are memoized too, not just the predicate
    steps = dataset.time_steps(return_all_combinations=False)
    edges = dataset._get_graph_for_frame(first_frame, steps)
    assert dataset._get_graph_for_frame(first_frame, steps) is edges


def test_from_images_and_masks_precomputes_edges(toy_images, toy_masks, basic_extractors):
    """The array entry point precomputes edges in memory unless a cache is requested."""
    edge_finder, hc_extractor = basic_extractors

    dataset = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        graph_search_radius=50,
        graph_num_steps=1,
    )

    assert dataset.precompute_edges
    assert isinstance(dataset.edge_data, pl.DataFrame)
    assert dataset._feature_tempdir is None

    on_the_fly = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        graph_search_radius=50,
        graph_num_steps=1,
        precompute_edges=False,
    )
    assert on_the_fly.edge_data is None


def test_from_images_and_masks_temp_cache(toy_images, toy_masks, basic_extractors):
    """'temp' writes a parquet store scanned lazily and removed with the dataset."""
    edge_finder, hc_extractor = basic_extractors

    dataset = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        sequence_id="toy",
        feature_dir="temp",
        graph_search_radius=50,
        graph_num_steps=1,
    )

    assert dataset._feature_tempdir is not None
    cache_root = Path(dataset._feature_tempdir.name)
    assert isinstance(dataset.edge_data, pl.LazyFrame)
    # node features and the percentiles are cached alongside the edges
    hc_dir = cache_root / "toy" / "seg"
    assert (hc_dir / "nodes.parquet").exists()
    assert (hc_dir / "percentiles.json").exists()

    # items still assemble from the lazily scanned store
    assert dataset[1] is not None

    del dataset
    gc.collect()
    assert not cache_root.exists()


def test_persistent_cache_invalidates_on_different_masks(
    toy_images, toy_masks, basic_extractors, tmp_path
):
    """A persistent `feature_dir` reused with different masks rebuilds, not reuses."""
    edge_finder, hc_extractor = basic_extractors

    first = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        sequence_id="toy",
        feature_dir=tmp_path,
        graph_search_radius=50,
        graph_num_steps=1,
    )
    first_labels = set(first.node_feats.filter(t=0)["label"].to_list())

    other_masks = toy_masks.copy()
    other_masks[0, 0:5, 0:5] = int(other_masks.max()) + 1

    second = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=other_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        sequence_id="toy",
        feature_dir=tmp_path,
        graph_search_radius=50,
        graph_num_steps=1,
    )
    second_labels = set(second.node_feats.filter(t=0)["label"].to_list())

    assert second_labels != first_labels


def test_trust_cache_skips_the_identity_check(
    toy_images, toy_masks, basic_extractors, tmp_path
):
    """`trust_cache=True` is the documented escape hatch: it reuses a stale cache."""
    edge_finder, hc_extractor = basic_extractors

    first = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        sequence_id="toy",
        feature_dir=tmp_path,
        graph_search_radius=50,
        graph_num_steps=1,
        trust_cache=True,
    )
    assert first.dataset_meta is None

    other_masks = toy_masks.copy()
    other_masks[0, 0:5, 0:5] = int(other_masks.max()) + 1

    second = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=other_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        sequence_id="toy",
        feature_dir=tmp_path,
        graph_search_radius=50,
        graph_num_steps=1,
        trust_cache=True,
    )

    # the stale cache from `first` is reused despite the different mask content
    assert second.node_feats.filter(t=0)["label"].to_list() == (
        first.node_feats.filter(t=0)["label"].to_list()
    )


def test_cached_deep_features_cover_every_item(toy_images, toy_masks, basic_extractors):
    """A cached run builds the deep cache for the whole sequence before any item.

    Regression: with a feature dir but no `prepare_data`, embeddings were extracted per
    temporal window and each window overwrote `embeddings.npy` with its own rows, so the
    next window read past the end of the file.
    """

    class _MeanEncoder(torch.nn.Module):
        arch = "MeanEncoder"

        def forward(self, imgs):
            return imgs.mean(dim=(2, 3))

    edge_finder, hc_extractor = basic_extractors
    deep_extractor = CellLevelExtractor(
        encoder=_MeanEncoder(), input_size_enc=32, verbose=False
    )

    dataset = GraphDataset.from_images_and_masks(
        images=toy_images,
        masks=toy_masks,
        edge_finder=edge_finder,
        handcrafted_feature_extractor=hc_extractor,
        deep_feature_extractor=deep_extractor,
        sequence_id="toy",
        feature_dir="temp",
        graph_search_radius=50,
        graph_num_steps=1,
    )

    n_nodes = int(dataset.node_feats["index"].to_numpy().max()) + 1
    assert deep_extractor.cache_covers(dataset.deep_feat_dir, n_nodes)

    reference = deep_extractor(
        image=toy_images,
        node_feats=dataset.node_feats,
        output_dir=None,
        image_percentiles=dataset.image_percentiles,
        spacing=dataset._spatial_spacing,
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item.empty_flag:
            continue
        assert torch.allclose(item.x_deep, reference[item.node_mapping], atol=1e-2)


def test_train_sampler_weighting_schemes():
    """Per-sequence sampling mass follows size ** alpha with an optional floor."""
    dm = TrackingDataset.__new__(TrackingDataset)
    dm.sampling_size_measure = "frames"

    class _Stub:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    # short sequence (10 items) combined with a long one (90 items)
    concat = ConcatDataset([_Stub(10), _Stub(90)])

    def seq_mass(alpha, floor=0.0):
        dm.sampling_alpha = alpha
        dm.sampling_floor = floor
        sampler = dm._get_train_sampler(concat)
        w = np.asarray(sampler.weights, dtype=float)
        w = w / w.sum()
        return float(w[:10].sum()), float(w[10:].sum())  # (short, long)

    # alpha 0: equal mass per sequence regardless of length
    short, long = seq_mass(0.0)
    assert short == pytest.approx(0.5)
    assert long == pytest.approx(0.5)

    # alpha 1 with frames measure: proportional to length -> uniform per item
    short, long = seq_mass(1.0)
    assert short == pytest.approx(0.1)
    assert long == pytest.approx(0.9)

    # alpha 0.5: between the two extremes
    short_half, _ = seq_mass(0.5)
    assert 0.1 < short_half < 0.5

    # floor lifts the rare short sequence above its proportional share
    short_floor, _ = seq_mass(1.0, floor=0.3)
    assert short_floor > 0.1
