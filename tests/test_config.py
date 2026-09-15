"""Test the instantiation of hydra configs."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from baclct.utils.config import resolve_and_merge_configs, resolve_checkpoint


def test_resolve_last_epoch_includes_interval_dir(tmp_path):
    """'last_epoch' picks the numerically highest epoch, incl. an interval/ subdir."""
    ckpt_dir = tmp_path / "checkpoints"
    (ckpt_dir / "interval").mkdir(parents=True)
    (ckpt_dir / "last.ckpt").touch()
    (ckpt_dir / "epoch=090.ckpt").touch()
    (ckpt_dir / "epoch=100.ckpt").touch()
    # the interval series carries a later snapshot than the top-level top-k copies
    (ckpt_dir / "interval" / "epoch=120.ckpt").touch()

    resolved = resolve_checkpoint("last_epoch", tmp_path)
    assert resolved.name == "epoch=120.ckpt"
    assert resolved.parent.name == "interval"


def test_resolve_epoch_sorts_numerically(tmp_path):
    """The 'epoch' fallback sorts by epoch number, not reverse-alphabetically."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch=090.ckpt").touch()
    (ckpt_dir / "epoch=100.ckpt").touch()

    assert resolve_checkpoint("epoch", tmp_path).name == "epoch=100.ckpt"


@pytest.mark.parametrize("name", ["swa", "swa.ckpt", "checkpoints/swa.ckpt"])
def test_resolve_explicit_name_finds_checkpoints_subdir(tmp_path, name):
    """A bare checkpoint name resolves to <run>/checkpoints/<name>.ckpt."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch=010.ckpt").touch()
    (ckpt_dir / "swa.ckpt").touch()

    resolved = resolve_checkpoint(name, tmp_path)
    assert resolved.name == "swa.ckpt"


@pytest.fixture
def experiment_cfg():
    """Experiment config in the shape a Hydra training job produces.

    The extractors are reached by interpolation, and the config is in struct mode, which
    `resolve_and_merge_configs` has to open before an override can add a key.
    """
    cfg = OmegaConf.create(
        {
            "features": {
                "deep": {
                    "_target_": "baclct.features.extractors.CellLevelExtractor",
                    "input_size_enc": 224,
                    "batch_size": 256,
                    "normalize_for_pretrained": True,
                },
                "handcrafted": {
                    "_target_": "baclct.features.extractors.HandcraftedExtractor",
                    "props": ["area", "eccentricity"],
                    "n_jobs": -1,
                },
            },
            "data": {
                "deep_feature_extractor": "${features.deep}",
                "handcrafted_feature_extractor": "${features.handcrafted}",
                "num_workers": 8,  # what training used
            },
            "hidden_dim": 128,
        }
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("batch_size", 512),
        ("num_workers", 2),
    ],
)
def test_runtime_params_override_locked_extractor(experiment_cfg, key, value):
    """Runtime parameters reach the extractor even though it is experiment-locked.

    `num_workers` is absent from configs saved before it existed, so the override has to
    create the missing key.
    """
    merged = resolve_and_merge_configs(
        experiment_cfg, {"features": {"deep": {key: value}}}
    )
    assert merged.data.deep_feature_extractor[key] == value


def test_feature_defining_params_stay_locked(experiment_cfg):
    """Params that change the extracted values are restored from the experiment config."""
    merged = resolve_and_merge_configs(
        experiment_cfg,
        {
            "features": {
                "deep": {
                    "input_size_enc": 999,
                    "normalize_for_pretrained": False,
                    "batch_size": 512,
                }
            }
        },
    )
    ext = merged.data.deep_feature_extractor
    assert ext.input_size_enc == 224
    assert ext.normalize_for_pretrained is True
    assert ext.batch_size == 512  # the runtime parameter still gets through


def test_handcrafted_runtime_and_locked_params(experiment_cfg):
    """The same split applies to the handcrafted extractor."""
    merged = resolve_and_merge_configs(
        experiment_cfg,
        {"features": {"handcrafted": {"n_jobs": 3, "props": ["area"]}}},
    )
    hc = merged.data.handcrafted_feature_extractor
    assert hc.n_jobs == 3
    assert list(hc.props) == ["area", "eccentricity"]


def test_locked_keys_untouched_without_override(experiment_cfg):
    """An override adding only a new top-level key leaves the extractors alone."""
    merged = resolve_and_merge_configs(experiment_cfg, {"sequences": ["03"]})
    assert list(merged.sequences) == ["03"]
    assert merged.data.deep_feature_extractor.batch_size == 256
    assert merged.data.deep_feature_extractor.input_size_enc == 224


def test_inference_defaults_apply_and_yield_to_overrides(experiment_cfg):
    """The inference overlay replaces training throughput values, overrides beat both.

    Only the parameters `track()` does not expose are in the overlay. The exposed ones
    arrive as its runtime layer instead.
    """
    defaults_only = resolve_and_merge_configs(experiment_cfg, inference=True)
    assert defaults_only.data.num_workers == 0
    assert defaults_only.features.edges.n_jobs == 1
    assert "batch_size" not in defaults_only.data

    overridden = resolve_and_merge_configs(
        experiment_cfg,
        {"data": {"num_workers": 4}, "features": {"edges": {"n_jobs": 8}}},
        inference=True,
        runtime_overrides={"data": {"batch_size": 1}},
    )
    assert overridden.data.num_workers == 4
    assert overridden.data.batch_size == 1
    # the unexposed parameters stay reachable for on-the-fly edge finding
    assert overridden.features.edges.n_jobs == 8


def test_inference_defaults_keep_extractor_interpolation(experiment_cfg):
    """`features.*` overrides must not collapse the `${features.*}` interpolation.

    Merging into `data.handcrafted_feature_extractor` directly would replace the
    interpolation with a literal copy, silently detaching it from `features.handcrafted`.
    """
    merged = resolve_and_merge_configs(
        experiment_cfg,
        inference=True,
        runtime_overrides={"features": {"handcrafted": {"n_jobs": 1}}},
    )
    raw = OmegaConf.to_container(merged, resolve=False)
    assert raw["data"]["handcrafted_feature_extractor"] == "${features.handcrafted}"
    assert merged.features.handcrafted.n_jobs == 1


def test_runtime_overrides_win_over_config_overrides(experiment_cfg):
    """`track()`'s throughput arguments are explicit, so they beat a config layer."""
    merged = resolve_and_merge_configs(
        experiment_cfg,
        {"features": {"handcrafted": {"n_jobs": 4}}, "data": {"batch_size": 8}},
        inference=True,
        runtime_overrides={"features": {"handcrafted": {"n_jobs": 2}}},
    )

    assert merged.features.handcrafted.n_jobs == 2
    # a key the runtime layer does not carry is left to the config layer
    assert merged.data.batch_size == 8


def test_overrides_from_a_yaml_path(experiment_cfg, tmp_path):
    """A path is loaded as an override layer, so a user config file can be passed."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data:\n  batch_size: 8\n  num_workers: 3\n")

    merged = resolve_and_merge_configs(experiment_cfg, config_file, inference=True)
    assert merged.data.batch_size == 8
    assert merged.data.num_workers == 3

    with pytest.raises(FileNotFoundError):
        resolve_and_merge_configs(experiment_cfg, tmp_path / "missing.yaml")


def test_overrides_set_a_null_leaf(experiment_cfg):
    """Only whole sub-configs are pruned, so a null scalar is still overridable.

    `device` and `checkpoint` are null in a trained config and are set exactly this way.
    """
    merged = resolve_and_merge_configs(
        experiment_cfg, {"device": "cpu", "checkpoint": "best"}, inference=True
    )

    assert merged.device == "cpu"
    assert merged.checkpoint == "best"


def test_overrides_skip_a_component_the_model_lacks(experiment_cfg):
    """A model trained without deep features must not gain a half-built extractor.

    `features=handcrafted_only` records `features.deep: null`, and `_worker_plan` sets
    `num_workers` on it, which merging would revive as a partial config.
    """
    from baclct.api import _worker_plan

    experiment_cfg.features.deep = None

    merged = resolve_and_merge_configs(
        experiment_cfg, inference=True, runtime_overrides=_worker_plan(4)
    )
    assert merged.features.deep is None
    assert merged.data.deep_feature_extractor is None
    # the parameters the model does have still apply
    assert merged.features.handcrafted.n_jobs == 4


def test_compose_package_config_restores_global_hydra():
    """Composing must leave an already-initialized Hydra untouched (notebooks, tests)."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    from baclct.utils.config import compose_package_config

    with hydra.initialize_config_module(
        config_module="baclct.config", job_name="outer", version_base="1.3"
    ):
        outer = GlobalHydra.instance().hydra
        cfg = compose_package_config(["dataset=spores", "task=tracking"])
        assert cfg.dataset_name == "spores"
        # the surrounding instance is the same object, still usable
        assert GlobalHydra.instance().hydra is outer
        assert hydra.compose(
            config_name="default", overrides=["dataset=spores", "task=tracking"]
        )
