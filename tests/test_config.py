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
        ("cells_per_item", 64),
        ("prefetch_factor", 4),
    ],
)
def test_runtime_params_override_locked_extractor(experiment_cfg, key, value):
    """Runtime parameters reach the extractor even though it is experiment-locked.

    `num_workers`/`cells_per_item`/`prefetch_factor` are absent from configs saved before
    they existed, so the override has to introduce the key, not just change it.
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
    """The inference overlay replaces training throughput values, overrides beat both."""
    defaults_only = resolve_and_merge_configs(experiment_cfg, inference=True)
    assert defaults_only.data.num_workers == 0
    assert defaults_only.data.handcrafted_feature_extractor.n_jobs == 1

    overridden = resolve_and_merge_configs(
        experiment_cfg, {"data": {"num_workers": 4}}, inference=True
    )
    assert overridden.data.num_workers == 4
    # untouched keys keep the inference default rather than the training value
    assert overridden.data.batch_size == 1


def test_inference_defaults_keep_extractor_interpolation(experiment_cfg):
    """`features.*` overrides must not collapse the `${features.*}` interpolation.

    Merging into `data.handcrafted_feature_extractor` directly would replace the
    interpolation with a literal copy, silently detaching it from `features.handcrafted`.
    """
    merged = resolve_and_merge_configs(experiment_cfg, inference=True)
    raw = OmegaConf.to_container(merged, resolve=False)
    assert raw["data"]["handcrafted_feature_extractor"] == "${features.handcrafted}"
    assert merged.features.handcrafted.n_jobs == 1


def test_flat_runtime_keys_reach_their_config_paths(experiment_cfg):
    """A runtime parameter can be set by name, without knowing where it lives."""
    merged = resolve_and_merge_configs(
        experiment_cfg,
        {"num_jobs_features": 4, "num_workers_encode": 0, "num_workers_predict": 2},
        inference=True,
    )

    assert merged.features.handcrafted.n_jobs == 4
    assert merged.features.deep.num_workers == 0
    assert merged.data.num_workers == 2
    # the flat name is consumed, not left behind as a stray top-level key
    assert "num_jobs_features" not in merged


def test_overrides_from_a_yaml_path(experiment_cfg, tmp_path):
    """A path is loaded as an override layer, so a user config file can be passed."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("num_jobs_features: 4\ndata:\n  batch_size: 8\n")

    merged = resolve_and_merge_configs(experiment_cfg, config_file, inference=True)
    assert merged.features.handcrafted.n_jobs == 4
    assert merged.data.batch_size == 8

    with pytest.raises(FileNotFoundError):
        resolve_and_merge_configs(experiment_cfg, tmp_path / "missing.yaml")


def test_flat_runtime_keys_skip_a_component_the_model_lacks(experiment_cfg):
    """A model trained without deep features must not gain a half-built extractor.

    `features=handcrafted_only` records `features.deep: null`, and the overlay sets
    `num_workers_encode`, which merging onto that node would revive as a partial config.
    """
    experiment_cfg.features.deep = None

    merged = resolve_and_merge_configs(
        experiment_cfg, {"num_workers_encode": 4}, inference=True
    )
    assert merged.features.deep is None
    assert merged.data.deep_feature_extractor is None
    # the parameters the model does have still apply
    assert merged.features.handcrafted.n_jobs == 1


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
