"""End-to-end tests of experiment runs (slow).

Several commonly used configs are unified in `baclct.config.experiment` (e.g., tracking-
only methods and several other ablations). Each one gets a short train-then-track loop,
so a config whose components no longer fit together fails here.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from baclct.api import BacLCT
from baclct.data.dataset import GraphPatchDataset
from baclct.io import load_images_and_masks

# no numbers are asserted here, and strict determinism raises on MPS
_DETERMINISM_OVERRIDE = "trainer.deterministic=warn"

# query Hydra's config search path directly, so no filesystem layout is assumed
with hydra.initialize_config_module(
    config_module="baclct.config", version_base="1.3", job_name="collect_experiments"
):
    _EXPERIMENT_NAMES = sorted(
        GlobalHydra.instance().config_loader().get_group_options("experiment")
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "features_config, model_config, patch_size",
    [
        ("dino_only", "deep", None),
        ("handcrafted_only", "handcrafted", 64),
    ],
)
@pytest.mark.parametrize("task", ["tracking", "tracking_with_states"])
def test_config_variant_runs(
    reproducibility_data_dir, features_config, model_config, patch_size, task
):
    """Core component combinations stay compatible over a train-and-track cycle.

    `patch_size` tiles each frame instead of building one whole-frame graph, on both the
    training and the inference path.
    """
    data_dir, feature_dir = reproducibility_data_dir
    run_dir = data_dir.parent / f"run_{features_config}_{model_config}"

    overrides = [
        "dataset=spores",
        f"task={task}",
        "data.precompute_edges=true",
        f"paths.output_dir={run_dir!s}",
        f"paths.data_dir={data_dir!s}",
        f"paths.feature_dir={feature_dir!s}",
        f"features={features_config}",
        f"model={model_config}",
        f"data.use_patches={patch_size is not None}",
        "debug=tests",
        "callbacks=tests",
        _DETERMINISM_OVERRIDE,
    ]
    if patch_size is not None:
        overrides.append(f"data.patch_size={patch_size}")

    with hydra.initialize_config_module(
        config_module="baclct.config",
        job_name="e2e_train_test",
        version_base="1.3",
    ):
        cfg_train = hydra.compose(config_name="default", overrides=overrides)

        # mimic @hydra.main by saving the config to disk for the prediction step
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg_train, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        pipeline_train = BacLCT(cfg_train)
        checkpoint_path, _ = pipeline_train.run_training()

    assert checkpoint_path is not None and checkpoint_path != ".", (
        "Training did not produce a checkpoint path."
    )
    assert Path(checkpoint_path).exists(), (
        f"Checkpoint file not found at {checkpoint_path}"
    )

    images, masks = load_images_and_masks(
        data_dir, "01", data_format="ctc", segmentation_name="GT", lazy=True
    )
    pipeline_pred = BacLCT(
        run_dir,
        config_overrides={
            "checkpoint": str(checkpoint_path),
            "data": {"graph_num_steps": cfg_train.data.graph_num_steps},
        },
    )
    pipeline_pred.track(
        images,
        masks,
        output_dir=run_dir / "tracking_results",
        sequence_id="01",
        export_format="ctc",
        patch_size=patch_size,
    )
    patched = isinstance(pipeline_pred.dataset, GraphPatchDataset)
    assert patched == (patch_size is not None)

    prediction_file = run_dir / "tracking_results" / "01" / "res_track.txt"
    assert prediction_file.exists(), f"Prediction file not found at {prediction_file}"


@pytest.mark.slow
@pytest.mark.parametrize("experiment", _EXPERIMENT_NAMES)
def test_experiment_config_train_and_predict(
    tmp_path: Path,
    reproducibility_data_dir,
    experiment: str,
):
    """Every experiment config trains a checkpoint and tracks with it.

    The configs are read from Hydra's `experiment` group, so the test follows whatever
    is in `config/experiment/`.
    """
    data_dir, _ = reproducibility_data_dir
    run_dir = tmp_path / f"run_{experiment}"
    feature_dir = tmp_path / f"features_{experiment}"

    overrides = [
        f"experiment={experiment}",
        f"paths.output_dir={run_dir!s}",
        f"paths.data_dir={data_dir!s}",
        f"paths.feature_dir={feature_dir!s}",
        "debug=tests",
        "callbacks=tests",
        _DETERMINISM_OVERRIDE,
    ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize_config_module(
        config_module="baclct.config",
        job_name=f"exp_train_{experiment}",
        version_base="1.3",
    ):
        cfg_train = hydra.compose(config_name="default", overrides=overrides)

        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg_train, hydra_dir / "config.yaml")

        L.seed_everything(1510, workers=True)
        pipeline_train = BacLCT(cfg_train)
        checkpoint_path, _ = pipeline_train.run_training()

    assert checkpoint_path is not None and checkpoint_path != ".", (
        f"[{experiment}] Training did not produce a checkpoint."
    )
    assert Path(checkpoint_path).exists(), (
        f"[{experiment}] Checkpoint not found at {checkpoint_path}"
    )

    images, masks = load_images_and_masks(
        data_dir, "01", data_format="ctc", segmentation_name="GT", lazy=True
    )
    pipeline_pred = BacLCT(
        run_dir,
        config_overrides={
            "checkpoint": str(checkpoint_path),
            "data": {"graph_num_steps": cfg_train.data.graph_num_steps},
        },
    )
    pipeline_pred.track(
        images,
        masks,
        output_dir=run_dir / "tracking_results",
        sequence_id="01",
        export_format="ctc",
    )

    prediction_file = run_dir / "tracking_results" / "01" / "res_track.txt"
    assert prediction_file.exists(), (
        f"[{experiment}] Prediction file not found at {prediction_file}"
    )
