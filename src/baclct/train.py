"""Hydra CLI entry point for training the BacLCT."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import baclct
from baclct import BacLCT
from baclct.utils import hydra_compat  # noqa: F401  (patches argparse for hydra's parser)
from baclct.utils.data import set_multiprocessing_context
from baclct.utils.logger import configure_lightning_loggers, get_pylogger

logger = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    """Train a BacLCT tracking (and optionally classification) model.

    Hydra entry point (`baclct-train`). Composes the full config from
    `config/default.yaml` plus CLI overrides, then runs the Lightning training loop via
    `BacLCT.run_training`. At minimum, `dataset` and `task` must be specified (e.g.,
    `dataset=spores task=tracking_with_classification`). These can also be set in a
    global experiment config (e.g., `experiment=full`).

    After training, a `best.ckpt` symlink is created in the checkpoints directory.
    Training is bare-bone: it runs the loop and returns nothing, and errors propagate.
    Hyperparameter sweeps and tracking-based objectives belong in a dedicated runner that
    drives the pipeline (and reads metrics off `run_training`/the trainer) directly.
    """
    configure_lightning_loggers()
    set_multiprocessing_context()  # defaults to forkserver on unix
    torch.set_float32_matmul_precision("high")
    logger.info(f"Version: {baclct.__version__}")

    if "task_configured" not in cfg:
        raise ValueError(
            "Task was not configured. Please use "
            "`task=tracking` or `task=tracking_with_states`.\n"
            f"Full config:\n{OmegaConf.to_yaml(cfg)}"
        )

    L.seed_everything(cfg.get("seed", 1510), workers=True)
    pipeline = BacLCT(cfg)
    checkpoint, metrics = pipeline.run_training()

    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        logger.info(f"Run saved to: {run_dir}")
    except Exception:
        pass

    if checkpoint:
        best_ckpt = Path(checkpoint)
        logger.info(f"Best checkpoint: {best_ckpt.name}")
        # create a `best.ckpt` symlink so inference can select checkpoint="best"
        symlink = best_ckpt.parent / "best.ckpt"
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(best_ckpt.name)

    if metrics is not None:
        f1_val = metrics.get("val/f1_score")
        if f1_val is not None:
            logger.info(f"val/f1_score: {float(f1_val):.4f}")


if __name__ == "__main__":
    main()
