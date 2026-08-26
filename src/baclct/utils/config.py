"""Hydra and OmegaConf helpers for loading, resolving, and instantiating configs."""

from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import hydra
from hydra.core.global_hydra import GlobalHydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from baclct.io import get_sequences_from_path, get_sequences_from_split
from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# packaged configs, resolved relative to this file (src/baclct/utils -> src/baclct/config)
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
INFERENCE_CONFIG = CONFIG_DIR / "inference.yaml"

# tells "the config has no such key" apart from "the config sets it to None"
_ABSENT = object()

# The flat inference surface: one name per runtime parameter, and the path it drives.
# Users override these by name and never have to know the config tree
# (`config_overrides={'num_jobs_features': 4}`). Extend by adding a row here and a default
# with its rationale to `config/inference.yaml`; nested overrides keep working alongside.
RUNTIME_KEYS: dict[str, str] = {
    "num_workers_predict": "data.num_workers",
    "num_workers_encode": "features.deep.num_workers",
    "num_jobs_features": "features.handcrafted.n_jobs",
    "num_jobs_edges": "features.edges.n_jobs",
    "batch_size": "data.batch_size",
}

# keys from the experiment config that config_overrides must not override
# they are tied to the trained model's architecture and input dimensionality
_EXPERIMENT_LOCKED_KEYS = [
    # model architecture
    "model",
    "hidden_dim",
    "num_edge_classes",
    "num_node_classes",
    # feature extractors and names
    "data.handcrafted_feature_extractor",
    "data.deep_feature_extractor",
    "data.edge_finder.feature_names",
    "data.edge_finder.extra_features",
    # training split used to produce the model
    "fold",
]

# leaf keys inside a locked node that may still be overridden at predict time. these only
# affect throughput and placement, never the feature values a locked extractor produces,
# so restoring them would only pin runtime parameters to whatever training happened to
# use. allowlist by design: anything new is locked until it is shown to be value-neutral.
#
# these are the extractor's own keys (`data.deep_feature_extractor`,
# `data.handcrafted_feature_extractor`), not the identically named top-level ones.
# `data.batch_size` (subgraphs per batch) and `data.num_workers` sit outside every locked
# node and are already freely overridable.
_RUNTIME_OVERRIDABLE_KEYS = frozenset(
    {
        "batch_size",  # patches per encoder forward
        "num_workers",  # dataloader workers cropping patches
        "cells_per_item",  # cells per worker item
        "prefetch_factor",  # items queued per worker
        "n_jobs",  # joblib threads (handcrafted regionprops)
        "verbose",  # show progress bars
        "device",  # encoder placement
    }
)


def _expand_runtime_keys(layer: DictConfig, train_cfg: DictConfig) -> DictConfig:
    """Layer with every flat runtime key rewritten onto the path it drives.

    A key whose component the model does not have is dropped rather than created:
    `num_workers_encode` on a model trained without deep features would otherwise merge
    onto `features.deep: null` and revive it as a config nothing can be built from.
    """
    expanded = layer.copy()
    OmegaConf.set_struct(expanded, False)
    for key, path in RUNTIME_KEYS.items():
        if key not in expanded:
            continue
        value = expanded.pop(key)
        parent = path.rpartition(".")[0]
        if parent and OmegaConf.select(train_cfg, parent, default=_ABSENT) is None:
            logger.debug(f"Model has no '{parent}', ignoring '{key}'")
            continue
        OmegaConf.update(expanded, path, value, merge=True)
    return expanded


def _restore_locked_node(old_val, new_val):
    """Locked node with the runtime parameters carried over from the override.

    Everything that shapes the extracted features comes from the experiment config; only
    the keys in `_RUNTIME_OVERRIDABLE_KEYS` are taken from the inference-time override.
    """
    if not isinstance(old_val, DictConfig) or not isinstance(new_val, DictConfig):
        return old_val

    restored = old_val.copy()
    OmegaConf.set_struct(restored, False)  # the override may add a key training never set
    for key in _RUNTIME_OVERRIDABLE_KEYS:
        if key in new_val:
            restored[key] = new_val[key]
    return restored


@contextmanager
def _isolated_hydra():
    """Compose without disturbing an already-initialized Hydra.

    A notebook, a test, or a host application may hold its own `GlobalHydra` instance.
    Composing inside it would either raise or leak our config path into it, so the
    surrounding instance is set aside and put back afterwards.

    Alternative is to just override configs relying on `GlobalHydra`, specifically:
    `paths.output_dir` (takes `hydra:runtime.output_dir`) and `paths.work_dir` (takes
    `hydra:runtime.cwd`). These configs are only relevant for checkpointing during
    training.
    """
    global_hydra = GlobalHydra.instance()
    saved = global_hydra.hydra
    if saved is not None:
        global_hydra.clear()
    try:
        yield
    finally:
        global_hydra.clear()
        if saved is not None:
            global_hydra.initialize(saved)


def compose_package_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the packaged `default.yaml` with Hydra.

    `dataset` and `task` are mandatory in that config, so at least those two must be
    given as overrides (e.g. `['dataset=toiam', 'task=tracking']`).
    """
    with _isolated_hydra():
        # config_path is relative to this file (src/baclct/utils) -> src/baclct/config
        hydra.initialize(version_base=None, config_path="../config")
        return hydra.compose(config_name="default", overrides=list(overrides or []))


def resolve_data_config(
    cfg, phase: list[str] | Literal["train", "val", "test", "predict"] | None = "train"
):
    """Resolve data parameters for single or combined datasets.

    Unified logic that iterates over one (single mode) or multiple (combined mode)
    dataset configurations to aggregate paths and sequences.
    """
    included = cfg.get("included_datasets", [])
    configs = []

    # configs are processed in list for compatibility with single or multiple data sources
    if not included:
        configs = [cfg]
    else:
        # load specific sub-configs for combined datasets
        for ds_name in included:
            configs.append(
                compose_package_config(
                    # task is mandatory, pass task from base conf
                    [f"dataset={ds_name}", f"task={cfg.task_config_name}"]
                )
            )

    combined_kwargs = defaultdict(list)
    is_predict = phase == "predict"
    requested_seqs = cfg.get("sequences") if is_predict else None

    requested_seqs_list = None
    if requested_seqs is not None:
        if isinstance(requested_seqs, str):
            requested_seqs_list = [requested_seqs]
        else:
            requested_seqs_list = list(requested_seqs)

    for ds_cfg in configs:
        data_dir = Path(ds_cfg.paths.data_dir).resolve()
        split_path = data_dir / "splits.yaml"

        # resolve sequences based on phase and availability
        if is_predict and requested_seqs_list is not None:
            if all(s in ["train", "val", "test"] for s in requested_seqs_list):
                current_seqs = get_sequences_from_split(
                    split_path, cfg.fold, phase=requested_seqs_list
                )
            else:
                current_seqs = requested_seqs_list
        elif not is_predict and split_path.exists():
            # not optimal, but will get all sequences so that we can use train/val/test
            # within a single pipeline object
            current_seqs = get_sequences_from_split(split_path, cfg.fold)
        else:
            # fallback: load all from directory (predict=all, or no split file found)
            current_seqs = get_sequences_from_path(data_dir)

        if not current_seqs:
            continue

        # aggregate paths and params (broadcast to length of sequences), required for
        # multi-dataset training/inference
        count = len(current_seqs)
        combined_kwargs["sequence_ids"].extend(current_seqs)
        combined_kwargs["data_dir"].extend([data_dir] * count)
        combined_kwargs["feature_dir"].extend([Path(ds_cfg.paths.feature_dir)] * count)

        for key in ["segmentation_name", "data_format", "img_name"]:
            combined_kwargs[key].extend([ds_cfg.data.get(key)] * count)

    return dict(combined_kwargs)


def _as_dictconfig(overrides: DictConfig | dict | list[str] | str | Path) -> DictConfig:
    """Accept a YAML path, a dotlist, a plain dict, or a `DictConfig` as overrides."""
    # DictConfig first: it is list-like enough that an `isinstance(_, list)` test would
    # not tell the two apart
    if isinstance(overrides, DictConfig):
        return overrides
    if isinstance(overrides, (str, Path)):
        path = Path(overrides).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Override config not found at {path}")
        loaded = OmegaConf.load(path)
        assert isinstance(loaded, DictConfig), f"{path} must hold a mapping."
        return loaded
    if isinstance(overrides, dict):
        return OmegaConf.create(overrides)
    return OmegaConf.from_dotlist([str(item) for item in overrides])


def resolve_and_merge_configs(
    config: DictConfig | str | Path,
    config_overrides: DictConfig | dict | list[str] | str | Path | None = None,
    inference: bool = False,
) -> DictConfig:
    """Load a config and merge overrides on top of it.

    The base config can be a `DictConfig`, a path to a directory (loads
    `.hydra/config.yaml`), or a path to a YAML file. Overrides can be a `DictConfig`, a
    plain dict, a dotlist of 'key=value' strings, or a path to a YAML file.

    With `inference`, the packaged `inference.yaml` is merged in first, so a checkpoint's
    training values for the runtime parameters give way to inference-appropriate ones
    before the caller's overrides are applied. Keys in `_EXPERIMENT_LOCKED_KEYS` are then
    restored from the base config, so no override can pair a checkpoint with a feature
    setup or an architecture it was not trained on.

    Every layer may address a runtime parameter by its flat name from `RUNTIME_KEYS`
    instead of its position in the config tree.
    """
    assert config is not None, "Please provide a proper config or path. Received `None`."

    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        if config_path.is_dir():
            config_path = config_path / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)
    else:
        cfg = config.copy()

    layers = []
    if inference:
        inference_cfg = OmegaConf.load(INFERENCE_CONFIG)
        assert isinstance(inference_cfg, DictConfig)
        layers.append(inference_cfg)
    if config_overrides is not None:
        layers.append(_as_dictconfig(config_overrides))
    if not layers:
        return cfg

    train_cfg = cfg.copy()  # save to restore locked keys after merge
    # experiment configs saved by Hydra are in struct mode. disable it so that overrides
    # can introduce new top-level keys (checkpoint, device)
    OmegaConf.set_struct(cfg, False)
    for layer in layers:
        cfg = OmegaConf.merge(cfg, _expand_runtime_keys(layer, train_cfg))
    assert isinstance(cfg, DictConfig)

    for key in _EXPERIMENT_LOCKED_KEYS:
        old_val = OmegaConf.select(train_cfg, key)
        new_val = OmegaConf.select(cfg, key)
        if old_val is None or old_val == new_val:
            continue
        restored = _restore_locked_node(old_val, new_val)
        if restored == new_val:
            # only runtime parameters differed: leave the node alone so a config that
            # reaches the extractor by interpolation (${features.deep}) stays intact
            continue
        logger.debug(f"Restoring experiment-locked key: {key}")
        # .update() with merge=False fully replaces the node
        OmegaConf.update(cfg, key, restored, merge=False)

    return cfg


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiate every entry of a callbacks config that carries a `_target_`.

    Adapted from github.com/ashleve/lightning-hydra-template.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate every entry of a logger config that carries a `_target_`.

    Adapted from github.com/ashleve/lightning-hydra-template.
    """
    loggers: list[Logger] = []

    if not logger_cfg:
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def _epoch_of(path: Path) -> int:
    """Parse the epoch index from a checkpoint filename, or -1 if absent."""
    match = re.search(r"epoch[=_](\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(
    checkpoint: Path | str | Literal["best", "last", "last_epoch", "epoch"] | None,
    experiment_dir: Path,
    descending: bool = True,
) -> Path:
    """Get checkpoint from path or find checkpoints of previous run.

    Args:
        checkpoint: Path to checkpoint, checkpoint name, or metric_name.
        experiment_dir: Path to lightning experiment.
        descending: Order of checkpoints if searching for metrics.
    """
    literals = ["best", "last", "last_epoch", "epoch"]
    experiment_dir = Path(experiment_dir)
    available_checkpoints = sorted(experiment_dir.rglob("*.ckpt"))

    if not available_checkpoints:
        raise FileNotFoundError(f"Could not find any checkpoints in {experiment_dir}.")

    # explicit path check. ignores subdirs like checkpoints/interval
    if isinstance(checkpoint, (str, Path)) and str(checkpoint) not in literals:
        raw = Path(checkpoint)
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(experiment_dir / raw)
            candidates.append(experiment_dir / "checkpoints" / raw.name)
        if raw.suffix != ".ckpt":
            candidates += [c.with_suffix(".ckpt") for c in list(candidates)]

        for cand in candidates:
            if cand.exists():
                return cand

        if str(checkpoint).endswith(".ckpt"):
            raise FileNotFoundError(
                f"Could not find checkpoint {checkpoint}. "
                f"Available: {available_checkpoints}"
            )

    # metric search
    if isinstance(checkpoint, str) and checkpoint not in literals:
        ckpt_paths = [
            ckpt
            for ckpt in experiment_dir.rglob(f"*{checkpoint}=*")
            if ckpt.parent.name == "checkpoints"
        ]

        if ckpt_paths:
            ckpt_scores = [
                (float(match.group(1)), ckpt)
                for ckpt in ckpt_paths
                if (match := re.search(rf"{checkpoint}=([\d\.]+)", ckpt.name))
            ]
            if ckpt_scores:
                ckpt_scores.sort(key=lambda x: x[0], reverse=descending)
                return ckpt_scores[0][1]

        raise FileNotFoundError(
            f"Could not find checkpoint for metric '{checkpoint}'. "
            f"Available: {available_checkpoints}"
        )

    # "best": look for the best.ckpt symlink created during training
    if checkpoint == "best":
        ckpt_paths = [
            ckpt
            for ckpt in experiment_dir.rglob("best.ckpt")
            if ckpt.parent.name == "checkpoints"
        ]
        if ckpt_paths:
            return ckpt_paths[0]
        logger.warning(
            f"Checkpoint 'best' requested but best.ckpt not found in "
            f"{experiment_dir}. Falling back to 'last'."
        )

    # "last" or None: symlink to last saved checkpoint (e.g., top3 metrics)
    if checkpoint in ("last", "best") or checkpoint is None:
        ckpt_paths = [
            ckpt
            for ckpt in experiment_dir.rglob("last.ckpt")
            if ckpt.parent.name == "checkpoints"
        ]
        if ckpt_paths:
            return ckpt_paths[0]

    # "last_epoch": highest-epoch checkpoint anywhere,
    # including subdirs like checkpoints/interval
    if checkpoint == "last_epoch":
        epoch_ckpts = [
            ckpt for ckpt in experiment_dir.rglob("epoch*.ckpt") if _epoch_of(ckpt) >= 0
        ]
        if epoch_ckpts:
            return max(epoch_ckpts, key=_epoch_of)

    # "epoch" fallback (or if 'last' was requested but missing)
    ckpt_paths = sorted(
        [
            ckpt
            for ckpt in experiment_dir.rglob("epoch*.ckpt")
            if ckpt.parent.name == "checkpoints"
        ],
        key=_epoch_of,
        reverse=True,
    )
    if ckpt_paths:
        return ckpt_paths[0]

    raise FileNotFoundError(
        f"Could not find valid checkpoints in {experiment_dir}. "
        f"Available: {available_checkpoints}"
    )
