"""Command line entry point for tracking a sequence with a trained model."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import yaml
from tqdm.auto import tqdm

from baclct.api import BacLCT
from baclct.io import get_sequences_from_path, load_images_and_masks
from baclct.utils.data import set_multiprocessing_context
from baclct.utils.pretrained import resolve_model_dir


def _configure_process() -> None:
    """Apply the process-wide settings this CLI needs, but a library import must not."""
    # `fork` (the default on linux) deadlocks with polars and the memory-mapped caches,
    # and windows offers `spawn` alone
    set_multiprocessing_context()
    # interpreter shutdown can close the log stream before the last records are flushed
    logging.raiseExceptions = False
    warnings.filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called in a multiprocessing",
        category=UserWarning,
    )


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    """Pick the sequence ids from the flags, the split file, or the dataset directory."""
    if args.sequences:
        return list(args.sequences)

    if args.split_file:
        if args.fold is None:
            raise ValueError("Must provide --fold when using --split-file.")
        with args.split_file.open() as file:
            return yaml.safe_load(file)[args.fold][args.phase]

    if args.data_format != "ctc":
        raise ValueError(
            "Sequence auto-discovery is only supported for --data-format ctc. "
            "Pass --sequences or --split-file for 'flat'/'dirs' layouts."
        )
    sequences = get_sequences_from_path(args.data_dir, data_format=args.data_format)
    if not sequences:
        raise ValueError(f"No sequences found in {args.data_dir}.")
    return sequences


def _resolve_search_radius(radius: str) -> int | str | None:
    """Resolve the radius flag, where 'trained' keeps the value from the model config."""
    if radius == "trained":
        return None
    try:
        return int(radius)
    except ValueError:
        return radius


def _resolve_prune_edges_by(
    method: str, param: str | None
) -> str | tuple[str, float | str] | None:
    """Normalize the pruning flags into what `BacLCT.track()` expects."""
    if method == "default":
        return None
    if method == "off":
        return "off"
    if method == "dilated_overlap":
        return (method, param if param is not None else "thickness")
    if method == "overlap":
        return (method, float(param) if param is not None else 0.1)
    return (method, float(param) if param is not None else 3.0)


def _build_track_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the flags `BacLCT.track()` takes as arguments."""
    return {
        "classify_states": args.classify_states,
        "graph_search_radius": _resolve_search_radius(args.graph_search_radius),
        "prune_edges_by": _resolve_prune_edges_by(args.prune_edges_by, args.prune_param),
        "patch_size": args.patch_size,
        "device": args.device,
        "export_format": args.export_format,
        "cache_dir": args.cache_dir,
    }


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the flags that are runtime config keys, leaving unset ones at the default.

    See `config/inference.yaml` for those defaults and what raising each one costs.
    """
    # encoder workers need an `if __name__ == '__main__'` guard, which this CLI has
    overrides: dict[str, Any] = {"num_workers_encode": 2}
    runtime = {
        "num_jobs_features": args.num_jobs_features,
        "num_workers_encode": args.num_workers_encode,
        "num_workers_predict": args.num_workers_predict,
        "batch_size": args.batch_size,
    }
    overrides.update({k: v for k, v in runtime.items() if v is not None})

    tracker: dict[str, Any] = {}
    if args.thr_corr is not None:
        tracker["thr_corr"] = args.thr_corr
    if args.thr_div is not None:
        tracker["thr_div"] = args.thr_div
    # the config holds a list of correction steps, so the flag can only switch them off
    if args.segmentation_correction is False:
        tracker["segmentation_correction"] = None
    if tracker:
        overrides["tracker"] = tracker

    return overrides


def _track_sequence(
    sequence_id: str,
    args: argparse.Namespace,
    experiment_dir: Path,
    overrides: dict[str, Any],
    track_kwargs: dict[str, Any],
) -> tuple[bool, str | None]:
    """Track a single sequence.

    Returns:
        Whether it succeeded, and a message to print if there is one.
    """
    results_dir = args.output_dir or args.data_dir / "baclct_results"
    output_dir = results_dir / args.segmentation_name
    seq_output_dir = output_dir / sequence_id

    if not args.overwrite and seq_output_dir.is_dir() and any(seq_output_dir.iterdir()):
        return True, "output already exists, skipping"

    try:
        images, masks, *_ = load_images_and_masks(
            args.data_dir,
            sequence_id,
            data_format=args.data_format,
            segmentation_name=args.segmentation_name,
            img_name=args.img_name,
            lazy=False,
        )
        output_dir.mkdir(exist_ok=True, parents=True)
    except Exception as err:
        return False, f"could not load ({args.segmentation_name}): {err}"

    try:
        pipeline = BacLCT(experiment_dir, config_overrides=overrides)
        pipeline.track(
            cast(np.ndarray, images),
            cast(np.ndarray, masks),
            output_dir=output_dir,
            sequence_id=sequence_id,
            **track_kwargs,
        )
    except Exception as err:
        return False, f"tracking failed: {err}"

    return True, None


def _build_parser() -> argparse.ArgumentParser:
    """Build the `baclct-track` parser."""
    parser = argparse.ArgumentParser(description="Run BacLCT.")

    # model and data
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Registered model name (e.g. 'baclct_track') or path to a model directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing image sequences.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequence ids to operate on, space-separated (e.g. --sequences 01 02). "
        "Defaults to every sequence found in --data-dir (CTC layout).",
    )

    # data format
    parser.add_argument(
        "--data-format",
        type=str,
        default="ctc",
        choices=["ctc", "flat", "dirs"],
        help="On-disk layout of images and masks.",
    )
    parser.add_argument(
        "--segmentation-name",
        type=str,
        default="GT",
        help="Segmentation to use. With 'ctc' the suffix after the sequence id "
        "(e.g. 'GT'), with 'dirs' the mask subdirectory name (e.g. 'masks'), with "
        "'flat' the filename suffix (e.g. 'masks').",
    )
    parser.add_argument(
        "--img-name",
        type=str,
        default=None,
        help="Image subdirectory name ('dirs') or filename suffix ('flat'). "
        "Required only for 'dirs'.",
    )

    # graph params
    parser.add_argument(
        "--graph-search-radius",
        type=str,
        default="trained",
        help="Maximum distance for an edge between two cells, in pixels or as a multiple "
        "of the expected cell size (e.g. '2.5x'). Pass 'trained' to use the value the "
        "model was trained with.",
    )
    parser.add_argument(
        "--prune-edges-by",
        type=str,
        default="default",
        choices=["default", "overlap", "dilated_overlap", "ellipse", "radius", "off"],
        help="How candidate edges are pruned. 'default' keeps what the model was trained "
        "with, 'off' keeps every candidate inside the search radius.",
    )
    parser.add_argument(
        "--prune-param",
        type=str,
        default=None,
        help="Parameter of --prune-edges-by: an area overlap threshold for 'overlap' "
        "(e.g. 0.1), a scaling factor for 'ellipse' and 'radius' (e.g. 3.0), or the "
        "dilation column for 'dilated_overlap' (e.g. 'thickness').",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Tile each frame into patches of this size instead of building one "
        "whole-frame graph. Needed when a full frame exceeds memory.",
    )

    # tracking
    parser.add_argument(
        "--classify-states",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Classify life cycle states. Detected from the model unless passed.",
    )
    parser.add_argument(
        "--thr-corr",
        type=float,
        default=None,
        help="Correspondence threshold of the LAP tracker.",
    )
    parser.add_argument(
        "--thr-div",
        type=float,
        default=None,
        help="Division threshold of the LAP tracker.",
    )
    parser.add_argument(
        "--segmentation-correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Correct over- and undersegmentation from the division and correspondence "
        "predictions. Follows the model configuration unless passed.",
    )

    # output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save tracking results. Falls back to 'baclct_results' inside "
        "--data-dir.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="ctc",
        choices=["ctc", "flat"],
        help="Layout of the exported results: a CTC sequence directory, or "
        "'{sequence_id}_tracks.csv' plus '{sequence_id}_tracks.tif'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tracking results.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where to cache node, deep, and edge features. A path caches persistently, "
        "'temp' uses a temporary directory removed after each sequence. Omit to keep "
        "the precomputed edges in memory and write nothing.",
    )

    # compute
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'auto' (default), 'cpu', 'cuda', 'cuda:1', or 'mps'.",
    )
    parser.add_argument(
        "--num-jobs-features",
        type=int,
        default=None,
        help="Frames whose handcrafted features are extracted in parallel (1). Use -1 "
        "for all cores.",
    )
    parser.add_argument(
        "--num-workers-encode",
        type=int,
        default=None,
        help="Workers cropping cell patches while the encoder runs (2).",
    )
    parser.add_argument(
        "--num-workers-predict",
        type=int,
        default=None,
        help="Dataloader workers during GNN prediction (0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Subgraphs per prediction batch (1).",
    )
    parser.add_argument(
        "--seed", type=int, default=1510, help="Seed for all random number generators."
    )

    # benchmark utils
    parser.add_argument(
        "--split-file", type=Path, default=None, help="Path to a splits.yaml file."
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Fold to use from the split file."
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Phase to use from the split file.",
    )

    return parser


def main():
    """Run BacLCT tracking from the command line.

    Loads the model named by `--model` and applies it to one or more image sequences. For
    each sequence, features are extracted from scratch (and cached only when `--cache-dir`
    is given), the GNN predicts edge and node classes, and a LAP tracker produces the
    final trajectories and (if the model was trained for it) life cycle state
    classifications. Tracked masks and lineage are exported in CTC or flat format,
    alongside the full single-cell trajectories and state classifications.

    Sequences come from `--sequences`, or from a `--split-file` together with `--fold` and
    `--phase`. If neither is given, every sequence found in `--data-dir` is tracked (CTC
    layout only). Existing results are skipped unless `--overwrite` is set.
    """
    args = _build_parser().parse_args()
    _configure_process()
    L.seed_everything(args.seed, workers=True)

    # accepts a registered model name as well as a path
    experiment_dir = resolve_model_dir(args.model)
    overrides = _build_overrides(args)
    track_kwargs = _build_track_kwargs(args)
    sequences = _resolve_sequences(args)

    failed: list[str] = []
    for sequence_id in tqdm(sequences, desc="Processing sequences"):
        ok, message = _track_sequence(
            sequence_id, args, experiment_dir, overrides, track_kwargs
        )
        if message:
            print(f"Sequence {sequence_id}: {message}")
        if not ok:
            failed.append(sequence_id)

    if failed:
        raise SystemExit(f"Failed on {len(failed)} of {len(sequences)}: {failed}")


if __name__ == "__main__":
    main()
