"""Command line entry points for tracking sequences and for the napari plugin."""

from __future__ import annotations

import importlib.util
import logging
import re
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import lightning as L
import numpy as np
import typer
import yaml
from tqdm.auto import tqdm

from baclct.api import BacLCT
from baclct.features.graph import EdgePruneArg
from baclct.io import (
    get_sequences_from_path,
    load_images_and_masks,
    resolve_export_suffix,
    shared_prefix,
)
from baclct.io.load import DATA_FORMATS, _read_frames
from baclct.utils.data import close_worker_pool, set_multiprocessing_context
from baclct.utils.pretrained import DEFAULT_MODEL

if TYPE_CHECKING:
    import dask.array as da

    _Frames = da.Array | np.ndarray

NAPARI_AVAILABLE = importlib.util.find_spec("napari") is not None

app = typer.Typer(add_completion=False, no_args_is_help=True)

# 'gt' is training-only. 'default' and 'off' are CLI spellings with no method behind them
PRUNE_METHODS = ("default", "off", *(m for m in get_args(EdgePruneArg) if m != "gt"))
EXPORT_FORMATS = ("ctc", "flat")

# panels the flags are grouped under in `--help`
_DATA = "Data"
_GRAPH = "Graph"
_TRACKER = "Tracker"
_THROUGHPUT = "Throughput"
_EXPORT = "Export"

# flags shared between track and napari entrypoints
ModelOpt = Annotated[
    str,
    typer.Option("--model", help="Registered model name or path to a model directory."),
]
GraphSearchRadiusOpt = Annotated[
    str | None,
    typer.Option(
        "--graph-search-radius",
        help="Maximum distance for an edge between two cells, in pixels or as a "
        "multiple of the expected cell size. 'trained' uses the model's own value.",
        rich_help_panel=_GRAPH,
    ),
]
PruneEdgesByOpt = Annotated[
    str | None,
    typer.Option(
        "--prune-edges-by",
        help="How candidate edges are pruned: 'default' (as trained), 'overlap', "
        "'dilated_overlap', 'ellipse', 'radius', or 'off'.",
        rich_help_panel=_GRAPH,
    ),
]
PruneParamOpt = Annotated[
    str | None,
    typer.Option(
        "--prune-param",
        help="Parameter of --prune-edges-by: an area overlap threshold for "
        "'overlap' (0.1), a scaling factor for 'ellipse' and 'radius' (3.0), or the "
        "dilation column for 'dilated_overlap' ('thickness').",
        rich_help_panel=_GRAPH,
        show_default=False,
    ),
]
ThrCorrOpt = Annotated[
    float | None,
    typer.Option(
        "--thr-corr",
        help="Correspondence threshold. Defaults to the model's own value.",
        rich_help_panel=_TRACKER,
        show_default=False,
    ),
]
ThrDivOpt = Annotated[
    float | None,
    typer.Option(
        "--thr-div",
        help="Division threshold. Lower it to recover missed divisions. Defaults to "
        "the model's own value.",
        rich_help_panel=_TRACKER,
        show_default=False,
    ),
]
SegmentationCorrectionOpt = Annotated[
    bool | None,
    typer.Option(
        "--segmentation-correction/--no-segmentation-correction",
        help="Correct over- and undersegmentation from the division and "
        "correspondence predictions. Follows the model unless passed.",
        rich_help_panel=_TRACKER,
        show_default=False,
    ),
]
DeviceOpt = Annotated[
    str | None,
    typer.Option(
        "--device",
        help="Device to run on: 'cpu', 'cuda', 'cuda:1', or 'mps'. Detected by default.",
        rich_help_panel=_THROUGHPUT,
        show_default=False,
    ),
]
CacheDirOpt = Annotated[
    str | None,
    typer.Option(
        "--cache-dir",
        help="Where node, deep, and edge features are cached: 'temp' for a "
        "temporary directory removed afterwards, 'memory' to cache nothing, or a "
        "path to keep them for repeated runs (e.g., for tuning thresholds).",
        rich_help_panel=_THROUGHPUT,
    ),
]
PatchSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--patch-size",
        help="Tile each frame into patches of this size instead of building one "
        "whole-frame graph. Needed when a full frame exceeds memory.",
        rich_help_panel=_THROUGHPUT,
        show_default=False,
    ),
]
BatchSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--batch-size",
        help="Subgraphs per prediction batch. Raise it on a large-memory machine.",
        rich_help_panel=_THROUGHPUT,
    ),
]
NumWorkersOpt = Annotated[
    int | None,
    typer.Option(
        "--num-workers",
        help="Worker processes for feature extraction: frames are handled this many "
        "at a time, and the encoder gets a fixed 2. Worth raising on a long "
        "sequence. -1 uses every core, unbounded.",
        rich_help_panel=_THROUGHPUT,
    ),
]
OutputDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Directory to write results to. Defaults to 'baclct_results' inside "
        "--data-dir, or next to MASKS.",
        rich_help_panel=_EXPORT,
        show_default=False,
    ),
]
ExportFormatOpt = Annotated[
    str | None,
    typer.Option(
        "--export-format",
        help="Layout of the results: 'ctc' for a sequence directory, 'flat' for "
        "'{id}_tracks.csv' plus '{id}_tracks.tif'. Follows the input layout by "
        "default.",
        rich_help_panel=_EXPORT,
        show_default=False,
    ),
]
ExportSuffixOpt = Annotated[
    str | None,
    typer.Option(
        "--export-suffix",
        help="Appended to the sequence id, naming the results "
        "'{id}{suffix}'. Defaults to '_RES' ('ctc') or '_tracks' ('flat'). Pass an "
        "empty string to name them after the sequence id alone.",
        rich_help_panel=_EXPORT,
        show_default=False,
    ),
]
SeedOpt = Annotated[
    int, typer.Option("--seed", help="Seed for all random number generators.")
]
TrackLazyOpt = Annotated[
    bool,
    typer.Option(
        "--lazy/--no-lazy",
        help="Read each frame as it is needed, so a sequence larger than memory still "
        "tracks. --no-lazy holds the whole sequence instead, which saves a second "
        "decode when the model uses image intensities. --dry-run is always lazy.",
        rich_help_panel=_DATA,
    ),
]

# `napari` has no --data-dir and no input layout to follow, so these two say something
# different from their `track` counterparts and are declared separately.
NapariOutputDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Directory to write results to. Without it the results are only added "
        "back as layers.",
        rich_help_panel=_EXPORT,
        show_default=False,
    ),
]
NapariExportFormatOpt = Annotated[
    str | None,
    typer.Option(
        "--export-format",
        help="Layout of the results: 'ctc' for a sequence directory, 'flat' for "
        "'{id}_tracks.csv' plus '{id}_tracks.tif'.",
        rich_help_panel=_EXPORT,
        show_default=False,
    ),
]


def _configure_process() -> None:
    """Apply the process-wide settings this CLI needs, but a library import must not."""
    # `fork` (the default on linux) deadlocks with polars and the memory-mapped caches,
    # and windows only has `spawn`
    set_multiprocessing_context()
    # interpreter shutdown can close the log stream before the last records are flushed
    logging.raiseExceptions = False
    warnings.filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called in a multiprocessing",
        category=UserWarning,
    )


def _checked(value: str | None, flag: str, choices: tuple[str, ...]) -> str | None:
    """A flag value from a closed set, rejected as a usage error when it is not."""
    if value is not None and value not in choices:
        raise typer.BadParameter(
            f"Unknown {flag} '{value}', expected one of {', '.join(choices)}."
        )
    return value


def _resolve_search_radius(radius: str | None) -> int | str | None:
    """Resolve the radius flag, where 'trained' keeps the value from the model config."""
    # None reaches here only from the shared `| None` alias, and means the same thing
    if radius is None or radius == "trained":
        return None
    try:
        return int(radius)
    except ValueError:
        return radius


def _resolve_prune_edges_by(
    method: str | None, param: str | None
) -> str | tuple[str, float | str] | None:
    """Normalize the pruning flags into what `BacLCT.track()` expects."""
    if method is None or method == "default":
        return None
    if method == "off":
        return "off"
    if method == "dilated_overlap":
        return (method, param if param is not None else "thickness")
    if method == "overlap":
        return (method, float(param) if param is not None else 0.1)
    return (method, float(param) if param is not None else 3.0)


def _resolve_cache_dir(cache_dir: str | None) -> Path | str | None:
    """Resolve the cache flag, where 'memory' keeps the features out of any directory."""
    if cache_dir is None or cache_dir == "memory":
        return None
    return cache_dir if cache_dir == "temp" else Path(cache_dir)


def _split_sequences(sequences: list[str] | None) -> list[str] | None:
    """Split every --sequences value, so one flag can carry several ids ("01,02").

    Bare "--sequences 01 02" cannot work: IMAGES is positional, so it would take "02".
    """
    if sequences is None:
        return None
    return [seq for value in sequences for seq in re.split(r"[,\s]+", value) if seq]


def _sequences_from_split(split_file: Path, fold: int | None, phase: str) -> list[str]:
    """Read the sequence ids of one fold and phase from a splits.yaml."""
    if fold is None:
        raise typer.BadParameter("--fold is required together with --split-file.")
    with split_file.open() as file:
        return yaml.safe_load(file)[fold][phase]


def _load_paths(
    images: Path | None, masks: Path, _unused_id: str = "", lazy: bool = False
) -> tuple[_Frames | None, _Frames]:
    """Load one sequence from two explicit paths, each a tif stack or a directory."""
    loaded_masks = _read_frames(masks, lazy=lazy)
    if loaded_masks is None:
        raise ValueError(f"No .tif masks at {masks}.")
    loaded_images = _read_frames(images, lazy=lazy) if images is not None else None
    return cast("_Frames | None", loaded_images), cast("_Frames", loaded_masks)


def _sequence_id(images: Path, masks: Path) -> str | None:
    """The sequence both paths belong to, or `None` when they share nothing.

    The id names the napari layers and the exported results, so it has to say which
    sequence a run was about, which the leaf of a path rarely does: '01_GT/TRA' is the
    segmentation of '01', and 'Segmentation' is the one beside 'BF'.
    """
    if images.parent == masks.parent:
        # 'flat': '01_images.tif' beside '01_masks.tif'
        prefix = shared_prefix(images.stem, masks.stem)
        if prefix is not None:
            return prefix
        # 'dirs': '01/BF' beside '01/Segmentation', so the sequence is their parent
        if images.is_dir() and masks.is_dir():
            return images.parent.name or None
        return None
    # 'ctc': '01' beside '01_GT/TRA'
    return shared_prefix(images.name, masks.parent.name)


def _describe(images: _Frames | None, masks: _Frames) -> str:
    """One line naming what a dry run resolved for a sequence."""
    shown = "no images" if images is None else f"images {images.shape} {images.dtype}"
    # the pair is loaded from two independent sources, so a partial export on either
    # side shows up here instead of halfway through a run
    mismatch = (
        " -- frame counts differ"
        if images is not None and len(images) != len(masks)
        else ""
    )
    # dtype alone, so a dry run still never reads a frame
    if masks.dtype == bool:
        mismatch += " -- masks are boolean and carry no instance labels"
    return f"{shown}, masks {masks.shape} {masks.dtype}{mismatch}"


def _results_exist(
    output_dir: Path, sequence_id: str, export_format: str, export_suffix: str | None
) -> bool:
    """Whether a previous run already wrote this sequence's results.

    Both flat files have to be there: a run interrupted halfway through its export left
    one of them behind, and that is worth redoing rather than skipping.
    """
    name = f"{sequence_id}{resolve_export_suffix(export_suffix, export_format)}"
    if export_format == "ctc":
        target = output_dir / name
        return target.is_dir() and any(target.iterdir())
    return all((output_dir / f"{name}{ext}").is_file() for ext in (".csv", ".tif"))


def _track_sequence(
    pipeline: BacLCT,
    sequence_id: str,
    load: Callable[[str], tuple[_Frames | None, _Frames]],
    output_dir: Path,
    overwrite: bool,
    track_kwargs: dict[str, Any],
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """Track a single sequence, or only resolve its inputs under `dry_run`.

    Returns:
        Whether it succeeded, and a message to print if there is one.
    """
    already_there = (
        not dry_run
        and not overwrite
        and _results_exist(
            output_dir,
            sequence_id,
            track_kwargs["export_format"],
            track_kwargs["export_suffix"],
        )
    )
    if already_there:
        return True, "output already exists, skipping"

    try:
        loaded_images, loaded_masks = load(sequence_id)
        if not dry_run:
            output_dir.mkdir(exist_ok=True, parents=True)
    except Exception as err:
        return False, f"could not load: {err}"

    if dry_run:
        return True, _describe(loaded_images, loaded_masks)

    try:
        pipeline.track(
            loaded_images,  # type: ignore[arg-type]
            loaded_masks,
            output_dir=output_dir,
            sequence_id=sequence_id,
            **track_kwargs,
        )
    except Exception as err:
        return False, f"tracking failed: {err}"

    return True, None


@app.command("track")
def track_main(
    # data
    images: Annotated[
        Path | None,
        typer.Argument(
            help="Image sequence: a tif stack or a directory of tifs. Omit together "
            "with MASKS and pass --data-dir to track a whole dataset.",
            show_default=False,
        ),
    ] = None,
    masks: Annotated[
        Path | None,
        typer.Argument(
            help="Instance segmentation masks for IMAGES, as a tif stack or a "
            "directory of tifs.",
            show_default=False,
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Dataset directory to track every sequence of, arranged as "
            "--data-format says.",
            rich_help_panel=_DATA,
            show_default=False,
        ),
    ] = None,
    sequences: Annotated[
        list[str] | None,
        typer.Option(
            "--sequences",
            help="Sequence ids to track, instead of every one found in --data-dir. "
            "Comma-separated, or the flag repeated.",
            rich_help_panel=_DATA,
            show_default=False,
        ),
    ] = None,
    data_format: Annotated[
        str,
        typer.Option(
            "--data-format",
            help="Layout of --data-dir: 'ctc' ('01/' beside '01_GT/TRA'), 'dirs' "
            "('01/BF' and '01/Segmentation'), or 'flat' ('01_images.tif' and "
            "'01_masks.tif').",
            rich_help_panel=_DATA,
        ),
    ] = "ctc",
    segmentation_name: Annotated[
        str | None,
        typer.Option(
            "--segmentation-name",
            help="Segmentation to track: the suffix after the sequence id ('ctc', "
            "'flat') or the subdirectory name ('dirs'). Defaults to 'GT', 'masks', and "
            "'Segmentation'.",
            rich_help_panel=_DATA,
            show_default=False,
        ),
    ] = None,
    img_name: Annotated[
        str | None,
        typer.Option(
            "--img-name",
            help="Image subdirectory ('dirs') or filename suffix ('flat'). Defaults to "
            "'BF' and 'images'.",
            rich_help_panel=_DATA,
            show_default=False,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve and open the inputs of every selected sequence, report what "
            "was found, and stop without tracking.",
            rich_help_panel=_DATA,
        ),
    ] = False,
    lazy: TrackLazyOpt = True,
    model: ModelOpt = DEFAULT_MODEL,
    # graph
    graph_search_radius: GraphSearchRadiusOpt = "2.5x",
    prune_edges_by: PruneEdgesByOpt = "default",
    prune_param: PruneParamOpt = None,
    # tracker
    thr_corr: ThrCorrOpt = None,
    thr_div: ThrDivOpt = None,
    segmentation_correction: SegmentationCorrectionOpt = None,
    # throughput
    device: DeviceOpt = None,
    cache_dir: CacheDirOpt = "temp",
    patch_size: PatchSizeOpt = None,
    batch_size: BatchSizeOpt = 1,
    num_workers: NumWorkersOpt = 0,
    # export
    output_dir: OutputDirOpt = None,
    export_format: ExportFormatOpt = None,
    export_suffix: ExportSuffixOpt = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite", help="Overwrite existing results.", rich_help_panel=_EXPORT
        ),
    ] = False,
    seed: SeedOpt = 1510,
    # benchmark utilities, hidden from the user-facing surface
    split_file: Annotated[Path | None, typer.Option("--split-file", hidden=True)] = None,
    fold: Annotated[int | None, typer.Option("--fold", hidden=True)] = None,
    phase: Annotated[str, typer.Option("--phase", hidden=True)] = "test",
) -> None:
    """Track bacteria and classify their life cycle states with a trained model.

    Takes an image sequence and its instance segmentation masks, either as two paths or
    as a --data-dir holding several sequences in a --data-format layout. For each
    sequence, features are extracted, the GNN predicts edge and node classes, and the
    tracker produces the final trajectories. Life cycle states follow the model: they are
    classified whenever it was trained for them. Results are written in CTC or flat
    format. --dry-run stops after opening the inputs, to check a layout before spending a
    run on it. Frames are read as they are needed, so a sequence larger than memory
    still tracks. --no-lazy loads it up front instead.
    """
    if (images is None) != (masks is None):
        raise typer.BadParameter(
            "Pass both IMAGES and MASKS, or neither with --data-dir."
        )
    if (masks is None) == (data_dir is None):
        raise typer.BadParameter(
            "Pass either IMAGES and MASKS, or --data-dir, but not both."
        )
    if data_format not in DATA_FORMATS:
        raise typer.BadParameter(
            f"Unknown --data-format '{data_format}', expected one of "
            f"{', '.join(DATA_FORMATS)}."
        )
    _checked(prune_edges_by, "--prune-edges-by", PRUNE_METHODS)
    _checked(export_format, "--export-format", EXPORT_FORMATS)

    sequences = _split_sequences(sequences)

    _configure_process()
    L.seed_everything(seed, workers=True)

    if data_dir is not None:
        if not data_dir.is_dir():
            raise typer.BadParameter(f"Not a dataset directory: {data_dir}")
        available = get_sequences_from_path(data_dir, data_format, segmentation_name)
        if split_file is not None:
            if sequences:
                raise typer.BadParameter(
                    "--sequences and --split-file both select the sequences to track, "
                    "so pass only one of them."
                )
            selected = _sequences_from_split(split_file, fold, phase)
        else:
            selected = list(sequences) if sequences else available
        if not selected:
            raise typer.BadParameter(f"No '{data_format}' sequences in {data_dir}.")
        missing = [seq for seq in selected if seq not in available]
        if missing:
            raise typer.BadParameter(
                f"Not a '{data_format}' sequence in {data_dir}: {missing}"
            )

        load = partial(
            load_images_and_masks,
            data_dir,
            data_format=data_format,
            segmentation_name=segmentation_name,
            img_name=img_name,
            # a dry run only reports what it resolved, so the frames stay unread
            lazy=lazy or dry_run,
            strict=False,
        )
        results_dir = output_dir or data_dir / "baclct_results"
        default_format = "ctc" if data_format == "ctc" else "flat"
    else:
        # with explicit paths there is one sequence, so an id only names the output
        if sequences and len(sequences) > 1:
            raise typer.BadParameter(
                "IMAGES and MASKS are a single sequence, so --sequences takes at most "
                "one id, used to name the results."
            )
        masks = cast(Path, masks)
        images = cast(Path, images)
        load = partial(_load_paths, images, masks, lazy=lazy or dry_run)
        selected = (
            list(sequences) if sequences else [_sequence_id(images, masks) or masks.stem]
        )
        results_dir = output_dir or masks.parent
        default_format = "flat"

    track_kwargs: dict[str, Any] = {
        "model": model,
        "graph_search_radius": _resolve_search_radius(graph_search_radius),
        "prune_edges_by": _resolve_prune_edges_by(prune_edges_by, prune_param),
        "thr_corr": thr_corr,
        "thr_div": thr_div,
        "segmentation_correction": segmentation_correction,
        "device": device,
        "cache_dir": _resolve_cache_dir(cache_dir),
        "patch_size": patch_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "export_format": export_format or default_format,
        "export_suffix": export_suffix,
        # closed once below, after every sequence: reuse across the loop skips worker
        # start-up on every call instead of paying it per sequence
        "_close_pool_after": False,
    }

    # one pipeline for every sequence: it holds nothing sequence-specific, and reuse
    # keeps the resolved config and the downloaded model
    pipeline = BacLCT()
    failed: list[str] = []
    desc = "Checking sequences" if dry_run else "Processing sequences"
    try:
        for sequence_id in tqdm(selected, desc=desc):
            ok, message = _track_sequence(
                pipeline, sequence_id, load, results_dir, overwrite, track_kwargs, dry_run
            )
            if message:
                print(f"Sequence {sequence_id}: {message}")
            if not ok:
                failed.append(sequence_id)
    finally:
        close_worker_pool()

    if failed:
        raise SystemExit(f"Failed on {len(failed)} of {len(selected)}: {failed}")


def _open_viewer(
    images: Path | None,
    masks: Path | None,
    lazy: bool,
    settings: dict[str, Any],
) -> tuple[Any, Any]:
    """Build the viewer, add the layers, dock the widget, and preset it.

    Split out of `napari_main` so the command stays argument handling and this stays
    testable without a running Qt event loop. The widget is reached through napari's own
    plugin registration instead of a direct import, so it is docked exactly as it would
    be from the Plugins menu.
    """
    import napari

    viewer = napari.Viewer()
    if masks is not None:
        loaded_images, loaded_masks = _load_paths(images, masks, "", lazy=lazy)
        sequence = _sequence_id(images, masks) if images is not None else None
        # napari renders a dask array frame by frame, so a lazy load stays lazy here
        if loaded_images is not None:
            images = cast(Path, images)
            name = f"{sequence} images" if sequence is not None else images.stem
            viewer.add_image(loaded_images, name=name)
        name = f"{sequence} masks" if sequence is not None else masks.stem
        viewer.add_labels(_as_labels(loaded_masks), name=name)

    _, widget = viewer.window.add_plugin_dock_widget("baclct", "BacLCT")
    widget.apply_settings(**settings)
    return viewer, widget


def _as_labels(masks: _Frames) -> _Frames:
    """Cast a mask stack to an integer dtype, which is all a Labels layer accepts."""
    if np.issubdtype(masks.dtype, np.integer):
        return masks
    if not np.issubdtype(masks.dtype, np.floating):
        raise typer.BadParameter(
            "Masks must be an instance segmentation with integer labels, got dtype "
            f"{masks.dtype}. Label the connected components first, e.g. "
            "skimage.measure.label()."
        )
    # a float label image is common enough from other tools, and the cast stays lazy
    return masks.astype(np.int32)


@app.command("napari")
def napari_main(
    # data
    images: Annotated[
        Path | None,
        typer.Argument(
            help="Image sequence: a tif stack or a directory of tifs. Omit together "
            "with MASKS to open an empty viewer with the plugin docked.",
            show_default=False,
        ),
    ] = None,
    masks: Annotated[
        Path | None,
        typer.Argument(
            help="Instance segmentation masks for IMAGES, as a tif stack or a "
            "directory of tifs.",
            show_default=False,
        ),
    ] = None,
    channel: Annotated[
        int | None,
        typer.Option(
            "--channel",
            help="Channel to track, for an image with a channel axis the segmentation "
            "does not have.",
            rich_help_panel=_DATA,
            show_default=False,
        ),
    ] = None,
    lazy: Annotated[
        bool,
        typer.Option(
            "--lazy/--no-lazy",
            help="Load the frames on demand, so opening is instant and only the frames "
            "in use are held. --no-lazy reads the whole stack up front.",
            rich_help_panel=_DATA,
        ),
    ] = True,
    model: ModelOpt = DEFAULT_MODEL,
    # graph
    graph_search_radius: GraphSearchRadiusOpt = None,
    prune_edges_by: PruneEdgesByOpt = None,
    prune_param: PruneParamOpt = None,
    # tracker
    thr_corr: ThrCorrOpt = None,
    thr_div: ThrDivOpt = None,
    segmentation_correction: SegmentationCorrectionOpt = None,
    # throughput
    device: DeviceOpt = None,
    cache_dir: CacheDirOpt = None,
    patch_size: PatchSizeOpt = None,
    batch_size: BatchSizeOpt = None,
    num_workers: NumWorkersOpt = None,
    # export
    output_dir: NapariOutputDirOpt = None,
    export_format: NapariExportFormatOpt = None,
    export_suffix: ExportSuffixOpt = None,
    seed: SeedOpt = 1510,
) -> None:
    """Open a sequence in napari with the BacLCT plugin docked and preset.

    Takes the same IMAGES and MASKS as `track`, loads them lazily so a long movie opens
    at once, docks the tracking widget, and fills it in from whichever flags were
    passed. A flag left out keeps the widget's own setting: the search radius and the
    pruning are filled in from the model, while the thresholds and segmentation
    correction are not, unless --thr-corr or --thr-div says otherwise. Press 'Start
    tracking' in the widget to run the same pipeline `track` runs.
    """
    if not NAPARI_AVAILABLE:
        raise typer.BadParameter(
            'napari is not installed. Install it with: pip install "baclct[napari]"'
        )
    if (images is None) != (masks is None):
        raise typer.BadParameter(
            "Pass both IMAGES and MASKS, or neither to open an empty viewer."
        )
    _checked(prune_edges_by, "--prune-edges-by", PRUNE_METHODS)
    _checked(export_format, "--export-format", EXPORT_FORMATS)

    _configure_process()
    L.seed_everything(seed, workers=True)

    # None means "leave the widget's own default alone", so unset flags are dropped
    settings = {
        key: value
        for key, value in {
            "model": model,
            "channel": channel,
            "graph_search_radius": graph_search_radius,
            "prune_edges_by": prune_edges_by,
            "prune_param": prune_param,
            "thr_corr": thr_corr,
            "thr_div": thr_div,
            "segmentation_correction": segmentation_correction,
            "device": device,
            "cache_dir": cache_dir,
            "patch_size": patch_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "output_dir": output_dir,
            "export_format": export_format,
            "export_suffix": export_suffix,
        }.items()
        if value is not None
    }

    try:
        import napari
    except ImportError as err:  # the package is there, so this is a broken Qt stack
        raise typer.BadParameter(
            f"napari is installed but could not be imported: {err}"
        ) from err

    try:
        _open_viewer(images, masks, lazy, settings)
    except ValueError as err:
        # the remaining settings land in widget fields with closed sets of their own
        raise typer.BadParameter(str(err)) from err
    napari.run()


def _run_standalone(command: Callable[..., None], prog_name: str) -> None:
    """Run one subcommand as its own program, for the hyphenated entry points.

    A fresh single-command app parses and renders exactly what the subcommand does
    inside the group, only without the group name in the usage line.
    """
    standalone = typer.Typer(add_completion=False)
    standalone.command()(command)
    standalone(prog_name=prog_name)


def track_entry() -> None:
    """`baclct-track`, the standalone alias for `baclct track`."""
    _run_standalone(track_main, "baclct-track")


def napari_entry() -> None:
    """`baclct-napari`, the standalone alias for `baclct napari`."""
    _run_standalone(napari_main, "baclct-napari")


if __name__ == "__main__":
    app()
