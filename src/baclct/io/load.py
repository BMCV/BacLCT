"""Discover sequences, load images and masks, and normalize intensities."""

from __future__ import annotations

import hashlib
import json
from itertools import chain
from pathlib import Path
from typing import Literal, overload

import dask.array as da
import dask.array.image
import numpy as np
import yaml
from tifffile import imread

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def get_sequences_from_path(data_dir: Path, data_format: str = "ctc"):
    """Return available sequence IDs in a CTC dataset folder."""
    if data_format == "ctc":
        return sorted({fp.stem.split("_")[0] for fp in data_dir.iterdir() if fp.is_dir()})

    raise NotImplementedError


def get_sequences_from_split(
    split_file: Path, fold: int, phase: str | list[str] | None = None
):
    """Get sequences defined for fold in cross-validation split yaml file."""
    with split_file.open("r") as file:
        splits = yaml.safe_load(file)

    if fold not in splits:
        raise ValueError(
            f"Fold {fold} not found in split file. "
            f"Available folds: {list(splits.keys())}."
        )

    fold_data = splits[fold]
    if phase is None:
        return sorted(set(chain.from_iterable(fold_data.values())))

    if isinstance(phase, str):
        phases = [phase]
    else:
        phases = phase

    seqs = []
    for p in phases:
        if p in fold_data:
            seqs.extend(fold_data[p])
        else:
            raise ValueError(
                f"Phase '{p}' not found in fold {fold}. "
                f"Available phases: {list(fold_data.keys())}."
            )
    return sorted(set(seqs))


def _histogram_percentiles(
    images: np.ndarray | da.Array, percentiles: tuple[float, float]
) -> tuple[float, float]:
    """Percentiles of an integer sequence, read off a histogram over intensity levels."""
    info = np.iinfo(images.dtype)
    offset = int(info.min)
    values = images.ravel()  # type: ignore
    if offset:
        values = values.astype(np.int64) - offset

    counts = (
        da.bincount(values, minlength=int(info.max) - offset + 1)
        if isinstance(images, da.Array)
        else np.bincount(values, minlength=int(info.max) - offset + 1)
    )
    cumulative = np.cumsum(counts.compute() if isinstance(counts, da.Array) else counts)
    n = int(cumulative[-1])
    if n == 0:
        raise ValueError("Cannot compute percentiles of an empty image sequence.")

    out = []
    for q in percentiles:
        # numpy divides by 100 first, and its lerp is two-sided. both forms are needed to
        # land on the same float it returns.
        virtual = (q / 100.0) * (n - 1)
        low_idx = int(np.floor(virtual))
        low = float(np.searchsorted(cumulative, low_idx + 1) + offset)
        high = float(np.searchsorted(cumulative, int(np.ceil(virtual)) + 1) + offset)
        weight = virtual - low_idx
        out.append(
            high - (high - low) * (1 - weight)
            if weight >= 0.5
            else low + (high - low) * weight
        )

    return out[0], out[1]


def get_percentiles(
    images: np.ndarray | da.Array, percentiles: tuple[float, float] = (0.05, 99.95)
) -> tuple[float, float]:
    """Intensity percentiles over the entire image sequence.

    Returns the values `np.percentile` returns for the fully materialized stack. 8- and
    16-bit integer input, which is what microscopy produces, is reduced without
    materializing it.
    """
    # wider integers would need a bin per level, so they take the materializing path
    if np.issubdtype(images.dtype, np.integer) and images.dtype.itemsize <= 2:
        return _histogram_percentiles(images, percentiles)

    if isinstance(images, da.Array):
        logger.debug(f"{images.dtype} input: percentiles materialize the whole sequence.")
        images = images.compute()

    p_low, p_high = np.percentile(images, list(percentiles))
    return float(p_low), float(p_high)


def scale_percentiles(
    img: np.ndarray, percentiles: tuple[float, float] | None
) -> np.ndarray:
    """Normalize a single frame to `[0, 1]` between the values in `percentiles`."""
    if percentiles is None:
        return img

    img_float = img.astype(np.float32)
    p_low, p_high = percentiles
    denom = p_high - p_low
    normalized = (img_float - p_low) / (denom if denom > 0 else 1.0)
    return np.clip(normalized, 0, 1)


def frame_hash(img: np.ndarray | None, msk: np.ndarray | None) -> str:
    """Hash a frame and its mask, identifying the input a feature cache was built from."""
    assert img is not None or msk is not None
    hasher = hashlib.blake2b()
    if img is not None:
        hasher.update(img.tobytes())
    if msk is not None:
        hasher.update(msk.tobytes())
    return hasher.hexdigest()


# frames to sample for `dataset_identity`: first, last, and evenly spaced in between,
# bounding hash cost on long sequences (e.g. toiam's 800 frames, one tif file each)
_N_HASH_FRAMES = 16


def dataset_identity(
    images: da.Array | np.ndarray | None,
    masks: da.Array | np.ndarray,
    *,
    data_dir: Path | None = None,
    sequence_id: str | None = None,
) -> dict:
    """Content identity of an image/mask sequence, used to guard feature caches.

    Hashes a fixed-size, deterministic sample of frames (`_N_HASH_FRAMES`) rather than
    the whole sequence, so validation cost stays bounded on long sequences. The sampled
    indices are a pure function of sequence length, so a write and a later validate
    against the same data always hash the same frames.
    """
    n = len(masks)
    sample_t = sorted(
        {int(t) for t in np.linspace(0, n - 1, min(n, _N_HASH_FRAMES)).astype(int)}
    )
    frame_hashes = {}
    for t in sample_t:
        img_t = None
        if images is not None:
            img_t = images[t].compute() if isinstance(images[t], da.Array) else images[t]
        msk_t = masks[t].compute() if isinstance(masks[t], da.Array) else masks[t]
        frame_hashes[str(t)] = frame_hash(img_t, msk_t)

    return {
        "data_dir": str(data_dir) if data_dir is not None else None,
        "sequence_id": sequence_id,
        "image_shape": list(images.shape) if images is not None else None,
        "masks_shape": list(masks.shape),
        "frame_hashes": frame_hashes,
    }


def dataset_identity_matches(current: dict, cached: dict | None) -> bool:
    """Whether a cache built under `cached`'s identity still matches `current`."""
    if not cached:
        return False
    if current.get("image_shape") != cached.get("image_shape"):
        return False
    if current.get("masks_shape") != cached.get("masks_shape"):
        return False
    cached_hashes = cached.get("frame_hashes", {})
    return all(
        cached_hashes.get(t) == h for t, h in current.get("frame_hashes", {}).items()
    )


_MISSING_MSG = (
    "Could not find {what} in {where}.\n"
    "If you intend to load images or masks only, use strict=False."
)

# per-format defaults for the image and segmentation source names
_SOURCE_DEFAULTS = {
    "ctc": (None, "GT"),
    "flat": ("images", "masks"),
    "dirs": ("BF", "Segmentation"),
}


def _ctc_segmentation_dir(data_dir: Path, seq_id: str, segmentation_name: str) -> Path:
    """Locate the CTC segmentation directory, preferring tracking over segmentation."""
    seg_dir = data_dir / f"{seq_id}_{segmentation_name}" / "TRA"
    if not seg_dir.exists() or not any(seg_dir.iterdir()):
        logger.debug(
            f"Could not find tracking data for {seg_dir.parent}. Loading segmentations. "
            "Training needs tracking annotations, so ST/ERR_SEG CTC data has to be "
            "converted into a TRA directory first."
        )
        seg_dir = seg_dir.with_name("SEG")
    if not seg_dir.exists() or not any(seg_dir.iterdir()):
        logger.debug("Trying base dir.")
        seg_dir = seg_dir.parent
    return seg_dir


def _resolve_sources(
    data_dir: Path,
    seq_id: str,
    data_format: str,
    img_name: str | None,
    segmentation_name: str | None,
) -> tuple[Path, Path]:
    """Locate the image and segmentation sources, each either a directory or a file."""
    if data_format not in _SOURCE_DEFAULTS:
        raise ValueError(f"Unknown data format: {data_format}")

    img_default, seg_default = _SOURCE_DEFAULTS[data_format]
    img_name = img_name or img_default
    segmentation_name = segmentation_name or seg_default
    assert segmentation_name

    if data_format == "ctc":
        return data_dir / seq_id, _ctc_segmentation_dir(
            data_dir, seq_id, segmentation_name
        )
    if data_format == "flat":
        return (
            data_dir / f"{seq_id}_{img_name}.tif",
            data_dir / f"{seq_id}_{segmentation_name}.tif",
        )
    assert img_name
    return data_dir / seq_id / img_name, data_dir / seq_id / segmentation_name


def _read_tifs(source: Path, lazy: bool) -> da.Array | np.ndarray | None:
    """Read a .tif file or a directory of them as `(T, ...)`, or `None` if absent."""
    if not source.exists():
        return None

    if source.is_dir():
        files = sorted(source.glob("*.tif"))
        if not files:
            return None
        if lazy:
            return dask.array.image.imread(str(source / "*.tif"), imread=imread)
        return np.stack([imread(fp) for fp in files])

    if not lazy:
        return imread(source)

    # a lazy read stacks the file list, so a multi-page file gains a leading axis
    stack = dask.array.image.imread(str(source), imread=imread)
    return stack[0] if stack.ndim > 3 else stack


# without percentiles
@overload
def load_images_and_masks(
    data_dir: Path,
    seq_id: str,
    data_format: str = ...,
    lazy: bool = ...,
    return_percentiles: Literal[False] = ...,
    segmentation_name: str | None = ...,
    img_name: str | None = ...,
    percentile_file: Path | None = ...,
    strict: bool = ...,
) -> tuple[da.Array | np.ndarray | None, da.Array | np.ndarray]: ...


# with percentiles
@overload
def load_images_and_masks(
    data_dir: Path,
    seq_id: str,
    data_format: str = ...,
    lazy: bool = ...,
    *,
    return_percentiles: Literal[True],
    segmentation_name: str | None = ...,
    img_name: str | None = ...,
    percentile_file: Path | None = ...,
    strict: bool = ...,
) -> tuple[
    da.Array | np.ndarray | None, da.Array | np.ndarray, tuple[float, float] | None
]: ...


def load_images_and_masks(
    data_dir: Path,
    seq_id: str,
    data_format: str = "ctc",
    lazy: bool = True,
    return_percentiles: bool = False,
    segmentation_name: str | None = None,
    img_name: str | None = None,
    percentile_file: Path | None = None,
    strict: bool = True,
) -> (
    tuple[da.Array | np.ndarray | None, da.Array | np.ndarray]
    | tuple[
        da.Array | np.ndarray | None, da.Array | np.ndarray, tuple[float, float] | None
    ]
):
    """Load images and masks for a sequence from the given on-disk layout.

    Args:
        data_dir: Root directory containing the dataset.
        seq_id: Sequence ID to load.
        data_format: On-disk layout of images and masks. One of 'ctc' (dirs, e.g.,
            '01' for images and '01_GT/TRA' for tracks), 'flat' (tifs, e.g.,
            '01_images.tif' and '01_masks.tif'), or 'dirs' (same as CTC, but dirs
            may contain tif stacks and does not check for 'TRA').
        lazy: If `True`, return dask arrays that read frames on demand. If
            `False`, stack into a numpy array eagerly.
        return_percentiles: If `True`, also return intensity percentiles computed
            over the full image sequence.
        segmentation_name: Name of the segmentation directory or filename suffix.
            With `data_format='ctc'` it is the suffix after `seq_id` (e.g.,
            'GT'). With `data_format='dirs'` it is the subdirectory name. With
            `data_format='flat'` it is the filename suffix (e.g., 'masks').
        img_name: Name of the image subdirectory (`data_format='dirs'`) or suffix
            (`data_format='flat'`).
        percentile_file: JSON file caching `(p_low, p_high)`. Loaded if it
            exists, otherwise percentiles are computed and written.
        strict: If `True`, raise when images cannot be found. If `False`, return
            `None` for images and proceed.

    Returns:
        `(images, masks)`, or `(images, masks, percentiles)` when
        `return_percentiles=True`. `percentiles` is `None` if `strict=False` and no
        images were found.
    """
    img_src, seg_src = _resolve_sources(
        data_dir, seq_id, data_format, img_name, segmentation_name
    )
    images = _read_tifs(img_src, lazy)
    masks = _read_tifs(seg_src, lazy)

    if images is None or len(images) == 0:
        images = None
        if strict:
            raise ValueError(_MISSING_MSG.format(what="images", where=img_src))
        logger.debug(f"Could not find images in {img_src}. Returning None for images.")

    if masks is None or len(masks) == 0:
        raise ValueError(_MISSING_MSG.format(what="masks", where=seg_src))

    if return_percentiles:
        return images, masks, cached_percentiles(images, percentile_file)

    return images, masks


def cached_percentiles(
    images: np.ndarray | da.Array | None, percentile_file: Path | None
) -> tuple[float, float] | None:
    """Intensity percentiles of `images`, read from or written to `percentile_file`."""
    if percentile_file is not None and percentile_file.exists():
        with percentile_file.open("r") as file:
            p_low, p_high = json.load(file)
        return float(p_low), float(p_high)
    if images is None:
        return None

    percentiles = get_percentiles(images)
    if percentile_file is not None:
        percentile_file.parent.mkdir(exist_ok=True, parents=True)
        with percentile_file.open("w") as file:
            json.dump(percentiles, file)
    return percentiles
