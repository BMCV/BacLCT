"""Handcrafted and cell-level feature extractors."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import dask.array as da
import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from skimage.measure import regionprops_table
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from baclct.features import normalization
from baclct.features.custom_features import CUSTOM_NODE_PROPS, CUSTOM_NODE_TRANSFORMS
from baclct.features.patches import CellPatchDataset, PatchWeighting
from baclct.io import dataset_identity_matches, load_lineage, scale_percentiles
from baclct.models.encoder import MaskedDINOEncoder, normalize_imagenet
from baclct.utils.data import get_device, get_multiprocessing_context
from baclct.utils.logger import get_pylogger
from baclct.utils.progress import ProgressCallback, report, track_iter
from baclct.utils.spacing import needs_spacing

logger = get_pylogger(__name__)

RoiStrategy: TypeAlias = Literal["bbox", "axis_major_length", "len_init", "median_size"]

# an in-memory stack above this size is copied into every dataloader worker, so cropping
# it inline beats paying that per worker
_MAX_WORKER_COPY_BYTES = 512 * 1024**2

# regionprops reductions that a bare prop name (e.g. "intensity") may expand to
_EXPANDABLE_STAT_SUFFIXES = ("min", "max", "mean", "median", "std")


def _matches_feature(column: str, feat_name: str) -> bool:
    """Whether a node-feature column belongs under a requested feature name.

    Matches an exact name, a `_norm` variant, vector components (`center-0`,
    `center-1`), and regionprops reductions (`intensity_min/max/mean`), but not
    arbitrary same-prefix columns.
    """
    base = column.removesuffix("_norm")
    if base == feat_name:
        return True
    rest = base.removeprefix(feat_name)
    if rest == base:  # feat_name is not a prefix of this column
        return False
    if rest.startswith("-") and rest[1:].isdigit():  # vector component, e.g. center-0
        return True
    return rest.startswith("_") and rest[1:] in _EXPANDABLE_STAT_SUFFIXES


class BaseExtractor(ABC):
    """Abstract base class for feature extractors.

    Subclasses implement the main feature extraction logic, including caching (`_load`,
    `_validate`, `_save`), feature computation from scratch (`_compute`), and an optional
    transform for the model (`_transform`) that applies feature selection, normalization,
    and scaling. Calling the extractor loads cached features and returns them after
    validation, or computes them from scratch and (optionally) saves the result.
    """

    def __init__(self, name: str, required_feats: list[str]):
        """Initialize extractor.

        Args:
            name: Identifier for extracted features.
            required_feats: Node features required for feature extraction (e.g., `bbox`).
        """
        self.name = name
        self.required_feats = required_feats
        # optional progress sink, attached by callers (see `baclct.utils.progress`)
        self.progress: ProgressCallback | None = None
        super().__init__()

    def __getstate__(self) -> dict:
        """Drop the progress sink when pickling (e.g. to joblib or dataloader workers).

        The extractor is pickled to worker processes because the per-frame tasks reference
        bound methods. A sink is typically a closure over a GUI (unpicklable), and workers
        have nothing to report to anyway: only the parent consumes progress.
        """
        state = self.__dict__.copy()
        state["progress"] = None
        return state

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the feature extraction."""
        raise NotImplementedError

    def _transform(self, features: Any) -> torch.Tensor | None:
        """Transforms raw features into tensors for the model."""
        return features

    def _validate(self, loaded_data: Any, *args: Any, **kwargs: Any) -> Any | None:
        """Validate loaded data against current inputs."""
        if loaded_data is not None:
            return loaded_data
        return None

    @abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the features from scratch."""
        raise NotImplementedError

    @abstractmethod
    def _load(self, *args: Any, **kwargs: Any) -> Any | None:
        """Load features from cache."""
        raise NotImplementedError

    @abstractmethod
    def _save(self, features: Any, *args: Any, **kwargs: Any) -> None:
        """Save features to cache."""
        raise NotImplementedError


class HandcraftedExtractor(BaseExtractor):
    """Compute per-cell morphological and intensity features.

    Runs `skimage.measure.regionprops_table` per frame to produce one row per segmented
    cell with the requested `props` (e.g., area, axis lengths, orientation, intensity
    statistics) and any `extra_props` defined in `custom_features`. Dataframe-level
    `extra_transforms` (e.g., aspect ratio) and optional global normalization (e.g.,
    scaling by initial spore size) are applied afterwards.

    The feature subset returned by `_transform` is selected by name with anchored
    expansion (e.g., `centroid` matches `centroid-0` and `centroid-1`, `intensity` matches
    `intensity_min/max/mean`), see `_matches_feature` for the exact rules.
    """

    def __init__(
        self,
        props: list[str] | tuple[str] | None = None,
        extra_props: list[str] | tuple[str] | None = None,
        extra_transforms: list[str] | tuple[str] | None = None,
        feature_names: list[str] | tuple[str] | None = None,
        feature_norm_fn: str | None = "scale_relative_size",
        verbose: bool = True,
        n_jobs: int = -1,
        min_node_area: int | None = None,
        **kwargs,
    ):
        """Initialize handcrafted extractor.

        Feature names and functions are passed to `skimage.measure.regionprops` to obtain
        handcrafted and positional features.

        Args:
            props: Extracted regionprops properties. If `None`, extracts the basic
                positional features required for graph construction and processing.
            extra_props: Custom property functions defined in `custom_features`.
            extra_transforms: Dataframe-level feature transforms defined in
                `CUSTOM_NODE_TRANSFORMS` (e.g., `aspect_ratio`). Applied after feature
                extraction to derive additional features from existing ones.
            feature_names: Features used by the model and returned by `_transform`.
                Matched with anchored expansion (see `_matches_feature`): exact
                names, `_norm` variants, vector components (`centroid` to
                `centroid-0`, `centroid-1`), and custom rules (`intensity` to
                `intensity_min`, `intensity_max`, `intensity_mean`). Columns that merely
                share a prefix are not expanded, so a transform-derived sibling has to be
                named explicitly.
            feature_norm_fn: Function used to normalize features in `_transform`. Features
                are normalized independent of their cache (i.e. after loading), so that
                a single cache is compatible with multiple normalization methods.
                Normalized features are suffixed with `_norm`.
            verbose: Print additional info and progress during feature extraction.
            n_jobs: Number of parallel threads (joblib) used during regionprops
                extraction across frames.
            min_node_area: Drop cells smaller than this area before indexing.
            **kwargs: Additional arguments passed to `regionprops`.
        """
        super().__init__(
            name="handcrafted",
            required_feats=[
                "label",
                "bbox",
                "centroid",
                "axis_minor_length",
                "axis_major_length",
                "area",
            ],
        )
        # ordered and deduplicated: regionprops_table emits columns in this order, so a
        # set would make the extracted (and exported) column order vary between runs
        self.props: list[str] = list(
            dict.fromkeys([*self.required_feats, *(props or [])])
        )

        self.extra_props = extra_props
        self.extra_props_fns: list[Callable] = []
        if extra_props is not None:
            for prop_name in extra_props:
                if prop_name in CUSTOM_NODE_PROPS:
                    self.extra_props_fns.append(CUSTOM_NODE_PROPS[prop_name])
                else:
                    raise KeyError(
                        f"Custom property '{prop_name}' not found in CUSTOM_NODE_PROPS."
                    )
            logger.debug(f"Computing extra props: {self.extra_props_fns}")

        self.extra_transforms = extra_transforms
        self.extra_transforms_fns: list[Callable] = []
        if extra_transforms is not None:
            for t_name in extra_transforms:
                if t_name in CUSTOM_NODE_TRANSFORMS:
                    self.extra_transforms_fns.append(CUSTOM_NODE_TRANSFORMS[t_name])
                else:
                    raise KeyError(
                        f"Custom transform '{t_name}' not found in "
                        "CUSTOM_NODE_TRANSFORMS."
                    )
            logger.debug(f"Computing extra transforms: {self.extra_transforms}")

        self.feature_names = feature_names
        self.feature_norm_fn = feature_norm_fn
        self.regionprops_kwargs = dict(**kwargs)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_node_area = min_node_area

        # store names of features returned after first `_transform`
        self.extracted_features = None

    def _validate(
        self,
        loaded_data: pl.DataFrame,
        masks: da.Array | np.ndarray,
    ) -> pl.DataFrame | None:
        if loaded_data["t"].to_numpy().max() != len(masks) - 1:
            logger.warning("Size mismatch between masks and features.")
            # some datasets have more frames than annotations. keep.

        for p in self.props:
            if not any(
                col.startswith(p.replace("_local", "")) for col in loaded_data.columns
            ):
                return

        if self.extra_props:
            for p in self.extra_props:
                if not any(
                    col.startswith(p.replace("_local", "")) for col in loaded_data.columns
                ):
                    return

        # extra_transforms are applied on load (see __call__), so they can't be verified

        return loaded_data

    def _load(
        self,
        filepath: Path | None,
        masks: da.Array | np.ndarray,
        validate: bool,
    ) -> pl.DataFrame | None:
        if (filepath is None) or (not filepath.exists()):
            return

        logger.debug(f"Trying to load features from {filepath}.")
        feats = pl.read_parquet(filepath)
        if not validate:
            return feats
        return self._validate(feats, masks)

    @staticmethod
    def _meta_path(filepath: Path) -> Path:
        return filepath.parent / "meta.json"

    def _read_meta(self, filepath: Path) -> dict:
        meta_path = self._meta_path(filepath)
        return json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def _cached_spacing_matches(
        self, filepath: Path, spacing: tuple[float, ...] | None
    ) -> bool:
        """Whether the cache at `filepath` is compatible with the requested spacing."""
        recorded = self._read_meta(filepath).get("spacing")
        requested = list(spacing) if needs_spacing(spacing) else None  # type: ignore
        return recorded == requested

    def _cached_dataset_matches(self, filepath: Path, dataset_meta: dict | None) -> bool:
        """Whether `filepath`'s cache was built from `dataset_meta`'s images/masks."""
        if dataset_meta is None:
            return True
        cached = self._read_meta(filepath).get("dataset")
        return dataset_identity_matches(dataset_meta, cached)

    def _write_meta(
        self,
        filepath: Path,
        spacing: tuple[float, ...] | None,
        dataset_meta: dict | None,
    ) -> None:
        meta = self._read_meta(filepath)
        if needs_spacing(spacing):
            meta["spacing"] = list(spacing)  # type: ignore
        else:
            meta.pop("spacing", None)
        if dataset_meta is not None:
            meta["dataset"] = dataset_meta
        meta_path = self._meta_path(filepath)
        if meta:
            meta_path.write_text(json.dumps(meta))
        elif meta_path.exists():
            meta_path.unlink()

    def _transform(self, features: pl.DataFrame) -> torch.Tensor | None:
        if self.feature_names is None:
            return

        logger.debug(f"Getting features from available columns: {features.columns}")
        cols_to_select = []
        for feat_name in self.feature_names:
            # name gets expanded: e.g., intensity -> intensity_min, intensity_max,
            # centroid -> centroid-0, centroid-1. expansion is anchored (see
            # _matches_feature), so same-prefix columns are not pulled in.
            # _init preprocessing columns are excluded.
            candidates = [
                c
                for c in features.columns
                if _matches_feature(c, feat_name) and not c.endswith("_init")
            ]

            best_matches = {}
            for c in candidates:
                is_norm = c.endswith("_norm")
                base_name = c.replace("_norm", "") if is_norm else c

                # prioritize normalized features
                if base_name not in best_matches or is_norm:
                    best_matches[base_name] = c

            # order selected columns by base name so the output is invariant to the
            # input column order (e.g. center-0 before center-1 regardless of how the
            # node-feature dataframe happens to be laid out).
            feat_names_to_select = [best_matches[k] for k in sorted(best_matches)]
            logger.debug(f"Extracting: {feat_name} -> {feat_names_to_select}")
            assert len(feat_names_to_select) > 0, (
                f"Could not find {feat_name} in node feature columns: {features.columns}."
            )
            cols_to_select.extend(feat_names_to_select)

        self.extracted_features = cols_to_select
        x = features.select(cols_to_select).to_numpy(writable=True)
        return torch.as_tensor(x, dtype=torch.float32)

    def _compute(
        self,
        images: da.Array | np.ndarray | None,
        masks: da.Array | np.ndarray,
        image_percentiles: tuple[float, float] | None = None,
        spacing: tuple[float, ...] | None = None,
    ) -> pl.DataFrame:
        # non-unit spacing makes regionprops sizes and centroids physical. bbox stays px.
        regionprops_kwargs = dict(self.regionprops_kwargs)
        if needs_spacing(spacing):
            regionprops_kwargs["spacing"] = spacing

        props = self.props
        if images is None:
            intensity_props = [p for p in props if "intensity" in p]
            if intensity_props:
                raise ValueError(
                    f"Cannot compute intensity features {intensity_props} without "
                    "images. Either provide images or remove intensity properties."
                )
        tasks = []
        images_iter = [None] * len(masks) if images is None else images
        for t, (img, msk) in enumerate(zip(images_iter, masks, strict=True)):  # type: ignore
            tasks.append(
                delayed(HandcraftedExtractor._regionprops_task)(
                    img,
                    msk,
                    t,
                    tuple(props),
                    self.extra_props_fns,
                    scale_percentiles,
                    image_percentiles,
                    **regionprops_kwargs,
                )
            )

        parallel = Parallel(n_jobs=self.n_jobs, return_as="generator")
        if self.verbose:
            node_feats_iter = tqdm(
                parallel(tasks), total=len(tasks), desc="Processing frames (handcrafted)"
            )
        else:
            node_feats_iter = parallel(tasks)
        node_feats = list(
            track_iter(
                node_feats_iter,
                self.progress,
                stage="features",
                total=len(tasks),
                message="Extracting features",
            )
        )
        node_feats = pl.concat(node_feats).with_row_index()

        for transform_fn in self.extra_transforms_fns:
            node_feats = transform_fn(node_feats)

        local_cols = [c for c in node_feats.columns if "_local" in c]
        if len(local_cols) > 0:
            logger.debug(f"Found local columns: {local_cols}. Shifting.")
            node_feats = node_feats.with_columns(
                # offset each local coord by min of bbox,
                # e.g., center-1 = center_local-1 + bbox-1
                [
                    pl.col(c_local) + pl.col(f"bbox-{c_local[-1]}")
                    for c_local in local_cols
                ]
            ).rename({c: c.replace("_local", "") for c in local_cols})

        # scale back coords to px for indexing etc.
        if needs_spacing(spacing):
            px_exprs = [
                (pl.col(c) / spacing[int(m.group(1))]).alias(f"{c}_px")  # type: ignore
                for c in node_feats.columns
                if (m := re.fullmatch(r"(?:centroid|center)-(\d+)", c))
            ]
            if px_exprs:
                node_feats = node_feats.with_columns(px_exprs)

        return node_feats

    @staticmethod
    def _regionprops_task(
        img, msk, t, props, extra_props, img_norm_fn, img_stats, **kwargs
    ):
        _img = img_norm_fn(
            img.compute() if isinstance(img, da.Array) else img, percentiles=img_stats
        )
        _msk = msk.compute() if isinstance(msk, da.Array) else msk
        return pl.DataFrame(
            regionprops_table(
                _msk,
                _img,
                properties=props,
                extra_properties=extra_props,
                **kwargs,
            )
        ).with_columns(t=pl.lit(t))

    @staticmethod
    def _load_lineage(lineage_file: Path | None, sequence_id: str | None):
        if not lineage_file:
            return

        if not lineage_file.exists():
            raise FileNotFoundError(
                f"Could not find lineage file at {lineage_file}. "
                "To extract features without lineage pass `linage_file=None`"
            )

        # check for None required: if seq is "00" reading from csv would cast to 0
        if (sequence_id is None) and (lineage_file.stem == "states"):
            raise ValueError("Loading state information requires passing `sequence_id`.")

        lineage = load_lineage(
            lineage_file, with_states=lineage_file.stem == "states", seq_id=sequence_id
        )
        assert isinstance(lineage, pl.DataFrame)

        return lineage

    @staticmethod
    def _validate_lineage(lineage, features):
        lf = features["label"]
        ll = lineage["label"]

        missing_traj = np.setdiff1d(lf, ll)
        missing_feat = np.setdiff1d(ll, lf)
        if len(missing_feat) > 0:
            raise ValueError(
                f"{len(missing_feat)} labels are missing from masks:\n"
                f"Missing: {features.filter(pl.col.label.is_in(ll.implode()))}"
            )
        if len(missing_traj) > 0:
            logger.warning(
                "Several trajectories do not have annotations. "
                "This might be intended if using weakly annotated data."
            )

    @staticmethod
    def _add_lineage_information(features, lineage):
        has_states = "state" in lineage.columns
        if has_states:
            return features.join(
                lineage.select("label", "t", "parent", "state"),
                on=("label", "t"),
                how="left",  # will be null for unannotated trajectories
            )

        return features.join(
            lineage.select("label", "parent"), on="label", how="left"
        ).with_columns(state=pl.lit(0))

    def _check_and_add_lineage_information(self, features, lineage):
        n_pre = features.height
        if lineage is not None:
            try:
                self._validate_lineage(lineage, features)
                logger.debug("Successfully loaded features. Adding lineage information.")

                out = self._add_lineage_information(features, lineage)
                assert out.height == n_pre
                return out
            except ValueError as err:
                logger.debug(
                    "Inconsistencies between features and lineage. "
                    f"Trying to recompute.\nError raised: {err}"
                )

        logger.debug(
            "Successfully loaded features. Did not specify lineage file or found "
            "incompatibility. Returning without state and lineage information."
        )
        return features.with_columns(state=pl.lit(0), parent=pl.lit(0))

    def _normalize_features(self, features: pl.DataFrame):
        if self.feature_norm_fn:
            norm_fn = getattr(normalization, self.feature_norm_fn)
            features_norm = norm_fn(features)
            assert not features_norm.is_empty(), "Error during normalization."
            return features_norm

        return features

    def _apply_area_filter(
        self, features: pl.DataFrame, min_node_area: int | None
    ) -> pl.DataFrame:
        """Drop cells below min_node_area and reassign a clean 0..M-1 index."""
        if min_node_area is None or "area" not in features.columns:
            return features
        filtered = features.filter(pl.col("area") >= min_node_area)
        if "index" in filtered.columns:
            filtered = filtered.drop("index")
        return filtered.with_row_index()

    def _save(self, features: pl.DataFrame, filepath: Path) -> None:
        filepath.parent.mkdir(exist_ok=True, parents=True)
        features.write_parquet(filepath)

    def __repr__(self) -> str:
        """Print summary."""
        props_list = sorted(self.props)
        info = [
            f"feature_names={self.feature_names}",
            f"feature_norm_fn={self.feature_norm_fn}",
            f"props={props_list}",
            f"extra_props={self.extra_props}",
            f"regionprops_kwargs={list(self.regionprops_kwargs.keys())}",
            f"min_node_area={self.min_node_area}",
        ]
        info_str = ",\n    ".join(info)
        return f"HandcraftedExtractor(\n    {info_str}\n  )"

    def __call__(
        self,
        image: da.Array | np.ndarray | None,
        masks: da.Array | np.ndarray,
        lineage_file: Path | None = None,
        sequence_id: str | None = None,
        filepath: Path | None = None,
        validate: bool = False,
        overwrite: bool = True,
        image_percentiles: tuple[float, float] | None = None,
        spacing: tuple[float, ...] | None = None,
        dataset_meta: dict | None = None,
    ) -> pl.DataFrame:
        """Extract handcrafted features for a full sequence.

        Runs regionprops per frame in parallel, applies extra props and
        dataframe-level transforms, normalizes, and joins lineage information.
        Cached parquets at `filepath` are reused when present and compatible.

        Args:
            image: Image sequence with shape `(T, H, W)`. If `None`,
                intensity-based props must not be requested.
            masks: Instance-segmentation masks for `image`.
            lineage_file: CTC `man_track.txt` or `states.txt`. When provided,
                adds `parent` (and `state` for states files) to the output.
            sequence_id: Sequence ID. Required when `lineage_file` is a states
                file.
            filepath: Cache path for the per-cell parquet. Reused if present and
                compatible, otherwise written after recomputation.
            validate: If `True`, verify that the cached parquet covers the
                requested props.
            overwrite: If `True`, write the computed features over an existing
                file. If `False`, only write when no file exists at `filepath`.
            image_percentiles: `(p_low, p_high)` used to scale intensities
                before regionprops.
            spacing: Physical voxel spacing in `(z,) y, x` order passed to regionprops so
                sizes and centroids are physical. Recorded in `meta.json`. When `spacing`
                is `None` spacing is loaded from cache. A mismatching `spacing` recomputes
                if `overwrite`, else raises (since this might affect graph construction).
            dataset_meta: `dataset_identity()` of `image`/`masks`, compared against the
                cache's `meta.json` before reusing it. `None` skips the check. A mismatch
                always recomputes and overwrites, ignoring `overwrite`.
        """
        lineage = self._load_lineage(lineage_file, sequence_id)
        features = self._load(filepath, masks, validate)
        if (
            features is not None
            and filepath is not None
            and spacing is not None
            and not self._cached_spacing_matches(filepath, spacing)
        ):
            if not overwrite:
                raise ValueError(
                    f"Cached node features at {filepath} were built with a different "
                    f"spacing than requested ({spacing}). Set overwrite=True to rebuild."
                )
            logger.warning("Cached node features use a different spacing. Recomputing.")
            features = None
        if (
            features is not None
            and filepath is not None
            and not self._cached_dataset_matches(filepath, dataset_meta)
        ):
            logger.warning(
                "Cached node features were built from different images/masks. "
                "Recomputing."
            )
            features = None
            overwrite = True
        if features is not None:
            # remove state and parent information for flexibility to use with multiple
            # lineage_files, i.e. possibility to add different classes (tasks)
            features = features.select(pl.exclude("state", "parent"))
            logger.debug(
                f"Loading features for {features.height} nodes of "
                f"{np.unique(features['label']).size} trajectories."
            )

            # if features could be loaded, check if already normalized and compatible with
            # provided lineage and recompute or replace information as required
            if features is not None:
                if self.feature_norm_fn and "len_init" not in features.columns:
                    features = self._normalize_features(features)

                for transform_fn in self.extra_transforms_fns:
                    features = transform_fn(features)

                return self._check_and_add_lineage_information(features, lineage)

        if not self.verbose:
            logger.debug(
                f"Computing handcrafted features from scratch. Saving to {filepath}."
            )
        features = self._compute(image, masks, image_percentiles, spacing=spacing)

        if features.is_empty() or features is None:
            raise ValueError(
                "Could not find handcrafted features for given config:\n"
                f"- Images: {image[:5] if isinstance(image, list) else image}\n"
                f"- Masks: {masks[:5] if isinstance(masks, list) else masks}\n"
                f"- Lineage: {lineage_file}"
            )

        # filter before normalization so statistics are computed only on kept cells
        features = self._apply_area_filter(features, self.min_node_area)
        features = self._normalize_features(features)
        features = self._check_and_add_lineage_information(features, lineage)
        assert not features.is_empty(), "Error during adding lineage information."

        if filepath is not None and (not filepath.exists() or overwrite):
            self._save(features, filepath)
            self._write_meta(filepath, spacing, dataset_meta)

        return features


class CellLevelExtractor(BaseExtractor):
    """Compute deep features per cell from individual crops.

    By default uses a DINO-pretrained ViT (see `DINOEncoder` / `MaskedDINOEncoder`). For
    each cell, a square crop is taken around the chosen center (`centroid` or medial axis
    `center`) with a size derived from `input_size_img` (fixed value, a per-cell feature
    like `axis_major_length`, the cell's bounding box (default), or the median across
    cells), resized to `input_size_enc`, optionally normalized to ImageNet statistics, and
    encoded. Due to large object size differences, crops may contain multiple cells. In
    that case `MaskedDINOEncoder` uses the segmentation mask to suppress background and
    neighbors. Embeddings are cached per sequence to a `embeddings.npy` indexed by node
    index, with `meta.json` containing metadata for validation and loading, e.g., a
    sampled content hash of the images/masks the cache was built from.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: str | None = None,
        input_size_img: int | RoiStrategy | tuple[RoiStrategy, int | float] = "bbox",
        input_size_enc: int = 224,
        padding: str | int | None = "constant",
        batch_size: int = 64,
        verbose: bool = True,
        center_name: Literal["centroid", "center"] = "centroid",
        normalize_for_pretrained: bool = True,
        num_workers: int = 0,
        cells_per_item: int = 128,
        prefetch_factor: int = 2,
    ):
        """Initialize cell-level feature extractor.

        Args:
            encoder: Single-cell image encoder.
            device: Torch device.
            input_size_img: Size of bounding box around center of single cell. If `str`,
                will use name of feature (e.g., axis_major_length). It is possible to
                provide an additional scaling factor using a tuple (e.g.,
                `(axis_major_length, 1.5)`). If `None`, will use the bounding box.
            input_size_enc: Size of image for encoder. Bounding box is resized prior to
                encoding. Resize maintains cell aspect ratio, i.e. if input bounding box
                is not square, the short axis is expanded symmetrically.
            padding: Padding employed when bounding box partially lies outside of image.
                If `None`, will shift the bounding box, so that the cell is not centered,
                anymore.
            batch_size: Patches per encoder forward. Every forward holds exactly this
                many except the last of a run, so it also caps the encoder's memory.
            verbose: Print additional info and progress during feature extraction.
            center_name: Type of center point to use, `centroid` (mean of coords) or
                `center` (midpoint on medial axis).
            normalize_for_pretrained: Normalize images to match the expected stats of
                natural images of ImageNet. Used for pretrained models such as DINO.
            num_workers: Worker processes that crop, resize and weight cell patches
                while the encoder runs. `0` does that work inline, which is markedly
                slower since it no longer overlaps the encoder.
            cells_per_item: Cells a worker crops per dataloader item, bounding its memory.
            prefetch_factor: Items each worker keeps queued ahead of the encoder.
        """
        self.encoder = encoder
        self.device = get_device(device)
        self.model_name = self.encoder.arch

        masked = isinstance(self.encoder, MaskedDINOEncoder)
        self.requires_masks = masked

        self.input_size_img = input_size_img
        self.input_size_img_strategy = None
        self.input_size_img_scaling = 1.0

        if isinstance(input_size_img, tuple):
            self.input_size_img_strategy, self.input_size_img_scaling = input_size_img
        elif isinstance(input_size_img, str):
            self.input_size_img_strategy = input_size_img

        self.padding = padding
        self.input_size_enc = input_size_enc
        self.center_name = center_name

        img_str = self.input_size_img_strategy or str(input_size_img)
        if self.input_size_img_scaling != 1:
            img_str += f"x{self.input_size_img_scaling}".replace(".", "_")

        name = (
            f"{self.model_name}_"
            f"{'masked' if masked else ''}_"
            f"img-{img_str}_"
            f"enc-{input_size_enc}_"
            f"pad-{padding}_"
            f"{center_name}"
        )
        if not normalize_for_pretrained:
            name += "_nonorm"
        super().__init__(name=name, required_feats=[center_name])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cells_per_item = cells_per_item
        self.prefetch_factor = prefetch_factor
        self.normalize_for_pretrained = normalize_for_pretrained
        # patch weights depend only on the mask, so they are built alongside the crops
        self.weighting = (
            PatchWeighting(gamma=encoder.gamma, patchsize=encoder.patchsize)
            if isinstance(encoder, MaskedDINOEncoder)
            else None
        )
        self.verbose = verbose

        # opened embedding memmaps keyed by cache dir, so the per-item load path does
        # not reopen the file on every __getitem__. one extractor instance is shared
        # across sequences, hence the dict.
        self._open_embedding_caches: dict[str, np.memmap] = {}

    def __getstate__(self) -> dict:
        """Drop the open embedding caches when pickling (e.g., to dataloader workers)."""
        # np.memmap does not pickle across processes, so forkserver workers reopen their
        # own (sharing the OS page cache).
        state = super().__getstate__()
        state["_open_embedding_caches"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state with no embedding caches open yet."""
        self.__dict__.update(state)
        self._open_embedding_caches = {}

    def _load_encoder(self):
        self.encoder.to(self.device)
        self.encoder.eval()

    def _unload_encoder(self):
        if hasattr(self, "encoder") and hasattr(self.encoder, "_unload_model"):
            self.encoder._unload_model()  # type: ignore
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _px_center(self, node_feats, axis: int) -> np.ndarray:
        """Pixel-space center on a spatial axis, preferring the `_px` column."""
        col = f"{self.center_name}-{axis}"
        src = f"{col}_px" if f"{col}_px" in node_feats.columns else col
        return node_feats[src].to_numpy().astype(int)

    def _get_boxes_and_padding(self, node_feats, image_size, spacing=None):
        h_img, w_img = image_size

        ndim = sum(
            bool(re.fullmatch(rf"{re.escape(self.center_name)}-\d+", c))
            for c in node_feats.columns
        )
        volumetric = ndim == 3
        ay, ax = (1, 2) if volumetric else (0, 1)

        cz = self._px_center(node_feats, 0) if volumetric else None
        cy = self._px_center(node_feats, ay)
        cx = self._px_center(node_feats, ax)
        indices = node_feats["index"].to_numpy()
        time_indices = node_feats["t"].to_numpy()
        labels = node_feats["label"].to_numpy()

        ymin = node_feats[f"bbox-{ay}"].to_numpy()
        xmin = node_feats[f"bbox-{ax}"].to_numpy()
        ymax = node_feats[f"bbox-{ndim + ay}"].to_numpy()
        xmax = node_feats[f"bbox-{ndim + ax}"].to_numpy()

        strategy = self.input_size_img_strategy
        size_features = ("axis_major_length", "len_init", "median_size")
        # convert lengths to px (crop box is in pixels, assumes square in-plane pixels)
        feat_to_px = float(np.mean(spacing[-2:])) if needs_spacing(spacing) else 1.0  # type: ignore
        if strategy == "bbox":
            # overwrite centers to use the in-plane bbox center
            cy, cx = (ymax + ymin) // 2, (xmax + xmin) // 2
            base_size = np.maximum(ymax - ymin, xmax - xmin)
            s_full = (base_size * self.input_size_img_scaling).astype(int)

        elif strategy in ("axis_major_length", "len_init") and not volumetric:
            # use feature to determine bounding box size
            feat = node_feats[strategy].to_numpy() / feat_to_px
            s_full = (feat * self.input_size_img_scaling).astype(int)

        elif strategy == "median_size" and not volumetric:
            median_s = np.median(node_feats["axis_major_length"]) / feat_to_px
            s_full = int(median_s * self.input_size_img_scaling)

        elif strategy in size_features and volumetric:
            logger.debug(f"3D crop sizing uses in-plane bbox for strategy {strategy!r}.")
            base_size = np.maximum(ymax - ymin, xmax - xmin)
            s_full = (base_size * self.input_size_img_scaling).astype(int)

        else:
            assert isinstance(self.input_size_img, int)
            s_full = self.input_size_img

        # one size per cell, so cropping never has to branch on scalar vs per-cell sizing
        s_full = np.broadcast_to(s_full, cy.shape).astype(int)
        s_half = s_full // 2

        # calculate box coords ignoring image borders
        # if padding is None, the box will be shifted so that it touches the image border
        if self.padding is None:
            # if the box goes out of bounds, we slide it back in
            ymin_ideal = np.clip(cy - s_half, 0, h_img - s_full)
            xmin_ideal = np.clip(cx - s_half, 0, w_img - s_full)
        else:
            ymin_ideal = cy - s_half
            xmin_ideal = cx - s_half

        ymax_ideal = ymin_ideal + s_full
        xmax_ideal = xmin_ideal + s_full

        # source image (out-of-bounds is clipped, since no pixels)
        src_ymin = np.clip(ymin_ideal, 0, h_img)
        src_xmin = np.clip(xmin_ideal, 0, w_img)
        src_ymax = np.clip(ymax_ideal, 0, h_img)
        src_xmax = np.clip(xmax_ideal, 0, w_img)

        # target patch (out-of-bounds leads to padding)
        dst_ymin = src_ymin - ymin_ideal
        dst_xmin = src_xmin - xmin_ideal
        h_valid = src_ymax - src_ymin
        w_valid = src_xmax - src_xmin
        dst_ymax = dst_ymin + h_valid
        dst_xmax = dst_xmin + w_valid

        return {
            "indices": indices,
            "labels": labels,
            "t": time_indices,
            "z": cz,
            "src": np.stack([src_ymin, src_xmin, src_ymax, src_xmax], axis=1),
            "dst": np.stack([dst_ymin, dst_xmin, dst_ymax, dst_xmax], axis=1),
            "pad_needed": (h_valid != s_full) | (w_valid != s_full),
            "s_full": s_full,
        }

    def _build_patch_dataset(
        self,
        image: Any,
        coords: dict,
        masks: Any,
        image_percentiles: tuple[float, float] | None,
        for_encoder: bool,
    ) -> CellPatchDataset:
        """Dataset producing the crops for `coords`.

        `for_encoder` keeps patches single-channel and swaps mask patches for the
        precomputed weights the encoder consumes.
        """
        return CellPatchDataset(
            image,
            masks,
            coords,
            input_size_enc=self.input_size_enc,
            padding=self.padding,
            cells_per_item=self.cells_per_item,
            image_percentiles=image_percentiles,
            weighting=self.weighting if for_encoder else None,
            expand_channels=not for_encoder,
        )

    def _effective_num_workers(self, image: Any) -> int:
        """Workers that can prepare patches for this input, or 0 where they would hurt."""
        if self.num_workers <= 0:
            return 0

        if get_worker_info() is not None:
            # this call is already inside a worker
            return 0

        if isinstance(image, da.Array):
            if image.chunksize[0] != 1:
                # a coarser chunk means every item decodes more than the frame it needs
                logger.debug("Frames are not chunked individually, cropping inline.")
                return 0
        elif getattr(image, "nbytes", 0) > _MAX_WORKER_COPY_BYTES:
            # forkserver pickles an in-memory stack into every worker
            logger.debug("Image stack is too large to copy per worker, cropping inline.")
            return 0

        return self.num_workers

    def _weighting_key(self) -> str | None:
        """Cache discriminator for how patch weights are derived from the mask."""
        if self.weighting is None:
            return None
        return f"per-sample_gamma-{self.weighting.gamma}_patch-{self.weighting.patchsize}"

    def get_single_cell_images(
        self,
        image: list[Path | np.ndarray] | np.ndarray,
        coords: dict,
        masks: list[Path | np.ndarray] | np.ndarray | None = None,
        image_percentiles: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
        """Extract and pad single cell images."""
        n_nodes = len(coords["indices"])
        enc = self.input_size_enc

        patches = torch.zeros((n_nodes, 3, enc, enc), dtype=torch.float32)
        mask_patches = (
            torch.zeros((n_nodes, 1, enc, enc), dtype=torch.float32)
            if masks is not None
            else None
        )

        dataset = self._build_patch_dataset(
            image, coords, masks, image_percentiles, for_encoder=False
        )
        for chunk in dataset:
            rows = chunk["rows"]
            if not len(rows):
                continue

            chunk_patches = chunk["patches"]
            if self.normalize_for_pretrained:
                chunk_patches = normalize_imagenet(chunk_patches)
            patches[rows] = chunk_patches

            if mask_patches is not None and chunk["masks"] is not None:
                mask_patches[rows] = chunk["masks"]

        return patches, mask_patches, coords["indices"].tolist()

    def _predict_batch(
        self, patches: torch.Tensor, aux: torch.Tensor | None
    ) -> torch.Tensor:
        imgs = patches.to(self.device, non_blocking=True)
        if imgs.shape[1] == 1:
            imgs = imgs.expand(-1, 3, -1, -1)
        if self.normalize_for_pretrained:
            imgs = normalize_imagenet(imgs)

        if aux is not None:
            embeds = self.encoder((imgs, aux.to(self.device, non_blocking=True)))
        else:
            embeds = self.encoder(imgs)
        return embeds.cpu()

    def _compute(
        self,
        image: da.Array | np.ndarray,
        coords: dict,
        masks: da.Array | np.ndarray | None,
        image_percentiles: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        if self.encoder is None:
            raise RuntimeError("Predictor not loaded. Call _load_predictor() first.")

        num_workers = self._effective_num_workers(image)
        dataset = self._build_patch_dataset(
            image, coords, masks, image_percentiles, for_encoder=True
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
            persistent_workers=False,
            multiprocessing_context=get_multiprocessing_context(num_workers),
        )

        all_embeds: list[torch.Tensor] = []
        all_indices: list[int] = []

        patch_buffer: list[torch.Tensor] = []
        aux_buffer: list[torch.Tensor] = []

        # progress counts encoded cells, not items: an item only buffers patches, so an
        # item counter runs ahead a whole batch and then stalls through the forward
        total = dataset.n_cells
        message = f"Encoding cells ({self.model_name})"
        bar = tqdm(total=total, desc=message, unit="cell") if self.verbose else None

        def flush(drain: bool = False) -> None:
            """Forward buffered patches in exact `batch_size` slices.

            Only the last forward of a run is smaller. Emitting whole items instead would
            vary the batch by up to `cells_per_item`, and a caching allocator keeps a
            block per distinct shape, so the pool grows far past what the tensors need.
            """
            nonlocal n_buffered, n_encoded
            if not patch_buffer:
                return
            patches = torch.cat(patch_buffer)
            aux = torch.cat(aux_buffer) if aux_buffer else None

            start = 0
            whole = len(patches) - len(patches) % self.batch_size
            limit = len(patches) if drain else whole
            while start < limit:
                stop = min(start + self.batch_size, len(patches))
                with torch.no_grad():
                    all_embeds.append(
                        self._predict_batch(
                            patches[start:stop],
                            None if aux is None else aux[start:stop],
                        )
                    )
                n_encoded += stop - start
                if bar is not None:
                    bar.update(stop - start)
                report(self.progress, "encode", n_encoded, total, message)
                start = stop

            patch_buffer.clear()
            aux_buffer.clear()
            if start < len(patches):
                patch_buffer.append(patches[start:])
                if aux is not None:
                    aux_buffer.append(aux[start:])
            n_buffered = len(patches) - start

        report(self.progress, "encode", 0, total, message)
        n_buffered = 0
        n_encoded = 0
        for chunk in loader:
            rows = chunk["rows"]
            if len(rows):
                patch_buffer.append(chunk["patches"])
                aux = chunk["weights"] if chunk["weights"] is not None else chunk["masks"]
                if aux is not None:
                    aux_buffer.append(aux)
                all_indices.extend(coords["indices"][rows.numpy()].tolist())
                n_buffered += len(rows)

            if n_buffered >= self.batch_size:
                flush()
            # repeats the count rather than advancing it: an item is also where a
            # cancelling callback gets to raise
            report(self.progress, "encode", n_encoded, total, message)

        flush(drain=True)
        if bar is not None:
            bar.close()

        if not all_embeds:
            return torch.empty(0), []

        return torch.cat(all_embeds, dim=0), all_indices

    def _open_embeddings(self, cache_dir: Path) -> np.memmap:
        """Memmap the cache's `embeddings.npy` once, then reuse it.

        The memmap is kept per cache dir so the per-item load path does not reopen the
        file on every `__getitem__`. Only touched rows are paged in by the OS.
        """
        key = str(cache_dir)
        embeddings = self._open_embedding_caches.get(key)
        if embeddings is None:
            embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
            self._open_embedding_caches[key] = embeddings
        return embeddings

    def _validate(self, meta: dict, dataset_meta: dict | None) -> bool:
        """Check metadata and dataset-identity compatibility of a cache."""
        compatible = (
            meta.get("model_name") == self.model_name
            and meta.get("input_size_img") == self.input_size_img
            and meta.get("input_size_enc") == self.input_size_enc
            and meta.get("weighting") == self._weighting_key()
        )
        if not compatible:
            logger.warning("Cache has incompatible metadata.")
            return False

        if dataset_meta is not None and not dataset_identity_matches(
            dataset_meta, meta.get("dataset")
        ):
            logger.warning("Cache was built from different images/masks.")
            return False
        return True

    def cache_covers(
        self,
        output_dir: Path | None,
        num_nodes: int,
        dataset_meta: dict | None = None,
    ) -> bool:
        """Whether the cache under `output_dir` is compatible and holds `num_nodes` rows.

        Used to decide whether the full-sequence cache still has to be built.
        """
        if output_dir is None:
            return False
        cache_dir = output_dir / self.name
        meta_file = cache_dir / "meta.json"
        if not (cache_dir / "embeddings.npy").exists() or not meta_file.exists():
            return False
        try:
            meta = json.loads(meta_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return self._validate(meta, dataset_meta) and meta.get("n_nodes", 0) >= num_nodes

    def _load(
        self,
        cache_dir: Path,
        indices: list[int],
        dataset_meta: dict | None,
    ) -> torch.Tensor | None:
        if not cache_dir.exists():
            return None

        # already-open memmap doubles as "already validated this process"
        if str(cache_dir) not in self._open_embedding_caches:
            meta = json.loads((cache_dir / "meta.json").read_text())
            if not self._validate(meta, dataset_meta):
                return None

        embeddings = self._open_embeddings(cache_dir)
        idx = np.asarray(indices)
        if len(idx) and int(idx.max()) >= len(embeddings):
            raise IndexError(
                f"Deep feature cache at {cache_dir} covers {len(embeddings)} nodes but "
                f"index {int(idx.max())} was requested. The cache is incomplete; delete "
                "it so it is rebuilt for the full sequence."
            )
        embeds = np.asarray(embeddings[idx]).astype(np.float32)
        return torch.from_numpy(embeds)

    def _save(
        self,
        features: torch.Tensor,
        cache_dir: Path,
        indices: list[int],
        dataset_meta: dict | None,
        n_nodes: int | None = None,
    ):
        if not self.verbose:
            logger.debug(f"Saving {len(features)} DINO features to {cache_dir}")

        cache_dir.mkdir(exist_ok=True, parents=True)
        idx = np.asarray(indices)
        feats = features.cpu().numpy()
        # size by the requested node count, not by the computed indices: a cell without an
        # extractable patch would otherwise truncate the array and force a recompute
        if n_nodes is None:
            n_nodes = int(idx.max()) + 1 if len(idx) else 0
        dim = feats.shape[1] if feats.ndim == 2 else 0
        embeddings = np.zeros((n_nodes, dim), dtype=np.float16)
        embeddings[idx] = feats.astype(np.float16)
        np.save(cache_dir / "embeddings.npy", embeddings)

        meta = {
            "model_name": self.model_name,
            "input_size_img": self.input_size_img,
            "input_size_enc": self.input_size_enc,
            "weighting": self._weighting_key(),
            "dataset": dataset_meta,
            "dim": dim,
            "n_nodes": n_nodes,
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta))

        # drop any stale open memmap so a later load reflects the freshly written data.
        self._open_embedding_caches.pop(str(cache_dir), None)

    def __repr__(self) -> str:
        """Print summary of feature encoder."""
        info = [
            f"model_name={self.model_name!r}",
            f"output_name={self.name!r}",
            f"input_size_img={self.input_size_img}",
            f"input_size_enc={self.input_size_enc}",
        ]
        info_str = ",\n    ".join(info)
        return f"CellLevelExtractor(\n    {info_str}\n  )"

    def __call__(
        self,
        image: da.Array | np.ndarray,
        node_feats: pl.DataFrame,
        masks: da.Array | np.ndarray | None = None,
        output_dir: Path | None = None,
        dataset_meta: dict | None = None,
        timepoints: Sequence[int] | np.ndarray | None = None,
        image_percentiles: tuple[float, float] | None = None,
        spacing: tuple[float, ...] | None = None,
    ):
        """Compute deep embeddings per cell from individual crops.

        For each requested cell (optionally filtered to `timepoints`), a crop is
        taken around the chosen center, resized to `input_size_enc`, optionally
        masked, and encoded in batches. Cached embeddings under `output_dir` are
        reused when present and compatible.

        Args:
            image: Image sequence with shape `(T, H, W)`.
            node_feats: Position (tyx) and handcrafted single-cell features.
                Must include the center and bbox columns used to place each crop.
            masks: Instance-segmentation masks for `image`. Required when the
                encoder consumes masks (e.g., `MaskedDINOEncoder`).
            output_dir: Directory under which the `{name}/` cache is written, where
                `name` encodes the extractor configuration. If `None`,
                embeddings are not cached and are recomputed on every call.
            dataset_meta: `dataset_identity()` of `image`/`masks`, compared against
                the cache's `meta.json` before reusing it. `None` skips the check.
            timepoints: Frame indices to process. If `None`, all rows of
                `node_feats` are processed. A windowed call never writes the cache,
                since its embeddings would only cover part of the sequence.
            image_percentiles: `(p_low, p_high)` used to scale intensities
                before encoding.
            spacing: Physical voxel spacing in `(z,) y, x` order, used to convert physical
                size features back to pixels for crop sizing.
        """
        if self.requires_masks and masks is None:
            raise ValueError(f"{self.model_name} requires masks, but none were provided.")

        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)

        positions = (
            node_feats
            if timepoints is None
            else node_feats.filter(pl.col("t").is_in(timepoints))
        )
        if positions.height == 0:
            logger.warning("No nodes found for the given timepoints.")
            return torch.empty(0)

        cache_dir = output_dir / self.name if output_dir else None
        indices_requested = positions["index"].to_numpy().tolist()

        if cache_dir and cache_dir.exists():
            embeds = self._load(cache_dir, indices_requested, dataset_meta)
            if embeds is not None:
                logger.debug(f"Loaded {len(embeds)} features from compatible cache.")
                return embeds

        if not self.verbose:
            logger.debug("Cache missing or incompatible. Computing from scratch.")

        img_frame_0 = image[0].compute() if isinstance(image[0], da.Array) else image[0]
        img_size = img_frame_0.shape[-2:]
        pos_coords = self._get_boxes_and_padding(positions, img_size, spacing)

        try:
            self._load_encoder()
            # pass full images list/array, logic handles extracting patches frame by frame
            embeds_computed, indices_computed = self._compute(
                image, pos_coords, masks, image_percentiles
            )

            # a windowed call only covers part of the sequence, so writing it would
            # replace a full cache with a partial one
            if cache_dir and timepoints is None:
                self._save(
                    embeds_computed,
                    cache_dir,
                    indices_computed,
                    dataset_meta,
                    n_nodes=int(positions["index"].to_numpy().max()) + 1,
                )

            # reorder to match requested indices
            if indices_computed != indices_requested:
                idx_map = {idx: i for i, idx in enumerate(indices_computed)}
                reorder_idx = [idx_map[idx] for idx in indices_requested]
                return embeds_computed[reorder_idx]

            return embeds_computed

        except Exception as err:
            logger.error("Could not compute deep features.")
            raise err

        finally:
            # encoders are only used in prepare_data, which is single-threaded, so this
            # frees the weights rather than saving memory during extraction
            self._unload_encoder()


def _get_deep_dir(
    deep_feat_extractor: BaseExtractor, feature_dir: Path, seq_id: str, seg: str
) -> Path:
    assert hasattr(deep_feat_extractor, "name")

    if isinstance(deep_feat_extractor, CellLevelExtractor):
        # cell-level features depend on segmentation
        return _get_seg_dir(feature_dir, seq_id, seg) / "embeds"
    # image-level features are segmentation-independent and already fully named
    return feature_dir / seq_id / "embeds" / deep_feat_extractor.name


def _get_seg_dir(feature_dir: Path, seq_id: str, seg: str) -> Path:
    """Root directory for a sequence's segmentation-dependent caches (nodes, edges)."""
    return feature_dir / seq_id / seg
