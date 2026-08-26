"""Trajectory visualization: annotated frames, videos, and napari layers.

`TrajectoryVisualizer` is a utility class to visualize tracking results using static
overlays (`show`), videos (`export_video`), or Napari layers (`show_napari`). The class
can be instantiated from arrays, from a `GraphDataset` with `from_dataset`, or from a
results directory with `from_dir`. For a single sequence, the class can contain multiple
results as well as the ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import dask.array as da
import numpy as np
import polars as pl
from skimage.measure import find_contours
from tqdm.auto import tqdm

try:
    import cmap
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes

    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

from baclct.features.extractors import HandcraftedExtractor
from baclct.io import (
    coordinate_columns,
    find_lineage_file,
    get_percentiles,
    load_images_and_masks,
    scale_percentiles,
)

if TYPE_CHECKING:
    import napari
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes

    from baclct.data.dataset import GraphDataset


class TrajectoryVisualizer:
    """Class to load and visualize cell trajectories and segmentations."""

    def __init__(
        self,
        images: np.ndarray | da.Array,
        masks: np.ndarray | da.Array | dict[str, np.ndarray | da.Array],
        tracks: pl.DataFrame | dict[str, pl.DataFrame] | None = None,
        name: str | None = "trajectory",
    ):
        """Initialize TrajectoryVisualizer.

        Every key of `masks` and `tracks` is a different result for the same sequence
        (e.g. ground truth next to a prediction), so they share `images`.

        Args:
            images: Image sequence of shape (T, H, W).
            masks: Mask sequence of shape (T, H, W).
            tracks: Tracking dataframe or a dict mapping prediction names to dataframes.
            name: Name of the visualizer instance.
        """
        if not _VIZ_AVAILABLE:
            raise ImportError(
                "TrajectoryVisualizer requires visualization extras. "
                "Install with: pip install baclct[train]"
            )

        if name is None:
            name = "traj"

        self.images = images
        self.masks: dict[str, np.ndarray | da.Array] = (
            dict(masks) if isinstance(masks, dict) else {name: masks}
        )
        self.name = name

        self.tracks: dict[str, pl.DataFrame]
        if tracks is None:
            extractor = HandcraftedExtractor(verbose=False, feature_norm_fn=None)
            self.tracks = {
                k: extractor(images, cast("np.ndarray", msk))
                for k, msk in self.masks.items()
            }
        elif isinstance(tracks, pl.DataFrame):
            self.tracks = {name: tracks}
        else:
            self.tracks = tracks

        self.tracks = {k: self._prepare_tracks(v) for k, v in self.tracks.items()}
        self.coords = coordinate_columns(next(iter(self.tracks.values())))

    @classmethod
    def from_dataset(
        cls,
        dataset: GraphDataset,
        name: str | None = None,
    ) -> TrajectoryVisualizer:
        """Initialize from GraphDataset."""
        if name is None:
            name = dataset.sequence_id

        images = dataset.images
        images = images.compute() if isinstance(images, da.Array) else images
        images = cast(np.ndarray, images)
        perc = dataset.image_percentiles or get_percentiles(images)
        images = scale_percentiles(images, perc)

        masks = dataset.masks
        masks = masks.compute() if isinstance(masks, da.Array) else masks
        tracks = dataset.node_feats

        return cls(images, masks, tracks, name=name)

    @classmethod
    def from_dir(
        cls,
        data_dir: Path | str | dict[str, Path | str],
        seq_id: str,
        format: Literal["ctc", "flat", "dirs"] = "ctc",
        segmentation_name: str | None | dict[str, str] = None,
        img_name: str | None = None,
        name: str | None = None,
        with_states: bool = False,
        require_images: bool = False,
    ) -> TrajectoryVisualizer:
        """Initialize from directory."""
        if not isinstance(data_dir, dict):
            data_dirs = {seq_id: data_dir}
            seg_names = {seq_id: segmentation_name}
        else:
            data_dirs = data_dir
            seg_names = (
                segmentation_name
                if isinstance(segmentation_name, dict)
                else dict.fromkeys(data_dir, segmentation_name)
            )

        masks_dict = {}
        track_dict = {}
        images_loaded = None

        for n, dd in data_dirs.items():
            seq_name = n or seq_id
            dd = Path(dd)

            # resolved per directory, so a derived name does not leak into the next
            seg_name = seg_names[n]
            if not seg_name:
                if format == "flat":
                    tracked = list(dd.glob(f"{seq_id}_tracked.tif"))
                    seg_name = "tracked" if tracked else "masks"
                elif format == "ctc":
                    seg_name = "GT"

            images, masks, features = TrajectoryVisualizer._process_dir(
                dd,
                seq_id,
                format,
                cast(str | None, seg_name),
                img_name,
                with_states,
                require_images,
            )
            masks_dict[seq_name] = masks
            track_dict[seq_name] = features

            if images is not None and images_loaded is None:
                images_loaded = images

        # use an empty black image array if images were not loaded
        if images_loaded is None:
            first_mask = next(iter(masks_dict.values()))
            images_loaded = np.zeros_like(first_mask, dtype=np.uint8)
        else:
            images_loaded = scale_percentiles(
                images_loaded, get_percentiles(images_loaded)
            )

        return cls(images_loaded, masks_dict, track_dict, name=name or seq_id)

    def show(
        self,
        t: int,
        n_frames: int = 10,
        plot_contours: bool = True,
        color_map: cmap.Colormap | None = None,
        seg_color_map: cmap.Colormap | None = None,
        shuffle_colors: bool = False,
        verbose: bool = True,
        prediction_key: str | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Show trajectories for a single frame."""
        prediction_key = self._resolve_prediction_key(prediction_key, verbose)
        tracks = self.tracks[prediction_key]
        masks = self.masks[prediction_key]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title(f"{self.name} ({prediction_key}) - t={t}")

        _plot_frame(
            ax,
            tracks,
            self.images[t],
            t,
            n_frames,
            self.coords,
            masks=masks[t],
            plot_contours=plot_contours,
            color_map=color_map,
            seg_color_map=seg_color_map,
            shuffle_colors=shuffle_colors,
            verbose=verbose,
        )

        return fig, ax

    def export_video(
        self,
        output_path: Path | str,
        n_frames: int = 30,
        fps: int = 10,
        plot_contours: bool = True,
        color_map: cmap.Colormap | None = None,
        seg_color_map: cmap.Colormap | None = None,
        shuffle_colors: bool = False,
        prediction_key: str | None = None,
    ):
        """Export trajectory visualization to a video file.

        Args:
            output_path: Destination path for the video (.mp4, .avi, etc.).
            n_frames: Number of trailing frames to show in trajectories.
            fps: Frames per second for the output video.
            plot_contours: Whether to overlay segmentation contours.
            color_map: Colormap for track trajectories.
            seg_color_map: Colormap for segmentation outlines.
            shuffle_colors: Shuffle colormap assignment.
            prediction_key: Which prediction set to visualise.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prediction_key = self._resolve_prediction_key(prediction_key, verbose=False)
        tracks = self.tracks[prediction_key]
        masks = self.masks[prediction_key]
        num_frames = len(self.images)

        fig, ax = plt.subplots(figsize=(10, 10))

        def update(t):
            ax.clear()
            _plot_frame(
                ax,
                tracks,
                self.images[t],
                t,
                n_frames,
                self.coords,
                masks=masks[t],
                plot_contours=plot_contours,
                color_map=color_map,
                seg_color_map=seg_color_map,
                shuffle_colors=shuffle_colors,
                verbose=False,
            )

        anim = FuncAnimation(fig, update, frames=num_frames, blit=False)
        writer = animation.FFMpegWriter(fps=fps)

        with tqdm(total=num_frames, desc=f"Saving {output_path.name}") as pbar:
            anim.save(
                str(output_path),
                writer=writer,
                progress_callback=lambda i, n: pbar.update(1),
            )

        plt.close(fig)

    def show_napari(
        self,
        prediction_key: str | None = None,
        viewer: napari.Viewer | None = None,
        scale: tuple[float, ...] | None = None,
    ) -> napari.Viewer:
        """Open the images and one labels/tracks layer pair per prediction in napari.

        Args:
            prediction_key: Show only this prediction. All of them are added if `None`.
            viewer: Viewer to add the layers to. A new one is opened if `None`.
            scale: Physical size of one pixel, passed through to every layer.
        """
        try:
            import napari

            from baclct.napari._layers import add_result_layers
        except ImportError as err:
            raise ImportError(
                "show_napari requires napari. Install with: pip install baclct[napari]"
            ) from err

        keys = (
            [self._resolve_prediction_key(prediction_key)]
            if prediction_key is not None
            else list(self.tracks)
        )

        if viewer is None:
            viewer = napari.Viewer()
        if self.name not in viewer.layers:
            viewer.add_image(self.images, name=self.name, scale=scale)

        for key in keys:
            masks = self.masks[key]
            if masks.shape != self.images.shape:
                raise ValueError(
                    f"Masks of '{key}' have shape {masks.shape}, but the images have "
                    f"shape {self.images.shape}."
                )
            add_result_layers(viewer, self.tracks[key], masks, key, scale=scale)

        return viewer

    @staticmethod
    def _prepare_tracks(df: pl.DataFrame) -> pl.DataFrame:
        """Normalize to the `BacLCT.track()` schema: `t`, `label`, `parent`, positions."""
        # a corrected frame keeps the mask label in `label`, so the track id wins
        for track_col, col in (("label_track", "label"), ("parent_track", "parent")):
            if track_col in df.columns:
                df = df.drop(col, strict=False).rename({track_col: col})

        coords = coordinate_columns(df)
        if not coords:
            raise ValueError("DataFrame must contain 'center-*' or 'centroid-*' columns.")
        for col in ("t", "label"):
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain column '{col}'")

        if "parent" not in df.columns:
            df = df.with_columns(parent=pl.lit(0))

        return df

    @staticmethod
    def _process_dir(
        data_dir: Path,
        seq_id: str,
        format: Literal["ctc", "flat", "dirs"] = "ctc",
        segmentation_name: str | None = None,
        img_name: str | None = None,
        with_states: bool = False,
        require_images: bool = True,
    ):
        images, masks, *_ = load_images_and_masks(
            data_dir,
            seq_id,
            data_format=format,
            lazy=False,
            segmentation_name=segmentation_name,
            img_name=img_name,
            strict=require_images,
        )
        lineage_file, _ = find_lineage_file(
            data_dir,
            seq_id,
            data_format=format,
            segmentation_name=segmentation_name,
            with_states=with_states,
        )
        extractor = HandcraftedExtractor(verbose=False, feature_norm_fn=None)
        features = extractor(
            image=images,
            masks=masks,
            lineage_file=lineage_file,
            sequence_id=seq_id,
            validate=False,
            overwrite=False,
        )

        return images, masks, features

    def _resolve_prediction_key(
        self, prediction_key: str | None, verbose: bool = True
    ) -> str:
        """Return the prediction key, defaulting to the first available."""
        if prediction_key is None:
            prediction_key = str(next(iter(self.tracks.keys())))
            if len(self.tracks) > 1 and verbose:
                print(f"Multiple predictions found. Defaulting to '{prediction_key}'.")
        return prediction_key


def _plot_frame(
    ax: Axes,
    df: pl.DataFrame,
    images: np.ndarray,
    t: int,
    n_frames: int,
    coords: list[str],
    masks: np.ndarray | None = None,
    plot_contours: bool = False,
    color_map: cmap.Colormap | None = None,
    seg_color_map: cmap.Colormap | None = None,
    shuffle_colors: bool = False,
    verbose: bool = True,
):
    """Plots a single frame of the tracking results.

    Args:
        ax: Matplotlib axes to plot on.
        df: Dataframe containing tracking data.
        images: Single image frame that will be plotted.
        t: The current time point.
        n_frames: The number of previous frames to show in the lineage.
        coords: Position columns, in axis order.
        masks: The segmentation masks for the current time point.
        plot_contours: Whether to plot the contours of the segmentation masks. Otherwise,
            masks are filled.
        color_map: Sequential colormap for trajectories and contours.
        seg_color_map: Sequential (or static) colormap for segmentation outlines.
        shuffle_colors: Should colormap be shuffled or correspond to labels.
        verbose: Output print statements.
    """
    if verbose:
        print(f"Plotting frame t={t}")
    ax.imshow(images, cmap="gray")

    # get a colormap
    unique_tracks = df["label"].unique().to_list()
    if shuffle_colors:
        rng = np.random.default_rng()
        rng.shuffle(unique_tracks)

    if color_map is None:
        color_map = cmap.Colormap("glasbey:glasbey")

    track_color_map = {
        track_id: np.asarray(color_map((i % 256) / 255))[:3]
        for i, track_id in enumerate(unique_tracks)
    }
    if seg_color_map is None:
        seg_color_map_dict = {}
    else:
        seg_color_map_dict = {
            track_id: np.asarray(seg_color_map(i / len(unique_tracks)))[:3]
            for i, track_id in enumerate(unique_tracks)
        }

    _plot_lineage(ax, df, t, n_frames, coords, track_color_map, verbose=verbose)
    _plot_divisions(ax, df, t, n_frames, coords, verbose=verbose)

    if plot_contours and masks is not None:
        _plot_contours(
            ax,
            masks,
            df.filter(pl.col("t") == t),
            seg_color_map_dict or track_color_map,
        )

    ax.axis("off")


def _plot_lineage(
    ax: Axes,
    df: pl.DataFrame,
    t: int,
    n_frames: int,
    coords: list[str],
    track_color_map: dict,
    verbose: bool = True,
):
    """Plots the lineage of the tracks.

    Args:
        ax: The matplotlib axes to plot on.
        df: The dataframe with the tracking data.
        t: The current time point.
        n_frames: The number of previous frames to show in the lineage.
        coords: Position columns, in axis order.
        track_color_map: A dictionary mapping track IDs to colors.
        verbose: Output print statements.
    """
    lineage_df = df.filter(pl.col("t").is_between(t - n_frames, t))
    if verbose:
        print(f"Found {len(lineage_df)} points for lineage plot.")

    for (track_id,), track_df in lineage_df.group_by("label"):
        if track_id not in track_color_map:
            print(f"Could not find label for {track_id}.")
            continue

        color = track_color_map[track_id]
        points = track_df.sort("t")[[coords[1], coords[0]]].to_numpy()

        alphas = np.linspace(0.2, 1.0, len(points))
        for i in range(len(points) - 1):
            ax.plot(
                points[i : i + 2, 0],
                points[i : i + 2, 1],
                color=color,
                alpha=alphas[i],
                linewidth=1,
            )


def _plot_divisions(
    ax: Axes,
    df: pl.DataFrame,
    t: int,
    n_frames: int,
    coords: list[str],
    verbose: bool = True,
):
    """Plots the divisions as white lines with decaying alpha.

    Args:
        ax: The matplotlib axes to plot on.
        df: The dataframe with the tracking data.
        t: The current time point.
        n_frames: The number of previous frames to show in the lineage.
        coords: Position columns, in axis order.
        verbose: Output print statements.
    """
    t_min = max(0, t - n_frames)
    division_df = (
        df.filter(pl.col("t").is_between(t_min, t), pl.col("parent") != 0)
        .sort("t", descending=True)
        .unique("label", keep="first")
    )
    if verbose:
        print(f"Found {len(division_df)} points for division plot.")

    for daughter_cell in division_df.iter_rows(named=True):
        t_div = daughter_cell["t"]
        parent_cell = df.filter(
            (pl.col("t") == t_div - 1) & (pl.col("label") == daughter_cell["parent"])
        )
        if parent_cell.is_empty():
            continue

        parent_pos = parent_cell[[coords[1], coords[0]]].to_numpy()[0]
        daughter_pos = np.array([daughter_cell[coords[1]], daughter_cell[coords[0]]])

        # calculate alpha based on the age of the division
        alpha = 0.2 + 0.8 * ((t_div - t_min) / n_frames)
        alpha = max(0.0, min(1.0, alpha))  # ensure alpha is in [0,1]

        ax.plot(
            [parent_pos[0], daughter_pos[0]],
            [parent_pos[1], daughter_pos[1]],
            color="white",
            linewidth=1.5,
            alpha=alpha,
        )


def _plot_contours(
    ax: Axes,
    masks: np.ndarray,
    df_t: pl.DataFrame,
    track_color_map: dict,
):
    """Plots the contours of the segmentation masks.

    Args:
        ax: The matplotlib axes to plot on.
        masks: The segmentation masks for the current time point.
        df_t: The dataframe filtered for the current time point.
        track_color_map: A dictionary mapping track IDs to colors.
    """
    labels = np.unique(masks)
    for label in labels:
        if label == 0:
            continue

        track_id_result = df_t.filter(pl.col("label") == label)
        color: Any = "white"
        if not track_id_result.is_empty():
            track_id = track_id_result["label"][0]
            color = track_color_map.get(track_id, "white")

        contour_mask = masks == label
        for contour in find_contours(contour_mask):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, alpha=0.8)
