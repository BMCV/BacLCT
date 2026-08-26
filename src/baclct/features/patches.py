"""Dataset that turns cell positions into encoder-ready image patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import InterpolationMode, Resize

from baclct.io import scale_percentiles
from baclct.models.encoder import compute_patch_weights

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PatchWeighting:
    """Parameters of the per-patch weights consumed by `MaskedDINOEncoder`."""

    gamma: float | None
    patchsize: int


def padded_patch(
    raw_patch: np.ndarray,
    dst_box: np.ndarray,
    patch_size: int,
    pad_mode: str | int | None,
) -> np.ndarray:
    """Pad a crop out to `patch_size` where it ran past the image border."""
    dy1, dx1, dy2, dx2 = dst_box

    kwargs: dict[str, Any] = {}
    if pad_mode is None or pad_mode == "constant":
        pad_mode = "constant"
        kwargs["constant_values"] = 0

    pad_width = ((dy1, patch_size - dy2), (dx1, patch_size - dx2))
    if raw_patch.ndim == 3:
        pad_width = ((0, 0), *pad_width)  # type: ignore[assignment]

    if all(p == 0 for p_dim in pad_width for p in p_dim):
        return raw_patch

    return np.pad(raw_patch, pad_width, mode=pad_mode, **kwargs)  # type: ignore


class CellPatchDataset(Dataset):
    """Crops, pads, and resizes one encoder batch of single-cell patches per item.

    Items are chunks of at most `cells_per_item` cells taken from a single frame, so a
    worker reads each frame once. Patches keep a single channel whenever the source frame
    is grayscale. The caller broadcasts to three channels and normalizes on the device.
    """

    def __init__(
        self,
        image: da.Array | np.ndarray | list[Path | np.ndarray],
        masks: da.Array | np.ndarray | list[Path | np.ndarray] | None,
        coords: dict,
        input_size_enc: int,
        padding: str | int | None,
        cells_per_item: int = 128,
        image_percentiles: tuple[float, float] | None = None,
        weighting: PatchWeighting | None = None,
        expand_channels: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            image: Image sequence with shape `(T, H, W)`.
            masks: Instance-segmentation masks for `image`.
            coords: Crop geometry from `CellLevelExtractor._get_boxes_and_padding`.
            input_size_enc: Side length the patches are resized to.
            padding: `np.pad` mode for crops that run past the image border.
            cells_per_item: Cells per item, bounding the memory a worker holds at once.
            image_percentiles: `(p_low, p_high)` used to scale intensities.
            weighting: If set, return per-patch weights instead of mask patches.
            expand_channels: If `True`, return three-channel patches rather than
                broadcasting on the device.
        """
        self.image = image
        self.masks = masks
        self.coords = coords
        self.input_size_enc = input_size_enc
        self.padding = padding
        self.image_percentiles = image_percentiles
        self.weighting = weighting
        self.expand_channels = expand_channels

        self.image_resize = Resize(input_size_enc, antialias=True)
        self.mask_resize = Resize(
            input_size_enc, interpolation=InterpolationMode.NEAREST, antialias=False
        )

        self.volumetric = coords.get("z") is not None
        self._chunks = self._build_chunks(coords["t"], coords["s_full"], cells_per_item)

    @staticmethod
    def _build_chunks(
        times: np.ndarray, sizes: np.ndarray, cells_per_item: int
    ) -> list[tuple[int, np.ndarray]]:
        """Split the rows of each frame into chunks of at most `cells_per_item` cells.

        Cells whose crop collapses to zero pixels are dropped here, so every row of a
        chunk yields a patch.
        """
        chunks = []
        for t in np.unique(times):
            rows = np.where((times == t) & (sizes > 0))[0]
            n_splits = max(1, int(np.ceil(len(rows) / cells_per_item)))
            chunks.extend((int(t), part) for part in np.array_split(rows, n_splits))
        return chunks

    def __len__(self) -> int:
        """Number of chunks, i.e. of encoder batches."""
        return len(self._chunks)

    @property
    def n_cells(self) -> int:
        """Cells the items yield, fewer than `coords` holds when a crop collapsed."""
        return sum(len(rows) for _, rows in self._chunks)

    def _read_frame(self, source: Any, t: int) -> np.ndarray | None:
        """Materialize frame `t`, keeping dask work on the calling thread."""
        if source is None:
            return None
        frame = source[t]
        if isinstance(frame, da.Array):
            # workers must not each spin up a full thread pool
            frame = frame.compute(scheduler="synchronous")
        return np.asarray(frame)

    def _crop_image(
        self, frame: np.ndarray, src_box: np.ndarray, dst_box: np.ndarray, size: int
    ) -> torch.Tensor:
        """Crop one cell out of `frame` and resize it to the encoder's input size."""
        sy1, sx1, sy2, sx2 = src_box
        patch = padded_patch(frame[..., sy1:sy2, sx1:sx2], dst_box, size, self.padding)
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensor = self.image_resize(tensor)
        if self.expand_channels and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def _crop_mask(
        self,
        frame: np.ndarray,
        src_box: np.ndarray,
        dst_box: np.ndarray,
        size: int,
        label: int,
    ) -> torch.Tensor:
        """Crop the binary mask of `label`, matching the geometry of `_crop_image`."""
        sy1, sx1, sy2, sx2 = src_box
        patch = padded_patch(frame[sy1:sy2, sx1:sx2], dst_box, size, None)
        patch = (patch == label).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).float().unsqueeze(0)
        return self.mask_resize(tensor)

    def __getitem__(self, index: int) -> dict:
        """Build one encoder batch.

        Returns:
            `patches` (n, C, S, S), `weights` (n, P) or `masks` (n, 1, S, S), the `rows`
            of `coords` that produced them, and the frame index `t`.
        """
        t, rows = self._chunks[index]

        raw_frame = self._read_frame(self.image, t)
        assert raw_frame is not None  # only the masks are optional
        img_frame = scale_percentiles(raw_frame, self.image_percentiles)
        mask_frame = self._read_frame(self.masks, t)

        src_boxes = self.coords["src"][rows]
        dst_boxes = self.coords["dst"][rows]
        labels = self.coords["labels"][rows]
        sizes = self.coords["s_full"][rows]
        z_t = self.coords["z"][rows] if self.volumetric else None

        patches: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        weights: list[np.ndarray] = []

        for k in range(len(rows)):
            size = int(sizes[k])

            if self.volumetric:
                assert z_t is not None
                z_i = int(np.clip(z_t[k], 0, img_frame.shape[0] - 1))
                img_src = img_frame[z_i]
                mask_src = mask_frame[z_i] if mask_frame is not None else None
            else:
                img_src, mask_src = img_frame, mask_frame

            patches.append(self._crop_image(img_src, src_boxes[k], dst_boxes[k], size))

            if mask_src is None:
                continue
            mask = self._crop_mask(mask_src, src_boxes[k], dst_boxes[k], size, labels[k])
            if self.weighting is None:
                masks.append(mask)
            else:
                weights.append(
                    compute_patch_weights(
                        mask.squeeze(0).numpy(),
                        self.weighting.gamma,
                        self.weighting.patchsize,
                    )
                )

        return {
            "patches": torch.stack(patches)
            if patches
            else torch.zeros(0, 1, self.input_size_enc, self.input_size_enc),
            "masks": torch.stack(masks) if masks else None,
            "weights": torch.from_numpy(np.stack(weights)).float() if weights else None,
            "rows": torch.from_numpy(rows).long(),
            "t": t,
        }
