"""DINO feature encoders for single-cell image embeddings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, no_type_check

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOEncoder(nn.Module):
    """Single-object feature extractor based on a pretrained DINO ViT.

    Loads a ViT from `torch.hub` (DINOv1 or DINOv2 with optional registers, frozen by
    default) and returns the model's final embedding for a single-cell image. No
    task-specific fine-tuning is applied and the public ImageNet weights are used
    directly.

    Model configurations and details: https://github.com/facebookresearch/dino.
    """

    def __init__(
        self,
        size: Literal["s", "b", "l"] = "b",
        patchsize: Literal[8, 14, 16] = 16,
        version: Literal[1, 2] = 1,
        freeze: bool = True,
        with_registers: bool = False,
        forward_norm_fn: Callable | None = None,
        foreground_only: bool = False,
    ) -> None:
        """Initialize Model Loader.

        Args:
            size: Size of the ViT backbone.
            patchsize: Size of embedded patches. 8 or 16 for DINOv1, 14 for DINOv2.
            version: Version of DINO weights.
            freeze: If `True`, freezes the weights of the encoder.
            with_registers: If `True`, uses register tokens. Only for DINOv2.
            forward_norm_fn: Normalization function applied to the network outputs.
            foreground_only: If `True` only gets features for the foreground determined
                by the input segmentation mask.
        """
        super().__init__()

        self.version = version
        self.dino_prefix = "dino" if version == 1 else "dinov2"

        self.patchsize = patchsize
        self.arch = f"{self.dino_prefix}_vit{size}{patchsize}"

        self.registers = with_registers
        if with_registers and (version == 2):
            self.arch += "_reg"

        self.foreground_only = foreground_only

        self.freeze = freeze
        self.forward_norm_fn = forward_norm_fn

        # torch.hub relies on sys.path which is not propagated on fork
        # thus, we lazy load for compatibility with multi-worker extractors
        # lazy load is done in first forward, calls:
        #   prepare_data: single worker, only uses forward when features don't exist
        #   setup: forward should never be called, memory footprint stays low
        self.model = None

    def _init_model(self, device):
        try:
            model = torch.hub.load(
                f"facebookresearch/{self.dino_prefix}:main", self.arch, verbose=False
            )
            assert isinstance(model, nn.Module)
        except RuntimeError as err:
            raise ValueError(
                f"{self.arch} is not a valid configuration of DINO."
            ) from err

        if self.freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.model = model.to(device)

    def _unload_model(self):
        """Frees the torch.hub model to prevent multiprocessing serialization issues."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None

    @no_type_check
    def get_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve tokens of the final attention layer.

        Returns:
            qkv: Tensor with Key, Query and Value tokens of the final attention layer.
        """
        with torch.no_grad():
            if self.version == 1:
                x = self.model.prepare_tokens(x)
            else:
                x = self.model.prepare_tokens_with_masks(x)
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                if len(self.model.blocks) - i <= 1:
                    break

            blk = self.model.blocks[-1]
            attn_layer = self.model.blocks[-1].attn

            x_norm = blk.norm1(x)
            if (self.registers) & (self.version == 2):
                cls_token = x_norm[:, 0]
                patch_tokens = x_norm[:, self.model.num_register_tokens + 1 :]
                x_norm = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)

            B, N, C = x_norm.shape
            qkv = (
                attn_layer.qkv(x_norm)
                .reshape(B, N, 3, attn_layer.num_heads, C // attn_layer.num_heads)
                .permute(2, 0, 3, 1, 4)
            )

        return qkv

    def get_tokens(
        self, x: torch.Tensor, tkn: Literal[0, 1, 2] = 1, include_cls_tkn: bool = False
    ) -> torch.Tensor:
        """Get Key, Query, or Value tokens without class tokens.

        Returns:
            Tokens with shape (`n_tokens`, `n_heads` * `n_patches`), optionally with class
            token.
        """
        qkv = self.get_qkv(x)
        n_qkv, n_batch, n_heads, n_patches, token_dim = qkv.shape
        tkns = (
            qkv[tkn].permute(0, 2, 1, 3).reshape(n_batch, n_patches, n_heads * token_dim)
        )

        if include_cls_tkn:
            return tkns

        return tkns[:, 1:, :]

    def forward(self, inputs: tuple[torch.Tensor, Any]) -> torch.Tensor:
        """Get ViT embeddings.

        Returns:
            ViT embeddings prior to classification layer. Shape is (n_samples, 768) for
            ViT-B16 backbone.
        """
        if self.model is None:
            self._init_model(inputs[0].device)

        x, msk = inputs
        if self.foreground_only:
            # convert tv tensors to tensors
            # mask: BYX, img: BCYX
            x = torch.tensor(x) * torch.tensor(msk).unsqueeze(1)

        assert self.model is not None
        outputs = self.model(x)
        if self.forward_norm_fn is not None:
            return self.forward_norm_fn(outputs)

        return outputs


class MaskedDINOEncoder(DINOEncoder):
    """DINO ViT with mask-weighted patch aggregation.

    Extends `DINOEncoder` to produce a single per-object embedding by aggregating
    patch-wise key/query/value tokens of the final attention layer with a weight map
    derived from the segmentation mask. The weights themselves are computed by
    `compute_patch_weights` outside the model, so `forward` consumes them ready-made.
    """

    def __init__(
        self,
        token: Literal["q", "k", "v"],
        background_gamma: float | None = None,
        size: Literal["s", "b", "l"] = "b",
        patchsize: Literal[8, 14, 16] = 16,
        version: Literal[1, 2] = 1,
        freeze: bool = True,
        with_registers: bool = False,
        forward_norm_fn: Callable[..., Any] | None = None,
        foreground_only: bool = False,
    ) -> None:
        """Initialize module.

        Args:
            token: Name of used token.
            background_gamma: Optional scaling factor for background suppression.
            size: Size of the ViT backbone.
            patchsize: Size of embedded patches. 8 or 16 for DINOv1, 14 for DINOv2.
            version: Version of DINO weights.
            freeze: If `True`, freezes the weights of the encoder.
            with_registers: If `True`, uses register tokens. Only for DINOv2.
            forward_norm_fn: Normalization function applied to the network outputs.
            foreground_only: If `True` only gets features for the foreground determined
                by the input segmentation mask.
        """
        super().__init__(
            size,
            patchsize,
            version,
            freeze,
            with_registers,
            forward_norm_fn,
            foreground_only,
        )
        self.token = {"q": 0, "k": 1, "v": 2}[token]
        self.gamma = background_gamma

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Aggregate patch-wise tokens into one embedding per object.

        Args:
            inputs: Input image and per-patch weights from `compute_patch_weights`, the
                latter shaped `(n_samples, n_patches)`.
        """
        if self.model is None:
            self._init_model(inputs[0].device)

        x, patch_weights = inputs

        tokens = self.get_tokens(x, self.token)  # type: ignore
        weights = patch_weights.unsqueeze(-1).expand_as(tokens).to(tokens.device)

        outputs = torch.sum(tokens * weights, dim=1) / weights.sum(dim=1)
        if self.forward_norm_fn is not None:
            return self.forward_norm_fn(outputs)

        return outputs


def normalize_imagenet(imgs: torch.Tensor) -> torch.Tensor:
    """Scale three-channel patches to the statistics the pretrained encoders expect."""
    mean = torch.tensor(IMAGENET_MEAN, device=imgs.device, dtype=imgs.dtype)
    std = torch.tensor(IMAGENET_STD, device=imgs.device, dtype=imgs.dtype)
    return (imgs - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def compute_patch_weights(
    mask: np.ndarray, gamma: float | None, patchsize: int
) -> np.ndarray:
    """Weight each ViT patch by how much of the object it covers.

    With `gamma`, weights follow the crop's distance transform normalized against its own
    maximum, so background and neighboring objects are suppressed.

    Returns:
        Flattened patch weights of length `(mask.shape[-1] // patchsize) ** 2`.
    """
    if gamma is None:
        dist_norm = mask.astype(np.float32)
    else:
        dist = distance_transform_edt(mask == 0)
        assert isinstance(dist, np.ndarray)
        d_max = float(dist.max())
        if d_max > 0 and mask.any():
            dist_norm = (1.0 - dist / d_max).astype(np.float32) ** gamma
        else:
            # a crop that is all foreground or all background has no gradient to weight
            # by, so every patch counts equally
            dist_norm = np.ones(mask.shape, dtype=np.float32)

    n = mask.shape[-1] // patchsize
    size = n * patchsize
    pooled = dist_norm[:size, :size].reshape(n, patchsize, n, patchsize).mean(axis=(1, 3))
    return pooled.ravel()
