"""Encoding and/or fusion of handcrafted and deep features embedded into nodes for GNN."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """Initial node embedding for the GNN.

    Fuses handcrafted single-object features and deep (e.g. DINO) features into a single
    node embedding. Either branch may be absent, and in that case, only the present branch
    is processed.

    Fusion strategies:
      - `concat`: concatenate the two raw feature tensors directly.
      - `encode_concat`: pass each branch through its own encoder first, then
        concatenate.

    An optional `fusion_mlp` is applied to the (concatenated) features for further
    embedding and dimensionality adjustment.
    """

    def __init__(
        self,
        handcrafted_encoder: nn.Module | None = None,
        deep_encoder: nn.Module | None = None,
        fusion_mode: Literal["concat", "encode_concat"] = "concat",
        fusion_mlp: nn.Module | None = None,
    ):
        """Initialize encoder.

        Args:
            handcrafted_encoder: Optional handcrafted feature encoder.
            deep_encoder: Optional deep features encoder.
            fusion_mode: Fusion strategy. Raw or encoded deep and handcrafted features.
                Are concatenated (`concat`) or concatenated and encoded by a fusion MLP
                (`encode_concat`).
            fusion_mlp: Optional encoder for concatenated features.
        """
        super().__init__()
        self.handcrafted_encoder = handcrafted_encoder
        self.deep_encoder = deep_encoder
        self.fusion_mode = fusion_mode
        self.fusion_mlp = fusion_mlp

        if fusion_mode not in ["concat", "encode_concat"]:
            raise ValueError(f"Invalid fusion_mode: {fusion_mode}")

    def forward(
        self,
        x_handcrafted: torch.Tensor | None = None,
        x_deep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse deep and handcrafted features."""
        if x_handcrafted is None and x_deep is None:
            raise ValueError("At least one of x_handcrafted or x_deep must be provided.")

        # single feature cases
        if x_handcrafted is None:
            assert x_deep is not None
            if self.deep_encoder:
                x_deep = self.deep_encoder(x_deep)
            if self.fusion_mlp:
                x_deep = self.fusion_mlp(x_deep)
            return x_deep

        if x_deep is None:
            if self.handcrafted_encoder:
                x_handcrafted = self.handcrafted_encoder(x_handcrafted)
            if self.fusion_mlp:
                x_handcrafted = self.fusion_mlp(x_handcrafted)
            return x_handcrafted

        # both features are present, perform fusion
        if self.fusion_mode == "concat":
            x_fused = torch.cat([x_handcrafted, x_deep], dim=1)

        elif self.fusion_mode == "encode_concat":
            x_hc_encoded = (
                self.handcrafted_encoder(x_handcrafted)
                if self.handcrafted_encoder
                else x_handcrafted
            )
            x_deep_encoded = self.deep_encoder(x_deep) if self.deep_encoder else x_deep
            x_fused = torch.cat([x_hc_encoded, x_deep_encoded], dim=1)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        if self.fusion_mlp:
            x_fused = self.fusion_mlp(x_fused)

        return x_fused
