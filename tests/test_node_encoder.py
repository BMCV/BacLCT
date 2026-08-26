"""Test node encoders and feature fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from baclct.models.node_encoder import NodeEncoder

N, HC, DEEP = 10, 4, 256


@pytest.mark.parametrize(
    "encoder, use_hc, use_deep, expected_dim",
    [
        # a single feature type passes through unchanged
        (NodeEncoder(), True, False, HC),
        (NodeEncoder(), False, True, DEEP),
        # concat stacks both feature blocks
        (NodeEncoder(fusion_mode="concat"), True, True, HC + DEEP),
        # encode_concat encodes each block before concatenation
        (
            NodeEncoder(
                handcrafted_encoder=nn.Linear(HC, 16),
                deep_encoder=nn.Linear(DEEP, 64),
                fusion_mode="encode_concat",
            ),
            True,
            True,
            16 + 64,
        ),
        # a missing encoder leaves that block unencoded
        (
            NodeEncoder(deep_encoder=nn.Linear(DEEP, 64), fusion_mode="encode_concat"),
            True,
            True,
            HC + 64,
        ),
        # fusion_mlp projects the concatenated features
        (
            NodeEncoder(fusion_mode="concat", fusion_mlp=nn.Linear(HC + DEEP, 128)),
            True,
            True,
            128,
        ),
    ],
)
def test_node_encoder_fusion_modes(encoder, use_hc, use_deep, expected_dim):
    """Node encoder fuses handcrafted and deep features per fusion mode."""
    x_handcrafted = torch.randn(N, HC) if use_hc else None
    x_deep = torch.randn(N, DEEP) if use_deep else None
    output = encoder(x_handcrafted=x_handcrafted, x_deep=x_deep)

    assert output.shape == (N, expected_dim)

    # a single feature type is returned unchanged
    if use_hc and not use_deep:
        assert x_handcrafted is not None
        assert torch.equal(output, x_handcrafted)
    if use_deep and not use_hc:
        assert x_deep is not None
        assert torch.equal(output, x_deep)
