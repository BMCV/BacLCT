"""Test custom node features."""

from __future__ import annotations

import numpy as np
from skimage.morphology import dilation, disk, skeletonize

from baclct.features.extractors import HandcraftedExtractor, _matches_feature


def debug_grid(mask, skel, center_local, win=10):
    """Text-based grid around a center, to see pixel alignment on a failure."""
    y_c, x_c = int(round(center_local[0])), int(round(center_local[1]))
    y_start, y_end = max(0, y_c - win), min(mask.shape[0], y_c + win)
    x_start, x_end = max(0, x_c - win), min(mask.shape[1], x_c + win)

    rows = []
    for r in range(y_start, y_end):
        line = ""
        for c in range(x_start, x_end):
            if r == y_c and c == x_c:
                line += "X"  # center
            elif skel[r, c]:
                line += "S"  # skel
            elif mask[r, c]:
                line += "."  # mask
            else:
                line += " "
        rows.append(line)

    return "\n".join(rows)


def test_handcrafted_extractor_with_extra_props(toy_images, toy_masks):
    """Custom props are computed, renamed to global coords, and the center is on-cell."""
    extractor = HandcraftedExtractor(
        props=["intensity_mean", "bbox"],
        extra_props=["center_local", "thickness"],
        n_jobs=1,
    )
    features_df = extractor._compute(images=toy_images, masks=toy_masks)
    assert "intensity_mean" in features_df.columns
    assert "area" in features_df.columns  # mandatory prop
    assert "center-0" in features_df.columns  # local renaming
    assert "center-1" in features_df.columns
    assert "thickness" in features_df.columns  # custom prop
    assert features_df["thickness"].sum() > 0
    assert features_df["center-0"].sum() > 0

    # TODO: Maybe refactor into dedicated test, but would either require synth
    #       data or more cell types. 100x100px arrays and low number might be
    #      sufficient and tiny with compression.
    for t, mask in enumerate(toy_masks):
        y, x, label = np.transpose(
            features_df.filter(t=t).select("center-0", "center-1", "label")
        ).astype(int)

        for i, (yi, xi, labeli) in enumerate(zip(y, x, label, strict=True)):
            assert mask[yi, xi] == labeli

            if i < 5:
                # dilation due to rounding (bbox coords, skel coords, center coords)
                skel = skeletonize(mask == labeli)
                skel = dilation(skel, disk(1))

                assert skel[yi, xi], (
                    "Center point not on skeleton:\n"
                    f"{debug_grid(mask == labeli, skel, (yi, xi))}"
                )


def test_matches_feature_anchored_expansion():
    """Feature-name matching expands stats/components but not derived siblings."""
    # exact name and its normalized variant
    assert _matches_feature("area", "area")
    assert _matches_feature("area_norm", "area")
    # vector components and regionprops reductions expand under the bare prop
    assert _matches_feature("centroid-0", "centroid")
    assert _matches_feature("intensity_mean", "intensity")
    assert _matches_feature("intensity_min", "intensity")
    # a derived sibling must NOT be swallowed by the prop it came from
    assert not _matches_feature("intensity_ratio", "intensity")
    # but it still resolves under its own (and _norm) name
    assert _matches_feature("intensity_ratio", "intensity_ratio")
    assert _matches_feature("intensity_ratio_norm", "intensity_ratio")
    # unrelated columns sharing a prefix are not captured
    assert not _matches_feature("aspect_ratio", "area")
