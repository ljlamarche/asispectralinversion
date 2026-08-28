import numpy as np

from asi_shared.background import (
    corner_mask,
    estimate_scalar_background,
    physical_offsky_mask,
)


def test_physical_offsky_mask_does_not_include_positive_low_elevation():
    elevation = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 22.0], [0.0, 0.0, 0.0]])
    mask = physical_offsky_mask(elevation)
    assert mask[1, 0]
    assert not mask[1, 1]
    assert not mask[1, 2]


def test_scalar_background_uses_masked_median_and_robust_sigma():
    image = np.array([[10.0, 10.0], [12.0, 1000.0]])
    mask = np.array([[True, True], [True, False]])
    background, sigma, values = estimate_scalar_background(
        image, mask, minimum_pixels=3
    )
    assert background == 10.0
    assert sigma == 0.0
    assert values.size == 3


def test_corner_mask_selects_four_corners():
    mask = corner_mask((10, 12), corner_size_px=2)
    assert np.count_nonzero(mask) == 16
    assert not mask[5, 6]
