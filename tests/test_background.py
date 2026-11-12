import numpy as np

from asi_shared.background import (
    corner_mask,
    estimate_scalar_background,
    physical_offsky_mask,
)


def test_physical_offsky_mask_does_not_include_positive_low_elevation():
    elevation = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 22.0], [0.0, 0.0, 0.0]])
    mask = physical_offsky_mask(elevation, edge_buffer_px=0)
    assert mask[1, 0]
    assert not mask[1, 1]
    assert not mask[1, 2]


def test_constant_background_uses_median_fallback():
    image = np.array([[10.0, 10.0], [10.0, 1000.0]])
    mask = np.array([[True, True], [True, False]])
    background, sigma, values = estimate_scalar_background(image, mask)
    assert background == 10.0
    assert sigma == 0.0
    assert values.size == 3


def test_fixed_buffer_excludes_positive_sky_and_nonfinite_elevation():
    elevation = np.zeros((200, 200))
    elevation[:, 100:] = 1.0  # Below display cutoff, but never background.
    elevation[0, 0] = np.nan
    elevation[-1, 0] = np.inf
    mask = physical_offsky_mask(elevation)
    assert mask[100, 49]
    assert not mask[100, 50]  # Exactly 50 pixels from sky: excluded.
    assert not mask[:, 100:].any()
    assert not mask[0, 0]
    assert not mask[-1, 0]


def test_gaussian_histogram_center_and_no_input_mutation():
    from asi_shared.background import estimate_corner_background
    rng = np.random.default_rng(42)
    image = rng.normal(120, 7, (400, 400))
    original = image.copy()
    mask = np.ones(image.shape, dtype=bool)
    result = estimate_corner_background(image, mask)
    assert not result['fit_failed']
    assert abs(result['center'] - 120) < 0.2
    assert abs(result['sigma'] - 7) < 0.2
    center, sigma, _ = estimate_scalar_background(image, mask)
    assert center == result['center']
    assert sigma == result['sigma']
    np.testing.assert_array_equal(image, original)


def test_no_samples_raises_without_relaxing_buffer():
    import pytest
    elevation = np.ones((100, 100))
    elevation[:, :30] = 0
    mask = physical_offsky_mask(elevation)
    assert not mask.any()
    with pytest.raises(ValueError, match='only 0 finite pixels'):
        estimate_scalar_background(np.ones(elevation.shape), mask)


def test_corner_mask_selects_four_corners():
    mask = corner_mask((10, 12), corner_size_px=2)
    assert np.count_nonzero(mask) == 16
    assert not mask[5, 6]
