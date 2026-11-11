"""Background selection regressions for the color-contour diagnostic."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "comparisons" / "color_contours.py"
SPEC = importlib.util.spec_from_file_location("color_contours_test_module", SCRIPT)
color_contours = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color_contours)


class TestColorContourBackground(unittest.TestCase):
    def setUp(self):
        # Scale the same 100%, 75%, 50%, 25%, 0% margin policy to tiny images.
        for name, value in (("BACKGROUND_EDGE_BUFFER_PX", 4), ("BACKGROUND_MINIMUM_PIXELS", 10)):
            patcher = patch.object(color_contours, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def image_and_elevation(self, offsky_columns):
        elevation = np.full((12, 12), 30.0)
        elevation[:, -offsky_columns:] = -1.0
        image = np.full(elevation.shape, 10000.0)
        image[:, -offsky_columns:] = np.arange(12 - offsky_columns, 12) + 100.0
        return image, elevation

    def estimate(self, image, elevation, channel="green"):
        output = io.StringIO()
        with redirect_stdout(output):
            result = color_contours.estimate_channel_background(image, elevation, channel)
        return result, output.getvalue()

    def test_preferred_buffer_is_preserved_when_it_has_enough_samples(self):
        image, elevation = self.image_and_elevation(offsky_columns=6)
        (background, sigma, values), message = self.estimate(image, elevation)
        np.testing.assert_array_equal(values, image[:, 10:].ravel())
        self.assertEqual(background, 110.5)
        self.assertAlmostEqual(sigma, 1.4826 * 0.5)
        self.assertEqual(message, "")

    def test_empty_buffer_relaxes_without_sampling_positive_elevation_sky(self):
        image, elevation = self.image_and_elevation(offsky_columns=4)
        # Even positive sky below the mapping cutoff must not become background.
        elevation[:, :8] = 1.0
        original = image.copy()
        (background, sigma, values), message = self.estimate(image, elevation)
        np.testing.assert_array_equal(values, image[:, 11])
        self.assertEqual(background, 111.0)
        self.assertEqual(sigma, 0.0)
        self.assertIn("green: reduced background buffer from 4 to 3 pixels", message)
        self.assertIn("12 finite off-sky samples", message)
        self.assertFalse(np.any(values == 10000.0))
        np.testing.assert_array_equal(image, original)

    def test_sample_threshold_counts_finite_image_values_not_just_mask_pixels(self):
        image, elevation = self.image_and_elevation(offsky_columns=6)
        image[:, 10] = np.nan
        image[:, 11] = np.inf
        (background, _sigma, values), message = self.estimate(image, elevation)
        np.testing.assert_array_equal(values, image[:, 9])
        self.assertEqual(background, 109.0)
        self.assertTrue(np.isfinite(values).all())
        self.assertIn("from 4 to 3 pixels", message)

    def test_all_sky_or_all_nonfinite_images_raise_instead_of_using_corners(self):
        image, elevation = self.image_and_elevation(offsky_columns=6)
        cases = (
            (image, np.full(elevation.shape, 1.0)),
            (np.full(image.shape, np.nan), elevation),
        )
        for candidate_image, candidate_elevation in cases:
            with self.subTest(finite_pixels=np.count_nonzero(np.isfinite(candidate_image))):
                with self.assertRaisesRegex(ValueError, "blue: only 0 finite off-sky"):
                    self.estimate(candidate_image, candidate_elevation, channel="blue")

    def test_too_few_offsky_samples_still_raise_at_zero_buffer(self):
        image = np.full((12, 12), 10000.0)
        elevation = np.full(image.shape, 20.0)
        elevation[0, :5] = -1.0
        with self.assertRaisesRegex(ValueError, "red: only 5 finite off-sky"):
            self.estimate(image, elevation, channel="red")

    def test_shape_mismatch_names_the_channel(self):
        with self.assertRaisesRegex(ValueError, "green: image/starmap shape mismatch"):
            self.estimate(np.ones((2, 3)), np.ones((3, 2)))

    def test_denoising_uses_relaxed_background_and_keeps_sky_and_input_unchanged(self):
        image, elevation = self.image_and_elevation(offsky_columns=4)
        original = image.copy()
        valid = elevation > 22.0
        geometry = {
            "elevation": elevation,
            "invalid": ~valid,
            "coordinate_valid": valid,
        }
        # Background selection is real; avoid expensive wavelet work in this test.
        with patch.object(color_contours, "cycle_spin", side_effect=lambda filled, **_kwargs: filled.copy()):
            with redirect_stdout(io.StringIO()):
                prepared, background, sigma = color_contours.denoise_like_notebook(
                    image, geometry, "green"
                )
        self.assertEqual(background, 111.0)
        self.assertEqual(sigma, 0.0)
        np.testing.assert_array_equal(prepared[valid], image[valid])
        self.assertTrue(np.isnan(prepared[~valid]).all())
        np.testing.assert_array_equal(image, original)


if __name__ == "__main__":
    unittest.main()
