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
    def test_fixed_buffer_excludes_bright_low_elevation_sky(self):
        elevation = np.zeros((200, 200))
        elevation[:, 100:] = 1.0
        image = np.full(elevation.shape, 20.0)
        image[:, 50:] = 10000.0
        original = image.copy()
        background, sigma, values = color_contours.estimate_channel_background(image, elevation, 'green')
        self.assertEqual(background, 20.0)
        self.assertEqual(sigma, 0.0)
        self.assertEqual(values.size, 200 * 50)
        np.testing.assert_array_equal(image, original)

    def test_insufficient_fixed_buffer_raises(self):
        elevation = np.ones((100, 100))
        elevation[:, :30] = 0
        with self.assertRaisesRegex(ValueError, 'blue: Background mask contains only 0'):
            color_contours.estimate_channel_background(np.ones(elevation.shape), elevation, 'blue')

    def test_shape_mismatch_names_channel(self):
        with self.assertRaisesRegex(ValueError, 'red: image/starmap shape mismatch'):
            color_contours.estimate_channel_background(np.ones((2, 3)), np.ones((3, 2)), 'red')
