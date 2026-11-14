import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from asispectralinversion.tiff_catalog import TiffPageCountCache


class TestTiffPageCountCache(unittest.TestCase):
    def test_persistent_reuse_and_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'stack.tif'
            catalog = root / 'cache' / 'catalog.json'
            tifffile.imwrite(image, np.zeros((3, 8, 8), dtype='uint16'), photometric='minisblack')
            self.assertEqual(TiffPageCountCache(catalog).page_count(image), 3)
            cache = TiffPageCountCache(catalog)
            with patch('asispectralinversion.tiff_catalog.tifffile.TiffFile', side_effect=AssertionError('cache miss')):
                self.assertEqual(cache.page_count(image), 3)
            self.assertEqual(cache.hits, 1)
            stat = image.stat()
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            self.assertEqual(cache.page_count(image), 3)
            self.assertEqual(cache.misses, 1)
            tifffile.imwrite(image, np.zeros((5, 8, 8), dtype='uint16'), photometric='minisblack')
            self.assertEqual(cache.page_count(image), 5)
            self.assertEqual(TiffPageCountCache(catalog).page_count(image), 5)

    def test_corrupt_catalog_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'stack.tif'
            catalog = root / 'catalog.json'
            tifffile.imwrite(image, np.zeros((8, 8), dtype='uint16'))
            catalog.write_text('{broken')
            with self.assertWarns(UserWarning):
                cache = TiffPageCountCache(catalog)
            self.assertEqual(cache.page_count(image), 1)
            self.assertEqual(TiffPageCountCache(catalog).page_count(image), 1)
