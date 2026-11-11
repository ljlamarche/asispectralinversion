"""Blue timing and RGB notebook integration checks without mission starmaps."""
import ast
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import tifffile

from asispectralinversion.blue import blue_stack_catalog, coadd_blue_frames
from asi_shared.calibration import BLUE_RAYLEIGH_SECONDS_PER_COUNT
from asi_shared.background import physical_offsky_mask, estimate_scalar_background


class TestBlue(unittest.TestCase):
    def test_rgb_background_relaxes_margin_without_using_sky(self):
        notebook = json.loads((Path(__file__).parents[1] / 'gneiss_rgb.ipynb').read_text())
        nodes = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'estimate_background']
        ns = dict(np=np, physical_offsky_mask=physical_offsky_mask,
                  estimate_scalar_background=estimate_scalar_background,
                  background_edge_buffer_px=100)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook>', 'exec'), ns)
        elevation = np.full((20, 20), 30.)
        elevation[:, :10] = 0
        image = np.where(elevation > 0, 10000., 12.)
        self.assertEqual(ns['estimate_background'](image, elevation), (12., 0.))
        with self.assertRaisesRegex(ValueError, 'off-sky'):
            ns['estimate_background'](image, np.ones((20, 20)))

    def test_current_calibration_values(self):
        self.assertEqual(BLUE_RAYLEIGH_SECONDS_PER_COUNT,
                         {'ARV': 100., 'VEE': 40., 'BVR': 10.})

    def test_log_timing_exposure_and_wavelength(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera_filename.tif'
            with tifffile.TiffWriter(path) as writer:
                for i in range(3):
                    writer.write(np.full((2, 2), i, dtype=np.uint16))
            log = path.with_suffix('.log')
            text = ('Date 2026-02-10\nTimeStart 10:00:00.005\nExposure(s) 0.9\n'
                    'ImageCadence(s) 0.90178\nNumberOfImages 3\nFilter(Ang) 4278\n')
            log.write_text(text)
            catalog = blue_stack_catalog(directory, '20260210')
            self.assertEqual(catalog[0]['start'].microsecond, 5000)
            self.assertEqual(catalog[0]['exposure'], .9)
            image, _, indices, _, exposure = coadd_blue_frames(
                catalog, catalog[0]['start'] + dt.timedelta(seconds=.90178), 3,
                lambda path, index: tifffile.imread(path, key=index))
            np.testing.assert_array_equal(image, np.ones((2, 2)))
            self.assertEqual(indices, [0, 1, 2])
            self.assertEqual(exposure, .9)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                coadd_blue_frames(catalog, dt.datetime(2026, 2, 10, 11), 3, Mock())
            log.write_text(text.replace('4278', '8446'))
            with self.assertRaisesRegex(ValueError, 'expected 4278'):
                blue_stack_catalog(directory, '20260210')
            with self.assertRaisesRegex(ValueError, 'expected 4278'):
                blue_stack_catalog(directory, '20260210', verified_stale_filter=7774)
            log.write_text(text.replace('4278', '7774'))
            catalog = blue_stack_catalog(directory, '20260210', verified_stale_filter=7774)
            self.assertEqual(catalog[0]['reported_filter_ang'], 7774)
            self.assertEqual(catalog[0]['verified_filter_ang'], 4278)

    def test_rgb_process_channel_calibrates_without_dark_frame(self):
        notebook = json.loads((Path(__file__).parents[1] / 'gneiss_rgb.ipynb').read_text())
        nodes = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'process_channel']
        grid = np.ones((2, 2))
        valid = np.ones((2, 2), dtype=bool)
        ns = dict(np=np, coadd_blue_frames=Mock(return_value=(grid * 100, [], [], [], .9)),
                  read_tiff_frame=Mock(), frames_to_coadd=3,
                  estimate_background=Mock(return_value=(10., 1.)),
                  denoise_image=lambda image, *args: image,
                  regrid_channel=lambda image, *args: image, wavelet_max_shifts={'blue': 0})
        sites = tuple(BLUE_RAYLEIGH_SECONDS_PER_COUNT)
        ns.update(blue_catalogs={s: [] for s in sites},
                  channel_starmaps_by_site={s: {'blue': (grid, grid, valid, None)} for s in sites},
                  blue_skymaps_by_site={s: {'elev': grid} for s in sites},
                  red_skymaps_by_site={}, green_skymaps_by_site={},
                  precomputed_regridders={s: {'blue': None} for s in sites},
                  site_config={s: {'calibration_r_s_per_count': {'blue': factor}}
                               for s, factor in BLUE_RAYLEIGH_SECONDS_PER_COUNT.items()})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook>', 'exec'), ns)
        for site, factor in BLUE_RAYLEIGH_SECONDS_PER_COUNT.items():
            result, metadata = ns['process_channel'](site, 'blue', dt.datetime(2026, 2, 10))
            expected = (100 - 10) / .9 * factor
            np.testing.assert_allclose(result, expected)
            self.assertEqual(metadata['exposure_seconds'], .9)


if __name__ == '__main__':
    unittest.main()
