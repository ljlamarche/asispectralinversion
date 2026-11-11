"""Exercise actual spawned workers and the single-writer result contract."""
import datetime as dt
from contextlib import closing
import multiprocessing
import time
import tempfile
from pathlib import Path
import unittest

import numpy as np

from asispectralinversion.rgb_series import iter_rgb_series
from test_inversion_lookup import inversion_case


SOURCES = [
    '''def invert_site(site, target):
    factor = 1 + target.second / 100
    rgb = [value * factor for value in observations]
    q, e, sp, sh = invert_owned_rgb(*rgb, lookup, owner_site == site_codes[site], False)
    return dict(q=q, e0=e, sigp=sp, sigh=sh, valid=np.isfinite(q) & np.isfinite(e))
''',
    '''def combine_site_products(products):
    return combine_owned_products(products, owner_site, site_codes, False)
''',
]


class TestRGBSeries(unittest.TestCase):
    def assert_workers_stopped(self, before):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if {p.pid for p in multiprocessing.active_children()} == before:
                return
            time.sleep(0.05)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_spawned_notebook_readers_apply_detector_correction(self):
        import ast
        import json
        import tifffile
        from asi_shared.detector import correct_detector_frame

        notebook = Path(__file__).resolve().parents[1] / 'gneiss_rgb.ipynb'
        names = {'open_tiff_stack', 'read_tiff_frame_cached', 'read_tiff_frame'}
        sources = []
        for cell in json.loads(notebook.read_text())['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    sources.append(ast.unparse(node))
        self.assertEqual(len(sources), 3)
        sources += [
            'def invert_site(site, target):\n    return read_tiff_frame(frame_path, 0)\n',
            "def combine_site_products(products):\n    return products['ARV']\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ARV_blue_4278.tif'
            raw = np.full((512, 512), 1000, dtype=np.uint16)
            tifffile.imwrite(path, raw)
            expected = correct_detector_frame(raw.astype(float), path)
            self.assertEqual(expected[153, 281], 923)
            for cached in (True, False):
                state = dict(sites=('ARV',), frame_path=str(path),
                             cache_tiff_frames=cached, tiff_frame_cache_size=2)
                results = list(iter_rgb_series(state, sources, [(0, None)], 2))
                np.testing.assert_array_equal(results[0][1]['combined'], expected)
            np.testing.assert_array_equal(tifffile.imread(path), raw)

    def state(self):
        lookup, observations = inversion_case()
        return dict(lookup=lookup, observations=observations,
                    sites=('ARV','VEE'), site_codes={'ARV':1,'VEE':2},
                    owner_site=np.array([[1,2,1],[2,0,1]],dtype=np.uint8))

    def test_spawn_matches_serial_and_preserves_noncontiguous_indices(self):
        jobs=[(i,dt.datetime(2026,2,10,10,24,i)) for i in (1,4,7)]
        serial=dict(iter_rgb_series(self.state(),SOURCES,jobs,1))
        parallel=dict(iter_rgb_series(self.state(),SOURCES,jobs,2))
        self.assertEqual(set(parallel), {1,4,7})
        for index in serial:
            self.assertEqual(parallel[index]['time'],serial[index]['time'])
            for key in serial[index]['combined']:
                np.testing.assert_array_equal(parallel[index]['combined'][key], serial[index]['combined'][key])

    def test_worker_error_is_propagated_and_workers_stop(self):
        before={p.pid for p in multiprocessing.active_children()}
        bad=['def invert_site(site, target):\n    raise ValueError("bad frame")\n']
        with self.assertRaisesRegex(ValueError,'bad frame'):
            list(iter_rgb_series(self.state(),bad,[(0,None)],2))
        self.assert_workers_stopped(before)

    def test_closing_iterator_stops_remaining_workers(self):
        before={p.pid for p in multiprocessing.active_children()}
        sources=['''def invert_site(site, target):
    import time
    time.sleep(target)
    return {}
''','def combine_site_products(products):\n    return {}\n']
        start=time.monotonic()
        with closing(iter_rgb_series(dict(sites=('ARV',)),sources,[(0,0),(1,20)],2)) as results:
            index,_=next(results)
            self.assertEqual(index,0)
        self.assertLess(time.monotonic()-start,15)
        self.assert_workers_stopped(before)

    def test_config_worker_count(self):
        from asispectralinversion.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.toml'
            config.write_text('[gneiss]\n')
            self.assertEqual(load_config(config).timestamp_workers, 2)
            config.write_text('[gneiss]\ntimestamp_workers = 3\n')
            self.assertEqual(load_config(config).timestamp_workers, 3)
            for value in ('0', '-1', 'true', '1.5'):
                config.write_text(f'[gneiss]\ntimestamp_workers = {value}\n')
                with self.assertRaisesRegex(ValueError, 'timestamp_workers'):
                    load_config(config)

    def test_invalid_worker_counts_and_empty_jobs(self):
        for workers in (0,-1,True,1.5):
            with self.assertRaises(ValueError):list(iter_rgb_series({},[],[],workers))
        self.assertEqual(list(iter_rgb_series({},[],[],2)),[])


if __name__=='__main__':unittest.main()
