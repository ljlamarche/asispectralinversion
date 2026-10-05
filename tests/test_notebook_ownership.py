"""Exercise notebook ownership and output code without mission data or inversions."""

import ast
from contextlib import redirect_stdout
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

from asispectralinversion.ownership import (
    OWNERSHIP_METHOD,
    combine_owned_products,
    fixed_camera_ownership,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "gneiss.ipynb"


def notebook_sources():
    notebook = json.loads(NOTEBOOK.read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def execute_nodes(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)


class TestNotebookOwnership(unittest.TestCase):
    def setUp(self):
        self.sources = notebook_sources()
        self.target_time = dt.datetime(2026, 2, 10, 10, 24)
        self.owner = np.array([[1, 1, 2], [2, 0, 1]], dtype=np.uint8)
        self.ns = {
            "np": np,
            "json": json,
            "h5py": h5py,
            "OWNERSHIP_METHOD": OWNERSHIP_METHOD,
            "combine_owned_products": combine_owned_products,
            "fixed_camera_ownership": fixed_camera_ownership,
            "owner_site": self.owner,
            "site_codes": {"ARV": 1, "VEE": 2},
            "sites": ("ARV", "VEE"),
            "calculate_conductances": True,
            "target_times": [self.target_time],
            "decgridlat": np.full((2, 3), 67.0),
            "decgridlon": np.full((2, 3), -146.0),
            "date": "20260210",
            "start": "10:24:00",
            "end": "10:24:00",
            "step_seconds": 1,
            "apex_reference_time": "10:24:00",
            "common_mapping_altitude_km": 110.0,
            "target_grid_spacing_km": 2.0,
            "site_config": {
                site: {"calibration_r_s_per_count": {"red": 2.0, "green": 3.0}}
                for site in ("ARV", "VEE")
            },
            "lookup_by_site": {
                site: {"Params": np.arange(8)} for site in ("ARV", "VEE")
            },
        }
        functions = {
            "combine_site_products", "process_time", "create_float_cube",
            "initialize_output", "validate_output_ownership", "write_time_result",
        }
        definitions = [
            node
            for source in self.sources
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
        self.assertEqual({node.name for node in definitions}, functions)
        execute_nodes(definitions, self.ns)

    def products(self):
        products = {}
        for site, value in (("ARV", 1.0), ("VEE", 20.0)):
            products[site] = {
                "q": np.full((2, 3), value),
                "e0": np.full((2, 3), value * 100),
                "valid": np.ones((2, 3), dtype=bool),
                "sigp": np.full((2, 3), value * 2),
                "sigh": np.full((2, 3), value * 3),
            }
        products["ARV"]["q"][0, 1] = np.nan
        products["ARV"]["valid"][0, 1] = False
        return products

    def memory_output(self):
        return h5py.File("ownership-test.h5", "w", driver="core", backing_store=False)

    def test_geometric_ownership_uses_both_channels_and_the_inversion_grid(self):
        source = next(s for s in self.sources if "site_geometric_coverage =" in s)
        names = {"site_geometric_coverage", "owner_site"}
        nodes = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        ]
        namespace = dict(self.ns)
        namespace.update({
            "gridlat": np.zeros((2, 4)),
            "decimation": 2,
            "precomputed_regridders": {
                "ARV": {
                    "red": {"inside": np.ones(8, dtype=bool)},
                    "green": {"inside": np.array([False, True, True, True] * 2)},
                },
                "VEE": {
                    "red": {"inside": np.array([True, True, False, True] * 2)},
                    "green": {"inside": np.ones(8, dtype=bool)},
                },
            },
            "site_distance_km": {"ARV": np.ones((1, 2)), "VEE": np.full((1, 2), 2.0)},
        })
        execute_nodes(nodes, namespace)
        np.testing.assert_array_equal(namespace["owner_site"], [[2, 1]])

    def test_process_time_and_hdf_output_keep_invalid_owner_holes(self):
        products = self.products()
        self.ns["parallel_sites"] = False
        self.ns["invert_site"] = lambda site, *_args: products[site]
        result = self.ns["process_time"](self.target_time)
        with self.memory_output() as h5:
            self.ns["initialize_output"](h5)
            self.ns["validate_output_ownership"](h5)
            self.ns["write_time_result"](h5, 0, result)
            self.assertEqual(h5.attrs["schema_version"], "1.3")
            self.assertEqual(h5.attrs["ownership_method"], OWNERSHIP_METHOD)
            np.testing.assert_array_equal(h5["owner_site"][:], self.owner)
            np.testing.assert_array_equal(
                h5["combined/q_mW_m2"][0], [[1, np.nan, 20], [20, np.nan, 1]]
            )
            np.testing.assert_array_equal(
                h5["combined/source_site"][0], [[1, 0, 2], [2, 0, 1]]
            )
            self.assertEqual(h5["combined/ownership_count"][0, 0, 1], 1)
            self.assertTrue(np.isnan(h5["combined/sigp_S"][0, 0, 1]))
            self.assertTrue(h5["completed"][0])

    def test_resume_guard_rejects_legacy_policy_missing_map_and_changed_map(self):
        for change, message in (
            ("policy", "different ownership rule"),
            ("missing_map", "different ownership rule"),
            ("changed_map", "different fixed camera ownership map"),
        ):
            with self.subTest(change=change), self.memory_output() as h5:
                self.ns["initialize_output"](h5)
                if change == "policy":
                    h5.attrs["ownership_method"] = "nearest_valid_site_after_independent_inversion"
                elif change == "missing_map":
                    del h5["owner_site"]
                else:
                    h5["owner_site"][0, 0] = 2
                with self.assertRaisesRegex(ValueError, message):
                    self.ns["validate_output_ownership"](h5)

    def test_output_cell_recombines_cached_sites_instead_of_reusing_old_combined_data(self):
        cached_combined = {
            key: np.full((2, 3), 99.0)
            for key in ("q", "e0", "sigp", "sigh", "source_site", "ownership_count")
        }
        self.ns.update({
            "save_results": True,
            "overwrite_output": False,
            "diagnostic_result": {
                "time": self.target_time,
                "sites": self.products(),
                "combined": cached_combined,
            },
            "process_time": Mock(side_effect=AssertionError("Cached inversions should be reused")),
        })
        output_source = next(s for s in self.sources if "def initialize_output(" in s)
        with tempfile.TemporaryDirectory() as directory:
            self.ns["output_file"] = Path(directory) / "series.h5"
            with redirect_stdout(io.StringIO()):
                exec(compile(output_source, str(NOTEBOOK), "exec"), self.ns)
            self.ns["process_time"].assert_not_called()
            with h5py.File(self.ns["output_file"], "r") as h5:
                np.testing.assert_array_equal(
                    h5["combined/q_mW_m2"][0], [[1, np.nan, 20], [20, np.nan, 1]]
                )
                self.assertEqual(h5["combined/source_site"][0, 0, 1], 0)
            np.testing.assert_array_equal(cached_combined["q"], np.full((2, 3), 99.0))


if __name__ == "__main__":
    unittest.main()
