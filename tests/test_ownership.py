import unittest

import numpy as np

from asispectralinversion.ownership import (
    combine_owned_products,
    fixed_camera_ownership,
)


def product(q, e0=None, valid=None, sigp=None, sigh=None):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    return {
        "q": q,
        "e0": np.full(q.shape, 500.0) if e0 is None else np.atleast_2d(e0),
        "valid": np.ones(q.shape, dtype=bool) if valid is None else np.atleast_2d(valid),
        "sigp": np.full(q.shape, 2.0) if sigp is None else np.atleast_2d(sigp),
        "sigh": np.full(q.shape, 3.0) if sigh is None else np.atleast_2d(sigh),
    }


class TestFixedCameraOwnership(unittest.TestCase):
    def test_only_geometrically_eligible_cameras_can_own_cells(self):
        coverage = {
            "A": np.array([[True, True, False, False]]),
            "B": np.array([[True, True, True, False]]),
        }
        distances = {
            "A": np.array([[1.0, 4.0, 0.0, 1.0]]),
            "B": np.array([[3.0, 2.0, 5.0, 2.0]]),
        }
        owner = fixed_camera_ownership(coverage, distances, {"A": 1, "B": 2})
        np.testing.assert_array_equal(owner, [[1, 2, 2, 0]])
        self.assertEqual(owner.dtype, np.uint8)

    def test_ties_follow_coverage_insertion_order(self):
        coverage = {"B": np.ones((1, 2), dtype=bool), "A": np.ones((1, 2), dtype=bool)}
        distances = {"A": np.array([[5.0, 4.0]]), "B": np.array([[5.0, 4.0]])}
        codes = {"A": 1, "B": 2}
        np.testing.assert_array_equal(
            fixed_camera_ownership(coverage, distances, codes), [[2, 2]]
        )


class TestCombineOwnedProducts(unittest.TestCase):
    codes = {"A": 1, "B": 2}

    def test_invalid_owner_leaves_holes_despite_valid_overlapping_camera(self):
        products = {
            "A": product(
                [np.nan, 3, 4, 5],
                e0=[500, np.inf, 500, 500],
                valid=[True, True, False, True],
            ),
            "B": product([20, 30, 40, 50]),
        }
        result = combine_owned_products(products, np.array([[1, 1, 1, 1]]), self.codes)
        # A's invalid flag also covers missing or unusable measured brightness.
        for name in ("q", "e0", "sigp", "sigh"):
            self.assertTrue(np.isnan(result[name][0, :3]).all())
        np.testing.assert_array_equal(result["q"], [[np.nan, np.nan, np.nan, 5]])
        np.testing.assert_array_equal(result["source_site"], [[0, 0, 0, 1]])
        np.testing.assert_array_equal(result["ownership_count"], [[1, 1, 1, 2]])

    def test_valid_owner_supplies_all_values_and_uncovered_cells_stay_empty(self):
        products = {
            "A": product([1, 2, 3], e0=[100, 200, 300], sigp=[4, 5, 6], sigh=[7, 8, 9]),
            "B": product([10, 20, 30], e0=[1000, 2000, 3000], sigp=[40, 50, 60], sigh=[70, 80, 90]),
        }
        result = combine_owned_products(products, np.array([[1, 2, 0]]), self.codes)
        for name, expected in (
            ("q", [[1, 20, np.nan]]),
            ("e0", [[100, 2000, np.nan]]),
            ("sigp", [[4, 50, np.nan]]),
            ("sigh", [[7, 80, np.nan]]),
            ("source_site", [[1, 2, 0]]),
            ("ownership_count", [[2, 2, 2]]),
        ):
            np.testing.assert_array_equal(result[name], expected)

    def test_missing_owner_conductance_is_not_filled_from_another_camera(self):
        products = {
            "A": product([1, 2], sigp=[np.nan, 5], sigh=[7, np.nan]),
            "B": product([10, 20], sigp=[40, 50], sigh=[70, 80]),
        }
        result = combine_owned_products(products, np.array([[1, 1]]), self.codes)
        np.testing.assert_array_equal(result["sigp"], [[np.nan, 5]])
        np.testing.assert_array_equal(result["sigh"], [[7, np.nan]])
        np.testing.assert_array_equal(result["source_site"], [[1, 1]])

    def test_disabled_conductances_do_not_require_conductance_arrays(self):
        products = {"A": product([1, 2]), "B": product([10, 20])}
        for values in products.values():
            del values["sigp"]
            del values["sigh"]
        result = combine_owned_products(
            products, np.array([[1, 2]]), self.codes, calculate_conductances=False
        )
        self.assertIsNone(result["sigp"])
        self.assertIsNone(result["sigh"])
        np.testing.assert_array_equal(result["q"], [[1, 20]])

    def test_same_owner_grid_across_timesteps_without_mutating_site_products(self):
        coverage = {site: np.ones((1, 2), dtype=bool) for site in self.codes}
        distances = {"A": np.array([[1.0, 3.0]]), "B": np.array([[2.0, 1.0]])}
        owner = fixed_camera_ownership(coverage, distances, self.codes)
        owner.flags.writeable = False
        for a_valid, b_valid, expected_q, expected_source in (
            ([True, True], [True, True], [[1, 20]], [[1, 2]]),
            ([False, True], [True, False], [[np.nan, np.nan]], [[0, 0]]),
            ([True, True], [True, True], [[1, 20]], [[1, 2]]),
        ):
            products = {"A": product([1, 2], valid=a_valid), "B": product([10, 20], valid=b_valid)}
            snapshots = {
                site: {key: value.copy() for key, value in values.items()}
                for site, values in products.items()
            }
            for values in products.values():
                for value in values.values():
                    value.flags.writeable = False
            result = combine_owned_products(products, owner, self.codes)
            np.testing.assert_array_equal(result["q"], expected_q)
            np.testing.assert_array_equal(result["source_site"], expected_source)
            np.testing.assert_array_equal(owner, [[1, 2]])
            for site, values in products.items():
                for key, value in values.items():
                    np.testing.assert_array_equal(value, snapshots[site][key])


if __name__ == "__main__":
    unittest.main()
