import unittest

import numpy as np

from asispectralinversion import inversion


def inversion_case():
    """Small float32 tables with a nondegenerate red/green brightness surface."""
    qvec = np.array([0.5, 1, 2, 4, 8, 16], dtype=np.float32)
    e0vec = np.array([100, 200, 400, 800, 1600], dtype=np.float32)
    qgrid, e0grid = np.meshgrid(qvec, e0vec)
    lookup = {"Qvec": qvec, "E0vec": e0vec}
    for channel, brightness, background in (
        ("red", qgrid * (3 - e0grid * 0.0007), 2.75),
        ("green", qgrid * (1 + e0grid * 0.0002), 1.25),
        ("blue", qgrid * (1 + e0grid / 1600), 0.75),
    ):
        lookup[f"{channel}mat"] = brightness + np.float32(background)
        lookup[f"{channel}bright_airglow"] = np.array(
            [[background]], dtype=np.float32
        )

    q = np.array([[1.3, 3.5, 6], [10, 12, np.nan]])
    e0 = np.array([[350, 700, 1000], [1400, 600, np.nan]])
    observations = (
        q * (3 - e0 * 0.0007),
        q * (1 + e0 * 0.0002),
        q * (1 + e0 / 1600),
    )
    return lookup, observations


def run_inversion(mode, lookup, observations):
    if mode == "rgb":
        return inversion.calculate_E0_Q_v2(*observations, lookup, plot=False)
    red, green, _blue = observations
    return inversion.calculate_E0_Q_v2_RGonly(
        red, green, None, lookup, checkagreement=False, plot=False
    )


# Captured from the unmodified deepcopy implementation, using the real Q and E0
# interpolation routines. The RGB case also exercises automatic blue bounds.
EXPECTED = {
    "rgb": (
        [[0.7921875, 2.515625, 4.875], [9.375, 8.25, np.nan]],
        [
            [351.4020847790033, 705.263266808171, 1020.0001603016666],
            [1418.7502051305546, 607.1428817602416, np.nan],
        ],
    ),
    "rg": (
        [
            [1.3094780083593207, 3.4537434793929904, 6.203695763321068],
            [9.460425612317927, 10.750662051707039, np.nan],
        ],
        [
            [351.4020365190871, 705.2632412832313, 1020.0001603016666],
            [1418.7502051305546, 607.1428817602416, np.nan],
        ],
    ),
}


class UnusedLookupTable:
    def __deepcopy__(self, memo):
        raise AssertionError("Inversion must not copy unused lookup tables")


class TestInversionLookup(unittest.TestCase):
    def assert_repeated_inversions_preserve_results_and_inputs(self, mode):
        lookup, observations = inversion_case()
        original_objects = lookup.copy()
        snapshots = {name: value.copy() for name, value in lookup.items()}
        observation_snapshots = [value.copy() for value in observations]
        # Lookup arrays are shared between timesteps and should also support
        # read-only inputs without changing their dtype or subtracting twice.
        for value in lookup.values():
            value.flags.writeable = False

        for _ in range(2):
            result = run_inversion(mode, lookup, observations)
            for actual, expected in zip(result, EXPECTED[mode]):
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
            self.assertEqual(lookup.keys(), original_objects.keys())
            for name, expected in snapshots.items():
                self.assertIs(lookup[name], original_objects[name])
                self.assertEqual(lookup[name].dtype, expected.dtype)
                np.testing.assert_array_equal(lookup[name], expected)
            for actual, expected in zip(observations, observation_snapshots):
                np.testing.assert_array_equal(actual, expected)

    def test_rgb_repeated_inversions_preserve_results_and_inputs(self):
        self.assert_repeated_inversions_preserve_results_and_inputs("rgb")

    def test_rg_repeated_inversions_preserve_results_and_inputs(self):
        self.assert_repeated_inversions_preserve_results_and_inputs("rg")

    def assert_unused_lookup_tables_are_not_copied(self, mode):
        lookup, observations = inversion_case()
        unused = UnusedLookupTable()
        for name in ("sigPmat", "sigHmat", "edensmat", "SigPmat", "SigHmat"):
            lookup[name] = unused

        result = run_inversion(mode, lookup, observations)
        for actual, expected in zip(result, EXPECTED[mode]):
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
        self.assertIs(lookup["edensmat"], unused)

    def test_rgb_does_not_copy_unused_lookup_tables(self):
        self.assert_unused_lookup_tables_are_not_copied("rgb")

    def test_rg_does_not_copy_unused_lookup_tables(self):
        self.assert_unused_lookup_tables_are_not_copied("rg")


if __name__ == "__main__":
    unittest.main()
