"""CLI and output regressions for color-contour time series."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import datetime as dt
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "comparisons" / "color_contours.py"
SPEC = importlib.util.spec_from_file_location("color_contours_series_test_module", SCRIPT)
color_contours = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color_contours)


def example_config():
    return SimpleNamespace(
        date="20260210",
        start="10:19:00",
        end="10:19:02",
        step_seconds=1.0,
        diagnostic_time="10:25:20",
        apex_reference_time="10:24:00",
        target_grid_spacing_km=2.0,
        decimation=1,
        paths=SimpleNamespace(
            image_root=Path("images"), starmap_root=Path("starmaps")
        ),
    )


class TestColorContourSeriesArguments(unittest.TestCase):
    def targets(self, argv, config=None):
        return list(color_contours.target_times(
            color_contours.parse_args(argv), config or example_config()
        ))

    def test_default_mode_remains_one_diagnostic_time(self):
        self.assertFalse(color_contours.parse_args([]).series)
        self.assertEqual(self.targets([]), [dt.datetime(2026, 2, 10, 10, 25, 20)])
        for value, expected in (
            ("102030", dt.datetime(2026, 2, 10, 10, 20, 30)),
            ("010203", dt.datetime(2026, 2, 10, 1, 2, 3)),
            ("000000", dt.datetime(2026, 2, 10)),
        ):
            with self.subTest(value=value):
                self.assertEqual(self.targets(["--time", value]), [expected])

    def test_series_defaults_use_config_and_include_aligned_endpoint(self):
        start = dt.datetime(2026, 2, 10, 10, 19)
        self.assertEqual(
            self.targets(["--series"]),
            [start + dt.timedelta(seconds=offset) for offset in range(3)],
        )

    def test_explicit_fractional_cadence_does_not_overshoot_end(self):
        start = dt.datetime(2026, 2, 10, 0, 20)
        self.assertEqual(
            self.targets([
                "--series", "--start", "002000", "--end", "002001",
                "--step", "0.4",
            ]),
            [start + dt.timedelta(seconds=offset) for offset in (0, 0.4, 0.8)],
        )

    def test_equal_bounds_produce_one_frame(self):
        self.assertEqual(
            self.targets(["--series", "--start", "102000", "--end", "102000"]),
            [dt.datetime(2026, 2, 10, 10, 20)],
        )

    def test_series_options_require_series_and_time_is_single_only(self):
        cases = (
            ["--start", "101900"],
            ["--end", "101902"],
            ["--step", "1"],
            ["--series", "--time", "101900"],
        )
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    color_contours.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_reversed_range_and_invalid_clock_times_are_rejected(self):
        with self.assertRaises(ValueError):
            self.targets(["--series", "--start", "102000", "--end", "101900"])
        invalid_times = (
            "240000", "106000", "105960", "10200", "0102000", "10:20:00",
            "invalid", "１２３４５６",
        )
        for option in ("--time", "--start", "--end"):
            for value in invalid_times:
                argv = (["--series"] if option != "--time" else []) + [option, value]
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        color_contours.parse_args(argv)
                    self.assertEqual(raised.exception.code, 2)

    def test_nonpositive_nonfinite_and_unrepresentable_steps_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), 1e-8):
            with self.subTest(value=value):
                args = color_contours.parse_args(["--series"])
                args.step = value
                with self.assertRaises(ValueError):
                    list(color_contours.target_times(args, example_config()))

    def test_invalid_config_cadence_is_also_rejected(self):
        config = example_config()
        config.step_seconds = 0
        with self.assertRaises(ValueError):
            self.targets(["--series"], config=config)


class TestColorContourSeriesOutput(unittest.TestCase):
    def test_default_single_output_name_is_preserved(self):
        target = dt.datetime(2026, 2, 10, 10, 25, 20)
        self.assertEqual(
            color_contours.contour_output_path(None, target, False),
            SCRIPT.parent / "color_contours_VEE_20260210_102520.png",
        )

    def test_explicit_single_output_is_unchanged(self):
        output = Path("custom/contours.pdf")
        self.assertEqual(
            color_contours.contour_output_path(
                output, dt.datetime(2026, 2, 10, 10, 25, 20), False
            ),
            output,
        )

    def test_series_outputs_are_distinct_with_subsecond_cadence(self):
        first = dt.datetime(2026, 2, 10, 10, 25, 20)
        second = first + dt.timedelta(seconds=0.5)
        self.assertEqual(
            color_contours.contour_output_path(None, second, True),
            SCRIPT.parent / "color_contours_VEE_20260210_102520_500000.png",
        )
        for output, expected_first, expected_second in (
            (
                Path("custom/contours.pdf"),
                Path("custom/contours_20260210_102520.pdf"),
                Path("custom/contours_20260210_102520_500000.pdf"),
            ),
            (
                Path("custom/contours"),
                Path("custom/contours_20260210_102520.png"),
                Path("custom/contours_20260210_102520_500000.png"),
            ),
        ):
            with self.subTest(output=output):
                self.assertEqual(
                    color_contours.contour_output_path(output, first, True), expected_first
                )
                self.assertEqual(
                    color_contours.contour_output_path(output, second, True), expected_second
                )

    def test_main_reuses_setup_and_writes_one_output_per_target(self):
        config = example_config()
        grid_lat = np.zeros((2, 2))
        grid_lon = np.ones((2, 2))
        geometries = {("VEE", channel): object() for channel in ("red", "green", "blue")}
        catalogs = {camera: [f"{camera}-catalog"] for camera in geometries}
        magnetic_grid = object()
        with ExitStack() as stack:
            def patched(name, **kwargs):
                return stack.enter_context(patch.object(color_contours, name, **kwargs))

            patched("load_config", return_value=config)
            apex = patched("Apex")
            apex.return_value.map_to_height.return_value = (66.8142, -146.5428, 0.0)
            build_magnetic_grid = patched("build_magnetic_grid", return_value=magnetic_grid)
            load_geometry = patched(
                "load_geometry",
                side_effect=lambda channel, *_args, **_kwargs: geometries[("VEE", channel)],
            )
            global_geometry = patched("notebook_global_grid_geometries", return_value={})
            build_grid = patched("build_grid", return_value=(grid_lat, grid_lon))
            discover = patched(
                "discover_tiffs",
                side_effect=lambda _root, _date, channel, site="VEE": [(site, channel)],
            )
            catalog = patched(
                "stack_catalog", side_effect=lambda files, _cadence: catalogs[files[0]]
            )
            bias = stack.enter_context(patch.object(
                color_contours.iio, "imread", return_value=np.array([[10, 20], [30, 40]])
            ))
            plot = patched("plot_contours")
            stack.enter_context(redirect_stdout(io.StringIO()))
            color_contours.main(["--series", "--output", "custom/contours.png"])

        self.assertEqual(apex.call_count, 1)
        apex.return_value.map_to_height.assert_called_once_with(
            color_contours.SITE_LAT, color_contours.SITE_LON,
            0.0, color_contours.COMMON_ALTITUDE_KM,
        )
        build_magnetic_grid.assert_called_once_with(apex.return_value, color_contours.BOUNDS)
        self.assertEqual(load_geometry.call_count, 3)
        self.assertEqual(global_geometry.call_count, 1)
        self.assertEqual(build_grid.call_count, 1)
        self.assertEqual(discover.call_count, 3)
        self.assertEqual(catalog.call_count, 3)
        self.assertEqual(bias.call_count, 1)
        self.assertEqual(plot.call_count, 3)
        shared = plot.call_args_list[0].args
        for offset, invocation in enumerate(plot.call_args_list):
            target, geometry, latitude, longitude, frame_catalogs, blue_bias, output = invocation.args
            self.assertEqual(target, dt.datetime(2026, 2, 10, 10, 19, offset))
            self.assertEqual(geometry, geometries)
            self.assertIs(geometry, shared[1])
            self.assertIs(latitude, grid_lat)
            self.assertIs(longitude, grid_lon)
            self.assertEqual(frame_catalogs, catalogs)
            self.assertIs(frame_catalogs, shared[4])
            self.assertIs(blue_bias, shared[5])
            np.testing.assert_allclose(blue_bias, [[-1.5, -0.5], [0.5, 1.5]])
            self.assertEqual(output, Path(f"custom/contours_20260210_10190{offset}.png"))
            self.assertEqual(invocation.kwargs["magnetic_zeniths"], {"VEE": (66.8142, -146.5428)})
            self.assertIs(invocation.kwargs["magnetic_grid"], magnetic_grid)


if __name__ == "__main__":
    unittest.main()
