"""Independent camera overlays for the merged color-contour diagnostic."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import datetime as dt
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "comparisons" / "color_contours.py"
SPEC = importlib.util.spec_from_file_location("color_contours_merge_test_module", SCRIPT)
color_contours = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color_contours)

from matplotlib.axes import Axes
from matplotlib.figure import Figure


CAMERAS = {
    (site, channel)
    for site in ("ARV", "VEE", "BVR")
    for channel in ("red", "green")
} | {("VEE", "blue")}


class TestColorContourMergeSelection(unittest.TestCase):
    def test_site_and_altitude_arguments(self):
        args = color_contours.parse_args(["--sites", "bvr", "BVR", "vee", "--red-alt", "240"])
        self.assertEqual(args.sites, ["BVR", "VEE"])
        self.assertEqual(args.red_alt, 240.)
        self.assertEqual(color_contours.parse_args([]).red_alt, 180.)
        for argv in (["--merge"], ["--sites", "unknown"], ["--sites"],
                     ["--red-alt", "0"], ["--red-alt", "-1"],
                     ["--red-alt", "nan"], ["--red-alt", "inf"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    color_contours.parse_args(argv)

    def test_red_altitude_reaches_both_mapping_stages_only_for_red(self):
        coordinates = np.array([[67.]])
        for channel, expected in (("red", 240.), ("green", 110.), ("blue", 110.)):
            apex = Mock()
            apex.map_to_height.return_value = (np.array([67.]), np.array([-146.]), None)
            with patch.object(color_contours, "load_array", return_value=np.array([[45.]])), \
                 patch.object(color_contours, "azel2geo", return_value=(coordinates, coordinates)) as geo:
                color_contours.load_geometry(channel, ("az", "el"), apex, red_alt=240.)
            self.assertEqual(geo.call_args.kwargs["alt"], expected)
            self.assertEqual(apex.map_to_height.call_args.args[2:4], (expected, 110.))

    def test_merge_defaults_off_and_selects_seven_available_camera_maps(self):
        self.assertEqual(color_contours.parse_args([]).sites, ["VEE"])
        self.assertEqual(color_contours.parse_args(["--sites", "arv", "vee", "bvr"]).sites, ["ARV", "VEE", "BVR"])
        root = Path("starmaps")
        default = color_contours.camera_starmap_paths(root)
        merged = color_contours.camera_starmap_paths(root, sites=("ARV", "VEE", "BVR"))
        self.assertEqual(set(default), {("VEE", channel) for channel in ("red", "green", "blue")})
        self.assertEqual(set(merged), CAMERAS)
        self.assertEqual({key: merged[key] for key in default}, default)
        self.assertEqual(
            default[("VEE", "green")][0],
            root / "green/VEE/GNEISS/VEE_5577_20260210_050145_az.sav",
        )
        self.assertEqual(
            default[("VEE", "red")][0],
            root / "red/6300/VEE/VEE_6300_20260210_050100_az.sav",
        )
        self.assertEqual(
            default[("VEE", "blue")][0],
            root / "blue/VEE/VEE_4278_20260210_050100_az.sav",
        )
        for (site, channel), pair in merged.items():
            directory = root / ("red/6300" if channel == "red" else channel) / site
            self.assertEqual(len(pair), 2)
            self.assertTrue(all(path.is_relative_to(directory) for path in pair))

    def test_tiff_discovery_keeps_sites_wavelengths_and_dates_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_arv = root / "red/6300/ARV/ARV_630_20260210_101900.tiff"
            expected_vee = {
                root / "red/6300/VEE/VEE_630_20260210_101900.tiff",
                root / "red/6300/VEE/GNEISS/VEE_630_20260210_102800.tiff",
            }
            paths = [expected_arv, *expected_vee,
                root / "red/6300/BVR/BVR_630_20260210_101900.tiff",
                root / "red/8446/ARV/ARV_844_20260210_101900.tiff",
                root / "green/ARV/ARV_558_20260210_101900.tiff",
                root / "red/6300/ARV/ARV_630_20260209_101900.tiff",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(
                color_contours.discover_tiffs(root, "20260210", "red", site="ARV"),
                [str(expected_arv)],
            )
            self.assertEqual(
                set(color_contours.discover_tiffs(root, "20260210", "red")),
                {str(path) for path in expected_vee},
            )

    def test_merged_output_has_distinct_default_name_and_honors_explicit_output(self):
        target = dt.datetime(2026, 2, 10, 10, 25, 20, 500000)
        self.assertEqual(
            color_contours.contour_output_path(None, target, True, sites=("ARV", "VEE", "BVR")),
            SCRIPT.parent / "color_contours_ARV_VEE_BVR_20260210_102520_500000.png",
        )
        explicit = Path("custom/contours.png")
        self.assertEqual(
            color_contours.contour_output_path(explicit, target, False, sites=("ARV", "VEE", "BVR")), explicit
        )


class TestColorContourMergeProcessing(unittest.TestCase):
    def test_series_loads_all_cameras_once_and_reuses_them(self):
        config = SimpleNamespace(
            date="20260210", start="10:19:00", end="10:19:02", step_seconds=1.,
            apex_reference_time="10:24:00", target_grid_spacing_km=2., decimation=1,
            paths=SimpleNamespace(image_root=Path("images"), starmap_root=Path("starmaps")),
        )
        paths = color_contours.camera_starmap_paths(config.paths.starmap_root, sites=("ARV", "VEE", "BVR"))
        camera_for_paths = {pair: camera for camera, pair in paths.items()}
        geometries = {camera: object() for camera in paths}
        catalogs = {camera: object() for camera in paths}
        lat, lon = np.zeros((2, 2)), np.ones((2, 2))
        magnetic_grid = object()
        with ExitStack() as stack:
            def patched(name, **kwargs):
                return stack.enter_context(patch.object(color_contours, name, **kwargs))

            patched("load_config", return_value=config)
            apex = patched("Apex")
            apex.return_value.map_to_height.side_effect = lambda lat, lon, *_args: (lat - 0.2, lon - 0.1, 0.)
            magnetic = patched("build_magnetic_grid", return_value=magnetic_grid)
            load_geometry = patched("load_geometry", side_effect=lambda channel, pair, _apex, **_kwargs: geometries[camera_for_paths[pair]])
            global_geometry = patched("notebook_global_grid_geometries")
            grid = patched("build_grid", return_value=(lat, lon))
            discovery = patched("discover_tiffs", side_effect=lambda _root, _date, channel, site="VEE": [(site, channel)])
            catalog = patched("stack_catalog", side_effect=lambda files, _cadence: catalogs[files[0]])
            stack.enter_context(patch.object(color_contours.iio, "imread", return_value=np.ones((2, 2))))
            plot = patched("plot_contours")
            stack.enter_context(redirect_stdout(io.StringIO()))
            color_contours.main(["--sites", "arv", "vee", "bvr", "--series"])

        self.assertEqual(load_geometry.call_count, 7)
        self.assertEqual(discovery.call_count, 7)
        self.assertEqual(catalog.call_count, 7)
        self.assertEqual(apex.call_count, 1)
        self.assertEqual(apex.return_value.map_to_height.call_count, 3)
        magnetic.assert_called_once()
        global_geometry.assert_not_called()
        grid.assert_called_once_with(geometries, 2., 1)
        self.assertEqual(plot.call_count, 3)
        expected_zeniths = {site: (coords[0] - 0.2, coords[1] - 0.1) for site, coords in color_contours.SITE_COORDINATES.items()}
        shared = plot.call_args_list[0]
        for offset, call in enumerate(plot.call_args_list):
            target, frame_geometries, latitude, longitude, frame_catalogs, _bias, output = call.args
            self.assertEqual(target, dt.datetime(2026, 2, 10, 10, 19, offset))
            self.assertEqual(frame_geometries, geometries)
            self.assertIs(frame_geometries, shared.args[1])
            self.assertEqual(frame_catalogs, catalogs)
            self.assertIs(frame_catalogs, shared.args[4])
            self.assertIs(latitude, lat)
            self.assertIs(longitude, lon)
            self.assertEqual(call.kwargs["magnetic_zeniths"], expected_zeniths)
            self.assertIs(call.kwargs["magnetic_grid"], magnetic_grid)
            self.assertEqual(output.name, f"color_contours_ARV_VEE_BVR_20260210_10190{offset}.png")

    def test_overlapping_camera_contours_keep_each_calibration_and_blue_bias(self):
        paths = color_contours.camera_starmap_paths(Path("starmaps"), sites=("ARV", "VEE", "BVR"))
        geometries = {camera: object() for camera in paths}
        catalogs = {camera: camera for camera in paths}
        target = dt.datetime(2026, 2, 10, 10, 25, 20)
        selected = [(dt.timedelta(0), target, "synthetic", 0)]
        images = {
            camera: np.arange(16, dtype=float).reshape(4, 4) * 10 + 100 + index
            for index, camera in enumerate(paths)
        }
        blue_bias = np.tile([-2., -1., 1., 2.], (4, 1))
        longitude, latitude = np.meshgrid(np.linspace(-148., -144., 4), np.linspace(66.8, 67.8, 4))
        normalize = color_contours.normalize
        contour = Axes.contour
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(color_contours, "coadd_closest_frames", side_effect=lambda camera, _target: (images[camera].copy(), selected)))
            denoise = stack.enter_context(patch.object(color_contours, "denoise_like_notebook", side_effect=lambda image, *_args: (image.copy(), 10., 1.)))
            stack.enter_context(patch.object(color_contours, "regrid", side_effect=lambda image, *_args: image.copy()))
            normalized = stack.enter_context(patch.object(color_contours, "normalize", side_effect=normalize))
            drawn = stack.enter_context(patch.object(Axes, "contour", autospec=True, side_effect=contour))
            stack.enter_context(patch.object(Figure, "savefig"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            color_contours.plot_contours(
                target, geometries, latitude, longitude, catalogs, blue_bias,
                Path(directory) / "merged.png",
                magnetic_zeniths={site: coords for site, coords in color_contours.SITE_COORDINATES.items()},
                magnetic_grid=(longitude, latitude, []),
            )

        self.assertEqual(drawn.call_count, 7)
        self.assertEqual(normalized.call_count, 7)
        for index, (site, channel) in enumerate(paths):
            expected_input = images[(site, channel)].copy()
            if (site, channel) == ("VEE", "blue"):
                expected_input -= blue_bias
            np.testing.assert_allclose(denoise.call_args_list[index].args[0], expected_input)
            expected_calibrated = expected_input - 10.
            if channel != "blue":
                factors = (color_contours.RED_RAYLEIGH_SECONDS_PER_COUNT if channel == "red" else color_contours.GREEN_RAYLEIGH_SECONDS_PER_COUNT)
                expected_calibrated *= factors[site] / color_contours.EXPOSURE_TIME_S_BY_COLOR[channel]
            np.testing.assert_allclose(normalized.call_args_list[index].args[0], expected_calibrated)
            call = drawn.call_args_list[index]
            np.testing.assert_array_equal(call.args[1], longitude)
            np.testing.assert_array_equal(call.args[2], latitude)
            # Every camera retains all overlapping coordinates and its own normalized data.
            self.assertTrue(np.isfinite(call.args[3]).all())
            np.testing.assert_allclose(call.args[3], normalize(expected_calibrated)[0])
            self.assertEqual(call.kwargs["linestyles"], {"ARV": "dashed", "VEE": "solid", "BVR": "dotted"}[site])
        self.assertFalse(color_contours.plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
