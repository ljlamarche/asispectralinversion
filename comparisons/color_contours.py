#!/usr/bin/env python3
"""Plot normalized GNEISS color contours for VEE or overlapping ARV/VEE/BVR cameras."""

import argparse
import datetime as dt
import functools
import glob
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import imageio.v3 as iio
import numpy as np
import tifffile
from apexpy import Apex
from scipy.interpolate import griddata
from scipy.io import readsav
from skimage.restoration import cycle_spin, denoise_wavelet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name != "asispectralinversion":
    PROJECT_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asispectralinversion.config import load_config
from asi_shared.detector import correct_detector_frame
from asi_shared.background import BACKGROUND_EDGE_BUFFER_PX, estimate_scalar_background, physical_offsky_mask
from asi_shared.calibration import (
    EXPOSURE_TIME_S_BY_COLOR,
    GREEN_RAYLEIGH_SECONDS_PER_COUNT,
    RED_RAYLEIGH_SECONDS_PER_COUNT,
)
from asi_shared.starmaps import azel2geo, load_array


SITE = "VEE"
SITE_LAT = 67.013
SITE_LON = -146.407
SITE_COORDINATES = {
    "ARV": (68.127, -145.533),
    "VEE": (SITE_LAT, SITE_LON),
    "BVR": (66.360, -147.400),
}
SITE_LINESTYLES = {"ARV": "dashed", "VEE": "solid", "BVR": "dotted"}
FRAMES_TO_COADD = 3
COMMON_ALTITUDE_KM = 110.0
EMISSION_ALTITUDE_KM = {"red": 180.0, "green": 110.0, "blue": 110.0}
CADENCE_SECONDS = {"red": 0.9, "green": 0.3, "blue": 0.90178}
WAVELET_MAX_SHIFTS = {"red": 3, "green": 3, "blue": 4}
BACKGROUND_MINIMUM_PIXELS = 2
CONTOUR_LEVELS = (0.25, 0.5, 0.75, 0.9)
BOUNDS = (-149.0, -143.0, 66.6, 68.2)
MAGNETIC_GRID_SPACING_DEG = (0.5, 1.0)  # Magnetic latitude, longitude.
MAGNETIC_GRID_STYLE = {"color": "0.4", "linewidth": 0.6, "linestyle": "--", "alpha": 0.18}


def parse_utc(date, value):
    return dt.datetime.strptime(f"{date} {value}", "%Y%m%d %H:%M:%S")


def channel_starmap_paths(starmap_root, site=SITE):
    """Return the selected calibration pair for each mapped camera at a site."""
    if site == "ARV":
        return {
            "red": (
                starmap_root / "red/6300/ARV/ARV_GASI_630_20260209_Az.FIT",
                starmap_root / "red/6300/ARV/ARV_GASI_630_20260209_El.FIT",
            ),
            "green": (
                starmap_root / "green/ARV/ARV_GASI_20260209_063700_rot5_full_Az.sav",
                starmap_root / "green/ARV/ARV_GASI_20260209_063700_rot5_full_El.sav",
            ),
        }
    if site == "BVR":
        return {
            "red": (
                starmap_root / "red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_Az.sav",
                starmap_root / "red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_El.sav",
            ),
            "green": (
                starmap_root / "green/BVR/BVR_20260210_090000_750_rot5_Az.sav",
                starmap_root / "green/BVR/BVR_20260210_090000_750_rot5_El.sav",
            ),
        }
    if site != SITE:
        raise ValueError(f"Unknown site: {site}")
    return {
        "red": (
            starmap_root / "red/6300/VEE/VEE_6300_20260210_050100_az.sav",
            starmap_root / "red/6300/VEE/VEE_6300_20260210_050100_el.sav",
        ),
        "green": (
            starmap_root / "green/VEE/GNEISS/VEE_5577_20260210_050145_az.sav",
            starmap_root / "green/VEE/GNEISS/VEE_5577_20260210_050145_el.sav",
        ),
        "blue": (
            starmap_root / "blue/VEE/VEE_4278_20260210_050100_az.sav",
            starmap_root / "blue/VEE/VEE_4278_20260210_050100_el.sav",
        ),
    }


def camera_starmap_paths(starmap_root, sites=(SITE,)):
    """Select available camera starmaps for the requested sites."""
    return {
        (site, channel): pair
        for site in sites
        for channel, pair in channel_starmap_paths(starmap_root, site).items()
    }


def load_geometry(channel, paths, apex, site_lat=SITE_LAT, site_lon=SITE_LON, *, red_alt=180.0):
    emission_altitude = red_alt if channel == "red" else EMISSION_ALTITUDE_KM[channel]
    azimuth = load_array(paths[0])
    elevation = load_array(paths[1])
    minimum_elevation = 15.0 if channel == "blue" else 22.0
    invalid = ~np.isfinite(azimuth) | ~np.isfinite(elevation) | (elevation < minimum_elevation)
    source_lat, source_lon = azel2geo(
        site_lat,
        site_lon,
        azimuth,
        elevation,
        alt=emission_altitude,
    )
    valid = ~invalid & np.isfinite(source_lat) & np.isfinite(source_lon)
    latitude = np.full(source_lat.shape, np.nan)
    longitude = np.full(source_lon.shape, np.nan)
    latitude[valid], longitude[valid], _ = apex.map_to_height(
        source_lat[valid],
        source_lon[valid],
        emission_altitude,
        COMMON_ALTITUDE_KM,
    )
    return {
        "azimuth": azimuth,
        "elevation": elevation,
        "invalid": invalid,
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_valid": valid & np.isfinite(latitude) & np.isfinite(longitude),
    }


def notebook_global_grid_geometries(starmap_root, apex, *, red_alt=180.0):
    return {
        f"{site}_{channel}": load_geometry(
            channel,
            channel_starmap_paths(starmap_root, site)[channel],
            apex,
            site_lat=SITE_COORDINATES[site][0],
            site_lon=SITE_COORDINATES[site][1], red_alt=red_alt,
        )
        for site in SITE_COORDINATES
        for channel in ("red", "green")
    }


def discover_tiffs(image_root, date, channel, site=SITE):
    if channel == "red":
        directory = image_root / "red" / "6300" / site
    elif channel == "green":
        directory = image_root / "green" / site
    else:
        directory = image_root / "blue" / site
    directories = [directory]
    if site == SITE and channel in ("red", "green"):
        directories.insert(0, directory / "GNEISS")
    paths = {
        str(path)
        for directory in directories
        for pattern in ("*.tif", "*.tiff")
        for path in directory.glob(pattern)
        if (date in path.name or date[2:] in path.name)
        and "_101900_102848_" not in path.name
    }
    if not paths:
        raise FileNotFoundError(f"No {site} {channel} TIFFs found for {date}")
    return sorted(paths)


def timestamp_from_filename(path):
    name = Path(path).name
    match = re.search(r"_(\d{8})_(\d{6})\.tiff?$", name, re.IGNORECASE)
    if match:
        return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    match = re.match(r"^[A-Za-z]+(\d{6})_(\d{6})(\d{2})?_16bit\.tiff?$", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse TIFF timestamp from {path}")
    start = dt.datetime.strptime(match.group(1) + match.group(2), "%y%m%d%H%M%S")
    if match.group(3):
        start += dt.timedelta(milliseconds=int(match.group(3)) * 10)
    return start


@functools.lru_cache(maxsize=128)
def read_tiff_frame(path, index):
    with tifffile.TiffFile(path) as tif:
        return correct_detector_frame(tif.pages[index].asarray().astype(float), path)


def stack_catalog(paths, cadence):
    catalog = []
    for path in paths:
        start = timestamp_from_filename(path)
        log_path = Path(path).with_suffix(".log")
        if log_path.exists():
            values = {}
            for line in log_path.read_text(errors="replace").splitlines():
                pieces = line.split(None, 1)
                if len(pieces) == 2:
                    values[pieces[0]] = pieces[1]
            if values.get("Date") and values.get("TimeStart"):
                start = dt.datetime.fromisoformat(f"{values['Date']}T{values['TimeStart']}")
            cadence = float(values.get("ImageCadence(s)", cadence))
            page_count = int(values["NumberOfImages"])
        else:
            with tifffile.TiffFile(path) as tif:
                page_count = len(tif.pages)
        end = start + dt.timedelta(seconds=(page_count - 1) * cadence)
        catalog.append((path, start, end, page_count, cadence))
    return catalog


def coadd_closest_frames(catalog, target):
    candidates = []
    for path, start, _end, page_count, cadence in catalog:
        center = round((target - start).total_seconds() / cadence)
        nearby = range(
            max(0, center - FRAMES_TO_COADD),
            min(page_count, center + FRAMES_TO_COADD + 1),
        )
        for index in nearby:
            frame_time = start + dt.timedelta(seconds=index * cadence)
            candidates.append((abs(frame_time - target), frame_time, path, index))
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:FRAMES_TO_COADD]
    if len(selected) < FRAMES_TO_COADD:
        raise ValueError(f"Only {len(selected)} frames are available near {target}")
    selected.sort(key=lambda item: item[1])
    images = [read_tiff_frame(path, index) for _delta, _time, path, index in selected]
    return np.mean(images, axis=0), selected


def estimate_channel_background(image, elevation, channel):
    """Use the fixed shared 0-degree / 50-pixel background policy."""
    if np.shape(image) != np.shape(elevation):
        raise ValueError(f"{channel}: image/starmap shape mismatch")
    mask = physical_offsky_mask(elevation, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX)
    try:
        return estimate_scalar_background(image, mask, minimum_pixels=BACKGROUND_MINIMUM_PIXELS)
    except ValueError as exc:
        raise ValueError(f"{channel}: {exc}") from exc


def denoise_like_notebook(image, geometry, channel):
    background, sigma, _ = estimate_channel_background(
        image, geometry["elevation"], channel
    )
    filled = np.where(geometry["invalid"] | ~np.isfinite(image), background, image)
    prepared = cycle_spin(
        filled,
        func=denoise_wavelet,
        max_shifts=WAVELET_MAX_SHIFTS[channel],
        func_kw={
            "sigma": sigma,
            "method": "BayesShrink",
            "mode": "soft",
            "rescale_sigma": True,
        },
        workers=1,
        channel_axis=None,
    )
    prepared[~geometry["coordinate_valid"]] = np.nan
    return prepared, background, sigma


def build_grid(geometries, spacing_km, decimation):
    latitudes = np.concatenate([
        geometry["latitude"][geometry["coordinate_valid"]]
        for geometry in geometries.values()
    ])
    longitudes = np.concatenate([
        geometry["longitude"][geometry["coordinate_valid"]]
        for geometry in geometries.values()
    ])
    mean_latitude = 0.5 * (np.min(latitudes) + np.max(latitudes))
    latitude_step = spacing_km / 111.0
    longitude_step = spacing_km / (111.0 * np.cos(np.deg2rad(mean_latitude)))
    lat_axis = np.arange(np.min(latitudes), np.max(latitudes) + 0.5 * latitude_step, latitude_step)
    lon_axis = np.arange(np.min(longitudes), np.max(longitudes) + 0.5 * longitude_step, longitude_step)
    grid_lon, grid_lat = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    return grid_lat[::decimation, ::decimation], grid_lon[::decimation, ::decimation]


def regrid(image, geometry, grid_lat, grid_lon):
    valid = geometry["coordinate_valid"] & np.isfinite(image)
    points = np.column_stack((geometry["longitude"][valid], geometry["latitude"][valid]))
    return griddata(points, image[valid], (grid_lon, grid_lat), method="linear")


def normalize(image):
    valid = np.isfinite(image) & (image > 0)
    scale = float(np.percentile(image[valid], 99))
    return np.where(valid, np.clip(image / scale, 0, 1), np.nan), scale


def parse_cli_time(value):
    """Validate compact UTC input and normalize it to the config's time format."""
    if re.fullmatch(r"[0-9]{6}", value):
        try:
            return dt.datetime.strptime(value, "%H%M%S").strftime("%H:%M:%S")
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("expected a valid UTC time as HHMMSS (e.g. 100300)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--colors", nargs="+", choices=("red", "green", "blue"),
        default=["red", "green", "blue"],
        help="Colors to include, separated by spaces (default: red green blue)",
    )
    parser.add_argument(
        "--time", type=parse_cli_time, metavar="HHMMSS", default=None,
        help="UTC as HHMMSS; defaults to diagnostic_time",
    )
    parser.add_argument("--series", action="store_true", help="Create a series of contour plots")
    parser.add_argument(
        "--sites", nargs="+", type=str.upper, choices=tuple(SITE_COORDINATES),
        default=[SITE], help="Sites to include, separated by spaces (default: VEE)",
    )
    parser.add_argument(
        "--red-alt", type=float, default=EMISSION_ALTITUDE_KM["red"], metavar="KM",
        help="Assumed red emission altitude in km (default: 180); mapped to the common 110 km layer",
    )
    parser.add_argument(
        "--start", type=parse_cli_time, metavar="HHMMSS",
        help="Series start UTC as HHMMSS; defaults to config start",
    )
    parser.add_argument(
        "--end", type=parse_cli_time, metavar="HHMMSS",
        help="Series end UTC as HHMMSS (inclusive); defaults to config end",
    )
    parser.add_argument(
        "--step", type=float, help="Series spacing in seconds; defaults to config step_seconds"
    )
    parser.add_argument(
        "--bounds", type=float, nargs=4, default=BOUNDS,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Plot limits in degrees (default: -149 -143 66.6 68.2)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output file; series mode appends each UTC date/time to the filename stem",
    )
    args = parser.parse_args(argv)
    args.sites = list(dict.fromkeys(args.sites))
    if not np.isfinite(args.red_alt) or args.red_alt <= 0:
        parser.error("--red-alt must be a positive, finite altitude in km")
    if not args.series and any(value is not None for value in (args.start, args.end, args.step)):
        parser.error("--start, --end, and --step require --series")
    if args.series and args.time is not None:
        parser.error("--time cannot be used with --series; use --start and --end")
    if not np.isfinite(args.bounds).all():
        parser.error("--bounds values must be finite")
    lon_min, lon_max, lat_min, lat_max = args.bounds
    if lon_min >= lon_max or lat_min >= lat_max:
        parser.error("--bounds requires LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")
    return args


def target_times(args, config):
    """Select UTC targets on the configured date, including an aligned endpoint."""
    if not args.series:
        return [parse_utc(config.date, args.time or config.diagnostic_time)]
    start = parse_utc(config.date, args.start if args.start is not None else config.start)
    end = parse_utc(config.date, args.end if args.end is not None else config.end)
    if end < start:
        raise ValueError("--end must be at or after --start on the configured date")
    seconds = args.step if args.step is not None else config.step_seconds
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("--step must be a positive, finite number of seconds")
    try:
        step = dt.timedelta(seconds=seconds)
    except OverflowError as exc:
        raise ValueError("--step is too large") from exc
    if step <= dt.timedelta(0):
        raise ValueError("--step must be at least one microsecond")
    count = (end - start) // step + 1
    return (start + index * step for index in range(count))


def contour_output_path(output, target, series, sites=(SITE,)):
    timestamp = target.strftime("%Y%m%d_%H%M%S")
    if target.microsecond:
        timestamp += f"_{target.microsecond:06d}"
    if output is None:
        selection = "_".join(sites)
        return PROJECT_ROOT / "comparisons" / f"color_contours_{selection}_{timestamp}.png"
    if series:
        return output.with_name(f"{output.stem}_{timestamp}{output.suffix or '.png'}")
    return output


def build_magnetic_grid(apex, bounds):
    """Sample magnetic coordinates at the common altitude over the visible map."""
    lon_min, lon_max, lat_min, lat_max = bounds
    longitude, latitude = np.meshgrid(
        np.linspace(lon_min, lon_max, 101),
        np.linspace(lat_min, lat_max, 101),
    )
    magnetic_lat, magnetic_lon = apex.geo2apex(latitude, longitude, COMMON_ALTITUDE_KM)
    # Keep a regional grid continuous when magnetic longitude crosses +/-180 degrees.
    magnetic_lon = np.rad2deg(
        np.unwrap(np.unwrap(np.deg2rad(magnetic_lon), axis=1), axis=0)
    )
    lines = []
    for coordinates, spacing in zip((magnetic_lat, magnetic_lon), MAGNETIC_GRID_SPACING_DEG):
        finite = coordinates[np.isfinite(coordinates)]
        if not finite.size:
            continue
        first = int(np.ceil(finite.min() / spacing))
        last = int(np.floor(finite.max() / spacing))
        levels = np.arange(first, last + 1) * spacing
        if levels.size:
            lines.append((np.ma.masked_invalid(coordinates), levels))
    return longitude, latitude, lines


def add_scale_bar(axis):
    """Add a local east-west scale at map-center latitude and mapping altitude."""
    lon_min, lon_max = axis.get_xlim()
    latitude = np.deg2rad(np.mean(axis.get_ylim()))
    # WGS84 prime-vertical radius, plus the altitude of the mapped contours.
    radius_km = 6378.137 / np.sqrt(1 - 6.69437999014e-3 * np.sin(latitude) ** 2)
    km_per_degree = np.deg2rad(1) * (radius_km + COMMON_ALTITUDE_KM) * np.cos(latitude)
    target_km = abs(lon_max - lon_min) * km_per_degree * 0.25
    magnitude = 10 ** np.floor(np.log10(target_km))
    length_km = max(value * magnitude for value in (1, 2, 5) if value * magnitude <= target_km)
    bar = AnchoredSizeBar(
        axis.transData, length_km / km_per_degree, f"{length_km:g} km",
        loc="lower left", pad=0.5, borderpad=0.8, sep=4,
        size_vertical=0.006 * abs(np.diff(axis.get_ylim())[0]),
        color="black", frameon=True,
    )
    bar.patch.set_facecolor("white")
    bar.patch.set_alpha(0.85)
    bar.patch.set_edgecolor("0.7")
    axis.add_artist(bar)


def plot_contours(
    target, geometries, grid_lat, grid_lon, catalogs, blue_bias, output,
    bounds=BOUNDS, *, magnetic_zeniths, magnetic_grid,
):
    sites = tuple(dict.fromkeys(site for site, _channel in geometries))
    merged = len(sites) > 1
    products = {}
    metadata = {}
    for camera, geometry in geometries.items():
        site, channel = camera
        try:
            image, selected = coadd_closest_frames(catalogs[camera], target)
        except ValueError as exc:
            raise ValueError(f"{site} {channel}: {exc}") from exc
        if camera == (SITE, "blue"):
            image = image - blue_bias
        prepared, background, sigma = denoise_like_notebook(
            image, geometry, channel
        )
        regridded = regrid(prepared, geometry, grid_lat, grid_lon)
        if channel in {"red", "green"}:
            factor = (
                RED_RAYLEIGH_SECONDS_PER_COUNT[site]
                if channel == "red"
                else GREEN_RAYLEIGH_SECONDS_PER_COUNT[site]
            )
            regridded = (
                (regridded - background)
                / EXPOSURE_TIME_S_BY_COLOR[channel]
                * factor
            )
        else:
            regridded = regridded - background
        # Keep every camera's full valid footprint and its own normalization.
        # There is deliberately no shared ownership mask or composite image.
        products[camera], scale = normalize(regridded)
        metadata[camera] = (selected, background, sigma, scale)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    green_products = [product for (_site, channel), product in products.items() if channel == "green"]
    for product in green_products:
        axis.pcolormesh(
            grid_lon, grid_lat, product, shading="nearest",
            cmap="Greys", vmin=0, vmax=1, alpha=0.35 / len(green_products),
        )
    magnetic_grid_lon, magnetic_grid_lat, magnetic_lines = magnetic_grid
    for coordinates, levels in magnetic_lines:
        axis.contour(
            magnetic_grid_lon, magnetic_grid_lat, coordinates, levels=levels,
            colors=MAGNETIC_GRID_STYLE["color"],
            linewidths=MAGNETIC_GRID_STYLE["linewidth"],
            linestyles=MAGNETIC_GRID_STYLE["linestyle"],
            alpha=MAGNETIC_GRID_STYLE["alpha"], zorder=1.5,
        )
    styles = {
        "red": ("#d73027", "solid", "630.0 nm red"),
        "green": ("#1a9850", "dashed", "557.7 nm green"),
        "blue": ("#2166ac", "dashdot", "427.8 nm blue"),
    }
    for (site, channel), product in products.items():
        color, linestyle, _label = styles[channel]
        if merged:
            linestyle = SITE_LINESTYLES[site]
        axis.contour(
            grid_lon, grid_lat, product, levels=CONTOUR_LEVELS,
            colors=color, linewidths=1.7, linestyles=linestyle,
        )
    zenith_handles = []
    for site, (magnetic_lat, magnetic_lon) in magnetic_zeniths.items():
        zenith_handles.append(axis.scatter(
            magnetic_lon, magnetic_lat, marker="X", s=120,
            color="yellow", edgecolor="black", linewidth=0.8, zorder=10,
            label=f"Magnetic zenith ({COMMON_ALTITUDE_KM:g} km)",
        ))
        if merged:
            axis.annotate(
                site, (magnetic_lon, magnetic_lat), xytext=(6, 6),
                textcoords="offset points", fontsize=8, zorder=10,
                annotation_clip=True,
            )
    lon_min, lon_max, lat_min, lat_max = bounds
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    add_scale_bar(axis)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    target_label = target.isoformat(sep=" ", timespec="microseconds" if target.microsecond else "seconds")
    axis.set_title(
        f"{'/'.join(sites)} normalized color contours at {target_label} UTC\n"
        f"common altitude {COMMON_ALTITUDE_KM:g} km | contours at 25th, 50th, 75th, 90th percentiles"
    )
    channel_handles = [
        Patch(
            facecolor="none", edgecolor=color,
            linestyle="solid" if merged else linestyle, label=label,
        )
        for channel, (color, linestyle, label) in styles.items()
        if any(camera_channel == channel for _site, camera_channel in products)
    ]
    site_handles = [
        Line2D([0], [0], color="0.35", linestyle=SITE_LINESTYLES[site], label=site)
        for site in sites
    ] if merged else []
    axis.legend(
        handles=channel_handles + site_handles + zenith_handles[:1] + [
            Line2D(
                [0], [0], label=f"Magnetic grid ({COMMON_ALTITUDE_KM:g} km)",
                **MAGNETIC_GRID_STYLE,
            ),
        ],
        loc="best",
    )
    axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for (site, channel), (selected, background, sigma, scale) in metadata.items():
        times = ", ".join(item[1].strftime("%H:%M:%S.%f") for item in selected)
        units = "R" if channel in {"red", "green"} else "background-subtracted counts"
        print(
            f"{site} {channel}: frames {times}; background={background:.2f}, "
            f"sigma={sigma:.2f}, p99={scale:.2f} {units}"
        )
    print(f"Saved {output}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(path=args.config, project_root=PROJECT_ROOT)
    try:
        targets = target_times(args, config)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    apex_time = parse_utc(config.date, config.apex_reference_time)
    apex = Apex(date=apex_time)
    paths = {
        camera: pair
        for camera, pair in camera_starmap_paths(
            config.paths.starmap_root, sites=args.sites
        ).items()
        if camera[1] in args.colors
    }
    if not paths:
        raise SystemExit("Error: no available cameras for the selected sites and colors (blue is VEE only)")
    sites = tuple(dict.fromkeys(site for site, _channel in paths))
    print("Using cameras: " + ", ".join(f"{site} {channel}" for site, channel in paths), flush=True)
    # Match the notebook: map the station's field line from ground to the common layer.
    magnetic_zeniths = {
        site: apex.map_to_height(*SITE_COORDINATES[site], 0.0, COMMON_ALTITUDE_KM)[:2]
        for site in sites
    }
    magnetic_grid = build_magnetic_grid(apex, args.bounds)
    geometries = {
        (site, channel): load_geometry(
            channel, pair, apex,
            site_lat=SITE_COORDINATES[site][0], site_lon=SITE_COORDINATES[site][1],
            red_alt=args.red_alt,
        )
        for (site, channel), pair in paths.items()
    }
    global_geometries = geometries if args.sites != [SITE] else notebook_global_grid_geometries(
        config.paths.starmap_root, apex, red_alt=args.red_alt,
    )
    grid_lat, grid_lon = build_grid(
        global_geometries, config.target_grid_spacing_km, config.decimation
    )
    blue_bias = None
    if (SITE, "blue") in paths:
        blue_bias = iio.imread(PROJECT_ROOT / "biasframes/blue_bias_processed.png").astype(float) / 10.0
        blue_bias -= np.mean(blue_bias)
    catalogs = {
        (site, channel): stack_catalog(
            discover_tiffs(config.paths.image_root, config.date, channel, site=site),
            CADENCE_SECONDS[channel],
        )
        for site, channel in paths
    }
    for target in targets:
        output = contour_output_path(args.output, target, args.series, sites=sites)
        print(f"Plotting {target.isoformat(sep=' ')} UTC", flush=True)
        plot_contours(
            target, geometries, grid_lat, grid_lon, catalogs, blue_bias, output,
            bounds=args.bounds, magnetic_zeniths=magnetic_zeniths,
            magnetic_grid=magnetic_grid,
        )


if __name__ == "__main__":
    main()
