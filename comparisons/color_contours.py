#!/usr/bin/env python3
"""Reproduce the GNEISS notebook's VEE normalized color-contour diagnostic."""

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
from matplotlib.patches import Patch
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
from asi_shared.background import estimate_scalar_background, physical_offsky_mask
from asi_shared.calibration import (
    EXPOSURE_TIME_S_BY_COLOR,
    GREEN_RAYLEIGH_SECONDS_PER_COUNT,
    RED_RAYLEIGH_SECONDS_PER_COUNT,
)
from asi_shared.starmaps import azel2geo, load_array


SITE = "VEE"
SITE_LAT = 67.013
SITE_LON = -146.407
FRAMES_TO_COADD = 3
COMMON_ALTITUDE_KM = 110.0
EMISSION_ALTITUDE_KM = {"red": 180.0, "green": 110.0, "blue": 107.0}
CADENCE_SECONDS = {"red": 0.9, "green": 0.3, "blue": 0.90178}
WAVELET_MAX_SHIFTS = {"red": 3, "green": 3, "blue": 4}
BACKGROUND_EDGE_BUFFER_PX = 100
CONTOUR_LEVELS = (0.25, 0.5, 0.75, 0.9)
BOUNDS = (-149.0, -143.0, 66.6, 68.2)


def parse_utc(date, value):
    return dt.datetime.strptime(f"{date} {value}", "%Y%m%d %H:%M:%S")


def channel_starmap_paths(starmap_root):
    return {
        "red": (
            starmap_root / "red/6300/VEE/VEE_GASI_630_20260210_045700_asistarcalibration_full_Az.sav",
            starmap_root / "red/6300/VEE/VEE_GASI_630_20260210_045700_asistarcalibration_full_El.sav",
        ),
        "green": (
            starmap_root / "green/VEE/GNEISS/VEE_GASI_20260210_050100_rot5_full_Az.sav",
            starmap_root / "green/VEE/GNEISS/VEE_GASI_20260210_050100_rot5_full_El.sav",
        ),
        "blue": (
            starmap_root / "blue/VEE/VEE_MOOSE_4278_20260210_050000_asistarcalibration_full_Az.sav",
            starmap_root / "blue/VEE/VEE_MOOSE_4278_20260210_050000_asistarcalibration_full_El.sav",
        ),
    }


def load_geometry(channel, paths, apex, site_lat=SITE_LAT, site_lon=SITE_LON):
    azimuth = load_array(paths[0])
    elevation = load_array(paths[1])
    minimum_elevation = 15.0 if channel == "blue" else 22.0
    invalid = ~np.isfinite(azimuth) | ~np.isfinite(elevation) | (elevation < minimum_elevation)
    source_lat, source_lon = azel2geo(
        site_lat,
        site_lon,
        azimuth,
        elevation,
        alt=EMISSION_ALTITUDE_KM[channel],
    )
    valid = ~invalid & np.isfinite(source_lat) & np.isfinite(source_lon)
    latitude = np.full(source_lat.shape, np.nan)
    longitude = np.full(source_lon.shape, np.nan)
    latitude[valid], longitude[valid], _ = apex.map_to_height(
        source_lat[valid],
        source_lon[valid],
        EMISSION_ALTITUDE_KM[channel],
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


def notebook_global_grid_geometries(starmap_root, apex):
    site_coordinates = {
        "ARV": (68.127, -145.533),
        "VEE": (67.013, -146.407),
        "BVR": (66.360, -147.400),
    }
    paths = {
        "ARV": {
            "red": (
                starmap_root / "red/6300/ARV/ARV_GASI_630_20260209_Az.FIT",
                starmap_root / "red/6300/ARV/ARV_GASI_630_20260209_El.FIT",
            ),
            "green": (
                starmap_root / "green/ARV/ARV_GASI_20260209_063700_rot5_full_Az.sav",
                starmap_root / "green/ARV/ARV_GASI_20260209_063700_rot5_full_El.sav",
            ),
        },
        "VEE": {
            "red": channel_starmap_paths(starmap_root)["red"],
            "green": channel_starmap_paths(starmap_root)["green"],
        },
        "BVR": {
            "red": (
                starmap_root / "red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_Az.sav",
                starmap_root / "red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_El.sav",
            ),
            "green": (
                starmap_root / "green/BVR/BVR_20260210_090000_750_rot5_Az.sav",
                starmap_root / "green/BVR/BVR_20260210_090000_750_rot5_El.sav",
            ),
        },
    }
    return {
        f"{site}_{channel}": load_geometry(
            channel,
            paths[site][channel],
            apex,
            site_lat=site_coordinates[site][0],
            site_lon=site_coordinates[site][1],
        )
        for site in ("ARV", "VEE", "BVR")
        for channel in ("red", "green")
    }


def discover_tiffs(image_root, date, channel):
    if channel == "red":
        directories = [
            image_root / "red/6300/VEE/GNEISS",
            image_root / "red/6300/VEE",
        ]
    elif channel == "green":
        directories = [
            image_root / "green/VEE/GNEISS",
            image_root / "green/VEE",
        ]
    else:
        directories = [image_root / "blue/VEE"]
    paths = {
        str(path)
        for directory in directories
        for pattern in ("*.tif", "*.tiff")
        for path in directory.glob(pattern)
        if (date in path.name or date[2:] in path.name)
        and "_101900_102848_" not in path.name
    }
    if not paths:
        raise FileNotFoundError(f"No VEE {channel} TIFFs found for {date}")
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
        return tif.pages[index].asarray().astype(float)


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


def denoise_like_notebook(image, geometry, channel):
    background_mask = physical_offsky_mask(
        geometry["elevation"], edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX
    )
    background, sigma, _ = estimate_scalar_background(image, background_mask)
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--time", default=None, help="UTC as HH:MM:SS; defaults to diagnostic_time")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(path=args.config, project_root=PROJECT_ROOT)
    time_text = args.time or config.diagnostic_time
    target = parse_utc(config.date, time_text)
    apex_time = parse_utc(config.date, config.apex_reference_time)
    apex = Apex(date=apex_time)
    paths = channel_starmap_paths(config.paths.starmap_root)
    geometries = {
        channel: load_geometry(channel, paths[channel], apex)
        for channel in ("red", "green", "blue")
    }
    global_geometries = notebook_global_grid_geometries(
        config.paths.starmap_root, apex
    )
    grid_lat, grid_lon = build_grid(
        global_geometries, config.target_grid_spacing_km, config.decimation
    )

    products = {}
    metadata = {}
    blue_bias = iio.imread(PROJECT_ROOT / "biasframes/blue_bias_processed.png").astype(float) / 10.0
    blue_bias -= np.mean(blue_bias)
    for channel in ("red", "green", "blue"):
        files = discover_tiffs(config.paths.image_root, config.date, channel)
        catalog = stack_catalog(files, CADENCE_SECONDS[channel])
        image, selected = coadd_closest_frames(catalog, target)
        if channel == "blue":
            image = image - blue_bias
        prepared, background, sigma = denoise_like_notebook(
            image, geometries[channel], channel
        )
        regridded = regrid(prepared, geometries[channel], grid_lat, grid_lon)
        if channel in {"red", "green"}:
            factor = (
                RED_RAYLEIGH_SECONDS_PER_COUNT[SITE]
                if channel == "red"
                else GREEN_RAYLEIGH_SECONDS_PER_COUNT[SITE]
            )
            regridded = (
                (regridded - background)
                / EXPOSURE_TIME_S_BY_COLOR[channel]
                * factor
            )
        else:
            regridded = regridded - background
        products[channel], scale = normalize(regridded)
        metadata[channel] = (selected, background, sigma, scale)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.pcolormesh(
        grid_lon, grid_lat, products["green"], shading="nearest",
        cmap="Greys", vmin=0, vmax=1, alpha=0.35,
    )
    styles = {
        "red": ("#d73027", "solid", "630.0 nm red"),
        "green": ("#1a9850", "dashed", "557.7 nm green"),
        "blue": ("#2166ac", "dashdot", "427.8 nm blue"),
    }
    for channel, (color, linestyle, _label) in styles.items():
        axis.contour(
            grid_lon, grid_lat, products[channel], levels=CONTOUR_LEVELS,
            colors=color, linewidths=1.7, linestyles=linestyle,
        )
    lon_min, lon_max, lat_min, lat_max = BOUNDS
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        f"VEE normalized color contours at {target:%Y-%m-%d %H:%M:%S} UTC\n"
        f"common altitude {COMMON_ALTITUDE_KM:g} km"
    )
    axis.legend(
        handles=[
            Patch(facecolor="none", edgecolor=color, linestyle=linestyle, label=label)
            for color, linestyle, label in styles.values()
        ],
        loc="best",
    )
    axis.grid(alpha=0.25)
    output = args.output or (
        PROJECT_ROOT / "comparisons" / f"color_contours_VEE_{config.date}_{target:%H%M%S}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for channel in ("red", "green", "blue"):
        selected, background, sigma, scale = metadata[channel]
        times = ", ".join(item[1].strftime("%H:%M:%S.%f") for item in selected)
        units = "R" if channel in {"red", "green"} else "background-subtracted counts"
        print(
            f"{channel}: frames {times}; background={background:.2f}, "
            f"sigma={sigma:.2f}, p99={scale:.2f} {units}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
