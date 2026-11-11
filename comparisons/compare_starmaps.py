#!/usr/bin/env python3
"""Compare two VEE starmap pairs in separate red, green, and blue contour plots.

The current pairs are those configured in gneiss.ipynb. The comparison pairs
are discovered as VEE_<wavelength>_*<date>*_<time>_az/el.sav. Both mappings use
the same three image frames, processing settings, geographic grid, and levels
as each other. Background masks and 99th-percentile normalization are evaluated
separately for each mapping, following color_contours.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Support execution by absolute path from the lab317 directory or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_contours as contours

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CHANNELS = {"red": "6300", "green": "5577", "blue": "4278"}
COLORS = {"red": "#d73027", "green": "#138a36", "blue": "#2166ac"}
LABELS = {"red": "630.0 nm red", "green": "557.7 nm green", "blue": "427.8 nm blue"}


def current_starmap_paths(root):
    """Use the notebook's current pairs, including its available 08:00 red pair."""
    return {
        "red": (
            root / "red/6300/VEE/VEE_GASI_630_20260210_080000_asistarcalibration_full_Az.sav",
            root / "red/6300/VEE/VEE_GASI_630_20260210_080000_asistarcalibration_full_El.sav",
        ),
        "green": (
            root / "green/VEE/GNEISS/VEE_GASI_20260210_050100_rot5_full_Az.sav",
            root / "green/VEE/GNEISS/VEE_GASI_20260210_050100_rot5_full_El.sav",
        ),
        "blue": (
            root / "blue/VEE/VEE_MOOSE_4278_20260210_050000_asistarcalibration_full_Az.sav",
            root / "blue/VEE/VEE_MOOSE_4278_20260210_050000_asistarcalibration_full_El.sav",
        ),
    }


def comparison_starmap_pair(root, channel, date, time=None):
    """Require one complete az/el pair with the same filename prefix and time."""
    directory = root / ("red/6300/VEE" if channel == "red" else f"{channel}/VEE")
    wavelength = CHANNELS[channel]
    pattern = re.compile(
        rf"^(VEE_{wavelength}_.*{re.escape(date)}.*_(\d{{6}}))_(az|el)\.sav$",
        re.IGNORECASE,
    )
    pairs = {}
    for path in sorted(directory.rglob("*")):
        match = pattern.fullmatch(path.name)
        if not path.is_file() or match is None or (time and match.group(2) != time):
            continue
        # Keep files in different directories separate; never mix calibrations.
        key = (path.parent, match.group(1).lower())
        components = pairs.setdefault(key, {})
        component = match.group(3).lower()
        if component in components:
            raise ValueError(f"Duplicate {component} starmaps: {components[component]} and {path}")
        components[component] = path
    incomplete = [prefix for (_directory, prefix), parts in pairs.items() if len(parts) != 2]
    if incomplete:
        raise FileNotFoundError(f"Incomplete {channel} az/el starmap pairs: {', '.join(incomplete)}")
    if not pairs:
        raise FileNotFoundError(
            f"No VEE_{wavelength}_*{date}*_{time or 'HHMMSS'}_az/el.sav pair under {directory}"
        )
    if len(pairs) != 1:
        choices = ', '.join(prefix for _directory, prefix in pairs)
        raise ValueError(
            f"Multiple {channel} starmap pairs: {choices}. "
            f"Choose one with --{channel}-starmap-time HHMMSS."
        )
    parts = next(iter(pairs.values()))
    return parts["az"], parts["el"]


def comparison_grid(spacing_km, decimation):
    """Use one regular geodetic grid over the color_contours.py plot bounds."""
    if spacing_km <= 0 or decimation < 1:
        raise ValueError("Grid spacing must be positive and decimation at least one")
    lon_min, lon_max, lat_min, lat_max = contours.BOUNDS
    mean_latitude = 0.5 * (lat_min + lat_max)
    lat_step = spacing_km / 111.0
    lon_step = spacing_km / (111.0 * np.cos(np.deg2rad(mean_latitude)))
    lat = np.arange(lat_min, lat_max + 0.5 * lat_step, lat_step)[::decimation]
    lon = np.arange(lon_min, lon_max + 0.5 * lon_step, lon_step)[::decimation]
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")
    return grid_lat, grid_lon


def mapped_product(image, geometry, channel, grid_lat, grid_lon):
    for coordinate in ("azimuth", "elevation"):
        if geometry[coordinate].shape != image.shape:
            raise ValueError(
                f"{channel}: image/{coordinate} starmap shape mismatch: "
                f"{image.shape} versus {geometry[coordinate].shape}"
            )
    prepared, background, sigma = contours.denoise_like_notebook(image, geometry, channel)
    mapped = contours.regrid(prepared, geometry, grid_lat, grid_lon) - background
    if channel in ("red", "green"):
        factors = (contours.RED_RAYLEIGH_SECONDS_PER_COUNT if channel == "red"
                   else contours.GREEN_RAYLEIGH_SECONDS_PER_COUNT)
        mapped *= factors[contours.SITE] / contours.EXPOSURE_TIME_S_BY_COLOR[channel]
    if not np.any(np.isfinite(mapped) & (mapped > 0)):
        raise ValueError(f"{channel}: no positive mapped brightness within the plot bounds")
    normalized, scale = contours.normalize(mapped)
    return normalized, {"background_counts": background, "sigma_counts": sigma, "p99": scale}


def plot_comparison(channel, target, grid_lat, grid_lon, products, pairs, output):
    fig, axis = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.23)
    axis.pcolormesh(
        grid_lon, grid_lat, products["current"], shading="nearest",
        cmap="Greys", vmin=0, vmax=1, alpha=0.18,
    )
    for name, color, style in (("current", COLORS[channel], "solid"),
                               ("comparison", "#303030", "dashed")):
        axis.contour(
            grid_lon, grid_lat, products[name], levels=contours.CONTOUR_LEVELS,
            colors=color, linestyles=style, linewidths=1.7,
        )
    axis.set_xlim(contours.BOUNDS[:2])
    axis.set_ylim(contours.BOUNDS[2:])
    axis.set_xlabel("Geodetic longitude (degrees east)")
    axis.set_ylabel("Geodetic latitude (degrees north)")
    axis.set_title(
        f"VEE {LABELS[channel]} — starmap comparison\n"
        f"{target:%Y-%m-%d %H:%M:%S} UTC; mapped to {contours.COMMON_ALTITUDE_KM:g} km"
    )
    axis.legend(handles=[
        Line2D([], [], color=COLORS[channel], lw=1.7, label="Current notebook starmap"),
        Line2D([], [], color="#303030", ls="--", lw=1.7,
               label=f"VEE_{CHANNELS[channel]} dated starmap"),
    ], loc="upper right")
    axis.grid(alpha=0.25)
    pair_names = {
        name: re.sub(r"_az\.sav$", "_az/el.sav", paths[0].name, flags=re.IGNORECASE)
        for name, paths in pairs.items()
    }
    fig.text(0.09, 0.135, f"Current: {pair_names['current']}", fontsize=9)
    fig.text(0.09, 0.11, f"Comparison: {pair_names['comparison']}", fontsize=9)
    fig.text(
        0.09, 0.06,
        "Each map normalized by its own positive-pixel 99th percentile within these bounds.\n"
        "Contour levels: " + ', '.join(f"{level:g}" for level in contours.CONTOUR_LEVELS),
        fontsize=9,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--time", default="10:03:00", help="UTC HH:MM:SS (default: 10:03:00)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    for channel in CHANNELS:
        parser.add_argument(
            f"--{channel}-starmap-time", metavar="HHMMSS",
            help=f"Select the {channel} comparison pair if multiple dated pairs exist",
        )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = contours.load_config(path=args.config, project_root=contours.PROJECT_ROOT)
    target = contours.parse_utc(config.date, args.time)
    current = current_starmap_paths(config.paths.starmap_root)
    comparison = {
        channel: comparison_starmap_pair(
            config.paths.starmap_root, channel, config.date,
            getattr(args, f"{channel}_starmap_time"),
        )
        for channel in CHANNELS
    }
    for paths in (*current.values(), *comparison.values()):
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Starmap not found: {path}")
    apex = contours.Apex(date=contours.parse_utc(config.date, config.apex_reference_time))
    grid_lat, grid_lon = comparison_grid(config.target_grid_spacing_km, config.decimation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "time_utc": target.isoformat(), "coordinate_system": "geodetic",
        "bounds": contours.BOUNDS, "contour_levels": contours.CONTOUR_LEVELS,
        "common_altitude_km": contours.COMMON_ALTITUDE_KM,
        "emission_altitude_km": contours.EMISSION_ALTITUDE_KM,
        "normalization": "per-map positive-pixel p99 within plot bounds",
        "channels": {},
    }
    for channel, wavelength in CHANNELS.items():
        print(f"Comparing VEE {channel} at {target:%H:%M:%S} UTC...", flush=True)
        paths = {"current": current[channel], "comparison": comparison[channel]}
        geometries = {
            name: contours.load_geometry(channel, pair, apex) for name, pair in paths.items()
        }
        files = contours.discover_tiffs(config.paths.image_root, config.date, channel)
        catalog = contours.stack_catalog(files, contours.CADENCE_SECONDS[channel])
        image, selected = contours.coadd_closest_frames(catalog, target)
        if channel == "blue":
            bias = contours.iio.imread(
                contours.PROJECT_ROOT / "biasframes/blue_bias_processed.png"
            ).astype(float) / 10.0
            image = image - (bias - np.mean(bias))
        products = {}
        details = {"frames": [
            {"time_utc": time.isoformat(), "file": path, "index": index}
            for _delta, time, path, index in selected
        ], "starmaps": {}}
        for name, geometry in geometries.items():
            products[name], stats = mapped_product(image, geometry, channel, grid_lat, grid_lon)
            details["starmaps"][name] = {
                "azimuth": str(paths[name][0]), "elevation": str(paths[name][1]), **stats,
            }
            print(f"  {name}: {paths[name][0].name}; background={stats['background_counts']:.2f}, "
                  f"sigma={stats['sigma_counts']:.2f}, p99={stats['p99']:.2f}", flush=True)
        output = args.output_dir / f"compare_starmaps_VEE_{wavelength}_{config.date}_{target:%H%M%S}.png"
        plot_comparison(channel, target, grid_lat, grid_lon, products, paths, output)
        details["plot"] = str(output)
        metadata["channels"][channel] = details
        print(f"Saved {output}", flush=True)
    metadata_path = args.output_dir / f"compare_starmaps_VEE_{config.date}_{target:%H%M%S}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Saved input and processing details to {metadata_path}")


if __name__ == "__main__":
    main()
