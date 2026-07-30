#!/usr/bin/env python3
"""Sample GNEISS inversion products at rocket magnetic footprints.

This is the inversion-product analogue of asi_mapping/scripts/
traj_brightness_series.py.  It uses the times already present in a GNEISS
inversion HDF5 or legacy NPZ file,
maps each rocket trajectory to the inversion's reference altitude, samples the
nearest valid inversion cells, and writes a structured HDF5 file and plot.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))
from asispectralinversion.config import load_config  # noqa: E402
from asi_shared.missions import (  # noqa: E402
    rocket_launch_datetime,
    rocket_time_window_datetimes,
    trajectory_config_tuples,
)
from asi_shared.trajectory import (  # noqa: E402
    build_traj_lookup,
    lookup_traj_geodetic_position,
    lookup_traj_position,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "gneiss_rg_20260210_101900_102840_step1s.h5"
)

PRODUCTS = {
    "q_mW_m2": ("energy_flux", "mW m-2", "Q"),
    "e0_eV": ("characteristic_energy", "eV", "E0"),
    "sigp_S": ("pedersen_conductance", "S", "SigP"),
    "sigh_S": ("hall_conductance", "S", "SigH"),
}


def datetime64_to_datetime(value):
    """Convert a numpy datetime64 scalar to a naive UTC datetime."""
    microseconds = value.astype("datetime64[us]").astype(np.int64)
    return dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=int(microseconds))


def iso_to_datetime(value):
    """Convert an HDF5 byte/string ISO timestamp to a naive UTC datetime."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dt.datetime.fromisoformat(str(value).removesuffix("Z"))


def open_inversion(path):
    """Open an HDF5 or legacy NPZ inversion without loading its cubes eagerly."""
    if h5py.is_hdf5(path):
        source = h5py.File(path, "r")
        required = {"time_iso", "gridlat", "gridlon", "completed", "combined"}
        missing = sorted(required - set(source.keys()))
        if missing:
            source.close()
            raise ValueError("HDF5 input is missing: " + ", ".join(missing))

        combined = source["combined"]
        product_info = {key: PRODUCTS[key] for key in PRODUCTS if key in combined}
        if not product_info:
            source.close()
            raise ValueError("HDF5 input contains no supported combined products")

        completed = np.asarray(source["completed"], dtype=bool)
        time_indices = np.flatnonzero(completed)
        if time_indices.size == 0:
            source.close()
            raise ValueError("HDF5 input contains no completed inversions")
        all_times = [iso_to_datetime(value) for value in source["time_iso"][:]]
        times = [all_times[index] for index in time_indices]
        mapping_altitude_km = float(source.attrs["common_mapping_altitude_km"])
        return {
            "source": source,
            "time_indices": time_indices,
            "times": times,
            "gridlat": np.asarray(source["gridlat"], dtype=float),
            "gridlon": np.asarray(source["gridlon"], dtype=float),
            "mapping_altitude_km": mapping_altitude_km,
            "product_info": product_info,
            "products": {key: combined[key] for key in product_info},
        }

    source = np.load(path)
    required = {"times", "gridlat", "gridlon", "common_mapping_altitude_km"}
    missing = sorted(required - set(source.files))
    if missing:
        source.close()
        raise ValueError("NPZ input is missing: " + ", ".join(missing))
    product_info = {key: PRODUCTS[key] for key in PRODUCTS if key in source.files}
    if not product_info:
        source.close()
        raise ValueError("NPZ input contains no supported inversion products")
    times = [datetime64_to_datetime(value) for value in source["times"]]
    return {
        "source": source,
        "time_indices": np.arange(len(times)),
        "times": times,
        "gridlat": np.asarray(source["gridlat"], dtype=float),
        "gridlon": np.asarray(source["gridlon"], dtype=float),
        "mapping_altitude_km": float(source["common_mapping_altitude_km"]),
        "product_info": product_info,
        "products": {key: source[key] for key in product_info},
    }


def time_argument(value):
    """Format a datetime for asi_mapping's nearest-trajectory lookup."""
    return value.strftime("%H%M%S.%f").rstrip("0").rstrip(".")


def validate_grid(gridlat, gridlon, product_shape):
    if gridlat.ndim != 2 or gridlon.ndim != 2:
        raise ValueError("gridlat and gridlon must both be two-dimensional")
    if gridlat.shape != gridlon.shape:
        raise ValueError(
            f"gridlat/gridlon shape mismatch: {gridlat.shape} vs {gridlon.shape}"
        )
    if gridlat.shape != product_shape:
        raise ValueError(
            f"Grid/product shape mismatch: {gridlat.shape} vs {product_shape}"
        )


def approximate_distance_km(lat, lon, target_lat, target_lon):
    """Local tangent-plane distance, sufficient for this small Alaska grid."""
    north_km = (lat - target_lat) * 111.0
    east_km = (lon - target_lon) * 111.0 * np.cos(np.deg2rad(target_lat))
    return np.hypot(north_km, east_km)


def grid_spacing_km(gridlat, gridlon):
    """Return the median distance between adjacent finite grid cells."""
    distances = []
    horizontal = (
        np.isfinite(gridlat[:, 1:])
        & np.isfinite(gridlat[:, :-1])
        & np.isfinite(gridlon[:, 1:])
        & np.isfinite(gridlon[:, :-1])
    )
    if np.any(horizontal):
        mean_lat = 0.5 * (gridlat[:, 1:] + gridlat[:, :-1])
        dx = np.diff(gridlon, axis=1) * 111.0 * np.cos(np.deg2rad(mean_lat))
        dy = np.diff(gridlat, axis=1) * 111.0
        distances.append(np.hypot(dx, dy)[horizontal])
    vertical = (
        np.isfinite(gridlat[1:, :])
        & np.isfinite(gridlat[:-1, :])
        & np.isfinite(gridlon[1:, :])
        & np.isfinite(gridlon[:-1, :])
    )
    if np.any(vertical):
        mean_lat = 0.5 * (gridlat[1:, :] + gridlat[:-1, :])
        dx = np.diff(gridlon, axis=0) * 111.0 * np.cos(np.deg2rad(mean_lat))
        dy = np.diff(gridlat, axis=0) * 111.0
        distances.append(np.hypot(dx, dy)[vertical])
    if not distances:
        raise ValueError("Could not estimate inversion-grid spacing")
    return float(np.nanmedian(np.concatenate(distances)))


def sample_nearest_cells(
    values,
    gridlat,
    gridlon,
    target_lat,
    target_lon,
    neighbors,
    maximum_nearest_distance_km,
):
    """Average the nearest valid cells, provided the footprint is on the grid."""
    valid = np.isfinite(values) & np.isfinite(gridlat) & np.isfinite(gridlon)
    if not np.any(valid) or target_lat is None or target_lon is None:
        return None

    rows, columns = np.nonzero(valid)
    distances = approximate_distance_km(
        gridlat[valid], gridlon[valid], target_lat, target_lon
    )
    count = min(neighbors, distances.size)
    nearest = np.argpartition(distances, count - 1)[:count]
    nearest = nearest[np.argsort(distances[nearest])]
    closest_distance = float(distances[nearest[0]])
    outside = closest_distance > maximum_nearest_distance_km
    if outside:
        return {
            "value": np.nan,
            "standard_deviation": np.nan,
            "closest_distance_km": closest_distance,
            "mean_distance_km": np.nan,
            "row": int(rows[nearest[0]]),
            "column": int(columns[nearest[0]]),
            "cell_count": 0,
            "outside_footprint": True,
        }

    selected_values = values[valid][nearest]
    return {
        "value": float(np.mean(selected_values)),
        "standard_deviation": float(np.std(selected_values)),
        "closest_distance_km": closest_distance,
        "mean_distance_km": float(np.mean(distances[nearest])),
        "row": int(rows[nearest[0]]),
        "column": int(columns[nearest[0]]),
        "cell_count": int(nearest.size),
        "outside_footprint": False,
    }


def compressed_dataset(group, name, values, units=None):
    values = np.asarray(values)
    options = (
        {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        if values.ndim
        else {}
    )
    dataset = group.create_dataset(name, data=values, **options)
    if units:
        dataset.attrs["units"] = units
    return dataset


def write_hdf5(
    path,
    input_path,
    times,
    mapping_altitude_km,
    neighbors,
    spacing_km,
    product_info,
    rocket_results,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "inversion_trajectory_footprint_series"
        h5.attrs["schema_version"] = "1.0"
        h5.attrs["mission"] = "GNEISS"
        h5.attrs["source_inversion_file"] = str(input_path.resolve())
        h5.attrs["mapped_altitude_km"] = mapping_altitude_km
        h5.attrs["neighbor_cells"] = neighbors
        h5.attrs["grid_spacing_km"] = spacing_km
        h5.attrs["maximum_nearest_distance_km"] = 1.5 * spacing_km
        h5.attrs["products_json"] = json.dumps(list(product_info))
        h5.create_dataset(
            "time_iso",
            data=np.asarray([value.isoformat() for value in times], dtype=object),
            dtype=string_dtype,
        )
        tg = np.asarray(
            [(value - rocket_launch_datetime("397")).total_seconds() for value in times]
        )
        compressed_dataset(h5, "time_since_tg_s", tg, "s")

        rockets_group = h5.create_group("rockets")
        for rocket, result in rocket_results.items():
            group = rockets_group.create_group(rocket)
            group.attrs["label"] = result["label"]
            group.attrs["source_trajectory_file"] = result["trajectory_file"]
            compressed_dataset(group, "footprint_latitude_deg", result["footprint_lat"], "degrees_north")
            compressed_dataset(group, "footprint_longitude_deg", result["footprint_lon"], "degrees_east")
            compressed_dataset(group, "geodetic_latitude_deg", result["geodetic_lat"], "degrees_north")
            compressed_dataset(group, "geodetic_longitude_deg", result["geodetic_lon"], "degrees_east")
            compressed_dataset(group, "geodetic_altitude_km", result["geodetic_alt"], "km")
            products_group = group.create_group("products")
            for npz_key, (dataset_name, units, _plot_label) in product_info.items():
                product_group = products_group.create_group(dataset_name)
                samples = result["products"][npz_key]
                compressed_dataset(product_group, "mean", samples["value"], units)
                compressed_dataset(product_group, "spatial_standard_deviation", samples["standard_deviation"], units)
                compressed_dataset(product_group, "closest_distance_km", samples["closest_distance_km"], "km")
                compressed_dataset(product_group, "mean_distance_km", samples["mean_distance_km"], "km")
                compressed_dataset(product_group, "nearest_row", samples["row"])
                compressed_dataset(product_group, "nearest_column", samples["column"])
                compressed_dataset(product_group, "cell_count", samples["cell_count"])
                compressed_dataset(product_group, "outside_footprint", samples["outside_footprint"])
    print(f"Wrote {path}")


def plot_results(path, times, product_info, rocket_results):
    products = list(product_info.items())
    fig, axes = plt.subplots(len(products), 1, figsize=(12, 3.6 * len(products)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (npz_key, (_dataset_name, units, label)) in zip(axes, products):
        for rocket, result in rocket_results.items():
            values = result["products"][npz_key]["value"]
            axis.plot(times, values, linestyle="-", linewidth=1.4, label=f"{rocket}")
        axis.set_ylabel(f"{label} ({units})")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("UTC")
    fig.suptitle("GNEISS inversion values at 110 km rocket magnetic footprints")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="GNEISS inversion HDF5 or legacy NPZ",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output HDF5 path")
    parser.add_argument("--neighbors", type=int, default=25, help="Number of nearest valid cells to average")
    parser.add_argument("--rockets", nargs="+", choices=["397", "398"], default=["397", "398"])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    run_config = load_config(project_root=PROJECT_ROOT)

    try:
        inversion = open_inversion(args.input)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    source = inversion["source"]
    time_indices = inversion["time_indices"]
    times = inversion["times"]
    gridlat = inversion["gridlat"]
    gridlon = inversion["gridlon"]
    mapping_altitude_km = inversion["mapping_altitude_km"]
    product_info = inversion["product_info"]
    products = inversion["products"]

    sample_product = next(iter(products.values()))
    if sample_product.ndim != 3:
        source.close()
        raise ValueError("Inversion products must have shape (time, latitude, longitude)")
    validate_grid(gridlat, gridlon, sample_product.shape[1:])
    for key, values in products.items():
        if values.shape != sample_product.shape:
            source.close()
            raise ValueError(f"Product shape mismatch for {key}: {values.shape}")
    if not np.isclose(mapping_altitude_km, 110.0):
        source.close()
        raise ValueError(
            f"Trajectory helper maps green footprints to 110 km, but input uses {mapping_altitude_km:g} km"
        )

    spacing_km = grid_spacing_km(gridlat, gridlon)
    maximum_distance_km = 1.5 * spacing_km
    trajectory_configs = {
        str(tag): (key, label, path)
        for key, tag, label, path in trajectory_config_tuples(
            "GNEISS",
            paths={
                "397": run_config.trajectory_397,
                "398": run_config.trajectory_398,
            },
            date=times[0].strftime("%Y%m%d"),
        )
    }
    rocket_results = {}
    for rocket in args.rockets:
        key, label, trajectory_path = trajectory_configs[rocket]
        if trajectory_path is None or not Path(trajectory_path).exists():
            parser.error(
                f"trajectory {rocket} is not configured or does not exist; "
                "set it under [trajectories] in config.toml"
            )
        lookup = build_traj_lookup(str(trajectory_path), color="green")
        window_start, window_end = rocket_time_window_datetimes(rocket)
        result = {
            "label": label,
            "trajectory_file": Path(trajectory_path).name,
            "footprint_lat": [], "footprint_lon": [],
            "geodetic_lat": [], "geodetic_lon": [], "geodetic_alt": [],
            "products": {
                name: {field: [] for field in (
                    "value", "standard_deviation", "closest_distance_km",
                    "mean_distance_km", "row", "column", "cell_count",
                    "outside_footprint",
                )}
                for name in product_info
            },
        }
        for time_index, timestamp in zip(time_indices, times):
            in_window = window_start <= timestamp <= window_end
            lookup_time = time_argument(timestamp)
            footprint = lookup_traj_position(lookup, lookup_time) if in_window else (None, None)
            geodetic = lookup_traj_geodetic_position(lookup, lookup_time) if in_window else (None, None, None)
            footprint_lat, footprint_lon = footprint
            result["footprint_lat"].append(np.nan if footprint_lat is None else footprint_lat)
            result["footprint_lon"].append(np.nan if footprint_lon is None else footprint_lon)
            for target, value in zip(("geodetic_lat", "geodetic_lon", "geodetic_alt"), geodetic):
                result[target].append(np.nan if value is None else value)
            for product_name, values in products.items():
                sample = sample_nearest_cells(
                    values[time_index], gridlat, gridlon,
                    footprint_lat, footprint_lon, args.neighbors, maximum_distance_km,
                )
                if sample is None:
                    sample = {
                        "value": np.nan, "standard_deviation": np.nan,
                        "closest_distance_km": np.nan, "mean_distance_km": np.nan,
                        "row": -1, "column": -1, "cell_count": 0,
                        "outside_footprint": True,
                    }
                for field, value in sample.items():
                    result["products"][product_name][field].append(value)
        for name in ("footprint_lat", "footprint_lon", "geodetic_lat", "geodetic_lon", "geodetic_alt"):
            result[name] = np.asarray(result[name], dtype=float)
        for samples in result["products"].values():
            for field, values in samples.items():
                dtype = bool if field == "outside_footprint" else int if field in {"row", "column", "cell_count"} else float
                samples[field] = np.asarray(values, dtype=dtype)
        rocket_results[rocket] = result
    source.close()

    output = args.output or args.input.with_name(args.input.stem + "_footprints.h5")
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        output = output.with_suffix(".h5")
    write_hdf5(
        output, args.input, times, mapping_altitude_km, args.neighbors,
        spacing_km, product_info, rocket_results,
    )
    if not args.no_plot:
        plot_results(output.with_suffix(".png"), times, product_info, rocket_results)

    for rocket, result in rocket_results.items():
        valid = np.isfinite(result["products"][next(iter(product_info))]["value"])
        print(f"36.{rocket}: {np.count_nonzero(valid)}/{len(times)} in-footprint samples")


if __name__ == "__main__":
    main()
