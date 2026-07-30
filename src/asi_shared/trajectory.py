"""Portable GNEISS CSV trajectory loading and magnetic-footprint lookup."""

import csv
import re

import numpy as np
from apexpy import Apex

GPS_TIME_RE = re.compile(r"^\d{3}\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$")
UTC_TIME_COLUMN = "Time (UTC / From GPS Receiver)"
FLIGHT_TIME_COLUMN = "Flight Time (Official T0)"
LAT_COLUMN = "Latitude"
LON_COLUMN = "Longitude"
ALT_COLUMN = "Altitude (km)"


def hhmmss_to_seconds(value):
    value = str(value).strip()
    base, _, fraction = value.partition(".")
    base = base.zfill(6)
    return (
        int(base[:2]) * 3600
        + int(base[2:4]) * 60
        + int(base[4:6])
        + (float(f"0.{fraction}") if fraction else 0.0)
    )


def parse_gps_utc_time(value):
    match = GPS_TIME_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid GPS UTC time: {value}")
    hour, minute, second = map(int, match.group(1, 2, 3))
    fraction = match.group(4) or ""
    return hour * 3600 + minute * 60 + second + (
        float(f"0.{fraction}") if fraction else 0.0
    )


def load_traj_records(filename):
    with open(filename, encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get(UTC_TIME_COLUMN)]
    if not rows:
        raise ValueError(f"No trajectory samples found in {filename}")
    return (
        np.asarray([parse_gps_utc_time(row[UTC_TIME_COLUMN]) for row in rows]),
        np.asarray([float(row[FLIGHT_TIME_COLUMN]) for row in rows]),
        np.asarray([float(row[LAT_COLUMN]) for row in rows]),
        np.asarray([float(row[LON_COLUMN]) for row in rows]),
        np.asarray([float(row[ALT_COLUMN]) for row in rows]),
    )


def build_traj_lookup(traj_path, color="green"):
    utc, flight, lat, lon, altitude = load_traj_records(traj_path)
    target_altitude = 180.0 if str(color).lower() == "red" else 110.0
    mapped_lat, mapped_lon, _ = Apex().map_to_height(
        lat, lon, altitude, target_altitude
    )
    return {
        "times": flight,
        "utc_times": utc,
        "lats": np.asarray(mapped_lat),
        "lons": np.asarray(mapped_lon),
        "raw_lats": lat,
        "raw_lons": lon,
        "raw_alts_km": altitude,
        "path": traj_path,
    }


def _nearest_index(lookup, map_time):
    seconds = hhmmss_to_seconds(map_time)
    utc = lookup["utc_times"]
    if seconds < utc[0] or seconds > utc[-1]:
        return None
    return int(np.argmin(np.abs(utc - seconds)))


def lookup_traj_position(lookup, map_time):
    index = _nearest_index(lookup, map_time)
    if index is None:
        return None, None
    return float(lookup["lats"][index]), float(lookup["lons"][index])


def lookup_traj_geodetic_position(lookup, map_time):
    index = _nearest_index(lookup, map_time)
    if index is None:
        return None, None, None
    return (
        float(lookup["raw_lats"][index]),
        float(lookup["raw_lons"][index]),
        float(lookup["raw_alts_km"][index]),
    )
