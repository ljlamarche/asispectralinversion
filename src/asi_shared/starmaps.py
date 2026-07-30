"""Generic azimuth/elevation starmap loading and projection."""

from pathlib import Path

import numpy as np
import pymap3d as pm
from astropy.io import fits
from scipy.io import readsav


def azel2geo(site_lat, site_lon, az, el, alt=110.0):
    """Intersect az/el sight lines with a WGS84 shell ``alt`` km above Earth."""
    x, y, z = pm.geodetic2ecef(site_lat, site_lon, 0.0)
    east, north, up = pm.aer2enu(az, el, 1.0)
    vx, vy, vz = pm.enu2uvw(east, north, up, site_lat, site_lon)
    earth = pm.Ellipsoid.from_name("wgs84")
    a2 = (earth.semimajor_axis + alt * 1000.0) ** 2
    c2 = (earth.semiminor_axis + alt * 1000.0) ** 2
    aa = vx**2 / a2 + vy**2 / a2 + vz**2 / c2
    bb = x * vx / a2 + y * vy / a2 + z * vz / c2
    cc = x**2 / a2 + y**2 / a2 + z**2 / c2 - 1
    distance = (np.sqrt(bb**2 - aa * cc) - bb) / aa
    latitude, longitude, _ = pm.ecef2geodetic(
        x + distance * vx, y + distance * vy, z + distance * vz
    )
    return latitude, longitude


def load_array(path):
    """Load the first array from an IDL SAV or a primary FITS image."""
    path = Path(path)
    if path.suffix.lower() in {".fit", ".fits"}:
        return np.asarray(fits.getdata(path), dtype=float)
    values = readsav(path, python_dict=True)
    return np.asarray(values[next(iter(values))], dtype=float)


def load_az_el(azimuth_path, elevation_path, minimum_elevation_deg=15.0):
    azimuth = load_array(azimuth_path)
    elevation = load_array(elevation_path)
    if azimuth.shape != elevation.shape:
        raise ValueError(
            f"Az/el shape mismatch: {azimuth.shape} vs {elevation.shape}"
        )
    return azimuth, elevation, elevation < float(minimum_elevation_deg)
