"""Pairwise camera-overlap masks, independent of plotting dependencies."""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def calculate_masks(
    site_lat_1,
    site_lon_1,
    az_1,
    el_1,
    site_lat_2,
    site_lon_2,
    az_2,
    el_2,
    altitude_1_km=110.0,
    altitude_2_km=110.0,
):
    lat1, lon1 = np.deg2rad([site_lat_1, site_lon_1])
    lat2, lon2 = np.deg2rad([site_lat_2, site_lon_2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    bearing1 = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    bearing2 = np.arctan2(
        np.sin(-dlon) * np.cos(lat1),
        np.cos(lat2) * np.sin(lat1)
        - np.sin(lat2) * np.cos(lat1) * np.cos(-dlon),
    )
    angular_distance = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2) ** 2
            + np.sin(dlon / 2) ** 2 * np.cos(lat1) * np.cos(lat2)
        )
    )

    def one_mask(azimuth, elevation, bearing, altitude):
        gamma = np.deg2rad(azimuth) - bearing
        lam = np.deg2rad(elevation)
        ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude)
        aa = np.sqrt(1 - ratio**2 * np.cos(lam) ** 2) / (
            ratio * np.cos(lam)
        )
        bb = (aa - np.tan(lam)) / (1 + aa * np.tan(lam)) * np.cos(gamma)
        return bb > np.tan(angular_distance / 2)

    return (
        one_mask(az_1, el_1, bearing1, altitude_1_km),
        one_mask(az_2, el_2, bearing2, altitude_2_km),
    )


def build_overlap_masks(skymaps, map_alt_km=None):
    for site, skymap in skymaps.items():
        skymap["extra_masks"] = {}
        for other_site, other in skymaps.items():
            if site == other_site:
                continue
            site_mask, _ = calculate_masks(
                skymap["site_lat"],
                skymap["site_lon"],
                skymap["azmt"],
                skymap["elev"],
                other["site_lat"],
                other["site_lon"],
                other["azmt"],
                other["elev"],
                map_alt_km or skymap.get("map_alt_km", 110.0),
                map_alt_km or other.get("map_alt_km", 110.0),
            )
            skymap["extra_masks"][other_site] = site_mask
