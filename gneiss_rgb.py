"""Run the GNEISS RGB notebook workflow as a standalone Python script.

Run with the lab317 environment::

    /Users/anniepflaum/lab317/.venv/bin/python gneiss_rgb.py

Reads config.toml beside this script (or ASISPECTRAL_CONFIG), writes the same
HDF5 inversion products as gneiss_rgb.ipynb, without inspection plots. The notebook's overwrite_output=True default is preserved.

The main guard prevents spawned timestamp workers from rerunning setup and
writing the output. Definitions stay in module scope so the existing worker
snapshot mechanism can capture their globals and source code.
"""

if __name__ == "__main__":
    # # GNEISS ARV/VEE/BVR RGB inversion
    # 
    # Blue 427.8 nm constrains Q; the red/green ratio constrains E0 at that Q. All three mapped footprints are required for fixed site ownership. Missing or invalid owner data remain holes. Blue calibration is (background-subtracted mean counts / exposure seconds) × factor: ARV 100, VEE 40, BVR 10.
    # 
    # All three sites use configured blue starmaps. ARV THA is user-verified 4278 Å despite stale 7774 Å log labels; both reported and verified wavelengths are recorded in output metadata. Blue exposure comes from shared calibration constants; cadence comes from sidecar logs. Outputs use a separate gneiss_rgb filename; gneiss.ipynb retains RG processing.
    # 
    # Only owned, finite positive RGB pixels are inverted. Site Q/E0 maps show owned regions.
    # 
    # Series timestamps use `timestamp_workers` separate processes (default 2), with sequential sites inside each worker and one HDF5 writer. Set 1 for sequential execution. Workers reuse prepared geometry and lookup tables; each has its own TIFF cache.

    # ## Imports

    import datetime as dt
    import functools
    import glob
    import json
    import os
    import re
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import h5py
    import numpy as np
    import pymap3d as pm
    import tifffile
    from astropy.io import fits
    from apexpy import Apex
    from scipy.io import readsav
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt, label as connected_component_labels
    from scipy.spatial import Delaunay
    from skimage.restoration import cycle_spin, denoise_wavelet

    project_root = Path(__file__).resolve().parent
    local_src = str(project_root / 'src')
    for import_path in (local_src,):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

    from asispectralinversion.inversion import (
        calculate_E0_Q_v2,
        calculate_Sig,
        load_lookup_tables,
        load_lookup_tables_directory,
    )
    from asispectralinversion.config import load_config
    from asispectralinversion.ownership import (
        OWNERSHIP_METHOD,
        fixed_camera_ownership,
        combine_owned_products,
        invert_owned_rgb,
    )
    from asi_shared.calibration import (
        EXPOSURE_TIME_S_BY_COLOR,
        GREEN_RAYLEIGH_SECONDS_PER_COUNT,
        RED_RAYLEIGH_SECONDS_PER_COUNT,
        BLUE_RAYLEIGH_SECONDS_PER_COUNT,
    )
    from asi_shared.background import BACKGROUND_EDGE_BUFFER_PX, estimate_scalar_background, physical_offsky_mask
    from asi_shared.starmaps import azel2geo
    from asispectralinversion.blue import blue_stack_catalog, coadd_blue_frames

    from contextlib import closing
    from asispectralinversion.rgb_series import snapshot_rgb_worker, iter_rgb_series

    from asispectralinversion.failure_reasons import FAILURE_REASONS

    # ## Run settings and blue calibration

    run_config = load_config(project_root=project_root)
    date = run_config.date
    start = run_config.start
    end = run_config.end
    step_seconds = run_config.step_seconds
    apex_reference_time = run_config.apex_reference_time

    station = 'VEE'
    sites = ('ARV', 'VEE', 'BVR')
    site_coordinates = {
        'ARV': {'latitude': 68.127, 'longitude': -145.533},
        'VEE': {'latitude': 67.013, 'longitude': -146.407},
        'BVR': {'latitude': 66.360, 'longitude': -147.400},
    }
    site_config = {
        site: {
            **site_coordinates[site],
            'calibration_r_s_per_count': {
                'red': RED_RAYLEIGH_SECONDS_PER_COUNT[site],
                'green': GREEN_RAYLEIGH_SECONDS_PER_COUNT[site],
                'blue': BLUE_RAYLEIGH_SECONDS_PER_COUNT[site],
            },
            'exposure_seconds': dict(EXPOSURE_TIME_S_BY_COLOR),
        }
        for site in sites
    }
    channels = ('red', 'green', 'blue')
    frames_to_coadd = 3
    common_mapping_altitude_km = 110.0
    target_grid_spacing_km = run_config.target_grid_spacing_km
    decimation = run_config.decimation
    diagnostic_time = run_config.diagnostic_time

    enable_wavelet_denoising = True
    wavelet_max_shifts = {'red': 3, 'green': 3, 'blue': 4, '8446': 3}
    background_edge_buffer_px = BACKGROUND_EDGE_BUFFER_PX

    # Performance switches; these do not change wavelet_max_shifts.
    use_precomputed_regridding = True
    cache_tiff_frames = True
    parallel_sites = False
    tiff_frame_cache_size = 128
    timestamp_workers = run_config.timestamp_workers  # Separate processes for series times

    # Keep 844.6 nm inactive; this notebook uses 427.8 nm blue.
    save_blue_diagnostics = False
    include_8446_diagnostics = False       # 844.6 nm: starmap and calibration do not yet exist

    calculate_conductances = True
    save_results = True
    overwrite_output = True  # Resume the strict-ownership file; set True to recompute it.
    output_file = run_config.paths.output_root / f'gneiss_rgb_{start.replace(":", "")}_{end.replace(":", "")}_step{step_seconds:g}s.h5'

    image_root = run_config.paths.image_root
    starmap_root = run_config.paths.starmap_root
    glow_root = run_config.paths.glow_root

    def parse_utc(value):
        return dt.datetime.strptime(f'{date} {value}', '%Y%m%d %H:%M:%S')

    start_datetime = parse_utc(start)
    end_datetime = parse_utc(end)
    if end_datetime < start_datetime:
        raise ValueError('end must not be earlier than start')
    if step_seconds <= 0:
        raise ValueError('step_seconds must be positive')

    target_times = []
    current = start_datetime
    while current <= end_datetime:
        target_times.append(current)
        current += dt.timedelta(seconds=step_seconds)

    print(f'{len(target_times)} inversions from {target_times[0]} through {target_times[-1]} UTC')
    print(f'Target global-grid spacing: {target_grid_spacing_km:g} km; decimation={decimation}')

    # ## Site GLOW lookup tables

    def one_match(pattern):
        matches = sorted(glob.glob(str(pattern)))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one match for {pattern}, found {len(matches)}')
        return matches[0]

    def load_site_lookup(site):
        site_dir = glow_root / site
        def main_file(stem):
            normal = sorted(site_dir.glob(f'{stem}*.bin'))
            if normal:
                return str(normal[0])
            # Some existing GLOW runs concatenated the site directory name to
            # the filename instead of writing inside it.
            return one_match(glow_root / f'{site}{stem}*.bin')
        airglow_dir = site_dir / 'airglow'
        lookup = load_lookup_tables(
            main_file('I6300'), main_file('I5577'), main_file('I4278'),
            main_file('ped3d'), main_file('hall3d'), main_file('edens3d'),
            plot=False,
        )
        airglow = load_lookup_tables(
            one_match(airglow_dir / 'I6300*.bin'),
            one_match(airglow_dir / 'I5577*.bin'),
            one_match(airglow_dir / 'I4278*.bin'),
            one_match(airglow_dir / 'ped3d*.bin'),
            one_match(airglow_dir / 'hall3d*.bin'),
            one_match(airglow_dir / 'edens3d*.bin'),
            plot=False,
        )
        lookup['redbright_airglow'] = airglow['redmat']
        lookup['greenbright_airglow'] = airglow['greenmat']
        lookup['bluebright_airglow'] = airglow['bluemat']
        lookup['sigP_bg'] = airglow['sigPmat']
        lookup['sigH_bg'] = airglow['sigHmat']
        lookup['SigP_bg'] = airglow['SigPmat']
        lookup['SigH_bg'] = airglow['SigHmat']
        return lookup

    lookup_by_site = {site: load_site_lookup(site) for site in sites}
    for site, lookup in lookup_by_site.items():
        print(site, 'GLOW params:', lookup['Params'][:8])

    # ## Camera geometry

    vee_site_lat = 67.013
    vee_site_lon = -146.407
    emission_altitude_km = {630.0: 180.0, 844.6: 180.0, 557.7: 110.0, 427.8: 107.0}
    starmap_files = {
        630.0: (
            starmap_root / 'red/6300/VEE/VEE_6300_20260210_050100_az.sav',
            starmap_root / 'red/6300/VEE/VEE_6300_20260210_050100_el.sav',
        ),
        557.7: (
            starmap_root / 'green/VEE/GNEISS/VEE_5577_20260210_050145_az.sav',
            starmap_root / 'green/VEE/GNEISS/VEE_5577_20260210_050145_el.sav',
        ),
        427.8: (
            starmap_root / 'blue/VEE/VEE_4278_20260210_050100_az.sav',
            starmap_root / 'blue/VEE/VEE_4278_20260210_050100_el.sav',
        ),
        844.6: None,  # Add the 844.6 nm az/el paths here when they become available.
    }

    def read_idl_starmap(wavelength_nm, minimum_elevation_deg=15.0):
        paths = starmap_files[wavelength_nm]
        if paths is None:
            raise FileNotFoundError(f'No {wavelength_nm} nm az/el starmaps are configured')
        az_path, el_path = paths
        az_data = readsav(az_path, python_dict=True)
        el_data = readsav(el_path, python_dict=True)
        az = np.asarray(az_data[next(iter(az_data))], dtype=float)
        el = np.asarray(el_data[next(iter(el_data))], dtype=float)
        if az.shape != el.shape:
            raise ValueError(f'Az/el shape mismatch for {wavelength_nm} nm')
        invalid = ~np.isfinite(az) | ~np.isfinite(el) | (el < minimum_elevation_deg)
        az[invalid] = np.nan
        el[invalid] = np.nan
        return az, el

    def azel_to_geographic(az, el, altitude_km):
        site_x, site_y, site_z = pm.geodetic2ecef(vee_site_lat, vee_site_lon, 0.0)
        east, north, up = pm.aer2enu(az, el, 1.0)
        vx, vy, vz = pm.enu2uvw(east, north, up, vee_site_lat, vee_site_lon)
        earth = pm.Ellipsoid.from_name('wgs84')
        a2 = (earth.semimajor_axis + altitude_km * 1000.0) ** 2
        c2 = (earth.semiminor_axis + altitude_km * 1000.0) ** 2
        qa = vx**2 / a2 + vy**2 / a2 + vz**2 / c2
        qb = site_x * vx / a2 + site_y * vy / a2 + site_z * vz / c2
        qc = site_x**2 / a2 + site_y**2 / a2 + site_z**2 / c2 - 1.0
        discriminant = qb**2 - qa * qc
        distance = np.where(discriminant >= 0, (np.sqrt(discriminant) - qb) / qa, np.nan)
        lat, lon, _ = pm.ecef2geodetic(site_x + distance * vx, site_y + distance * vy, site_z + distance * vz)
        invalid = ~np.isfinite(az) | ~np.isfinite(el)
        lat[invalid] = np.nan
        lon[invalid] = np.nan
        return lat, lon

    apex = Apex(date=parse_utc(apex_reference_time))

    def mapped_starmap(wavelength_nm):
        az, el = read_idl_starmap(wavelength_nm)
        source_altitude = emission_altitude_km[wavelength_nm]
        lat, lon = azel_to_geographic(az, el, source_altitude)
        valid = np.isfinite(lat) & np.isfinite(lon)
        mapped_lat = np.full(lat.shape, np.nan)
        mapped_lon = np.full(lon.shape, np.nan)
        mapped_lat[valid], mapped_lon[valid], _ = apex.map_to_height(
            lat[valid], lon[valid], source_altitude, common_mapping_altitude_km
        )
        return mapped_lat, mapped_lon

    red_lat, red_lon = mapped_starmap(630.0)
    green_lat, green_lon = mapped_starmap(557.7)
    blue_lat, blue_lon = mapped_starmap(427.8)
    print('Red starmap:', red_lat.shape, 'Green starmap:', green_lat.shape, 'Blue starmap:', blue_lat.shape)

    # Load the ARV/VEE/BVR red and green camera geometry.
    # Paths are explicit because the legacy asi_mapping loader does not include
    # the red/6300 directory component for ARV and BVR.
    site_starmap_files = {
        'ARV': {
            'red': (
                starmap_root / 'red/6300/ARV/ARV_GASI_630_20260209_Az.FIT',
                starmap_root / 'red/6300/ARV/ARV_GASI_630_20260209_El.FIT',
            ),
            'green': (
                starmap_root / 'green/ARV/ARV_GASI_20260209_063700_rot5_full_Az.sav',
                starmap_root / 'green/ARV/ARV_GASI_20260209_063700_rot5_full_El.sav',
            ),
        },
        'VEE': {
            'red': starmap_files[630.0],
            'green': starmap_files[557.7],
        },
        'BVR': {
            'red': (
                starmap_root / 'red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_Az.sav',
                starmap_root / 'red/6300/BVR/BVR_GASI_630_20260210_051800_asistarcalibration_full_El.sav',
            ),
            'green': (
                starmap_root / 'green/BVR/BVR_20260210_090000_750_rot5_Az.sav',
                starmap_root / 'green/BVR/BVR_20260210_090000_750_rot5_El.sav',
            ),
        },
    }

    site_starmap_files['ARV']['blue'] = (
        starmap_root / 'blue/ARV/ZOR_ARV_20260210_4278_Az.FIT',
        starmap_root / 'blue/ARV/ZOR_ARV_20260210_4278_El.FIT',
    )
    site_starmap_files['BVR']['blue'] = (
        starmap_root / 'blue/BVR/ARI_20260209_4278_Az.FIT',
        starmap_root / 'blue/BVR/ARI_20260209_4278_El.FIT',
    )
    site_starmap_files['VEE']['blue'] = starmap_files[427.8]
    missing_blue_maps = [str(path) for site in sites
                         for path in site_starmap_files[site]['blue'] if not path.exists()]
    if missing_blue_maps:
        raise FileNotFoundError('Add the blue starmaps or update site_starmap_files:\n' + '\n'.join(missing_blue_maps))

    def read_starmap_array(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in ('.fit', '.fits'):
            return np.asarray(fits.getdata(path), dtype=float)
        data = readsav(path, python_dict=True)
        return np.asarray(data[next(iter(data))], dtype=float)

    def load_site_skymap(site, channel):
        az_path, el_path = site_starmap_files[site][channel]
        az = read_starmap_array(az_path)
        el = read_starmap_array(el_path)
        if az.shape != el.shape:
            raise ValueError(f'{site} {channel} az/el shape mismatch: {az.shape} vs {el.shape}')
        minimum_elevation_deg = 15.0 if channel == 'blue' else 22.0
        mask = ~np.isfinite(az) | ~np.isfinite(el) | (el < minimum_elevation_deg)
        source_altitude = emission_altitude_km[{'red': 630.0, 'green': 557.7, 'blue': 427.8}[channel]]
        latitude = site_config[site]['latitude']
        longitude = site_config[site]['longitude']
        mapped_lat, mapped_lon = azel2geo(latitude, longitude, az, el, alt=source_altitude)
        return {
            'site_lat': latitude, 'site_lon': longitude,
            'azmt': az, 'elev': el, 'mask': mask,
            'lat': mapped_lat, 'lon': mapped_lon,
            'map_alt_km': source_altitude,
        }

    red_skymaps_by_site = {site: load_site_skymap(site, 'red') for site in sites}
    green_skymaps_by_site = {site: load_site_skymap(site, 'green') for site in sites}
    blue_skymaps_by_site = {site: load_site_skymap(site, 'blue') for site in sites}

    def map_site_coordinates_to_common(skymap, source_altitude_km):
        source_lat = np.asarray(skymap['lat'], dtype=float)
        source_lon = np.asarray(skymap['lon'], dtype=float)
        base_mask = np.asarray(skymap['mask'], dtype=bool)
        valid = np.isfinite(source_lat) & np.isfinite(source_lon) & ~base_mask
        mapped_lat = np.full(source_lat.shape, np.nan)
        mapped_lon = np.full(source_lon.shape, np.nan)
        mapped_lat[valid], mapped_lon[valid], _ = apex.map_to_height(
            source_lat[valid], source_lon[valid],
            source_altitude_km, common_mapping_altitude_km,
        )
        return mapped_lat, mapped_lon

    mapped_coordinates_by_site = {}
    native_masks_by_site = {}
    for site in sites:
        mapped_coordinates_by_site[site] = {
            'red': map_site_coordinates_to_common(
                red_skymaps_by_site[site], emission_altitude_km[630.0]
            ),
            'blue': map_site_coordinates_to_common(
                blue_skymaps_by_site[site], emission_altitude_km[427.8]
            ),
            'green': map_site_coordinates_to_common(
                green_skymaps_by_site[site], emission_altitude_km[557.7]
            ),
        }
        native_masks_by_site[site] = {}
        for channel, skymaps in (
            ('red', red_skymaps_by_site), ('green', green_skymaps_by_site),
            ('blue', blue_skymaps_by_site)
        ):
            base_mask = np.asarray(skymaps[site]['mask'], dtype=bool)
            native_masks_by_site[site][channel] = {
                'base': base_mask,
            }

    for site in sites:
        print(
            site,
            'red/green shapes:',
            mapped_coordinates_by_site[site]['red'][0].shape,
            mapped_coordinates_by_site[site]['green'][0].shape,
            'valid native FOV pixels:',
            f"red={np.count_nonzero(~native_masks_by_site[site]['red']['base']):,}",
            f"green={np.count_nonzero(~native_masks_by_site[site]['green']['base']):,}",
        )

    # ## Common grid and interpolation

    def coordinate_bounds(lat, lon):
        valid = np.isfinite(lat) & np.isfinite(lon)
        return np.nanmin(lat[valid]), np.nanmax(lat[valid]), np.nanmin(lon[valid]), np.nanmax(lon[valid])

    # Union of every site's geometric red and green field of view. Fixed camera
    # ownership is computed on this grid before processing any timestep.
    retained_latitudes = []
    retained_longitudes = []
    for site in sites:
        for channel in channels:
            channel_lat, channel_lon = mapped_coordinates_by_site[site][channel]
            retained = (
                np.isfinite(channel_lat) & np.isfinite(channel_lon)
                & ~native_masks_by_site[site][channel]['base']
            )
            retained_latitudes.append(channel_lat[retained])
            retained_longitudes.append(channel_lon[retained])
    all_latitudes = np.concatenate(retained_latitudes)
    all_longitudes = np.concatenate(retained_longitudes)
    lat_min, lat_max = np.min(all_latitudes), np.max(all_latitudes)
    lon_min, lon_max = np.min(all_longitudes), np.max(all_longitudes)
    mean_latitude = 0.5 * (lat_min + lat_max)
    latitude_step = target_grid_spacing_km / 111.0
    longitude_step = target_grid_spacing_km / (111.0 * np.cos(np.deg2rad(mean_latitude)))
    grid_lat_axis = np.arange(lat_min, lat_max + 0.5 * latitude_step, latitude_step)
    grid_lon_axis = np.arange(lon_min, lon_max + 0.5 * longitude_step, longitude_step)
    gridlon, gridlat = np.meshgrid(grid_lon_axis, grid_lat_axis, indexing='xy')
    decgridlat = gridlat[::decimation, ::decimation]
    decgridlon = gridlon[::decimation, ::decimation]

    def interpolation_geometry(lat, lon):
        valid = np.isfinite(lat) & np.isfinite(lon)
        points = np.column_stack((lon[valid], lat[valid]))
        targets = np.column_stack((gridlon.ravel(), gridlat.ravel()))
        return valid, points, targets

    red_coord_valid, red_points, grid_targets = interpolation_geometry(red_lat, red_lon)
    green_coord_valid, green_points, _ = interpolation_geometry(green_lat, green_lon)
    blue_coord_valid, blue_points, _ = interpolation_geometry(blue_lat, blue_lon)
    channel_starmaps_by_site = {}
    for site in sites:
        channel_starmaps_by_site[site] = {}
        for channel in channels:
            channel_lat, channel_lon = mapped_coordinates_by_site[site][channel]
            coordinate_valid, points, _ = interpolation_geometry(channel_lat, channel_lon)
            channel_starmaps_by_site[site][channel] = (
                channel_lat, channel_lon, coordinate_valid, points
            )

    def build_linear_regridder(points, targets):
        triangulation = Delaunay(points)
        simplex = triangulation.find_simplex(targets)
        inside = simplex >= 0
        vertices = np.zeros((targets.shape[0], 3), dtype=np.int32)
        weights = np.zeros((targets.shape[0], 3), dtype=np.float64)
        inside_simplex = simplex[inside]
        vertices[inside] = triangulation.simplices[inside_simplex]
        transform = triangulation.transform[inside_simplex]
        barycentric = np.einsum(
            'nij,nj->ni', transform[:, :2, :], targets[inside] - transform[:, 2, :]
        )
        weights[inside, :2] = barycentric
        weights[inside, 2] = 1.0 - barycentric.sum(axis=1)
        return {'inside': inside, 'vertices': vertices, 'weights': weights}

    precomputed_regridders = {
        site: {
            channel: build_linear_regridder(
                channel_starmaps_by_site[site][channel][3], grid_targets
            )
            for channel in channels
        }
        for site in sites
    }
    print('Global union grid:', gridlat.shape, 'Inversion grid:', decgridlat.shape)
    print(f'Grid spacing ≈ {target_grid_spacing_km:g} km')
    print('Fixed Apex reference time:', parse_utc(apex_reference_time), 'UTC')

    # ## Red/green frame loading and preprocessing

    from asi_shared.detector import correct_detector_frame
    def discover_site_tiffs(site, channel):
        if channel == 'red':
            directories = [image_root / 'red' / '6300' / site]
            if site == 'VEE':
                directories.insert(0, image_root / 'red' / '6300' / 'VEE' / 'GNEISS')
        elif channel == 'green':
            directories = [image_root / 'green' / site]
            if site == 'VEE':
                directories.insert(0, image_root / 'green' / 'VEE' / 'GNEISS')
        else:
            raise ValueError(f'Unsupported channel: {channel}')
        paths = {
            str(path)
            for directory in directories
            for pattern in ('*.tif', '*.tiff')
            for path in directory.glob(pattern)
            if date in path.name
        }
        return sorted(paths)

    channel_files = {
        site: {
            channel: discover_site_tiffs(site, channel)
            for channel in ('red', 'green')
        }
        for site in sites
    }
    channel_cadence_seconds = {'red': 0.9, 'green': 0.3}
    for site in sites:
        for channel in ('red', 'green'):
            if not channel_files[site][channel]:
                raise FileNotFoundError(f'No {site} {channel} TIFF stacks found for {date}')
            print(f'{site} {channel}: {len(channel_files[site][channel])} TIFF stacks')

    def timestamp_from_filename(path):
        match = re.search(r'_(\d{8})_(\d{6})\.tiff?$', Path(path).name, re.IGNORECASE)
        if not match:
            raise ValueError(f'Cannot parse TIFF timestamp from {path}')
        return dt.datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S')

    from asispectralinversion.tiff_catalog import TiffPageCountCache

    tiff_catalog_cache = TiffPageCountCache(
        project_root / '.cache' / 'tiff_page_counts.json'
    )
    print(f'TIFF page-count catalog: {tiff_catalog_cache.path}', flush=True)

    def stack_info(path, cadence_seconds):
        start_time = timestamp_from_filename(path)
        page_count = tiff_catalog_cache.page_count(path)
        end_time = start_time + dt.timedelta(seconds=(page_count - 1) * cadence_seconds)
        return start_time, end_time, page_count

    stack_catalog = {
        site: {
            channel: [
                (path, *stack_info(path, channel_cadence_seconds[channel]))
                for path in channel_files[site][channel]
            ]
            for channel in ('red', 'green')
        }
        for site in sites
    }

    print(f'TIFF catalog ready: {tiff_catalog_cache.hits} reused, '
          f'{tiff_catalog_cache.misses} indexed.', flush=True)

    @functools.lru_cache(maxsize=None)
    def open_tiff_stack(path):
        return tifffile.TiffFile(path)

    @functools.lru_cache(maxsize=tiff_frame_cache_size)
    def read_tiff_frame_cached(path, index):
        return correct_detector_frame(open_tiff_stack(path).pages[index].asarray().astype(float), path)

    def read_tiff_frame(path, index):
        if cache_tiff_frames:
            return read_tiff_frame_cached(path, index)
        with tifffile.TiffFile(path) as tif:
            return correct_detector_frame(tif.pages[index].asarray().astype(float), path)

    def coadd_closest_frames(site, channel, target_time):
        cadence = channel_cadence_seconds[channel]
        candidates = []
        for path, stack_start, _stack_end, page_count in stack_catalog[site][channel]:
            # Only indices near the target can be among the globally closest frames.
            center = round((target_time - stack_start).total_seconds() / cadence)
            nearby = range(
                max(0, center - frames_to_coadd),
                min(page_count, center + frames_to_coadd + 1),
            )
            for index in nearby:
                frame_time = stack_start + dt.timedelta(seconds=index * cadence)
                candidates.append((abs(frame_time - target_time), frame_time, path, index))
        if len(candidates) < frames_to_coadd:
            raise ValueError(f'Only {len(candidates)} {site} {channel} frames are available near {target_time.isoformat()} UTC')
        selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:frames_to_coadd]
        selected.sort(key=lambda item: item[1])

        images = [read_tiff_frame(path, index) for _offset, _time, path, index in selected]
        filenames = [Path(path).name for _offset, _time, path, _index in selected]
        indices = [index for _offset, _time, _path, index in selected]
        frame_times = [frame_time for _offset, frame_time, _path, _index in selected]
        return np.mean(images, axis=0), filenames, indices, frame_times

    def estimate_background(image, elevation):
        mask = physical_offsky_mask(elevation, edge_buffer_px=background_edge_buffer_px)
        background, sigma, _values = estimate_scalar_background(image, mask)
        return background, sigma


    def denoise_image(image, invalid_mask, background, sigma, max_shifts):
        if not enable_wavelet_denoising:
            output = image.astype(float).copy()
            output[invalid_mask] = np.nan
            return output
        filled = np.where(invalid_mask | ~np.isfinite(image), background, image)
        output = cycle_spin(
            filled, func=denoise_wavelet, max_shifts=max_shifts,
            func_kw={'sigma': sigma, 'method': 'BayesShrink', 'mode': 'soft', 'rescale_sigma': True},
            workers=1, channel_axis=None,
        )
        output[invalid_mask] = np.nan
        return output

    def regrid_channel(image, lat, lon, coordinate_valid, precomputed=None):
        if use_precomputed_regridding and precomputed is not None:
            source_values = image[coordinate_valid]
            inside = precomputed['inside']
            vertices = precomputed['vertices'][inside]
            vertex_values = source_values[vertices]
            output = np.full(grid_targets.shape[0], np.nan)
            usable = np.all(np.isfinite(vertex_values), axis=1)
            inside_indices = np.flatnonzero(inside)
            output[inside_indices[usable]] = np.einsum(
                'ni,ni->n', vertex_values[usable], precomputed['weights'][inside][usable]
            )
            return output.reshape(gridlat.shape)
        valid = coordinate_valid & np.isfinite(image)
        # Rebuild points only if the image contains additional invalid pixels.
        use_points = np.column_stack((lon[valid], lat[valid]))
        values = image[valid]
        return griddata(use_points, values, grid_targets, method='linear').reshape(gridlat.shape)

    def finite_percentile(image, percentile=99.5):
        values = np.asarray(image)[np.isfinite(image)]
        if values.size == 0:
            raise ValueError('Cannot calculate a percentile from an all-NaN image')
        return np.percentile(values, percentile)

    print('Calibration factors (R s/count):')
    for site in sites:
        print(f"  {site}: {site_config[site]['calibration_r_s_per_count']}")
    print('Exposure times (s):', dict(EXPOSURE_TIME_S_BY_COLOR))
    print('Blue exposure uses asi_shared.calibration.EXPOSURE_TIME_S_BY_COLOR; logs supply frame timing only.')

    # ## Blue acquisition logs

    # Logs supply per-stack UTC start, cadence, and wavelength; exposure comes from shared calibration.
    # Do not infer blue timestamps from the camera filename.
    blue_catalogs = {site: blue_stack_catalog(
        image_root / 'blue' / site, date,
        # User verified ARV THA is 4278 A; its 7774-A log label is stale.
        verified_stale_filter=7774 if site == 'ARV' else None,
    ) for site in sites}

    # ## RGB processing and fixed ownership

    site_codes = {'ARV': 1, 'VEE': 2, 'BVR': 3}

    def process_channel(site, channel, target_time):
        channel_lat, channel_lon, coordinate_valid, _points = channel_starmaps_by_site[site][channel]
        if channel == 'blue':
            image, filenames, indices, frame_times, exposure = coadd_blue_frames(
                blue_catalogs[site], target_time, frames_to_coadd, read_tiff_frame
            )
        else:
            image, filenames, indices, frame_times = coadd_closest_frames(site, channel, target_time)
            exposure = site_config[site]['exposure_seconds'][channel]
        if image.shape != channel_lat.shape:
            raise ValueError(f'{site} {channel} image/starmap shape mismatch: {image.shape} vs {channel_lat.shape}')
        native_skymap = (
            {'red': red_skymaps_by_site, 'green': green_skymaps_by_site,
             'blue': blue_skymaps_by_site}[channel][site]
        )
        background, sigma = estimate_background(image, native_skymap['elev'])
        prepared = denoise_image(
            image, ~coordinate_valid, background, sigma, wavelet_max_shifts[channel]
        )
        regridded = regrid_channel(
            prepared, channel_lat, channel_lon, coordinate_valid,
            precomputed_regridders[site][channel],
        )
        calibration = site_config[site]['calibration_r_s_per_count'][channel]
        rayleighs = (regridded - background) / exposure * calibration
        metadata = {
            'files': filenames, 'indices': indices, 'frame_times': frame_times,
            'background_counts': background, 'sigma_counts': sigma,
            'exposure_seconds': exposure,
        }
        return rayleighs, metadata

    def invert_site(site, target_time, collect_diagnostics=False):
        red, red_metadata = process_channel(site, 'red', target_time)
        green, green_metadata = process_channel(site, 'green', target_time)
        blue, blue_metadata = process_channel(site, 'blue', target_time)
        blue = blue[::decimation, ::decimation]
        red = red[::decimation, ::decimation]
        green = green[::decimation, ::decimation]
        red_finite = np.isfinite(red)
        green_finite = np.isfinite(green)
        positive_rg = red_finite & green_finite & (red > 0) & (green > 0)
        positive_rgb = positive_rg & np.isfinite(blue) & (blue > 0)
        lookup = lookup_by_site[site]
        owned = owner_site == site_codes[site]
        q, e0, sigp, sigh, failure_reason = invert_owned_rgb(
            red, green, blue, lookup, owned,
            calculate_conductances=calculate_conductances, return_reasons=True,
        )
        diagnostics = None
        if collect_diagnostics:
            observed_ratio = np.full(red.shape, np.nan)
            observed_ratio[positive_rg] = red[positive_rg] / green[positive_rg]
            lookup_red = lookup['redmat'] - lookup['redbright_airglow'][0][0]
            lookup_green = lookup['greenmat'] - lookup['greenbright_airglow'][0][0]
            lookup_ratio = np.divide(
                lookup_red, lookup_green,
                out=np.full_like(lookup_red, np.nan, dtype=float),
                where=np.isfinite(lookup_green) & (lookup_green != 0),
            )
            ratio_min, ratio_max = np.nanmin(lookup_ratio), np.nanmax(lookup_ratio)
            ratio_in_lookup = positive_rg & (observed_ratio >= ratio_min) & (observed_ratio <= ratio_max)
            diagnostics = {
                'red_finite': red_finite, 'green_finite': green_finite,
                'positive_rg': positive_rg, 'ratio_in_lookup': ratio_in_lookup,
                'q_finite': np.isfinite(q), 'e0_finite': np.isfinite(e0),
                'sigp_finite': np.isfinite(sigp) if sigp is not None else None,
                'sigh_finite': np.isfinite(sigh) if sigh is not None else None,
                'observed_ratio': observed_ratio,
                'observed_red': red, 'observed_green': green, 'observed_blue': blue,
                'lookup_ratio_bounds': (ratio_min, ratio_max),
            }
        return {
            'q': q, 'e0': e0, 'sigp': sigp, 'sigh': sigh,
            'inversion_pixel_selection': 'fixed_owner_only',
            'failure_reason': failure_reason,
            'valid': np.isfinite(q) | np.isfinite(e0),
            'red_metadata': red_metadata, 'green_metadata': green_metadata,
            'blue_metadata': blue_metadata,
            'diagnostics': diagnostics,
        }

    site_distance_km = {}
    for site in sites:
        north = (decgridlat - site_config[site]['latitude']) * 111.0
        east = ((decgridlon - site_config[site]['longitude']) * 111.0
                * np.cos(np.deg2rad(decgridlat)))
        site_distance_km[site] = np.hypot(north, east)

    # Ownership depends only on fixed camera geometry, never on a frame's brightness
    # or inversion success. The same owner keeps every pixel throughout the series.
    site_geometric_coverage = {
        site: (
            precomputed_regridders[site]['red']['inside']
            & precomputed_regridders[site]['green']['inside']
            & precomputed_regridders[site]['blue']['inside']
        ).reshape(gridlat.shape)[::decimation, ::decimation]
        for site in sites
    }
    owner_site = fixed_camera_ownership(site_geometric_coverage, site_distance_km, site_codes)


    def combine_site_products(site_products):
        return combine_owned_products(
            site_products, owner_site, site_codes,
            calculate_conductances=calculate_conductances,
        )

    site_executor = ThreadPoolExecutor(max_workers=len(sites), thread_name_prefix='gneiss-site')

    def process_time(target_time, collect_diagnostics=False):
        if parallel_sites:
            futures = {
                site: site_executor.submit(invert_site, site, target_time, collect_diagnostics)
                for site in sites
            }
            site_products = {site: futures[site].result() for site in sites}
        else:
            site_products = {
                site: invert_site(site, target_time, collect_diagnostics)
                for site in sites
            }
        combined = combine_site_products(site_products)
        return {'time': target_time, 'sites': site_products, 'combined': combined}

    # ## Save or resume RGB products

    def create_float_cube(group, name, shape):
        chunks = (1, min(shape[1], 128), min(shape[2], 128))
        return group.create_dataset(
            name, shape=shape, dtype='f4', chunks=chunks,
            compression='gzip', compression_opts=4, shuffle=True,
            fillvalue=np.nan,
        )

    def initialize_output(h5):
        nt = len(target_times)
        cube_shape = (nt, *decgridlat.shape)
        h5.attrs['format'] = 'gneiss_multisite_rgb_inversion'
        h5.attrs['inversion_pixel_selection'] = 'fixed_owner_only'
        h5.attrs['blue_dark_frame_subtracted'] = False
        h5.attrs['schema_version'] = '1.3'
        h5.attrs['blue_timing_json'] = json.dumps(blue_catalogs, default=str)
        h5.attrs['storage_mode'] = 'combined_only'
        h5.attrs['ownership_method'] = OWNERSHIP_METHOD
        h5.attrs['date'] = date
        h5.attrs['start'] = start
        h5.attrs['end'] = end
        h5.attrs['step_seconds'] = step_seconds
        h5.attrs['apex_reference_time'] = f'{date}T{apex_reference_time}'
        h5.attrs['common_mapping_altitude_km'] = common_mapping_altitude_km
        h5.attrs['target_grid_spacing_km'] = target_grid_spacing_km
        h5.attrs['site_codes_json'] = json.dumps(site_codes)
        h5.attrs['calibration_json'] = json.dumps({
            site: site_config[site]['calibration_r_s_per_count'] for site in sites
        })
        h5.attrs['glow_params_json'] = json.dumps({
            site: lookup_by_site[site]['Params'][:8].tolist() for site in sites
        })
        h5.create_dataset('time_iso', data=np.asarray(
            [value.isoformat() for value in target_times], dtype='S26'
        ))
        h5.create_dataset('completed', shape=(nt,), dtype='?')
        h5.create_dataset('gridlat', data=decgridlat.astype('f4'))
        h5.create_dataset('gridlon', data=decgridlon.astype('f4'))
        h5.create_dataset('owner_site', data=owner_site)
        h5['owner_site'].attrs['description'] = (
            'Fixed geometric camera owner; 0 means outside all red/green/blue footprints. '
            'Ownership does not change when that camera has invalid data.'
        )
        combined = h5.create_group('combined')
        reasons = combined.create_dataset('failure_reason', shape=cube_shape, dtype='u1',
            chunks=(1, min(cube_shape[1], 128), min(cube_shape[2], 128)),
            compression='gzip', compression_opts=4, shuffle=True, fillvalue=255)
        reasons.attrs['codes_json'] = json.dumps(FAILURE_REASONS)
        reasons.attrs['version'] = 1
        create_float_cube(combined, 'q_mW_m2', cube_shape)
        create_float_cube(combined, 'e0_eV', cube_shape)
        combined.create_dataset(
            'source_site', shape=cube_shape, dtype='u1',
            chunks=(1, min(cube_shape[1], 128), min(cube_shape[2], 128)),
            compression='gzip', compression_opts=4, shuffle=True,
        )
        combined['source_site'].attrs['description'] = (
            'Camera supplying a valid combined Q/E0 value; 0 means no valid owner value.'
        )
        if calculate_conductances:
            create_float_cube(combined, 'sigp_S', cube_shape)
            create_float_cube(combined, 'sigh_S', cube_shape)
        h5['gridlat'].attrs['coordinate_system'] = 'geodetic'
        h5['gridlat'].attrs['units'] = 'degrees_north'
        h5['gridlon'].attrs['coordinate_system'] = 'geodetic'
        h5['gridlon'].attrs['units'] = 'degrees_east'
        combined['q_mW_m2'].attrs['units'] = 'mW m-2'
        combined['e0_eV'].attrs['units'] = 'eV'
        if calculate_conductances:
            combined['sigp_S'].attrs['units'] = 'S'
            combined['sigh_S'].attrs['units'] = 'S'

    def validate_output_ownership(h5):
        if h5.attrs.get('inversion_pixel_selection') != 'fixed_owner_only':
            raise ValueError('Existing output predates owner-only inversion; use a new file or overwrite_output=True')
        if h5.attrs.get('blue_dark_frame_subtracted', True):
            raise ValueError('Existing output used blue dark-frame subtraction; use a new output file or overwrite_output=True')
        if h5.attrs.get('format') != 'gneiss_multisite_rgb_inversion':
            raise ValueError('Existing output is not an RGB inversion')
        expected_calibration = json.dumps({site: site_config[site]['calibration_r_s_per_count'] for site in sites})
        if h5.attrs.get('calibration_json') != expected_calibration:
            raise ValueError('Calibration changed; use a new output file or overwrite_output=True')
        if h5.attrs.get('blue_timing_json') != json.dumps(blue_catalogs, default=str):
            raise ValueError('Blue inputs or timing changed; use a new output file or overwrite_output=True')
        if h5.attrs.get('ownership_method') != OWNERSHIP_METHOD or 'owner_site' not in h5:
            raise ValueError(
                'Existing output uses a different ownership rule. Use a new output file '
                'or set overwrite_output=True to recompute with fixed camera ownership.'
            )
        if not np.array_equal(h5['owner_site'][:], owner_site):
            raise ValueError('Existing output has a different fixed camera ownership map')


    def write_time_result(h5, index, result):
        combined = result['combined']
        h5['combined/q_mW_m2'][index] = combined['q'].astype('f4')
        h5['combined/e0_eV'][index] = combined['e0'].astype('f4')
        h5['combined/source_site'][index] = combined['source_site']
        if calculate_conductances:
            h5['combined/sigp_S'][index] = combined['sigp'].astype('f4')
            h5['combined/sigh_S'][index] = combined['sigh'].astype('f4')
        h5['combined/failure_reason'][index] = combined['failure_reason']
        h5['completed'][index] = True
        h5.flush()

    if save_results:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if overwrite_output and output_file.exists():
            output_file.unlink()
        with h5py.File(output_file, 'a') as h5:
            if 'completed' not in h5:
                initialize_output(h5)
            elif h5['completed'].shape != (len(target_times),):
                raise ValueError('Existing output has a different time dimension')
            elif h5['gridlat'].shape != decgridlat.shape:
                raise ValueError('Existing output has a different spatial grid')
            validate_output_ownership(h5)
            if 'combined/failure_reason' not in h5:
                raise ValueError('Output predates saved failure codes; use a new file or overwrite_output=True.')
            if calculate_conductances and ('combined/sigp_S' not in h5 or 'combined/sigh_S' not in h5):
                cube_shape = (len(target_times), *decgridlat.shape)
                combined = h5['combined']
                if 'sigp_S' not in combined:
                    create_float_cube(combined, 'sigp_S', cube_shape).attrs['units'] = 'S'
                if 'sigh_S' not in combined:
                    create_float_cube(combined, 'sigh_S', cube_shape).attrs['units'] = 'S'
                h5.attrs['schema_version'] = '1.3'
                h5['completed'][:] = False
                h5.flush()
                print('Added SigP/SigH datasets; recomputing existing time steps.')
            pending_jobs = []
            cached_diagnostic = globals().get('diagnostic_result')
            for index, target_time in enumerate(target_times):
                if bool(h5['completed'][index]):
                    continue
                use_cached = (
                    cached_diagnostic is not None
                    and all('failure_reason' in p for p in cached_diagnostic['sites'].values())
                    and target_time == cached_diagnostic['time']
                    and all(product.get('inversion_pixel_selection') == 'fixed_owner_only'
                            for product in cached_diagnostic['sites'].values())
                    and (not calculate_conductances
                         or cached_diagnostic['combined'].get('sigp') is not None)
                )
                if use_cached:
                    write_time_result(h5, index, {
                        'time': target_time,
                        'combined': combine_site_products(cached_diagnostic['sites']),
                    })
                else:
                    pending_jobs.append((index, target_time))
            if pending_jobs:
                worker_state, worker_sources = snapshot_rgb_worker(globals())
                workers = min(timestamp_workers, len(pending_jobs))
                print(f'Processing {len(pending_jobs)} timestamps with {workers} worker process(es); sites run sequentially.', flush=True)
                with closing(iter_rgb_series(worker_state, worker_sources, pending_jobs, workers)) as results:
                    for index, result in results:
                        write_time_result(h5, index, result)
                        print(f'Saved [{index + 1:03d}/{len(target_times):03d}] {result["time"].time()} UTC', flush=True)
        print('Saved/resumed', output_file)
