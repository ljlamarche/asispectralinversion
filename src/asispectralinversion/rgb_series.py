"""Spawn-safe timestamp workers for the RGB notebook; HDF5 stays in the parent."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import functools
import inspect
import multiprocessing
import textwrap

import numpy as np
import tifffile
from pathlib import Path
from scipy.interpolate import griddata
from skimage.restoration import cycle_spin, denoise_wavelet

from asi_shared.background import physical_offsky_mask, estimate_scalar_background
from asi_shared.detector import correct_detector_frame
from .blue import coadd_blue_frames
from .ownership import invert_owned_rgb, combine_owned_products


FUNCTIONS = (
    'open_tiff_stack', 'read_tiff_frame_cached', 'read_tiff_frame',
    'coadd_closest_frames', 'estimate_background', 'denoise_image',
    'regrid_channel', 'process_channel', 'invert_site', 'combine_site_products',
)
STATE_KEYS = (
    'sites', 'site_codes', 'owner_site', 'site_config', 'lookup_by_site',
    'channel_starmaps_by_site', 'precomputed_regridders', 'grid_targets', 'gridlat',
    'decimation', 'calculate_conductances', 'frames_to_coadd', 'stack_catalog',
    'channel_cadence_seconds', 'blue_catalogs', 'tiff_frame_cache_size',
    'cache_tiff_frames', 'background_edge_buffer_px', 'enable_wavelet_denoising',
    'wavelet_max_shifts', 'use_precomputed_regridding',
)


def snapshot_rgb_worker(namespace):
    """Capture current settings and function definitions, never live TIFF/HDF5 handles.

    Jupyter stores executed cell sources in linecache, which inspect uses here.
    Capturing executed definitions also respects edits made in the running notebook.
    """
    state = {key: namespace[key] for key in STATE_KEYS}
    for color in ('red', 'green', 'blue'):
        key = f'{color}_skymaps_by_site'
        state[key] = {site: {'elev': value['elev']}
                      for site, value in namespace[key].items()}
    try:
        sources = [textwrap.dedent(inspect.getsource(namespace[name])) for name in FUNCTIONS]
    except (OSError, TypeError) as exc:
        raise RuntimeError('Cannot capture RGB worker functions; rerun the notebook setup cells') from exc
    return state, sources


def _namespace(state, sources):
    context = dict(globals())
    context.update(state)
    for source in sources:
        exec(compile(source, '<rgb-worker>', 'exec'), context)
    return context


def _initialize_worker(state, sources):
    global _worker_context
    _worker_context = _namespace(state, sources)


def _process(context, job):
    index, target = job
    # No nested site executor; each process owns one timestamp at a time.
    products = {site: context['invert_site'](site, target) for site in context['sites']}
    return index, {'time': target, 'combined': context['combine_site_products'](products)}


def _process_job(job):
    return _process(_worker_context, job)


def iter_rgb_series(state, sources, jobs, workers=2):
    """Yield (original index, merged result), with bounded, unordered completion.

    Only ``workers`` jobs are in flight. Closing this generator or interrupting
    it terminates workers on Python 3.14+, rather than leaving inversions running.
    The caller is the sole writer and records completion after each successful write.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('timestamp_workers must be a positive integer')
    jobs = iter(jobs)
    if workers == 1:
        context = _namespace(state, sources)
        for job in jobs:
            yield _process(context, job)
        return
    pool = ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize_worker, initargs=(state, sources))
    pending = set()
    try:
        for _ in range(workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.add(pool.submit(_process_job, job))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                job = next(jobs, None)
                if job is not None:
                    pending.add(pool.submit(_process_job, job))
    except BaseException:
        if hasattr(pool, 'terminate_workers'):
            pool.terminate_workers()
        else:
            pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
