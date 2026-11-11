"""ARV/ZORBAS 4278 detector correction, applied once at raw-image ingestion.

Source: fix_ARV_4278 in fix_ZOR_4278_2026.pro (2026 campaign).
Coordinates refer to unrotated, uncropped detector pixels, in DN.
"""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

ARV_BLUE_CORRECTION = "arv_4278_column_offsets_v1"
ARV_BLUE_OFFSETS = ((281, 153, 77), (282, 153, 47), (342, 146, 47),
                    (343, 146, 429), (344, 146, 10))


def is_arv_blue(path=None, *, site=None, color=None):
    """Identify by explicit camera/channel or the workspace's source naming."""
    p = Path(path) if path is not None else Path("")
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    if site is None:
        site = "ARV" if "arv" in parts or name.startswith(("arv_", "zor_", "zorbas_")) else None
        # GNEISS ARV blue acquisition filenames; THA alone is not sufficient
        # because it also names red GIRAFF acquisitions.
        if re.match(r"tha260210_\d+_16bit\.tiff?$", name):
            site = "ARV"
            if color is None:
                color = "blue"
    if color is None:
        color = "blue" if "blue" in parts or "4278" in name or "_blue_" in name else None
    return str(site).upper() == "ARV" and str(color).lower() == "blue"


@lru_cache(maxsize=256)
def _source_corrected(path, mtime_ns, size):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        return ARV_BLUE_CORRECTION in (tif.pages[0].description or "")


def source_is_corrected(path):
    p = Path(path)
    if p.suffix.lower() not in {".tif", ".tiff"} or not p.is_file():
        return False
    stat = p.stat()
    return _source_corrected(str(p.resolve()), stat.st_mtime_ns, stat.st_size)


def correct_detector_frame(image, path=None, *, site=None, color=None):
    """Correct a freshly read raw frame. Do not call again on processed arrays.

    Corrected TIFFs marked with ARV_BLUE_CORRECTION are passed through. Other
    cameras/channels are untouched. Original images and files are never mutated.
    """
    if not is_arv_blue(path, site=site, color=color):
        return image
    if path is not None and source_is_corrected(path):
        return image
    raw = np.asarray(image)
    if raw.shape != (512, 512):
        raise ValueError(f"ARV 4278 correction requires native 512x512 pixels, got {raw.shape}; correct before cropping/coadding")
    out = raw.astype(np.float32, copy=True)
    for column, first_row, offset in ARV_BLUE_OFFSETS:
        out[first_row:, column] -= offset
    return out
