"""Background-mask construction and robust scalar estimation for ASI images."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def inset_mask(mask, edge_buffer_px=0):
    """Keep mask pixels at least ``edge_buffer_px`` from its boundary."""
    mask = np.asarray(mask, dtype=bool)
    edge_buffer_px = float(edge_buffer_px)
    if edge_buffer_px < 0:
        raise ValueError("edge_buffer_px must be nonnegative")
    if edge_buffer_px == 0:
        return mask.copy()
    return mask & (distance_transform_edt(mask) > edge_buffer_px)


def physical_offsky_mask(elevation, edge_buffer_px=0):
    """Select pixels outside the calibrated sky disc in an elevation map.

    The ARV starmaps encode off-sky pixels as zero elevation. Nonfinite values
    are also treated as off-sky. Positive low-elevation pixels remain sky and
    are deliberately not selected.
    """
    elevation = np.asarray(elevation, dtype=float)
    offsky = ~np.isfinite(elevation) | (elevation <= 0.0)
    return inset_mask(offsky, edge_buffer_px=edge_buffer_px)


def corner_mask(shape, corner_size_px=100):
    """Return four square corner regions for a two-dimensional image."""
    if len(shape) != 2:
        raise ValueError("shape must describe a two-dimensional image")
    rows, columns = shape
    size = int(corner_size_px)
    if size <= 0 or size > min(rows, columns) // 2:
        raise ValueError("corner_size_px must be positive and at most half the image")
    mask = np.zeros((rows, columns), dtype=bool)
    mask[:size, :size] = True
    mask[:size, -size:] = True
    mask[-size:, :size] = True
    mask[-size:, -size:] = True
    return mask


def estimate_scalar_background(image, mask, minimum_pixels=100):
    """Return the masked median and MAD-based robust standard deviation."""
    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} versus {mask.shape}")
    values = image[mask & np.isfinite(image)]
    if values.size < int(minimum_pixels):
        raise ValueError(f"Background mask contains only {values.size} finite pixels")
    background = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - background)))
    return background, sigma, values
