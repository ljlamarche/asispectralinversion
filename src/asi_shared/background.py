"""Background-mask construction and robust scalar estimation for ASI images."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from skimage.restoration import estimate_sigma

BACKGROUND_ELEVATION_CUTOFF_DEG = 0.0
BACKGROUND_EDGE_BUFFER_PX = 50.0


def inset_mask(mask, edge_buffer_px=0):
    """Keep mask pixels at least ``edge_buffer_px`` from its boundary."""
    mask = np.asarray(mask, dtype=bool)
    edge_buffer_px = float(edge_buffer_px)
    if not np.isfinite(edge_buffer_px) or edge_buffer_px < 0:
        raise ValueError("edge_buffer_px must be nonnegative")
    if edge_buffer_px == 0:
        return mask.copy()
    return mask & (distance_transform_edt(mask) > edge_buffer_px)


def physical_offsky_mask(elevation, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    """Finite elevation <= 0 degrees, inset by 50 detector pixels by default.

    Zero-valued starmap regions are a selection proxy, not a physical horizon.
    Nonfinite elevations and positive low-elevation sky are never samples.
    """
    elevation = np.asarray(elevation, dtype=float)
    offsky = np.isfinite(elevation) & (elevation <= BACKGROUND_ELEVATION_CUTOFF_DEG)
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


def estimate_scalar_background(image, mask, minimum_pixels=2):
    """Return Gaussian histogram center, noise sigma, and selected finite DNs.

    The shared estimator uses the diagnostic's median fallback if fitting fails.
    Insufficient samples raise; the requested mask is never silently enlarged.
    """
    image = np.asarray(image)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} versus {mask.shape}")
    values = finite_values(image[mask])
    if values.size < minimum_pixels:
        raise ValueError(f"Background mask contains only {values.size} finite pixels")
    result = estimate_corner_background(image, mask)
    return result['center'], result['sigma'], values


def gauss(x, amp, center, sigma):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def estimate_corner_background(im, mask):
    imvec = finite_values(np.asarray(im)[mask])
    if imvec.size < 2:
        raise ValueError(f"corner mask produced only {imvec.size} finite pixels")

    data_min = float(np.nanmin(imvec))
    data_max = float(np.nanmax(imvec))
    center_guess = float(np.nanmedian(imvec))
    if data_max <= data_min:
        return {
            "center": center_guess,
            "sigma": 0.0,
            "median": center_guess,
            "fit_failed": True,
            "n_pixels": int(imvec.size),
        }

    bins = np.arange(np.floor(data_min), np.ceil(data_max) + 1)
    if bins.size < 3:
        bins = np.linspace(data_min, data_max + 1.0, 3)
    bincenters = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(imvec, bins=bins)

    sigma_guess = float(estimate_sigma(np.asarray(im)[np.isfinite(im)]))
    if not np.isfinite(sigma_guess) or sigma_guess <= 0:
        sigma_guess = float(np.nanstd(imvec))
    if not np.isfinite(sigma_guess) or sigma_guess <= 0:
        sigma_guess = 1.0

    fit_failed = False
    peak_guess = max(float(np.nanmax(hist)) / 5.0, 1.0)
    try:
        _peak, center, sigma = curve_fit(
            gauss,
            bincenters,
            hist,
            p0=[peak_guess, center_guess, sigma_guess],
            maxfev=10000,
        )[0]
        center = float(center)
        sigma = abs(float(sigma))
    except Exception:
        fit_failed = True
        center = center_guess
        sigma = sigma_guess

    hrange = float(bins[-1] - bins[0])
    if not np.isfinite(center) or not np.isfinite(sigma) or (hrange > 0 and (sigma > (hrange / 2.0) or abs(center - center_guess) > (hrange / 4.0))):
        fit_failed = True
        center = center_guess
        sigma = sigma_guess

    return {
        "center": center,
        "sigma": sigma,
        "median": center_guess,
        "fit_failed": fit_failed,
        "n_pixels": int(imvec.size),
    }



def load_hdf5_elevation(path):
    """Read native elevation without deriving background from geographic NaNs.

    Accept one unambiguous Elevation/elevation dataset anywhere in an HDF5
    starmap. Degrees and radians have the same zero-degree selection boundary.
    """
    import h5py
    with h5py.File(path, "r") as handle:
        names = []
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1].lower() == "elevation":
                names.append(name)
        handle.visititems(collect)
        if len(names) != 1:
            raise ValueError(
                f"{path}: expected one native Elevation dataset, found {len(names)}. "
                "Supply the original detector elevation map for 0-degree / 50-pixel background selection."
            )
        return np.asarray(handle[names[0]], dtype=float)
