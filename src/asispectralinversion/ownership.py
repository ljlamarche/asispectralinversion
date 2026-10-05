"""Combine independent inversions using fixed camera geometry."""

import numpy as np


OWNERSHIP_METHOD = "fixed_geometric_owner_no_fallback"


def fixed_camera_ownership(site_coverage, site_distance_km, site_codes):
    """Return the nearest eligible camera's code on a shared 2D grid.

    ``site_coverage`` contains each camera's static red-and-green geometric
    coverage. Brightness and inversion validity must not enter these masks.
    Equal distances favor the first camera in ``site_coverage`` insertion order.
    A zero code means no camera covers the cell at a finite distance.
    """
    if not site_coverage:
        raise ValueError("At least one camera coverage grid is required")
    shape = np.shape(next(iter(site_coverage.values())))
    owner_site = np.zeros(shape, dtype=np.uint8)
    nearest_distance = np.full(shape, np.inf)
    for site, coverage in site_coverage.items():
        distance = np.asarray(site_distance_km[site])
        eligible = np.asarray(coverage, dtype=bool) & np.isfinite(distance)
        nearer = eligible & (distance < nearest_distance)
        owner_site[nearer] = site_codes[site]
        nearest_distance[nearer] = distance[nearer]
    return owner_site


def combine_owned_products(
    site_products, owner_site, site_codes, calculate_conductances=True
):
    """Select only valid results from each cell's fixed geometric owner.

    An invalid owner leaves a hole even if another camera has a valid result.
    ``source_site`` is zero at those holes and outside geometric coverage.
    ``ownership_count`` retains the count of all independently valid cameras;
    it is an overlap diagnostic, not a measure of combined-output validity.
    Conductances come from the same owner, including that owner's NaNs.
    Input products and the owner grid are never modified.
    """
    owner_site = np.asarray(owner_site)
    q = np.full(owner_site.shape, np.nan)
    e0 = np.full(owner_site.shape, np.nan)
    sigp = np.full(owner_site.shape, np.nan) if calculate_conductances else None
    sigh = np.full(owner_site.shape, np.nan) if calculate_conductances else None
    source_site = np.zeros(owner_site.shape, dtype=np.uint8)
    ownership_count = np.zeros(owner_site.shape, dtype=np.uint8)

    for site, code in site_codes.items():
        product = site_products[site]
        valid = (
            np.asarray(product["valid"], dtype=bool)
            & np.isfinite(product["q"])
            & np.isfinite(product["e0"])
        )
        ownership_count += valid
        selected = (owner_site == code) & valid
        q[selected] = product["q"][selected]
        e0[selected] = product["e0"][selected]
        source_site[selected] = code
        if calculate_conductances:
            sigp[selected] = product["sigp"][selected]
            sigh[selected] = product["sigh"][selected]

    return {
        "q": q,
        "e0": e0,
        "sigp": sigp,
        "sigh": sigh,
        "source_site": source_site,
        "ownership_count": ownership_count,
    }
