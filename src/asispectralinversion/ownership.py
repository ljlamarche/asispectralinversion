"""Combine independent inversions using fixed camera geometry."""

import numpy as np


OWNERSHIP_METHOD = "fixed_geometric_owner_no_fallback_either_finite"


def invert_owned_rgb(red, green, blue, lookup, owned, calculate_conductances=True, return_reasons=False):
    """Invert only finite positive RGB samples owned by this camera.

    Return full-grid arrays for combination and plotting, leaving unselected
    pixels NaN. Image preprocessing must happen before applying this mask.
    """
    from .inversion import calculate_E0_Q_v2, calculate_Sig

    red, green, blue = (np.asarray(value) for value in (red, green, blue))
    owned = np.asarray(owned, dtype=bool)
    if any(value.shape != owned.shape for value in (red, green, blue)):
        raise ValueError("RGB arrays and ownership mask must have the same shape")
    selected = owned & np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    selected &= (red > 0) & (green > 0) & (blue > 0)
    q = np.full(owned.shape, np.nan)
    e0 = np.full(owned.shape, np.nan)
    sigp = np.full(owned.shape, np.nan) if calculate_conductances else None
    sigh = np.full(owned.shape, np.nan) if calculate_conductances else None
    reasons = np.full(owned.shape, 1, dtype=np.uint8)
    reasons[owned] = 2
    finite = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    reasons[owned & finite & ~selected] = 3
    if selected.any():
        result = calculate_E0_Q_v2(
            red[selected], green[selected], blue[selected], lookup,
            minE0=150, generous=False, plot=False,
            **({"return_reasons": True} if return_reasons else {}),
        )
        selected_q, selected_e0 = result[:2]
        if return_reasons: reasons[selected] = result[2]
        q[selected], e0[selected] = selected_q, selected_e0
        if calculate_conductances:
            sigp[selected], sigh[selected] = calculate_Sig(
                selected_q, selected_e0, lookup, generous=False, plot=False,
            )
    if return_reasons:
        if calculate_conductances:
            reasons[(reasons == 0) & (~np.isfinite(sigp) | ~np.isfinite(sigh))] = 11
        return q, e0, sigp, sigh, reasons
    return q, e0, sigp, sigh


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

    An owner is retained when its valid flag is set and either Q or E0 is finite.
    Each nonfinite component remains missing. No other camera fills its gaps.
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

    reasons = np.full(owner_site.shape, 1, dtype=np.uint8)
    has_reasons = all("failure_reason" in p for p in site_products.values())
    for site, code in site_codes.items():
        product = site_products[site]
        if has_reasons:
            owned = owner_site == code
            reasons[owned] = product["failure_reason"][owned]
        valid = (
            np.asarray(product["valid"], dtype=bool)
            & (np.isfinite(product["q"]) | np.isfinite(product["e0"]))
        )
        ownership_count += valid
        selected = (owner_site == code) & valid
        q[selected] = np.where(np.isfinite(product["q"][selected]), product["q"][selected], np.nan)
        e0[selected] = np.where(np.isfinite(product["e0"][selected]), product["e0"][selected], np.nan)
        source_site[selected] = code
        if calculate_conductances:
            sigp[selected] = product["sigp"][selected]
            sigh[selected] = product["sigh"][selected]

    return {
        **({"failure_reason": reasons} if has_reasons else {}),
        "q": q,
        "e0": e0,
        "sigp": sigp,
        "sigh": sigh,
        "source_site": source_site,
        "ownership_count": ownership_count,
    }
