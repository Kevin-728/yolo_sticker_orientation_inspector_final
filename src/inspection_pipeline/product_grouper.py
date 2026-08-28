# product_grouper.py
"""
Assign detected components to their respective products.

Strategy
--------
For each fuse_cover (one per product):
  - Compute centroid distance from every QR code / yellow_triangle centroid
    to the fuse_cover centroid.
  - Assign the N closest components (exclusive; no-conflict assumed by caller).

Expected counts per product:
    qrcode:          2
    yellow_triangle: 3

If fewer than the expected count are available for a product, the missing
slots are recorded as None so the inspector can emit UNKNOWN for them.
"""

import numpy as np

QR_PER_PRODUCT  = 2
TRI_PER_PRODUCT = 3


def _centroid(polygon):
    """Mean of polygon vertices as float array (2,)."""
    return np.mean(np.array(polygon, dtype=np.float64), axis=0)


def group_by_fuse_cover(detections):
    """
    Parameters
    ----------
    detections : dict
        Flat dict with keys 'fuse_cover', 'qrcode', 'yellow_triangle',
        each mapping to a list of polygons.

    Returns
    -------
    list of dict, one per fuse_cover:
        {
            "fuse_cover":      polygon,          # single polygon
            "qrcode":          [poly|None, ...], # length QR_PER_PRODUCT
            "yellow_triangle": [poly|None, ...], # length TRI_PER_PRODUCT
        }
    """
    fuse_polys = detections.get("fuse_cover", [])
    qr_polys   = list(detections.get("qrcode", []))
    tri_polys  = list(detections.get("yellow_triangle", []))

    if not fuse_polys:
        return []

    fuse_centroids = [_centroid(p) for p in fuse_polys]
    qr_centroids   = [_centroid(p) for p in qr_polys]
    tri_centroids  = [_centroid(p) for p in tri_polys]

    # Track which components are still unassigned
    qr_available  = list(range(len(qr_polys)))
    tri_available = list(range(len(tri_polys)))

    products = []

    for fuse_idx, fuse_center in enumerate(fuse_centroids):

        # ── Assign QR codes ────────────────────────────────────────────────
        assigned_qr = _assign_nearest(
            fuse_center, qr_centroids, qr_available, QR_PER_PRODUCT
        )
        qr_slots = [qr_polys[i] if i is not None else None
                    for i in assigned_qr]

        # ── Assign yellow triangles ────────────────────────────────────────
        assigned_tri = _assign_nearest(
            fuse_center, tri_centroids, tri_available, TRI_PER_PRODUCT
        )
        tri_slots = [tri_polys[i] if i is not None else None
                     for i in assigned_tri]

        products.append({
            "fuse_cover":      fuse_polys[fuse_idx],
            "qrcode":          qr_slots,
            "yellow_triangle": tri_slots,
        })

    return products


def _assign_nearest(fuse_center, all_centroids, available_indices, n_needed):
    """
    From *available_indices* pick the *n_needed* entries whose centroid is
    closest to *fuse_center*.  Removes selected indices from *available_indices*
    in-place.  Returns list of length *n_needed* (None for missing slots).
    """
    if not available_indices:
        return [None] * n_needed

    # Sort available by distance to fuse_center
    distances = [
        (np.linalg.norm(all_centroids[i] - fuse_center), i)
        for i in available_indices
    ]
    distances.sort(key=lambda t: t[0])

    selected = []
    for _, idx in distances[:n_needed]:
        selected.append(idx)
        available_indices.remove(idx)

    # Pad with None if not enough components were available
    while len(selected) < n_needed:
        selected.append(None)

    return selected
