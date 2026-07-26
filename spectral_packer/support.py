"""
Support (stability) constraint utilities for bin packing.

The spectral packer has no notion of gravity: a placement is valid as long
as the item does not overlap occupied voxels. This module provides the
helpers needed to reject physically unstable placements:

- items floating in mid-air (nothing underneath),
- items resting on a sliver of contact that would tip over in reality.

All functions here are pure NumPy/SciPy — they do not depend on the
compiled ``_core`` module, so they can be unit-tested on machines without
the CUDA build (see ``tests/test_support.py``).

Coordinate conventions match the packer: arrays are indexed ``[x, y, z]``
with axis 2 (``k``) pointing up. A placement ``position`` is the item's
min-corner offset inside the tray, exactly as passed to ``place_in_tray``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def downward_facing_mask(item: np.ndarray) -> np.ndarray:
    """Boolean mask of the item's downward-facing (contact) voxels.

    A voxel is downward-facing when it is occupied and the cell directly
    below it *within the item* is empty — i.e. it is a surface the item
    could rest on. This includes overhang undersides of L/T-shaped items,
    not just the lowest layer.

    Parameters
    ----------
    item : np.ndarray
        3D array, non-zero = occupied.

    Returns
    -------
    np.ndarray
        Boolean array of the same shape as ``item``.
    """
    occupied = item > 0
    below_empty = np.ones_like(occupied)
    below_empty[:, :, 1:] = ~occupied[:, :, :-1]
    return occupied & below_empty


def support_ratio_grid(
    occ: np.ndarray,
    item: np.ndarray,
    ground_at_z0: bool = True,
) -> Tuple[np.ndarray, int]:
    """Support ratio for every candidate placement origin.

    For a placement origin ``q``, the support ratio is::

        (# downward-facing voxels v of the item such that the tray cell
         directly below q+v is occupied, or q+v sits on the z=0 floor)
        / (total # downward-facing voxels)

    The count for all origins at once is a cross-correlation between the
    tray occupancy (shifted up by one voxel along z) and the item's
    downward-facing mask, computed with an FFT. Indexing matches the C++
    placement origin: ``corr[q] = sum_v shifted_occ[q + v] * D[v]``, which
    is ``fftconvolve(shifted_occ, D[::-1, ::-1, ::-1], mode="full")``
    sliced starting at ``D.shape - 1`` (linear convolution — no circular
    wrap-around).

    Parameters
    ----------
    occ : np.ndarray
        Tray occupancy (non-zero = occupied: walls, floor and placed items).
    item : np.ndarray
        The item to place (non-zero = occupied). Must NOT be cropped to its
        bounding box — the array's own origin defines the placement offset.
    ground_at_z0 : bool, default True
        Treat the plane below ``z=0`` as solid ground. Set to False for
        trays whose ``z=0`` layer is not a physical floor.

    Returns
    -------
    ratio : np.ndarray
        Float array with the same shape as ``occ``; ``ratio[q]`` is the
        support ratio when the item's min-corner is placed at ``q``.
        Origins where the item would extend outside the tray are NOT
        filtered here — combine with the collision feasibility mask.
    contact_count : int
        Total number of downward-facing voxels (0 for an empty item, in
        which case ``ratio`` is all zeros).
    """
    from scipy.signal import fftconvolve

    contact = downward_facing_mask(item)
    contact_count = int(contact.sum())
    if contact_count == 0:
        return np.zeros(occ.shape, dtype=np.float64), 0

    shifted_occ = np.zeros(occ.shape, dtype=np.float64)
    shifted_occ[:, :, 1:] = (occ[:, :, :-1] != 0)
    if ground_at_z0:
        shifted_occ[:, :, 0] = 1.0

    kernel = contact[::-1, ::-1, ::-1].astype(np.float64)
    full = fftconvolve(shifted_occ, kernel, mode="full")

    dx, dy, dz = contact.shape
    counts = full[
        dx - 1: dx - 1 + occ.shape[0],
        dy - 1: dy - 1 + occ.shape[1],
        dz - 1: dz - 1 + occ.shape[2],
    ]
    counts = np.clip(np.rint(counts), 0, contact_count)
    return counts / float(contact_count), contact_count


def _supported_contact_points(
    item: np.ndarray,
    occ: np.ndarray,
    position: Tuple[int, int, int],
    ground_at_z0: bool = True,
) -> np.ndarray:
    """Tray-space (x, y) coordinates of contact voxels that are supported.

    Coordinates are looked up voxel-by-voxel (``argwhere(D) + position``)
    rather than via array slices, so items with trailing empty padding are
    handled correctly.
    """
    contact_local = np.argwhere(downward_facing_mask(item))
    if len(contact_local) == 0:
        return np.empty((0, 2), dtype=np.int64)

    world = contact_local + np.asarray(position, dtype=np.int64)
    below_z = world[:, 2] - 1

    supported = np.zeros(len(world), dtype=bool)
    on_floor = below_z < 0
    if ground_at_z0:
        supported |= on_floor

    in_bounds = (
        ~on_floor
        & (world[:, 0] >= 0) & (world[:, 0] < occ.shape[0])
        & (world[:, 1] >= 0) & (world[:, 1] < occ.shape[1])
        & (below_z < occ.shape[2])
    )
    idx = np.flatnonzero(in_bounds)
    if idx.size:
        supported[idx] = occ[world[idx, 0], world[idx, 1], below_z[idx]] != 0

    return world[supported][:, :2]


def com_supported(
    item: np.ndarray,
    occ: np.ndarray,
    position: Tuple[int, int, int],
    ground_at_z0: bool = True,
) -> bool:
    """Check that the item's center of mass sits over its support region.

    The support region is the 2D convex hull of the supported contact
    voxels; the placement passes when the horizontal projection of the
    center of mass (voxel centroid, uniform density assumed) lies inside
    the hull. Degenerate contact sets (fewer than 3 points, or collinear
    points) fall back to an axis-aligned straddle test: the contact points
    must bracket the center of mass in both x and y.

    Returns False when no contact voxel is supported at all.
    """
    points = _supported_contact_points(item, occ, position, ground_at_z0)
    if len(points) == 0:
        return False

    voxels = np.argwhere(item > 0)
    com_xy = voxels[:, :2].mean(axis=0) + np.asarray(position[:2], dtype=np.float64)

    unique_pts = np.unique(points, axis=0).astype(np.float64)
    if len(unique_pts) >= 3:
        from scipy.spatial import Delaunay

        try:
            from scipy.spatial import QhullError
        except ImportError:  # scipy < 1.8
            from scipy.spatial.qhull import QhullError

        try:
            hull = Delaunay(unique_pts)
            return bool(hull.find_simplex(com_xy) >= 0)
        except QhullError:
            pass  # collinear points — fall through to straddle test

    # Straddle fallback: contact extent must bracket the COM on both axes.
    # Half-voxel tolerance: a contact voxel supports its full cell footprint.
    lo = unique_pts.min(axis=0) - 0.5
    hi = unique_pts.max(axis=0) + 0.5
    return bool(np.all(com_xy >= lo) and np.all(com_xy <= hi))
