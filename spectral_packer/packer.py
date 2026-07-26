"""
High-level bin packing interface.

This module provides the main BinPacker class for 3D bin packing operations,
along with the PackingResult dataclass for results.

Examples
--------
>>> from spectral_packer import BinPacker
>>> packer = BinPacker(tray_size=(100, 100, 100))
>>> result = packer.pack_files(["item1.stl", "item2.obj"])
>>> print(f"Packed {result.num_placed}/{result.num_placed + result.num_failed} items")
>>> print(f"Density: {result.density:.1%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .mesh_io import load_mesh
from .voxelizer import Voxelizer, VoxelizationInfo, LogEntry
from .rotations import get_orientations, make_contiguous


@dataclass
class PlacementInfo:
    """Information about a single item placement.

    Attributes
    ----------
    item_index : int
        Original index of the item in the input list.
    position : tuple or None
        (x, y, z) placement position, or None if placement failed.
    score : float or None
        Placement score (lower is better), or None if placement failed.
    success : bool
        Whether the item was successfully placed.
    volume : int
        Number of voxels in the item.
    orientation_index : int
        Index of the orientation used (0 = original orientation).
    """

    item_index: int
    position: Optional[Tuple[int, int, int]]
    score: Optional[float]
    success: bool
    volume: int = 0
    orientation_index: int = 0
    refined_position: Optional[Tuple[float, float, float]] = None


@dataclass
class MeshPlacementInfo:
    """Placement info with mesh-level metadata for Blender export.

    This dataclass extends PlacementInfo with the metadata needed to
    transform original mesh files to their packed positions for export
    to Blender or other 3D applications.

    Attributes
    ----------
    mesh_path : Path
        Path to the original mesh file.
    voxel_info : VoxelizationInfo
        Voxelization metadata for coordinate mapping.
    voxel_position : tuple of int
        (x, y, z) position in voxel coordinates.
    orientation_index : int
        Index of the orientation used (0-23).
    success : bool
        Whether the item was successfully placed.
    """

    mesh_path: Path
    voxel_info: VoxelizationInfo
    voxel_position: Optional[Tuple[int, int, int]]
    orientation_index: int
    success: bool
    refined_position: Optional[Tuple[float, float, float]] = None


@dataclass
class PackingResult:
    """Results from a packing operation.

    Attributes
    ----------
    tray : np.ndarray
        Final voxel grid with all placed items. Each item's voxels are
        marked with a unique ID (1, 2, 3, ...).
    placements : list of PlacementInfo
        List of placement information for each item.
    num_placed : int
        Number of items successfully placed.
    num_failed : int
        Number of items that could not be placed.
    density : float
        Packing density (occupied volume / bounding box volume).
    total_volume : int
        Total number of occupied voxels.
    bounding_box : tuple
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) of occupied region.
    """

    tray: np.ndarray
    placements: List[PlacementInfo] = field(default_factory=list)
    num_placed: int = 0
    num_failed: int = 0
    density: float = 0.0
    total_volume: int = 0
    bounding_box: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    mesh_placements: Optional[List[MeshPlacementInfo]] = None

    def get_item_mask(self, item_id: int) -> np.ndarray:
        """Get a binary mask for a specific item.

        Parameters
        ----------
        item_id : int
            The ID of the item (1-indexed).

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates voxels belonging to the item.
        """
        return self.tray == item_id

    def save_vox(self, path: Union[str, Path]) -> None:
        """Save result to MagicaVoxel .vox format.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        from . import save_vox
        save_vox(self.tray, str(path))

    def summary(self) -> str:
        """Get a human-readable summary of the packing result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"Packing Result Summary",
            f"=" * 40,
            f"Items placed:    {self.num_placed}",
            f"Items failed:    {self.num_failed}",
            f"Success rate:    {self.num_placed / max(1, self.num_placed + self.num_failed):.1%}",
            f"Packing density: {self.density:.1%}",
            f"Total volume:    {self.total_volume} voxels",
            f"Tray size:       {self.tray.shape}",
        ]
        if self.bounding_box:
            bbox_min, bbox_max = self.bounding_box
            lines.append(f"Bounding box:    {bbox_min} to {bbox_max}")
        return "\n".join(lines)


class BinPacker:
    """
    GPU-accelerated 3D bin packing using spectral (FFT) collision detection.

    This class provides the main interface for packing 3D items into a
    rectangular tray. It uses FFT-based algorithms for efficient collision
    detection and optimal placement finding.

    Parameters
    ----------
    tray_size : tuple of int
        Size of the packing tray as (x, y, z).
    voxel_resolution : int, default 128
        Resolution for mesh voxelization.
    height_penalty : float, default 1e8
        Penalty factor for height in placement scoring.
        Higher values encourage items to be placed lower.
    num_orientations : int, default 1
        Number of orientations to try for each item.
        Valid values: 1 (original only), 4 (Z-axis rotations),
        6 (one per face), 24 (all cube symmetries).
        More orientations = better packing but slower.
    interlocking_free : bool, default False
        If True, only place items in positions where they can be removed
        without colliding with other items (Section 4.3 of the paper).
        This results in lower packing density but guarantees no interlocking.
    support_threshold : float, default 0.0
        Minimum fraction of the item's downward-facing voxels that must
        rest on the tray floor or on occupied voxels (walls / other items)
        for a placement to be accepted. 0.0 disables the support constraint
        (original behavior). When enabled, the item's center of mass must
        also project inside the convex hull of its supported contacts.
        Not compatible with ``interlocking_free`` or
        ``continuous_refinement`` (v1).
    support_top_k : int, default 256
        Number of best-scoring candidate positions to test against the
        center-of-mass condition before giving up (support mode only).

    Attributes
    ----------
    tray_size : tuple
        The tray dimensions.
    voxel_resolution : int
        Voxelization resolution.
    height_penalty : float
        Height penalty factor.
    num_orientations : int
        Number of orientations to sample.
    interlocking_free : bool
        Whether to enforce interlocking-free placement.

    Examples
    --------
    Basic usage with file paths:

    >>> packer = BinPacker(tray_size=(100, 100, 100))
    >>> result = packer.pack_files(["item1.stl", "item2.obj"])
    >>> print(f"Density: {result.density:.1%}")

    With pre-voxelized items:

    >>> voxelizer = Voxelizer(resolution=64)
    >>> items = [voxelizer.voxelize_file(f) for f in files]
    >>> result = packer.pack_voxels(items)

    Accessing placement details:

    >>> for p in result.placements:
    ...     if p.success:
    ...         print(f"Item {p.item_index} at {p.position}, score={p.score:.2f}")
    """

    def __init__(
        self,
        tray_size: Tuple[int, int, int],
        voxel_resolution: int = 128,
        height_penalty: float = 50.0,
        num_orientations: int = 1,
        interlocking_free: bool = False,
        continuous_refinement: bool = False,
        pitch: Optional[float] = None,
        verbose: bool = False,
        support_threshold: float = 0.0,
        support_top_k: int = 256,
    ):
        if len(tray_size) != 3:
            raise ValueError(f"tray_size must be a 3-tuple, got {len(tray_size)} elements")
        if any(s <= 0 for s in tray_size):
            raise ValueError(f"tray_size dimensions must be positive, got {tray_size}")
        if voxel_resolution <= 0:
            raise ValueError(f"voxel_resolution must be positive, got {voxel_resolution}")
        if num_orientations not in (1, 4, 6, 24):
            raise ValueError(f"num_orientations must be 1, 4, 6, or 24, got {num_orientations}")
        if pitch is not None and pitch <= 0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                f"support_threshold must be in [0, 1], got {support_threshold}"
            )
        if support_threshold > 0.0 and interlocking_free:
            raise ValueError(
                "support_threshold cannot be combined with interlocking_free (v1)"
            )
        if support_threshold > 0.0 and continuous_refinement:
            raise ValueError(
                "support_threshold cannot be combined with continuous_refinement: "
                "sub-voxel refinement does not re-check the support condition"
            )
        if support_top_k <= 0:
            raise ValueError(f"support_top_k must be positive, got {support_top_k}")

        self.tray_size = tuple(tray_size)
        self.voxel_resolution = voxel_resolution
        self.height_penalty = height_penalty
        self.num_orientations = num_orientations
        self.interlocking_free = interlocking_free
        self.continuous_refinement = continuous_refinement
        self.support_threshold = float(support_threshold)
        self.support_top_k = int(support_top_k)
        self.pitch = pitch
        self.verbose = verbose
        self.log: List[LogEntry] = []
        self._voxelizer = Voxelizer(resolution=voxel_resolution, pitch=pitch)

    def _record(self, step: str, duration: float, success: bool = True, notes: str = ""):
        self.log.append(LogEntry(step=step, duration_sec=duration, success=success, notes=notes))

    def pack_files(
        self,
        paths: Sequence[Union[str, Path]],
        sort_by_volume: bool = True,
        validate_meshes: bool = True,
        repair_meshes: bool = True,
    ) -> PackingResult:
        """
        Pack meshes from file paths.

        Parameters
        ----------
        paths : sequence of str or Path
            Paths to mesh files (STL, OBJ, PLY, etc.).
        sort_by_volume : bool, default True
            Sort items by volume (largest first) before packing.
        validate_meshes : bool, default True
            Validate meshes during loading.
        repair_meshes : bool, default True
            Attempt to repair invalid meshes.

        Returns
        -------
        PackingResult
            Packing results including final tray and statistics.

        Raises
        ------
        FileNotFoundError
            If any mesh file does not exist.
        MeshLoadError
            If any mesh fails to load.
        """
        voxels = []
        for path in paths:
            voxel = self._voxelizer.voxelize_file(
                path,
                validate=validate_meshes,
                repair=repair_meshes,
            )
            voxels.append(voxel)

        return self.pack_voxels(voxels, sort_by_volume=sort_by_volume)

    def pack_files_for_export(
        self,
        paths: Sequence[Union[str, Path]],
        sort_by_volume: bool = True,
        validate_meshes: bool = True,
        repair_meshes: bool = True,
    ) -> PackingResult:
        """
        Pack meshes from file paths and preserve metadata for Blender export.

        This method is similar to `pack_files()` but also populates the
        `mesh_placements` field in the result, which contains the metadata
        needed to export packed objects to Blender with correct transforms.

        Parameters
        ----------
        paths : sequence of str or Path
            Paths to mesh files (STL, OBJ, PLY, etc.).
        sort_by_volume : bool, default True
            Sort items by volume (largest first) before packing.
        validate_meshes : bool, default True
            Validate meshes during loading.
        repair_meshes : bool, default True
            Attempt to repair invalid meshes.

        Returns
        -------
        PackingResult
            Packing results with `mesh_placements` populated for export.

        Raises
        ------
        FileNotFoundError
            If any mesh file does not exist.
        MeshLoadError
            If any mesh fails to load.
        """
        # Voxelize all meshes and collect metadata
        voxels = []
        voxel_infos = []
        for path in paths:
            path = Path(path)
            voxel, info = self._voxelizer.voxelize_file_with_info(
                path,
                validate=validate_meshes,
                repair=repair_meshes,
            )
            voxels.append(voxel)
            voxel_infos.append(info)

        # Pack using existing method
        result = self.pack_voxels(voxels, sort_by_volume=sort_by_volume)

        # Build mesh placements by matching item_index to original order
        # Note: placements are sorted by item_index in pack_voxels()
        mesh_placements = []
        for placement in result.placements:
            idx = placement.item_index
            mesh_placements.append(MeshPlacementInfo(
                mesh_path=voxel_infos[idx].mesh_path,
                voxel_info=voxel_infos[idx],
                voxel_position=placement.position,
                orientation_index=placement.orientation_index,
                success=placement.success,
                refined_position=placement.refined_position,
            ))

        result.mesh_placements = mesh_placements
        return result



    def pack_voxels(
        self,
        items: Sequence[np.ndarray],
        sort_by_volume: bool = True,
        initial_tray: Optional[np.ndarray] = None,
    ) -> PackingResult:
        """
        Pack pre-voxelized items.

        Parameters
        ----------
        items : sequence of np.ndarray
            List of 3D int/bool arrays representing voxelized items.
            Non-zero values indicate occupied voxels.
        sort_by_volume : bool, default True
            Sort items by volume (largest first) before packing.
        initial_tray : np.ndarray, optional
            Pre-filled tray to pack into. If provided, must match tray_size.
            Non-zero values are treated as obstacles. Object IDs will start
            after the maximum value in initial_tray.

        Returns
        -------
        PackingResult
            Packing results including final tray and statistics.

        Raises
        ------
        ValueError
            If items list is empty or contains invalid arrays.
        """
        from . import (
            fft_search_placement,
            fft_search_placement_with_cache,
            fft_search_batch_interlocking_free,
            place_in_tray,
            set_height_penalty,
        )

        # Propagate height_penalty to C++ layer
        set_height_penalty(self.height_penalty)

        if len(items) == 0:
            raise ValueError("items list cannot be empty")

        # Validate and convert items
        processed_items = []
        volumes = []
        for i, item in enumerate(items):
            if not isinstance(item, np.ndarray):
                raise ValueError(f"Item {i} is not a numpy array")
            if item.ndim != 3:
                raise ValueError(f"Item {i} must be 3D, got {item.ndim}D")
            item_int = item.astype(np.int32)
            processed_items.append(item_int)
            volumes.append(int(np.sum(item_int > 0)))

        # Sort by volume if requested
        if sort_by_volume:
            sorted_indices = np.argsort(volumes)[::-1]  # Largest first
            processed_items = [processed_items[i] for i in sorted_indices]
            volumes = [volumes[i] for i in sorted_indices]
            original_indices = list(sorted_indices)
        else:
            original_indices = list(range(len(items)))

        # Initialize tray
        if initial_tray is not None:
            if initial_tray.shape != self.tray_size:
                raise ValueError(
                    f"initial_tray shape {initial_tray.shape} does not match "
                    f"tray_size {self.tray_size}"
                )
            tray = initial_tray.astype(np.int32).copy()
            # Start object IDs after max value in initial tray
            id_offset = int(np.max(tray))
        else:
            tray = np.zeros(self.tray_size, dtype=np.int32)
            id_offset = 0
        generation = 0

        placements = []
        num_placed = 0
        pack_start = time.perf_counter()

        for idx, (item, orig_idx, volume) in enumerate(
            zip(processed_items, original_indices, volumes)
        ):
            item_start = time.perf_counter()
            # Try multiple orientations and find the best placement
            best_position = None
            best_score = float('inf')
            best_orientation_idx = 0
            best_rotated_item = item
            found_any = False

            orientations = get_orientations(item, self.num_orientations)

            # Pre-compute distance field ONCE per item (not per orientation)
            # Use only placed items (tray > id_offset) — exclude walls (= id_offset)
            # so that proximity scoring attracts items to each other, not to walls.
            # Support mode always needs it, even for a single orientation
            # (the C++ single-orientation path would use the wall-included
            # tray instead — the score therefore differs slightly there).
            support_enabled = self.support_threshold > 0.0
            if self.num_orientations > 1 or self.interlocking_free or support_enabled:
                item_only_tray = (tray > id_offset).astype(np.int32)
                tray_distance = self._compute_distance_field(item_only_tray)

            if self.interlocking_free:
                # Use batch interlocking-free search (Section 4.3)
                # Filter orientations that fit in tray
                valid_orientations = []
                valid_orient_indices = []
                for orient_idx, rotated_item in enumerate(orientations):
                    rotated_item = make_contiguous(rotated_item.astype(np.int32))
                    if not any(rotated_item.shape[i] > self.tray_size[i] for i in range(3)):
                        valid_orientations.append(rotated_item)
                        valid_orient_indices.append(orient_idx)

                if valid_orientations:
                    position, found, score, batch_orient_idx = fft_search_batch_interlocking_free(
                        valid_orientations, tray, tray_distance, generation
                    )
                    if found and batch_orient_idx >= 0:
                        best_position = position
                        best_score = score
                        # Map back to original orientation index
                        best_orientation_idx = valid_orient_indices[batch_orient_idx]
                        best_rotated_item = valid_orientations[batch_orient_idx]
                        found_any = True
            else:
                # Standard search (no interlocking constraint)
                for orient_idx, rotated_item in enumerate(orientations):
                    rotated_item = make_contiguous(rotated_item.astype(np.int32))

                    if any(rotated_item.shape[i] > self.tray_size[i] for i in range(3)):
                        continue

                    if support_enabled:
                        position, found, score = self._search_placement_supported(
                            rotated_item, tray, tray_distance
                        )
                    elif self.num_orientations > 1:
                        position, found, score = fft_search_placement_with_cache(
                            rotated_item, tray, tray_distance, generation
                        )
                    else:
                        position, found, score = fft_search_placement(rotated_item, tray)

                    if found and score < best_score:
                        best_position = position
                        best_score = score
                        best_orientation_idx = orient_idx
                        best_rotated_item = rotated_item
                        found_any = True

            if found_any:
                # Continuous refinement: sub-voxel position optimization (Section 4.2)
                refined_pos = None
                if self.continuous_refinement:
                    # Need distance field (compute if not already available)
                    if not (self.num_orientations > 1 or self.interlocking_free):
                        item_only_tray = (tray > id_offset).astype(np.int32)
                        tray_distance = self._compute_distance_field(item_only_tray)
                    refined_pos = self._refine_placement(
                        best_rotated_item, tray_distance, best_position
                    )

                item_id = id_offset + num_placed + 1
                tray = place_in_tray(best_rotated_item, tray, best_position, item_id)
                generation += 1
                num_placed += 1
                placements.append(PlacementInfo(
                    item_index=orig_idx,
                    position=best_position,
                    score=best_score,
                    success=True,
                    volume=volume,
                    orientation_index=best_orientation_idx,
                    refined_position=refined_pos,
                ))
                self._record(
                    f"place_item:{orig_idx}",
                    time.perf_counter() - item_start,
                    True,
                    f"pos={best_position} orient={best_orientation_idx} score={best_score:.2f}",
                )
            else:
                placements.append(PlacementInfo(
                    item_index=orig_idx,
                    position=None,
                    score=None,
                    success=False,
                    volume=volume,
                    orientation_index=0,
                ))
                self._record(
                    f"place_item:{orig_idx}",
                    time.perf_counter() - item_start,
                    False,
                    (
                        f"no supported placement (support_threshold={self.support_threshold})"
                        if support_enabled
                        else "no valid placement"
                    ),
                )

        # Sort placements by original index for consistent ordering
        placements.sort(key=lambda p: p.item_index)

        pack_elapsed = time.perf_counter() - pack_start
        self._record(
            "packing_total",
            pack_elapsed,
            True,
            f"placed={num_placed}/{len(items)}",
        )

        # Calculate statistics
        density, total_volume, bbox = self._calculate_stats(tray)

        result = PackingResult(
            tray=tray,
            placements=placements,
            num_placed=num_placed,
            num_failed=len(items) - num_placed,
            density=density,
            total_volume=total_volume,
            bounding_box=bbox,
        )

        return result

    def pack_single(
        self,
        item: np.ndarray,
        tray: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[int, int, int]], bool, float]:
        """
        Find optimal placement for a single item.

        Parameters
        ----------
        item : np.ndarray
            3D int/bool array representing the item.
        tray : np.ndarray, optional
            Current tray state. If None, uses an empty tray.

        Returns
        -------
        position : tuple or None
            (x, y, z) placement position, or None if no valid placement.
        found : bool
            Whether a valid placement was found.
        score : float
            Placement score (lower is better), or 0.0 if not found.
        """
        from . import fft_search_placement

        if tray is None:
            tray = np.zeros(self.tray_size, dtype=np.int32)

        item_int = item.astype(np.int32)
        position, found, score = fft_search_placement(item_int, tray)

        if found:
            return tuple(position), True, score
        return None, False, 0.0

    def _search_placement_supported(
        self,
        item: np.ndarray,
        tray: np.ndarray,
        tray_phi: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int]], bool, float]:
        """Find the best placement that satisfies the support constraint.

        Reimplements the C++ search score in Python so that candidate
        positions can be filtered before the argmin (the C++ search only
        returns its single best position):

            score(q) = proximity(q) / |item| + height_penalty * (q_z / L)^3

        where ``proximity = dft_corr3(tray_phi, padded_item)`` and the
        feasible set is ``collision_grid(tray, item) == 0`` (the binding
        marks out-of-bounds origins as colliding — linear convolution,
        no circular wrap). On top of that, candidates must have a support
        ratio >= ``support_threshold``; the best ``support_top_k`` of them
        (by score) are then tested against the center-of-mass condition.

        Returns
        -------
        (position, found, score) — same contract as ``fft_search_placement``.
        """
        from . import collision_grid, dft_corr3
        from .support import com_supported, support_ratio_grid

        tray_shape = tray.shape

        # Pad item to tray size at the origin corner (same as C++ padto3d).
        padded = np.zeros(tray_shape, dtype=np.int32)
        padded[: item.shape[0], : item.shape[1], : item.shape[2]] = item

        collision = np.asarray(collision_grid(tray, item))
        feasible = collision == 0

        occ = (tray != 0).astype(np.int32)
        ratio, contact_count = support_ratio_grid(occ, item, ground_at_z0=True)
        if contact_count == 0:
            return None, False, 0.0

        proximity = np.asarray(dft_corr3(tray_phi.astype(np.int32), padded))
        if collision.shape != tray_shape or proximity.shape != tray_shape:
            # The C++ bindings truncate their FFT results back to tray size
            # (dft_conv3 → truncateto3d); flat-index math below relies on it.
            raise RuntimeError(
                f"binding output shape mismatch: collision={collision.shape} "
                f"proximity={proximity.shape} tray={tray_shape}"
            )
        norm = float(max(1, int(np.count_nonzero(item))))
        depth = tray_shape[2]
        qz = (np.arange(depth, dtype=np.float64) / depth) ** 3
        score = proximity / norm + self.height_penalty * qz[np.newaxis, np.newaxis, :]

        # Numerical guard on the ratio comparison (FFT round-off).
        valid = feasible & (ratio >= self.support_threshold - 1e-9)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return None, False, 0.0

        flat_score = score.ravel()
        k = min(self.support_top_k, valid_idx.size)
        top = valid_idx[np.argpartition(flat_score[valid_idx], k - 1)[:k]]
        top = top[np.argsort(flat_score[top])]

        for flat in top:
            pos = np.unravel_index(flat, tray_shape)
            if com_supported(item, occ, pos, ground_at_z0=True):
                position = tuple(int(v) for v in pos)
                return position, True, float(flat_score[flat])

        return None, False, 0.0

    def _refine_placement(
        self,
        item: np.ndarray,
        tray_distance: np.ndarray,
        discrete_position: tuple,
    ) -> Optional[Tuple[float, float, float]]:
        """Continuous refinement of a discrete voxel placement (Section 4.2).

        Starting from the discrete integer position q*, minimizes the proximity
        energy E(t) = Σ_{v ∈ item} φ(q* + v + t) over a continuous offset t
        within [-0.5, 0.5]^3 using trilinear interpolation of the distance field.

        Returns the refined (float) position.
        """
        from scipy.optimize import minimize
        from scipy.ndimage import map_coordinates

        item_voxels = np.argwhere(item > 0).astype(float)  # (N, 3)
        if len(item_voxels) == 0:
            return None

        q = np.array(discrete_position, dtype=float)
        shape = np.array(tray_distance.shape, dtype=float)
        phi = tray_distance.astype(float)

        def energy(t):
            coords = q + item_voxels + t  # (N, 3)
            # Clamp to valid range for interpolation
            coords = np.clip(coords, 0, shape - 1)
            sampled = map_coordinates(phi, coords.T, order=1, mode='nearest')
            return float(np.sum(sampled))

        result = minimize(
            energy,
            x0=[0.0, 0.0, 0.0],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * 3,
        )
        refined = tuple((q + result.x).tolist())
        return refined

    def _compute_distance_field(self, tray: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance field using scipy.

        scipy.ndimage.distance_transform_edt gives exact L2 distances,
        whereas the C++ sweep approximates L1/Chebyshev distance.
        """
        from scipy.ndimage import distance_transform_edt
        field = distance_transform_edt(tray == 0)
        return np.round(field).astype(np.int32)

    def _calculate_stats(
        self, tray: np.ndarray
    ) -> Tuple[float, int, Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
        """Calculate packing statistics."""
        occupied = np.sum(tray > 0)
        total_volume = int(occupied)

        if occupied == 0:
            return 0.0, 0, None

        # Find bounding box of occupied voxels
        occupied_indices = np.argwhere(tray > 0)
        bbox_min = tuple(occupied_indices.min(axis=0).tolist())
        bbox_max = tuple(occupied_indices.max(axis=0).tolist())
        bbox_dims = tuple(mx - mn + 1 for mn, mx in zip(bbox_min, bbox_max))
        bbox_volume = np.prod(bbox_dims)

        density = float(occupied / bbox_volume) if bbox_volume > 0 else 0.0

        return density, total_volume, (bbox_min, bbox_max)

    def __repr__(self) -> str:
        return (
            f"BinPacker(tray_size={self.tray_size}, "
            f"voxel_resolution={self.voxel_resolution}, "
            f"num_orientations={self.num_orientations}, "
            f"height_penalty={self.height_penalty}, "
            f"continuous_refinement={self.continuous_refinement}, "
            f"support_threshold={self.support_threshold})"
        )
