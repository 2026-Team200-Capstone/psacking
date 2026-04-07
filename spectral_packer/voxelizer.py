"""
Voxelization utilities for converting meshes to voxel grids.

This module provides utilities for converting 3D triangle meshes to
voxel grids suitable for the spectral packing algorithm.

Examples
--------
>>> from spectral_packer import Voxelizer
>>> vox = Voxelizer(resolution=128)
>>> grid = vox.voxelize_file("model.stl")
>>> print(f"Voxel grid shape: {grid.shape}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .mesh_io import load_mesh


@dataclass
class LogEntry:
    """단일 작업의 타이밍 로그 항목."""
    step: str
    duration_sec: float
    success: bool
    notes: str = ""


@dataclass
class VoxelizationInfo:
    """Metadata for mapping voxel coordinates back to mesh coordinates.

    Attributes
    ----------
    mesh_path : Path
        Path to the original mesh file.
    mesh_bounds_min : np.ndarray
        Minimum corner of the mesh bounding box (shape: (3,)).
    mesh_bounds_max : np.ndarray
        Maximum corner of the mesh bounding box (shape: (3,)).
    pitch : float
        Size of each voxel in mesh coordinate units.
    voxel_shape : Tuple[int, int, int]
        Shape of the resulting voxel grid (x, y, z).
    """

    mesh_path: Path
    mesh_bounds_min: np.ndarray
    mesh_bounds_max: np.ndarray
    pitch: float
    voxel_shape: Tuple[int, int, int]


class Voxelizer:
    """
    Mesh to voxel grid converter.

    Parameters
    ----------
    resolution : int, default 128
        Maximum resolution along the longest axis.
    pitch : float, optional
        Fixed voxel size. If None, computed from resolution.
    verbose : bool, default False
        If True, print timing information and store log entries.
    """

    def __init__(
        self,
        resolution: int = 128,
        pitch: Optional[float] = None,
        verbose: bool = False,
    ):
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if pitch is not None and pitch <= 0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        self.resolution = resolution
        self.pitch = pitch
        self.verbose = verbose
        self.log: List[LogEntry] = []

    def _record(self, step: str, duration: float, success: bool = True, notes: str = ""):
        entry = LogEntry(step=step, duration_sec=duration, success=success, notes=notes)
        self.log.append(entry)
        if self.verbose:
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {step}: {duration:.2f}s  {notes}")

    def voxelize_space(self, path: Union[str, Path]) -> Tuple[np.ndarray, float, tuple]:
        """
        공간 메시를 복셀화해 initial_tray(장애물 그리드), pitch, tray_size를 반환합니다.

        내부(1) → free(0), 외부(0) → obstacle(1)로 반전해 반환합니다.

        Parameters
        ----------
        path : str or Path
            공간으로 사용할 메시 파일 경로.

        Returns
        -------
        initial_tray : np.ndarray[int32]
            0 = 아이템 배치 가능, 1 = 장애물(공간 외부)
        pitch : float
            voxel 1칸의 실세계 크기
        tray_size : tuple[int, int, int]
            initial_tray의 shape
        """
        import trimesh

        path = Path(path)
        t0 = time.perf_counter()

        vertices, faces = load_mesh(path, validate=True, repair=True)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        max_extent = float(mesh.extents.max())
        pitch = self.pitch if self.pitch is not None else max_extent / (self.resolution - 1)

        if self.verbose:
            print(f"\n[공간 복셀화] {path.name}")
            print(f"  바운딩박스 크기: {mesh.extents.tolist()}")
            print(f"  pitch          : {pitch:.4f}  (해상도 {self.resolution})")

        try:
            filled = mesh.voxelized(pitch=pitch).fill().matrix.astype("int32")
        except Exception as e:
            filled = mesh.voxelized(pitch=pitch).matrix.astype("int32")
            if self.verbose:
                print(f"  [경고] fill() 실패 → 표면 복셀 사용: {e}")

        initial_tray = (1 - filled).astype("int32")
        tray_size = filled.shape

        elapsed = time.perf_counter() - t0

        if self.verbose:
            print(f"  트레이 크기    : {tray_size}")
            print(f"  내부 free 비율 : {filled.mean():.1%}")

        self._record("space_voxelization", elapsed, notes=f"tray={tray_size}")
        return initial_tray, pitch, tray_size

    def voxelize_item(
        self,
        path: Union[str, Path],
        pitch: float,
    ) -> Tuple[Optional[np.ndarray], Optional[VoxelizationInfo]]:
        """
        아이템 메시를 지정된 pitch로 복셀화합니다.

        Parameters
        ----------
        path : str or Path
            메시 파일 경로.
        pitch : float
            voxel 크기 (공간과 동일한 값 사용).

        Returns
        -------
        grid : np.ndarray or None
            복셀 그리드. 실패 시 None.
        info : VoxelizationInfo or None
            Blender 내보내기용 메타데이터. 실패 시 None.
        """
        import trimesh

        path = Path(path)
        t0 = time.perf_counter()

        try:
            vertices, faces = load_mesh(path, validate=True, repair=True)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            bounds = mesh.bounds

            vox = mesh.voxelized(pitch=pitch)
            try:
                grid = vox.fill().matrix.astype("int32")
            except Exception:
                grid = vox.matrix.astype("int32")
                if self.verbose:
                    print(f"  [경고] {path.name} fill() 실패 → 표면 복셀 사용")

            info = VoxelizationInfo(
                mesh_path=path,
                mesh_bounds_min=bounds[0].copy(),
                mesh_bounds_max=bounds[1].copy(),
                pitch=pitch,
                voxel_shape=grid.shape,
            )

            elapsed = time.perf_counter() - t0
            notes = f"shape={grid.shape} voxels={int(grid.sum())}"
            self._record(f"item_voxelization:{path.name}", elapsed, True, notes)

            if self.verbose:
                print(f"  {path.name:40s} → {grid.shape}  ({int(grid.sum())} voxels)")

            return grid, info

        except Exception as e:
            elapsed = time.perf_counter() - t0
            self._record(f"item_voxelization:{path.name}", elapsed, False, str(e))
            if self.verbose:
                print(f"  [경고] {path.name} 복셀화 실패: {e} → 건너뜀")
            return None, None

    def voxelize_file(
        self,
        path: Union[str, Path],
        validate: bool = True,
        repair: bool = True,
    ) -> np.ndarray:
        """Voxelize a mesh from a file."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".stl" and self.pitch is None:
            try:
                from . import voxelize_stl
                return voxelize_stl(str(path), self.resolution)
            except Exception:
                pass

        vertices, faces = load_mesh(path, validate=validate, repair=repair)
        return self.voxelize_mesh(vertices, faces)

    def voxelize_file_with_info(
        self,
        path: Union[str, Path],
        validate: bool = True,
        repair: bool = True,
    ) -> Tuple[np.ndarray, VoxelizationInfo]:
        """Voxelize a mesh from a file and return metadata for coordinate mapping."""
        path = Path(path)

        vertices, faces = load_mesh(path, validate=validate, repair=repair)

        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required. Install with: pip install trimesh")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        bounds = mesh.bounds
        mesh_bounds_min = bounds[0].copy()
        mesh_bounds_max = bounds[1].copy()

        extents = mesh.extents
        max_extent = extents.max()
        if self.pitch is not None:
            max_voxels = max_extent / self.pitch
            if max_voxels > self.resolution:
                import warnings
                pitch = max_extent / (self.resolution - 1)
                warnings.warn(
                    f"Object would exceed resolution {self.resolution} with "
                    f"pitch={self.pitch} ({max_voxels:.0f} voxels needed). "
                    f"Falling back to pitch={pitch:.4f}.",
                    UserWarning,
                )
            else:
                pitch = self.pitch
        else:
            pitch = max_extent / (self.resolution - 1)

        try:
            voxelized = mesh.voxelized(pitch=pitch)
            voxelized = voxelized.fill()
            grid = voxelized.matrix.astype(np.int32)
        except Exception as e:
            import warnings
            warnings.warn(f"Trimesh voxelization failed: {e}. Using fallback method.")
            grid = self._fallback_voxelize(mesh, pitch)

        info = VoxelizationInfo(
            mesh_path=path,
            mesh_bounds_min=mesh_bounds_min,
            mesh_bounds_max=mesh_bounds_max,
            pitch=pitch,
            voxel_shape=grid.shape,
        )

        return grid, info

    def voxelize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        fill: bool = True,
    ) -> np.ndarray:
        """Voxelize a mesh from vertex and face arrays."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required. Install with: pip install trimesh")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        extents = mesh.extents
        max_extent = extents.max()
        if self.pitch is not None:
            max_voxels = max_extent / self.pitch
            if max_voxels > self.resolution:
                import warnings
                pitch = max_extent / (self.resolution - 1)
                warnings.warn(
                    f"Object would exceed resolution {self.resolution} with "
                    f"pitch={self.pitch} ({max_voxels:.0f} voxels needed). "
                    f"Falling back to pitch={pitch:.4f}.",
                    UserWarning,
                )
            else:
                pitch = self.pitch
        else:
            pitch = max_extent / (self.resolution - 1)

        try:
            voxelized = mesh.voxelized(pitch=pitch)
            if fill:
                voxelized = voxelized.fill()
            grid = voxelized.matrix.astype(np.int32)
        except Exception as e:
            import warnings
            warnings.warn(f"Trimesh voxelization failed: {e}. Using fallback method.")
            grid = self._fallback_voxelize(mesh, pitch)

        return grid

    def _fallback_voxelize(self, mesh, pitch: float) -> np.ndarray:
        """Fallback voxelization using point containment."""
        bounds = mesh.bounds
        grid_size = np.ceil((bounds[1] - bounds[0]) / pitch).astype(int) + 1
        grid_size = np.minimum(grid_size, self.resolution)

        x = np.linspace(bounds[0, 0], bounds[1, 0], grid_size[0])
        y = np.linspace(bounds[0, 1], bounds[1, 1], grid_size[1])
        z = np.linspace(bounds[0, 2], bounds[1, 2], grid_size[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        try:
            inside = mesh.contains(points)
        except Exception:
            inside = np.zeros(len(points), dtype=bool)

        return inside.reshape(grid_size).astype(np.int32)

    def __repr__(self) -> str:
        return (
            f"Voxelizer(resolution={self.resolution}, pitch={self.pitch}, "
            f"verbose={self.verbose})"
        )
