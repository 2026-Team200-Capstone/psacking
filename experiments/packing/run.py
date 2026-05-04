#!/usr/bin/env python3
"""
3D Bin Packing — 임의 공간 메시에 STL 아이템 패킹
=====================================================

아래 '설정' 블록만 바꾸고 실행하면 됩니다.

사용법:
    python experiments/packing/run.py

Blender에서 실행:
    blender --background --python experiments/packing/run.py
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Blender 환경에서만 sys.path 주입
if "bpy" in sys.modules:
    for _site in ["/usr/local/lib/python3.10/dist-packages", "/workspace"]:
        if _site not in sys.path:
            sys.path.insert(0, _site)

_HERE     = Path(__file__).parent
INPUT_DIR = _HERE.parent / "input_meshes"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ 설정 — 여기만 바꾸면 됩니다
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPACE_FILE = INPUT_DIR / "abstracted_trunk.stl"  # 패킹 공간 메시 (.obj/.stl/.ply 등)
RESOLUTION      = 128    # 복셀화 해상도 (가장 긴 축 기준 최대 복셀 수)
NUM_ORIENTATIONS = 6     # 아이템 방향 수 (1 / 4 / 6 / 24)
HEIGHT_PENALTY  = 50.0   # 높이 페널티 (클수록 아이템이 낮게 쌓임)
OUTPUT_DIR      = None   # 결과 저장 경로 (None → results/<space_stem>/ 자동 생성)

# 아이템 정의: INPUT_DIR 아래 test_box*.stl 파일을 모두 배치 대상으로 사용합니다.
# 각 타입은 최소 1개 이상 포함하고, 총 아이템 점유 볼륨이 내부공간의
# TARGET_ITEM_VOLUME_RATIO 배가 될 때까지 랜덤 타입을 골라 개수를 늘립니다.
ITEM_MESH_PATTERN = "test_box*.stl"
ITEM_COUNT_RANDOM_SEED = None  # None이면 매 실행마다 다른 랜덤 개수
MIN_ITEM_COUNT_PER_TYPE = 1
INITIAL_ITEM_COUNT_RANGE = (1, 4)
TARGET_ITEM_VOLUME_RATIO = 1.25  # 100% 초과로 넣어 일부 아이템은 배치 실패하도록 함

COLORS = [
    (0.90, 0.25, 0.20, 1.0),
    (0.20, 0.75, 0.30, 1.0),
    (0.20, 0.40, 0.90, 1.0),
    (0.95, 0.70, 0.10, 1.0),
    (0.70, 0.20, 0.80, 1.0),
    (0.10, 0.80, 0.80, 1.0),
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _natural_sort_key(path):
    """test_box2가 test_box10보다 먼저 오도록 파일명을 정렬합니다."""
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", Path(path).stem)
    ]


def discover_item_mesh_types(input_dir, pattern):
    item_paths = sorted(Path(input_dir).glob(pattern), key=_natural_sort_key)
    if not item_paths:
        raise FileNotFoundError(f"아이템 STL 파일 없음: {Path(input_dir) / pattern}")
    return [(path.stem, path) for path in item_paths]


def _make_item_count_rng(seed):
    resolved_seed = int(seed) if seed is not None else time.time_ns()
    return np.random.default_rng(resolved_seed), resolved_seed


def build_mesh_items(voxelizer, item_mesh_types, pitch, free_volume, target_ratio,
                     rng, min_count_per_type, initial_count_range):
    """
    STL 아이템을 같은 pitch로 복셀화하고, 목표 볼륨까지 타입을 섞어 복제합니다.

    meta tuple:
      (type_name, shape, type_idx, mesh_path, voxel_count, voxel_info)
    item_types tuple:
      (type_name, shape, count, mesh_path, voxel_count, voxel_info)
    """
    if target_ratio <= 1.0:
        raise ValueError("TARGET_ITEM_VOLUME_RATIO는 1.0보다 커야 배치 실패가 보장됩니다.")

    min_count_per_type = max(1, int(min_count_per_type))
    initial_low, initial_high = (int(initial_count_range[0]), int(initial_count_range[1]))
    if initial_low < 1 or initial_high < initial_low:
        raise ValueError("INITIAL_ITEM_COUNT_RANGE는 (1 이상, 하한 <= 상한)이어야 합니다.")

    prototypes = []
    for type_idx, (type_name, mesh_path) in enumerate(item_mesh_types):
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"아이템 파일 없음: {mesh_path}")

        grid, info = voxelizer.voxelize_item(mesh_path, pitch)
        if grid is None or info is None:
            raise RuntimeError(f"아이템 복셀화 실패: {mesh_path}")

        grid = (grid > 0).astype(np.int32)
        voxel_count = int(np.sum(grid > 0))
        if voxel_count <= 0:
            raise RuntimeError(f"아이템 복셀 수가 0입니다: {mesh_path}")

        prototypes.append({
            "type_name": type_name,
            "mesh_path": mesh_path,
            "grid": grid,
            "shape": tuple(int(v) for v in grid.shape),
            "voxel_count": voxel_count,
            "voxel_info": info,
            "type_idx": type_idx,
        })

    target_volume = max(1, int(np.ceil(free_volume * target_ratio)))
    counts = rng.integers(
        initial_low, initial_high + 1, size=len(prototypes)
    ).astype(int).tolist()
    counts = [max(min_count_per_type, count) for count in counts]
    total_volume = sum(
        proto["voxel_count"] * count
        for proto, count in zip(prototypes, counts)
    )
    while total_volume < target_volume:
        proto_idx = int(rng.integers(0, len(prototypes)))
        counts[proto_idx] += 1
        total_volume += prototypes[proto_idx]["voxel_count"]

    items = []
    meta = []
    item_types = []
    for proto, count in zip(prototypes, counts):
        item_types.append((
            proto["type_name"],
            proto["shape"],
            count,
            proto["mesh_path"],
            proto["voxel_count"],
            proto["voxel_info"],
        ))
        for _ in range(count):
            items.append(proto["grid"])
            meta.append((
                proto["type_name"],
                proto["shape"],
                proto["type_idx"],
                proto["mesh_path"],
                proto["voxel_count"],
                proto["voxel_info"],
            ))

    return items, meta, item_types


def voxelize_space(space_path, resolution):
    """메시를 복셀화하고 (voxelizer, initial_tray, pitch, tray_size, voxel_origin) 반환."""
    from spectral_packer.voxelizer import Voxelizer

    voxelizer = Voxelizer(resolution=resolution)
    initial_tray, pitch, tray_size, voxel_origin = voxelizer.voxelize_space(space_path)
    return voxelizer, initial_tray, pitch, tray_size, voxel_origin


def pack_items(items, initial_tray, tray_size, pitch, resolution, num_orientations,
               height_penalty):
    """아이템을 voxel tray에 패킹하고 (result, packer) 반환."""
    from spectral_packer import BinPacker

    packer = BinPacker(
        tray_size=tray_size,
        voxel_resolution=resolution,
        num_orientations=num_orientations,
        height_penalty=height_penalty,
        pitch=pitch,
    )
    result = packer.pack_voxels(items, initial_tray=initial_tray)
    return result, packer


# ── 회전 행렬 (run.py 내부에서만 사용) ───────────────────────────────────────
# get_orientations(item, 6) 순서와 일치
_ROT6 = [
    np.array([[1, 0, 0], [0,  1,  0], [0,  0,  1]], dtype=float),  # 0: I
    np.array([[1, 0, 0], [0,  0, -1], [0,  1,  0]], dtype=float),  # 1: RX
    np.array([[1, 0, 0], [0, -1,  0], [0,  0, -1]], dtype=float),  # 2: RX²
    np.array([[1, 0, 0], [0,  0,  1], [0, -1,  0]], dtype=float),  # 3: RX³
    np.array([[0, 0, 1], [0,  1,  0], [-1, 0,  0]], dtype=float),  # 4: RY
    np.array([[0, 0,-1], [0,  1,  0], [1,  0,  0]], dtype=float),  # 5: RY³
]

def _build_rot24():
    """get_24_orientations() 순서와 일치하는 24개 회전 행렬 생성."""
    I  = np.eye(3, dtype=float)
    RX = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
    RY = np.array([[0,0,1],[0,1,0],[-1,0,0]], float)
    RZ = np.array([[0,-1,0],[1,0,0],[0,0,1]], float)
    RX2, RX3 = RX@RX, RX@RX@RX
    RY3       = RY@RY@RY
    RZ2, RZ3  = RZ@RZ, RZ@RZ@RZ
    mats = []
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz.copy())
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz @ RX)
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz @ RX2)
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz @ RX3)
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz @ RY)
    for Rz in [I, RZ, RZ2, RZ3]:
        mats.append(Rz @ RY3)
    return mats

_ROT24 = _build_rot24()


def get_rotation_matrix(orientation_idx, num_orientations=6):
    rots = _ROT6 if num_orientations <= 6 else _ROT24
    if orientation_idx >= len(rots):
        raise ValueError(
            f"orientation_index {orientation_idx} >= {len(rots)} "
            f"(num_orientations={num_orientations})"
        )
    return rots[orientation_idx]


def get_rotated_shape(original_shape, orientation_idx, num_orientations=6):
    R = get_rotation_matrix(orientation_idx, num_orientations)
    # abs(R) @ shape → 90° 회전 후 축 정렬 바운딩박스 크기
    rotated = np.abs(R) @ np.array(original_shape, dtype=float)
    return tuple(int(round(v)) for v in rotated)


def _mesh_transform_matrix(voxel_info, position, orientation_idx, voxel_origin,
                           num_orientations):
    R = get_rotation_matrix(orientation_idx, num_orientations)
    mesh_min = np.array(voxel_info.mesh_bounds_min, dtype=float)
    mesh_max = np.array(voxel_info.mesh_bounds_max, dtype=float)
    mesh_center = (mesh_min + mesh_max) / 2.0
    mesh_half_extents = (mesh_max - mesh_min) / 2.0
    rotated_half_extents = np.abs(R) @ mesh_half_extents

    final_center = (
        np.array(voxel_origin, dtype=float)
        + np.array(position, dtype=float) * voxel_info.pitch
        + rotated_half_extents
    )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R
    transform[:3, 3] = final_center - R @ mesh_center
    return transform


def _import_blender_mesh(bpy, mesh_path, name):
    suffix = mesh_path.suffix.lower()
    bpy.ops.object.select_all(action='DESELECT')

    if suffix == '.obj':
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path), axis_forward='Y', axis_up='Z')
    elif suffix == '.stl':
        if hasattr(bpy.ops.wm, 'stl_import'):
            bpy.ops.wm.stl_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif suffix == '.ply':
        if hasattr(bpy.ops.wm, 'ply_import'):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    elif suffix in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    else:
        raise ValueError(f"Blender에서 {suffix} 직접 임포트 미지원: {mesh_path}")

    imported_objects = list(bpy.context.selected_objects)
    if not imported_objects:
        raise RuntimeError(f"Blender 임포트 실패: {mesh_path}")
    if len(imported_objects) > 1:
        bpy.context.view_layer.objects.active = imported_objects[0]
        bpy.ops.object.join()

    obj = bpy.context.active_object or imported_objects[0]
    obj.name = name
    return obj


def export_blender(result, meta, pitch, voxel_origin, space_path, out_dir, colors,
                   num_orientations):
    try:
        import bpy
        import mathutils
    except ImportError:
        print(
            "\n[경고] bpy 없음 → Blender export 건너뜀\n"
            "실행 방법:\n"
            "  blender --background --python experiments/packing/run.py -- <space_mesh>"
        )
        return

    output_blend = out_dir / "packed_result.blend"
    print(f"\n[Blender 내보내기] → {output_blend}")
    t0 = time.perf_counter()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 컨테이너 메시 임포트 (axis 변환 없이 → trimesh 복셀 좌표계와 일치)
    print(f"  [Blender] voxel_origin={voxel_origin.tolist()}")
    try:
        container_objs = [_import_blender_mesh(bpy, space_path, "Container")]
    except Exception as e:
        print(f"  [경고] 컨테이너 메시 임포트 실패: {e}")
        container_objs = []

    for obj in container_objs:
        obj.name = "Container"
        mat = bpy.data.materials.new(name="Container_Mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs["Alpha"].default_value = 0.3
        mat.blend_method = 'BLEND'
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    placed = [p for p in result.placements if p.success]
    for i, p in enumerate(placed):
        type_name, _shape, type_idx, mesh_path, _voxel_count, voxel_info = meta[p.item_index]
        obj = _import_blender_mesh(bpy, mesh_path, f"Item_{i:03d}_{type_name}")
        obj.name = f"Item_{i:03d}_{type_name}"
        transform = _mesh_transform_matrix(
            voxel_info, p.position, p.orientation_index,
            voxel_origin, num_orientations,
        )
        obj.matrix_world = mathutils.Matrix(transform.tolist())

        color = colors[type_idx % len(colors)]
        mat = bpy.data.materials.new(name=f"Mat_{type_name}_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.35
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"  저장 완료! ({time.perf_counter() - t0:.2f}s)")


def save_csv(voxelizer, packer, result, meta, tray_size, voxel_origin, initial_tray,
             total_elapsed, out_dir):
    from spectral_packer.voxelizer import LogEntry

    free_vol   = int((initial_tray == 0).sum())
    placed_vol = sum(
        int(meta[p.item_index][4])
        for p in result.placements if p.success
    )
    fill_rate  = placed_vol / free_vol if free_vol > 0 else 0.0

    origin_str = "[" + ", ".join(f"{v:.4f}" for v in voxel_origin) + "]"
    entries = voxelizer.log + packer.log + [
        LogEntry(
            step="summary",
            duration_sec=0.0,
            success=True,
            notes=(
                f"placed={result.num_placed}/{len(result.placements)} "
                f"fill={fill_rate:.1%} "
                f"tray={tray_size} "
                f"placed_vol={placed_vol} free_vol={free_vol} "
                f"voxel_origin={origin_str}"
            ),
        ),
        LogEntry(step="total", duration_sec=total_elapsed, success=True),
    ]

    output_csv = out_dir / "log.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "duration_sec", "success", "notes"]
        )
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "step":         e.step,
                "duration_sec": f"{e.duration_sec:.3f}",
                "success":      e.success,
                "notes":        e.notes,
            })
    print(f"[로그] → {output_csv}")


def save_json(result, meta, item_types, tray_size, pitch, voxel_origin, initial_tray,
              space_path, num_orientations, out_dir, item_mesh_pattern=None,
              item_count_seed=None, target_item_volume_ratio=None):
    free_volume = int((initial_tray == 0).sum())
    data = {
        "space_name": space_path.stem,
        "space_file": space_path.name,
        "tray_size":  [int(v) for v in tray_size],
        "pitch":      float(pitch),
        "voxel_origin": [float(v) for v in voxel_origin],
        "num_orientations": num_orientations,
        "item_mesh_pattern": item_mesh_pattern,
        "item_count_seed": item_count_seed,
        "target_item_volume_ratio": target_item_volume_ratio,
        # 아이템이 실제로 배치 가능한 복셀 수 (채움률 분모)
        "container_free_volume": free_volume,
        "items_are_meshes": True,
        "item_types": [
            {
                "type_name": tn,
                "shape": [int(v) for v in sh],
                "count": int(cnt),
                "mesh_file": Path(mesh_path).name,
                "voxel_count": int(voxel_count),
            }
            for tn, sh, cnt, mesh_path, voxel_count, _info in item_types
        ],
        "colors": COLORS,
        "meta": [
            {
                "type_name": tn,
                "shape": [int(v) for v in sh],
                "type_idx": int(ti),
                "mesh_file": Path(mesh_path).name,
                "voxel_count": int(voxel_count),
            }
            for tn, sh, ti, mesh_path, voxel_count, _info in meta
        ],
        "placements": [
            {
                "item_index":       int(p.item_index),
                "position":         [int(v) for v in p.position] if p.position is not None else None,
                "orientation_index": int(p.orientation_index),
                "success":          bool(p.success),
                "score":            float(p.score) if p.score is not None else None,
                "volume":           int(p.volume),
            }
            for p in result.placements
        ],
    }

    # 컨테이너 메시 형상 저장 — visualize.py에서 실제 3D 형태로 렌더링
    try:
        import trimesh as _trimesh
        _mesh = _trimesh.load(str(space_path), force="mesh")
        if isinstance(_mesh, _trimesh.Trimesh):
            data["container_mesh_vertices"] = [
                [round(float(v), 6) for v in vert]
                for vert in _mesh.vertices
            ]
            data["container_mesh_faces"] = _mesh.faces.tolist()
            print(f"[JSON] 컨테이너 메시: {len(_mesh.vertices)}개 정점, {len(_mesh.faces)}개 삼각형")
        else:
            print("[경고] 컨테이너 메시가 단일 Trimesh가 아님 → 메시 저장 건너뜀")
    except Exception as _e:
        print(f"[경고] 컨테이너 메시 저장 실패 (visualize에서 형상이 보이지 않을 수 있음): {_e}")

    # 아이템 메시 형상 저장 — visualize.py에서 실제 STL 형상으로 렌더링
    try:
        import trimesh as _trimesh
        item_meshes = {}
        for type_name, _shape, _count, mesh_path, _voxel_count, info in item_types:
            _mesh = _trimesh.load(str(mesh_path), force="mesh")
            if isinstance(_mesh, _trimesh.Trimesh):
                item_meshes[type_name] = {
                    "mesh_file": Path(mesh_path).name,
                    "vertices": [
                        [round(float(v), 6) for v in vert]
                        for vert in _mesh.vertices
                    ],
                    "faces": _mesh.faces.tolist(),
                    "bounds_min": [float(v) for v in info.mesh_bounds_min],
                    "bounds_max": [float(v) for v in info.mesh_bounds_max],
                }
        data["item_meshes"] = item_meshes
        print(f"[JSON] 아이템 메시: {len(item_meshes)}개 타입 저장")
    except Exception as _e:
        print(f"[경고] 아이템 메시 저장 실패 (visualize에서 박스로 표시됨): {_e}")

    output_json = out_dir / "packed_result.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[JSON] → {output_json}")


def main():
    t0 = time.perf_counter()

    space_path = Path(SPACE_FILE)
    if not space_path.exists():
        raise FileNotFoundError(f"공간 파일 없음: {space_path}")

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else _HERE / "results" / space_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[공간] {space_path}")
    print(f"[설정] resolution={RESOLUTION}  orientations={NUM_ORIENTATIONS}"
          f"  height_penalty={HEIGHT_PENALTY}")
    print(f"[출력] {out_dir}")
    item_mesh_types = discover_item_mesh_types(INPUT_DIR, ITEM_MESH_PATTERN)
    item_count_rng, item_count_seed = _make_item_count_rng(ITEM_COUNT_RANDOM_SEED)
    print(f"[아이템 파일] {ITEM_MESH_PATTERN} → "
          f"{', '.join(Path(path).name for _name, path in item_mesh_types)}")
    print(f"[랜덤] item_count_seed={item_count_seed}")

    # ── 1단계: 복셀화 → tray_size 확정 ────────────────────────────────────
    print("\n[복셀화 중...]")
    voxelizer, initial_tray, pitch, tray_size, voxel_origin = voxelize_space(
        space_path, RESOLUTION,
    )

    # ── 2단계: 같은 pitch로 STL 아이템 복셀화 및 개수 자동 증량 ─────────────
    free_vol = int((initial_tray == 0).sum())
    items, meta, item_types = build_mesh_items(
        voxelizer,
        item_mesh_types,
        pitch,
        free_vol,
        TARGET_ITEM_VOLUME_RATIO,
        item_count_rng,
        MIN_ITEM_COUNT_PER_TYPE,
        INITIAL_ITEM_COUNT_RANGE,
    )

    total_item_vol = sum(int(m[4]) for m in meta)
    print(f"\n[아이템] {len(items)}개  총 점유 볼륨 {total_item_vol:,} vox"
          f"  (내부공간 대비 {total_item_vol / max(1, free_vol):.1%})")
    for type_name, shape, count, mesh_path, voxel_count, _info in item_types:
        print(
            f"         {type_name:12s} {str(shape):18s} × {count:3d}개 "
            f"= {voxel_count * count:,} vox  ({Path(mesh_path).name})"
        )

    # ── 3단계: 패킹 ─────────────────────────────────────────────────────────
    print("\n[패킹 시작]")
    result, packer = pack_items(
        items, initial_tray, tray_size, pitch,
        RESOLUTION, NUM_ORIENTATIONS, HEIGHT_PENALTY,
    )

    placed_vol = sum(
        int(meta[p.item_index][4])
        for p in result.placements if p.success
    )
    fill_rate = placed_vol / free_vol if free_vol > 0 else 0.0
    print(f"[결과] 배치 성공: {result.num_placed}/{len(items)}개")
    print(f"[결과] 채움률:    {fill_rate:.1%}  (내부공간 {free_vol:,} vox 기준)")
    print(f"[결과] tray:      {tray_size}  pitch={pitch:.4f}")

    failed = [p for p in result.placements if not p.success]
    if failed:
        print(f"\n[실패] 배치 실패 아이템 {len(failed)}개:")
        from collections import Counter
        failed_types = Counter(meta[p.item_index][0] for p in failed)
        for type_name, count in failed_types.most_common():
            shape = next(m[1] for m in meta if m[0] == type_name)
            print(f"  - {type_name} {shape}  × {count}개")

    export_blender(result, meta, pitch, voxel_origin, space_path, out_dir, COLORS,
                   NUM_ORIENTATIONS)

    total_elapsed = time.perf_counter() - t0
    save_csv(voxelizer, packer, result, meta, tray_size, voxel_origin, initial_tray,
             total_elapsed, out_dir)
    save_json(result, meta, item_types, tray_size, pitch, voxel_origin, initial_tray,
              space_path, NUM_ORIENTATIONS, out_dir,
              item_mesh_pattern=ITEM_MESH_PATTERN,
              item_count_seed=item_count_seed,
              target_item_volume_ratio=TARGET_ITEM_VOLUME_RATIO)

    json_path = out_dir / "packed_result.json"
    print(f"\n[완료] {total_elapsed:.1f}s")
    print(f"[시각화] python experiments/packing/visualize.py {json_path}")


if __name__ == "__main__":
    main()
