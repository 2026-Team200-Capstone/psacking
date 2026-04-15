#!/usr/bin/env python3
"""
3D Bin Packing — 임의 공간 메시에 박스 아이템 패킹
=====================================================

아래 '설정' 블록만 바꾸고 실행하면 됩니다.

사용법:
    python experiments/packing/run.py

Blender에서 실행:
    blender --background --python experiments/packing/run.py
"""

import csv
import json
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

SPACE_FILE      = INPUT_DIR / "abstracted_trunk.ply"  # 패킹 공간 메시 (.obj/.stl/.ply 등)
RESOLUTION      = 128    # 복셀화 해상도 (가장 긴 축 기준 최대 복셀 수)
NUM_ORIENTATIONS = 6     # 아이템 방향 수 (1 / 4 / 6 / 24)
HEIGHT_PENALTY  = 50.0   # 높이 페널티 (클수록 아이템이 낮게 쌓임)
OUTPUT_DIR      = None   # 결과 저장 경로 (None → results/<space_stem>/ 자동 생성)

# 아이템 정의 (컨테이너 각 축 tray 크기에 대한 비율)
# (이름, (x비율, y비율, z비율), 개수)
# → 복셀화 후 실제 tray_size에 곱해 자동으로 절대 복셀 크기로 변환
# 총 이론 볼륨 ≈ 395,508 vox → 이론 채움률 73% (실제: 일부 배치 실패 예상)
ITEM_TYPES_RELATIVE = [
    ("large_box",  (0.60, 0.30, 0.28),  3),
    ("medium_box", (0.36, 0.22, 0.26),  4),
    ("flat_box",   (0.55, 0.25, 0.08),  5),
    ("tall_box",   (0.20, 0.14, 0.44),  4),
    ("small_box",  (0.25, 0.14, 0.11),  6),
    ("tiny_box",   (0.14, 0.09, 0.09),  8),
]

COLORS = [
    (0.90, 0.25, 0.20, 1.0),
    (0.20, 0.75, 0.30, 1.0),
    (0.20, 0.40, 0.90, 1.0),
    (0.95, 0.70, 0.10, 1.0),
    (0.70, 0.20, 0.80, 1.0),
    (0.10, 0.80, 0.80, 1.0),
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def scale_item_types(item_types_relative, tray_size):
    """
    상대 비율 기반 아이템 정의를 실제 복셀 크기로 변환합니다.

    Parameters
    ----------
    item_types_relative : list of (name, (fx, fy, fz), count)
    tray_size           : tuple of (nx, ny, nz)

    Returns
    -------
    list of (name, (vx, vy, vz), count)  — 복셀 단위 절대 크기
    """
    result = []
    for name, fracs, count in item_types_relative:
        shape = tuple(max(1, int(round(f * d))) for f, d in zip(fracs, tray_size))
        result.append((name, shape, count))
    return result


def make_box(shape):
    return np.ones(shape, dtype=np.int32)


def build_items(item_types):
    items = []
    meta  = []
    for type_idx, (type_name, shape, count) in enumerate(item_types):
        for _ in range(count):
            items.append(make_box(shape))
            meta.append((type_name, shape, type_idx))
    return items, meta


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


def get_rotated_shape(original_shape, orientation_idx, num_orientations=6):
    rots = _ROT6 if num_orientations <= 6 else _ROT24
    if orientation_idx >= len(rots):
        raise ValueError(
            f"orientation_index {orientation_idx} >= {len(rots)} "
            f"(num_orientations={num_orientations})"
        )
    R = rots[orientation_idx]
    # abs(R) @ shape → 90° 회전 후 축 정렬 바운딩박스 크기
    rotated = np.abs(R) @ np.array(original_shape, dtype=float)
    return tuple(int(round(v)) for v in rotated)


def export_blender(result, meta, pitch, voxel_origin, space_path, out_dir, colors,
                   num_orientations):
    try:
        import bpy
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
    suffix = space_path.suffix.lower()
    bpy.ops.object.select_all(action='DESELECT')
    if suffix == '.obj':
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=str(space_path), forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.import_scene.obj(filepath=str(space_path), axis_forward='Y', axis_up='Z')
    elif suffix == '.stl':
        bpy.ops.import_mesh.stl(filepath=str(space_path))
    elif suffix == '.ply':
        bpy.ops.import_mesh.ply(filepath=str(space_path))
    elif suffix in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=str(space_path))
    else:
        print(f"  [경고] Blender에서 {suffix} 직접 임포트 미지원, 컨테이너 메시 생략")

    container_objs = list(bpy.context.selected_objects)
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
        type_name, original_shape, type_idx = meta[p.item_index]
        rshape = get_rotated_shape(original_shape, p.orientation_index, num_orientations)
        center = voxel_origin + (np.array(p.position) + (np.array(rshape) - 1) / 2.0) * pitch

        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.name = f"Item_{i:03d}_{type_name}"
        obj.scale = (rshape[0] * pitch, rshape[1] * pitch, rshape[2] * pitch)
        obj.location = tuple(float(c) for c in center)
        bpy.ops.object.transform_apply(scale=True)

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
        int(np.prod(meta[p.item_index][1]))
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
              space_path, num_orientations, out_dir):
    free_volume = int((initial_tray == 0).sum())
    data = {
        "space_name": space_path.stem,
        "space_file": space_path.name,
        "tray_size":  [int(v) for v in tray_size],
        "pitch":      float(pitch),
        "voxel_origin": [float(v) for v in voxel_origin],
        "num_orientations": num_orientations,
        # 아이템이 실제로 배치 가능한 복셀 수 (채움률 분모)
        "container_free_volume": free_volume,
        "item_types": [
            {"type_name": tn, "shape": [int(v) for v in sh], "count": int(cnt)}
            for tn, sh, cnt in item_types
        ],
        "colors": COLORS,
        "meta": [
            {"type_name": tn, "shape": [int(v) for v in sh], "type_idx": int(ti)}
            for tn, sh, ti in meta
        ],
        "placements": [
            {
                "item_index":       int(p.item_index),
                "position":         [int(v) for v in p.position] if p.position is not None else None,
                "orientation_index": int(p.orientation_index),
                "success":          bool(p.success),
                "score":            float(p.score) if p.score is not None else None,
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

    # ── 1단계: 복셀화 → tray_size 확정 ────────────────────────────────────
    print("\n[복셀화 중...]")
    voxelizer, initial_tray, pitch, tray_size, voxel_origin = voxelize_space(
        space_path, RESOLUTION,
    )

    # ── 2단계: tray_size 기반 아이템 크기 자동 결정 ────────────────────────
    item_types = scale_item_types(ITEM_TYPES_RELATIVE, tray_size)
    items, meta = build_items(item_types)

    total_item_vol = sum(int(np.prod(m[1])) for m in meta)
    print(f"\n[아이템] {len(items)}개  총 볼륨 {total_item_vol:,} vox"
          f"  (tray {tray_size} 기준 자동 크기)")
    for type_name, shape, count in item_types:
        vol = int(np.prod(shape))
        print(f"         {type_name:12s} {str(shape):18s} × {count:2d}개 = {vol*count:,} vox")

    # ── 3단계: 패킹 ─────────────────────────────────────────────────────────
    print("\n[패킹 시작]")
    result, packer = pack_items(
        items, initial_tray, tray_size, pitch,
        RESOLUTION, NUM_ORIENTATIONS, HEIGHT_PENALTY,
    )

    placed_vol = sum(
        int(np.prod(meta[p.item_index][1]))
        for p in result.placements if p.success
    )
    free_vol  = int((initial_tray == 0).sum())
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
              space_path, NUM_ORIENTATIONS, out_dir)

    json_path = out_dir / "packed_result.json"
    print(f"\n[완료] {total_elapsed:.1f}s")
    print(f"[시각화] python experiments/packing/visualize.py {json_path}")


if __name__ == "__main__":
    main()
