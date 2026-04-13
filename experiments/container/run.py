#!/usr/bin/env python3
"""
실험: Container 공간에 합성 박스 아이템 3D Bin Packing
======================================================

12281_Container_v2_L2.obj 내부 공간에 numpy로 정의한 다양한 크기의 박스를
패킹하고 결과를 results/packed_result.blend 로 내보냅니다.

사용법:
    # 패킹만 (bpy 없어도 됨)
    python experiments/container/run.py

    # 패킹 + .blend 생성
    blender --background --python experiments/container/run.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Blender가 sys.path를 조작하는 경우 대비
for _site in ["/usr/local/lib/python3.10/dist-packages", "/workspace"]:
    if _site not in sys.path:
        sys.path.insert(0, _site)

_HERE        = Path(__file__).parent
INPUT_DIR    = _HERE.parent / "input_meshes"
OUTPUT_BLEND = _HERE / "results" / "packed_result.blend"
OUTPUT_CSV   = _HERE / "results" / "log.csv"
OUTPUT_JSON  = _HERE / "results" / "packed_result.json"

# ── 설정 ──────────────────────────────────────────────────────────────────────
SPACE_FILE        = "coffee_cup_obj.obj"
SPACE_RESOLUTION  = 128
NUM_ORIENTATIONS  = 6
HEIGHT_PENALTY    = 50.0

# 아이템 정의: (이름, 복셀 크기 (x,y,z), 개수)
# pitch는 컨테이너 복셀화 후 결정 → 아이템 크기도 pitch 기준
# 커피컵 내부가 약 90×90×127 복셀이므로 아이템은 그에 맞게 설정
ITEM_TYPES = [
    ("large_box",  (20, 15, 10),  2),
    ("medium_box", (12, 10,  8),  4),
    ("flat_box",   (18, 12,  3),  4),
    ("tall_box",   ( 6,  6, 15),  4),
    ("small_box",  ( 8,  6,  4),  6),
    ("tiny_box",   ( 5,  4,  3),  8),
]

# 아이템 타입별 색상 (RGBA)
COLORS = [
    (0.90, 0.25, 0.20, 1.0),
    (0.20, 0.75, 0.30, 1.0),
    (0.20, 0.40, 0.90, 1.0),
    (0.95, 0.70, 0.10, 1.0),
    (0.70, 0.20, 0.80, 1.0),
    (0.10, 0.80, 0.80, 1.0),
]
# ─────────────────────────────────────────────────────────────────────────────


def make_box(shape):
    return np.ones(shape, dtype=np.int32)


def build_items():
    items = []
    meta  = []
    for type_idx, (type_name, shape, count) in enumerate(ITEM_TYPES):
        for _ in range(count):
            items.append(make_box(shape))
            meta.append((type_name, shape, type_idx))
    return items, meta


def run_packing(items, space_path):
    from spectral_packer import BinPacker
    from spectral_packer.voxelizer import Voxelizer

    voxelizer = Voxelizer(resolution=SPACE_RESOLUTION)
    initial_tray, pitch, tray_size, voxel_origin = voxelizer.voxelize_space(space_path)

    packer = BinPacker(
        tray_size=tray_size,
        voxel_resolution=SPACE_RESOLUTION,
        num_orientations=NUM_ORIENTATIONS,
        height_penalty=HEIGHT_PENALTY,
        pitch=pitch,
    )

    result = packer.pack_voxels(items, initial_tray=initial_tray)
    return result, packer, voxelizer, pitch, voxel_origin, initial_tray


# get_orientations(item, 6) 순서와 일치하는 회전 행렬
# 0:identity, 1:RX, 2:RX², 3:RX³, 4:RY, 5:RY³
_ROT6 = [
    np.array([[1, 0, 0], [0,  1,  0], [0,  0,  1]], dtype=float),  # 0: I
    np.array([[1, 0, 0], [0,  0, -1], [0,  1,  0]], dtype=float),  # 1: RX
    np.array([[1, 0, 0], [0, -1,  0], [0,  0, -1]], dtype=float),  # 2: RX²
    np.array([[1, 0, 0], [0,  0,  1], [0, -1,  0]], dtype=float),  # 3: RX³
    np.array([[0, 0, 1], [0,  1,  0], [-1, 0,  0]], dtype=float),  # 4: RY
    np.array([[0, 0,-1], [0,  1,  0], [1,  0,  0]], dtype=float),  # 5: RY³
]


def get_rotated_shape(original_shape, orientation_idx):
    R = _ROT6[orientation_idx]
    rotated = np.abs(R) @ np.array(original_shape, dtype=float)
    return tuple(int(round(v)) for v in rotated)


def export_blender(result, meta, pitch, voxel_origin, space_path):
    try:
        import bpy
    except ImportError:
        print(
            "\n[경고] bpy 없음 → Blender export 건너뜀\n"
            "실행 방법:\n"
            "  blender --background --python experiments/container/run.py"
        )
        return

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Blender 내보내기] → {OUTPUT_BLEND}")
    t0 = time.perf_counter()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 컨테이너 메시 임포트
    # axis 변환 없이 임포트해야 trimesh voxel 좌표계와 일치함
    print(f"  [Blender] voxel_origin={voxel_origin.tolist()}")
    suffix = space_path.suffix.lower()
    bpy.ops.object.select_all(action='DESELECT')
    if suffix == '.obj':
        if hasattr(bpy.ops.wm, 'obj_import'):
            # Blender 4.0+: forward_axis='Y', up_axis='Z' → 축 변환 없음
            bpy.ops.wm.obj_import(filepath=str(space_path), forward_axis='Y', up_axis='Z')
        else:
            # Blender 3.x: axis_forward='Y', axis_up='Z' → 축 변환 없음
            bpy.ops.import_scene.obj(filepath=str(space_path), axis_forward='Y', axis_up='Z')
    container_objs = list(bpy.context.selected_objects)
    for obj in container_objs:
        obj.name = "Container"
        # 반투명 재질
        mat = bpy.data.materials.new(name="Container_Mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs["Alpha"].default_value = 0.3
        mat.blend_method = 'BLEND'
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    # 배치 성공 아이템 → box mesh 생성
    placed = [p for p in result.placements if p.success]
    for i, p in enumerate(placed):
        type_name, original_shape, type_idx = meta[p.item_index]
        rshape = get_rotated_shape(original_shape, p.orientation_index)

        # 복셀 좌표 → 월드 좌표 (center)
        # voxel_origin = 복셀(0,0,0)의 중심 좌표이므로
        # 아이템 중심 = voxel_origin + (position + (rshape-1)/2) * pitch
        center = voxel_origin + (np.array(p.position) + (np.array(rshape) - 1) / 2.0) * pitch

        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.name = f"Item_{i:03d}_{type_name}"
        obj.scale = (rshape[0] * pitch, rshape[1] * pitch, rshape[2] * pitch)
        obj.location = tuple(float(c) for c in center)
        bpy.ops.object.transform_apply(scale=True)

        color = COLORS[type_idx % len(COLORS)]
        mat = bpy.data.materials.new(name=f"Mat_{type_name}_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.35
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    print(f"  저장 완료! ({time.perf_counter() - t0:.2f}s)")


def save_csv(voxelizer, packer, result, meta, tray_size, voxel_origin, total_elapsed):
    from spectral_packer.voxelizer import LogEntry

    container_vol = int(np.prod(tray_size))
    placed_vol = sum(
        int(np.prod(meta[p.item_index][1]))
        for p in result.placements if p.success
    )
    fill_rate = placed_vol / container_vol

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
                f"placed_vol={placed_vol} container_vol={container_vol} "
                f"voxel_origin={origin_str}"
            ),
        ),
        LogEntry(step="total", duration_sec=total_elapsed, success=True),
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
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
    print(f"[로그] → {OUTPUT_CSV}")


def save_json(result, meta, tray_size, pitch, voxel_origin, initial_tray):
    data = {
        "tray_size": [int(v) for v in tray_size],
        "pitch": float(pitch),
        "voxel_origin": [float(v) for v in voxel_origin],
        "item_types": [
            {"type_name": tn, "shape": [int(v) for v in sh], "count": int(cnt)}
            for tn, sh, cnt in ITEM_TYPES
        ],
        "colors": COLORS,
        "meta": [
            {"type_name": tn, "shape": [int(v) for v in sh], "type_idx": int(ti)}
            for tn, sh, ti in meta
        ],
        "placements": [
            {
                "item_index": int(p.item_index),
                "position": [int(v) for v in p.position] if p.position is not None else None,
                "orientation_index": int(p.orientation_index),
                "success": bool(p.success),
                "score": float(p.score) if p.score is not None else None,
            }
            for p in result.placements
        ],
        # 장애물(벽) 복셀 좌표 — visualize.py에서 컨테이너 내부 형상 확인용
        "container_walls": np.argwhere(initial_tray == 1).tolist(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[JSON] → {OUTPUT_JSON}")


def main():
    t0 = time.perf_counter()

    space_path = INPUT_DIR / SPACE_FILE
    if not space_path.exists():
        raise FileNotFoundError(f"공간 파일 없음: {space_path}")

    items, meta = build_items()
    total_item_vol = sum(int(np.prod(m[1])) for m in meta)
    print(f"[아이템] {len(items)}개  총 볼륨 {total_item_vol:,} vox")
    for type_name, shape, count in ITEM_TYPES:
        vol = int(np.prod(shape))
        print(f"         {type_name:12s} {str(shape):15s} × {count:2d}개 = {vol*count:,} vox")

    print("\n[패킹 시작]")
    result, packer, voxelizer, pitch, voxel_origin, initial_tray = run_packing(items, space_path)

    tray_size = result.tray.shape
    placed_vol = sum(
        int(np.prod(meta[p.item_index][1]))
        for p in result.placements if p.success
    )
    fill_rate = placed_vol / int(np.prod(tray_size))
    print(f"[결과] 배치 성공: {result.num_placed}/{len(items)}개")
    print(f"[결과] 채움률:    {fill_rate:.1%}")
    print(f"[결과] tray:      {tray_size}  pitch={pitch:.4f}")

    export_blender(result, meta, pitch, voxel_origin, space_path)

    total_elapsed = time.perf_counter() - t0
    save_csv(voxelizer, packer, result, meta, tray_size, voxel_origin, total_elapsed)
    save_json(result, meta, tray_size, pitch, voxel_origin, initial_tray)
    print(f"[완료] {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
