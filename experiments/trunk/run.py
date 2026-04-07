#!/usr/bin/env python3
"""
실험 2: 트렁크(trunk_kona.ply)를 공간으로 사용한 3D Bin Packing
=============================================================

trunk_kona.ply 내부에 박스 아이템들을 패킹하고
결과를 results/packed_result.blend 로 내보냅니다.

사용법:
    blender --background --python experiments/trunk/run.py
    python experiments/trunk/run.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

for _site in ["/usr/local/lib/python3.10/dist-packages", "/workspace"]:
    if _site not in sys.path:
        sys.path.insert(0, _site)

# ── 경로 ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
OUTPUT_BLEND = _HERE / "results" / "packed_result.blend"
OUTPUT_CSV   = _HERE / "results" / "log.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── 공간 설정 ─────────────────────────────────────────────────────────────────
SPACE_FILE       = _HERE.parent / "input_meshes" / "trunk_kona.ply"
SPACE_RESOLUTION = 128
NUM_ORIENTATIONS  = 6
HEIGHT_PENALTY    = 4.0
INTERLOCKING_FREE = False
# ─────────────────────────────────────────────────────────────────────────────

# ── 박스 아이템 정의 ──────────────────────────────────────────────────────────
# 트레이 크기: (128, 75, 109) 기준
# (가로, 세로, 높이) 복셀 수, 개수, 레이블
BOXES = [
    ((45, 35, 30), 3, "large"),    # 큰 박스 3개
    ((28, 22, 20), 5, "medium"),   # 중간 박스 5개
    ((16, 13, 12), 6, "small"),    # 작은 박스 6개
]
# ─────────────────────────────────────────────────────────────────────────────


def make_box_items():
    """BOXES 설정에서 복셀 그리드 목록과 레이블 목록을 생성합니다."""
    voxels, labels = [], []
    for size, count, label in BOXES:
        for i in range(count):
            voxels.append(np.ones(size, dtype=np.int32))
            labels.append(f"{label}_{i + 1}  {size}")
    return voxels, labels


def save_csv(voxelizer, packer, total_elapsed):
    from spectral_packer.voxelizer import LogEntry
    entries = voxelizer.log + packer.log + [
        LogEntry(step="total", duration_sec=total_elapsed, success=True),
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "duration_sec", "success", "notes"])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "step": e.step,
                "duration_sec": f"{e.duration_sec:.3f}",
                "success": e.success,
                "notes": e.notes,
            })


def run_packing(voxels: list):
    from spectral_packer import BinPacker
    from spectral_packer.voxelizer import Voxelizer

    voxelizer = Voxelizer(resolution=SPACE_RESOLUTION)
    initial_tray, pitch, tray_size = voxelizer.voxelize_space(SPACE_FILE)

    print(f"\n[박스 아이템] {len(voxels)}개")
    for i, (_, label) in enumerate(zip(voxels, labels)):
        print(f"  {i:2d}. {label}")

    packer = BinPacker(
        tray_size=tray_size,
        voxel_resolution=SPACE_RESOLUTION,
        num_orientations=NUM_ORIENTATIONS,
        height_penalty=HEIGHT_PENALTY,
        interlocking_free=INTERLOCKING_FREE,
        pitch=pitch,
    )

    result = packer.pack_voxels(voxels, initial_tray=initial_tray)
    return result, voxelizer, packer


def export_blender(result):
    from spectral_packer import is_blender_available

    if not is_blender_available():
        print(
            "\n[경고] bpy 없음 → Blender 내보내기 건너뜀\n"
            "실행 방법:\n"
            "  blender --background --python experiments/trunk/run.py"
        )
        return

    import bpy

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    tray = result.tray

    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)

    colors = [
        (0.9, 0.25, 0.2, 1), (0.2, 0.75, 0.3, 1), (0.2, 0.4, 0.9, 1),
        (0.95, 0.7, 0.1, 1), (0.7, 0.2, 0.8, 1), (0.1, 0.8, 0.8, 1),
        (0.95, 0.5, 0.5, 1), (0.5, 0.8, 0.2, 1), (0.3, 0.3, 0.7, 1),
        (0.9, 0.6, 0.4, 1),
    ]

    item_ids = [p.item_index + 1 for p in result.placements if p.success]
    for item_id in item_ids:
        voxels_idx = np.argwhere(tray == item_id)
        if len(voxels_idx) == 0:
            continue
        mn = voxels_idx.min(axis=0)
        mx = voxels_idx.max(axis=0)
        size = (mx - mn + 1).tolist()
        center = ((mn + mx) / 2).tolist()

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(center[0], center[1], center[2]),
        )
        obj = bpy.context.active_object
        obj.scale = (size[0], size[1], size[2])
        obj.name = f"Box_{item_id}"

        mat = bpy.data.materials.new(name=f"Mat_{item_id}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = colors[(item_id - 1) % len(colors)]
        bsdf.inputs["Roughness"].default_value = 0.3
        obj.data.materials.append(mat)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    print(f"\n[Blender 저장] → {OUTPUT_BLEND}")

    render_scene()


def render_scene():
    import math
    import mathutils
    import bpy

    OUTPUT_RENDERS = OUTPUT_BLEND.parent / "renders"
    OUTPUT_RENDERS.mkdir(parents=True, exist_ok=True)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not mesh_objects:
        return

    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3
    for obj in mesh_objects:
        for v in obj.bound_box:
            world_v = obj.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                min_coords[i] = min(min_coords[i], world_v[i])
                max_coords[i] = max(max_coords[i], world_v[i])

    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    max_size = max(max_coords[i] - min_coords[i] for i in range(3))

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.image_settings.file_format = 'PNG'

    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.15, 0.15, 0.2, 1)
    bg.inputs['Strength'].default_value = 0.5
    out = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])

    bpy.ops.object.light_add(type='SUN', location=(50, -30, 80))
    sun = bpy.context.active_object
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(30))

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    scene.camera = camera
    camera.data.lens = 50

    def point_camera_at(cam, target):
        direction = mathutils.Vector(target) - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    cam_distance = max_size * 2.5
    views = [
        ("front", (center[0], center[1] - cam_distance, center[2] + max_size * 0.3)),
        ("iso",   (center[0] + cam_distance * 0.7, center[1] - cam_distance * 0.7, center[2] + cam_distance * 0.5)),
    ]

    print(f"\n[렌더링] → {OUTPUT_RENDERS}")
    for view_name, cam_loc in views:
        t0 = time.perf_counter()
        camera.location = cam_loc
        point_camera_at(camera, center)
        output_path = OUTPUT_RENDERS / f"{view_name}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        print(f"  저장: {output_path}  ({time.perf_counter() - t0:.2f}s)")


def main():
    total_start = time.perf_counter()

    if not SPACE_FILE.exists():
        raise FileNotFoundError(f"공간 파일 없음: {SPACE_FILE}")

    global labels
    voxels, labels = make_box_items()

    result, voxelizer, packer = run_packing(voxels)
    export_blender(result)

    total_elapsed = time.perf_counter() - total_start
    save_csv(voxelizer, packer, total_elapsed)


if __name__ == "__main__":
    main()
