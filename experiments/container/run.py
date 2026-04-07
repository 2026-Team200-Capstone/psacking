#!/usr/bin/env python3
"""
실험 1: Container를 공간으로 사용한 3D Bin Packing
=================================================

12281_Container_v2_L2.obj 내부에 나머지 아이템들을 패킹하고
결과를 results/packed_result.blend 로 내보냅니다.

사용법:
    # Blender 내장 Python으로 실행 (시각화 포함)
    blender --background --python experiments/container/run.py

    # 일반 Python으로 패킹만 실행 (bpy 없어도 됨)
    python experiments/container/run.py
"""

import csv
import sys
import time
from pathlib import Path

# Blender가 sys.path를 조작해 site-packages를 무시하는 경우를 대비해 직접 추가
for _site in ["/usr/local/lib/python3.10/dist-packages", "/workspace"]:
    if _site not in sys.path:
        sys.path.insert(0, _site)

# ── 경로 ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent                    # experiments/container/
INPUT_DIR    = _HERE.parent / "input_meshes"            # experiments/input_meshes/
OUTPUT_BLEND = _HERE / "results" / "packed_result.blend"
OUTPUT_CSV   = _HERE / "results" / "log.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 ──────────────────────────────────────────────────────────────────────
SPACE_FILE        = "12281_Container_v2_L2.obj"
SPACE_RESOLUTION  = 128
NUM_ORIENTATIONS  = 6
HEIGHT_PENALTY    = 4.0
INTERLOCKING_FREE = False
SKIP_FILES        = {"trunk_kona.ply"}
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".stl", ".obj", ".ply", ".gltf", ".glb", ".dae", ".3mf"}


def save_csv(voxelizer, packer, total_elapsed):
    from spectral_packer.voxelizer import LogEntry
    entries = voxelizer.log + packer.log + [
        LogEntry(
            step="total",
            duration_sec=total_elapsed,
            success=True,
            notes=f"placed={packer.log[-1].notes if packer.log else '?'}",
        )
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
    print(f"\n[로그 저장] → {OUTPUT_CSV}")


def collect_item_files() -> list:
    exclude = SKIP_FILES | {SPACE_FILE}
    return [
        f for f in sorted(INPUT_DIR.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTS and f.name not in exclude
    ]


def run_packing(item_paths: list, space_path: Path):
    from spectral_packer import BinPacker
    from spectral_packer.packer import MeshPlacementInfo
    from spectral_packer.voxelizer import Voxelizer

    voxelizer = Voxelizer(resolution=SPACE_RESOLUTION, verbose=True)

    print(f"\n[공간 복셀화] {space_path.name}")
    initial_tray, pitch, tray_size = voxelizer.voxelize_space(space_path)

    print(f"\n[아이템 복셀화] pitch={pitch:.4f}")
    voxels, voxel_infos = [], []
    for path in item_paths:
        grid, info = voxelizer.voxelize_item(path, pitch)
        if grid is not None:
            voxels.append(grid)
            voxel_infos.append(info)

    if not voxels:
        raise ValueError("복셀화된 아이템이 없습니다.")

    print(f"\n[패킹 시작]  아이템 {len(voxels)}개  트레이 {tray_size}")
    packer = BinPacker(
        tray_size=tray_size,
        voxel_resolution=SPACE_RESOLUTION,
        num_orientations=NUM_ORIENTATIONS,
        height_penalty=HEIGHT_PENALTY,
        interlocking_free=INTERLOCKING_FREE,
        pitch=pitch,
        verbose=True,
    )

    result = packer.pack_voxels(voxels, initial_tray=initial_tray)

    result.mesh_placements = [
        MeshPlacementInfo(
            mesh_path=voxel_infos[p.item_index].mesh_path,
            voxel_info=voxel_infos[p.item_index],
            voxel_position=p.position,
            orientation_index=p.orientation_index,
            success=p.success,
            refined_position=p.refined_position,
        )
        for p in result.placements
    ]

    return result, voxelizer, packer


def print_result(result):
    print("\n" + result.summary())
    print("\n[개별 배치 결과]")
    for p in result.placements:
        if p.success:
            print(f"  Item {p.item_index:2d}: 배치 완료  위치={p.position}  "
                  f"방향={p.orientation_index}  점수={p.score:.2f}")
        else:
            print(f"  Item {p.item_index:2d}: 배치 실패")


def export_blender(result):
    from spectral_packer import export_to_blend, is_blender_available

    if not is_blender_available():
        print(
            "\n[경고] bpy 없음 → Blender 내보내기 건너뜀\n"
            "실행 방법:\n"
            "  blender --background --python experiments/container/run.py"
        )
        return

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Blender 내보내기] → {OUTPUT_BLEND}")
    t0 = time.perf_counter()
    export_to_blend(result=result, output_path=OUTPUT_BLEND, include_tray_boundary=True)
    print(f"  저장 완료! ({time.perf_counter() - t0:.2f}s)")

    render_scene(result)


def render_scene(result):
    import math
    import mathutils
    import bpy

    OUTPUT_RENDERS = OUTPUT_BLEND.parent / "renders"
    OUTPUT_RENDERS.mkdir(parents=True, exist_ok=True)

    mesh_objects = [
        obj for obj in bpy.data.objects
        if obj.type == 'MESH' and not obj.name.startswith('Tray')
    ]
    if not mesh_objects:
        print("  [경고] 렌더링할 메시 오브젝트 없음")
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

    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)
    bpy.ops.object.light_add(type='SUN', location=(50, -30, 80))
    sun = bpy.context.active_object
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(30))

    colors = [
        (0.9, 0.25, 0.2, 1), (0.2, 0.75, 0.3, 1), (0.2, 0.4, 0.9, 1),
        (0.95, 0.7, 0.1, 1), (0.7, 0.2, 0.8, 1), (0.1, 0.8, 0.8, 1),
        (0.95, 0.5, 0.5, 1), (0.5, 0.8, 0.2, 1), (0.3, 0.3, 0.7, 1),
        (0.9, 0.6, 0.4, 1),
    ]
    for i, obj in enumerate(mesh_objects):
        mat = bpy.data.materials.new(name=f"Mat_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = colors[i % len(colors)]
        bsdf.inputs["Roughness"].default_value = 0.4
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    for obj in bpy.data.objects:
        if obj.name.startswith('Tray'):
            obj.hide_render = True

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

    print("=" * 55)
    print("  실험 1: Container 공간 패킹")
    print("=" * 55)

    space_path = INPUT_DIR / SPACE_FILE
    if not space_path.exists():
        raise FileNotFoundError(f"공간 파일 없음: {space_path}")

    item_files = collect_item_files()
    print(f"\n[공간 파일] {SPACE_FILE}")
    print(f"[아이템 파일]")
    for f in item_files:
        print(f"  - {f.name}")

    result, voxelizer, packer = run_packing(item_files, space_path)
    print_result(result)
    export_blender(result)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[전체 소요 시간] {total_elapsed:.2f}s")
    print("\n완료!")

    save_csv(voxelizer, packer, total_elapsed)


if __name__ == "__main__":
    main()
