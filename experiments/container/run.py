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

from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent                       # experiments/container/
INPUT_DIR = _HERE.parent / "input_meshes"               # experiments/input_meshes/
OUTPUT_BLEND = _HERE / "results" / "packed_result.blend"
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 공간으로 사용할 파일 (input_meshes/ 기준)
SPACE_FILE = "12281_Container_v2_L2.obj"

# 공간 메시 복셀화 해상도 (긴 축 기준 voxel 수)
# → 이 값이 컨테이너와 아이템 사이의 공통 pitch를 결정
SPACE_RESOLUTION = 128

# 회전 수: 1(고정) / 4(Z축) / 6(면별) / 24(전방향)
NUM_ORIENTATIONS = 6

# 높이 패널티 (클수록 낮게 쌓으려 함)
HEIGHT_PENALTY = 4.0

# True: 꺼낼 수 있는 위치만 허용
INTERLOCKING_FREE = False

# 패킹에서 제외할 파일 (아직 사용하지 않을 파일)
SKIP_FILES = {"trunk_kona.ply"}
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".stl", ".obj", ".ply", ".gltf", ".glb", ".dae", ".3mf"}


def collect_item_files() -> list[Path]:
    """공간 파일·스킵 파일을 제외한 아이템 목록을 반환합니다."""
    exclude = SKIP_FILES | {SPACE_FILE}
    return [
        f for f in sorted(INPUT_DIR.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTS and f.name not in exclude
    ]


def voxelize_space(space_path: Path):
    """
    공간 메시를 복셀화해 (내부 free, 외부 obstacle) initial_tray와 pitch를 반환합니다.

    Returns
    -------
    initial_tray : np.ndarray[int32]
        0 = 아이템 배치 가능, 1 = 장애물(컨테이너 외부)
    pitch : float
        voxel 1칸의 실세계 크기 (아이템 복셀화에 동일하게 사용)
    tray_size : tuple[int, int, int]
    """
    import trimesh
    from spectral_packer import load_mesh

    print(f"\n[공간 복셀화] {space_path.name}")

    vertices, faces = load_mesh(space_path, validate=True, repair=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    max_extent = float(mesh.extents.max())
    pitch = max_extent / (SPACE_RESOLUTION - 1)

    print(f"  바운딩박스 크기: {mesh.extents.tolist()}")
    print(f"  pitch          : {pitch:.4f}  (해상도 {SPACE_RESOLUTION})")

    # fill=True → 내부를 채운 solid 복셀 (1=내부, 0=외부)
    filled = mesh.voxelized(pitch=pitch).fill().matrix.astype("int32")

    # initial_tray: 외부(0) → obstacle(1), 내부(1) → free(0)
    initial_tray = (1 - filled).astype("int32")
    tray_size = filled.shape

    print(f"  트레이 크기    : {tray_size}")
    print(f"  내부 free 비율 : {filled.mean():.1%}")

    return initial_tray, pitch, tray_size


def voxelize_items(item_paths: list[Path], pitch: float):
    """
    아이템들을 공통 pitch로 복셀화합니다.

    Returns
    -------
    voxels      : list[np.ndarray]
    voxel_infos : list[VoxelizationInfo]
    """
    import trimesh
    from spectral_packer import load_mesh
    from spectral_packer.voxelizer import VoxelizationInfo

    voxels, voxel_infos = [], []

    print(f"\n[아이템 복셀화] pitch={pitch:.4f}")
    for path in item_paths:
        vertices, faces = load_mesh(path, validate=True, repair=True)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        bounds = mesh.bounds
        try:
            grid = mesh.voxelized(pitch=pitch).fill().matrix.astype("int32")
        except Exception as e:
            print(f"  [경고] {path.name} 복셀화 실패: {e} → 건너뜀")
            continue

        voxels.append(grid)
        voxel_infos.append(VoxelizationInfo(
            mesh_path=path,
            mesh_bounds_min=bounds[0].copy(),
            mesh_bounds_max=bounds[1].copy(),
            pitch=pitch,
            voxel_shape=grid.shape,
        ))
        print(f"  {path.name:40s} → {grid.shape}  ({int(grid.sum())} voxels)")

    return voxels, voxel_infos


def run_packing(item_paths: list[Path], space_path: Path):
    """컨테이너 내부에 아이템들을 패킹하고 PackingResult를 반환합니다."""
    from spectral_packer import BinPacker
    from spectral_packer.packer import MeshPlacementInfo

    # 1. 공간 복셀화
    initial_tray, pitch, tray_size = voxelize_space(space_path)

    # 2. 아이템 복셀화 (동일 pitch → 비율 보장)
    voxels, voxel_infos = voxelize_items(item_paths, pitch)
    if not voxels:
        raise ValueError("복셀화된 아이템이 없습니다.")

    # 3. 패킹
    print(f"\n[패킹 시작]  아이템 {len(voxels)}개  트레이 {tray_size}")
    packer = BinPacker(
        tray_size=tray_size,
        voxel_resolution=SPACE_RESOLUTION,
        num_orientations=NUM_ORIENTATIONS,
        height_penalty=HEIGHT_PENALTY,
        interlocking_free=INTERLOCKING_FREE,
        pitch=pitch,
    )
    result = packer.pack_voxels(voxels, initial_tray=initial_tray)

    # 4. Blender 내보내기용 메타데이터
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

    return result


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
    export_to_blend(result=result, output_path=OUTPUT_BLEND, include_tray_boundary=True)
    print("  저장 완료!")

    render_scene(result)


def render_scene(result):
    """Blender 씬에 조명·카메라·머티리얼을 설정하고 PNG를 렌더링합니다."""
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

    # 씬 바운딩박스 계산
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

    # 렌더 설정
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.image_settings.file_format = 'PNG'

    # 월드 배경
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

    # 기존 조명 제거 후 새 조명 추가
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)
    bpy.ops.object.light_add(type='SUN', location=(50, -30, 80))
    sun = bpy.context.active_object
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(30))

    # 아이템별 머티리얼 색상
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

    # 트레이 경계선 렌더에서 숨기기
    for obj in bpy.data.objects:
        if obj.name.startswith('Tray'):
            obj.hide_render = True

    # 카메라 설정
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
        camera.location = cam_loc
        point_camera_at(camera, center)
        output_path = OUTPUT_RENDERS / f"{view_name}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        print(f"  저장: {output_path}")


def main():
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

    result = run_packing(item_files, space_path)
    print_result(result)
    export_blender(result)

    print("\n완료!")


if __name__ == "__main__":
    main()
