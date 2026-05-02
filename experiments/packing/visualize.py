#!/usr/bin/env python3
"""
패킹 결과 Plotly 인터랙티브 시각화
=====================================
packed_result.json을 읽어 브라우저에서 3D로 시각화합니다.
spectral_packer 없이 plotly + numpy만 있으면 실행 가능합니다.
어떤 형태의 컨테이너 메시든 올바르게 렌더링합니다.

사용법:
    python experiments/packing/visualize.py                          # results/ 에서 최근 JSON 자동 탐색
    python experiments/packing/visualize.py results/coffee_cup_obj/  # 결과 디렉토리 지정
    python experiments/packing/visualize.py results/coffee_cup_obj/packed_result.json
    python experiments/packing/visualize.py coffee_cup_obj           # space 이름으로 탐색
    python experiments/packing/visualize.py http://server:8000/jobs/<job_id>/result
    python experiments/packing/visualize.py --server http://server:8000 --job <job_id>
"""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

_HERE        = Path(__file__).parent
_RESULTS_DIR = _HERE / "results"


# ── 회전 행렬 ─────────────────────────────────────────────────────────────────

def _build_rot_matrices():
    """
    6-방향 및 24-방향 회전 행렬을 생성합니다.
    run.py / spectral_packer/rotations.py 의 순서와 일치합니다.

    Returns
    -------
    rot6  : list of 6 (3×3) ndarray
    rot24 : list of 24 (3×3) ndarray
    """
    I  = np.eye(3, dtype=float)
    RX = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
    RY = np.array([[0,0,1],[0,1,0],[-1,0,0]], float)
    RZ = np.array([[0,-1,0],[1,0,0],[0,0,1]], float)
    RX2, RX3 = RX@RX, RX@RX@RX
    RY3       = RY@RY@RY
    RZ2, RZ3  = RZ@RZ, RZ@RZ@RZ

    # 6-방향: I, RX, RX², RX³, RY, RY³  (get_orientations num=6 순서)
    rot6 = [I, RX, RX2, RX3, RY, RY3]

    # 24-방향: get_24_orientations() / _generate_rotation_matrices() 순서
    rot24 = []
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz.copy())
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz @ RX)
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz @ RX2)
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz @ RX3)
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz @ RY)
    for Rz in [I, RZ, RZ2, RZ3]:
        rot24.append(Rz @ RY3)

    return rot6, rot24


_ROT6, _ROT24 = _build_rot_matrices()


def get_rotated_shape(shape, orientation_idx, num_orientations=6):
    """회전 인덱스에 따라 아이템의 축 정렬 바운딩박스 크기를 반환합니다."""
    rots = _ROT6 if num_orientations <= 6 else _ROT24
    if orientation_idx >= len(rots):
        raise ValueError(
            f"orientation_index {orientation_idx} >= {len(rots)} "
            f"(num_orientations={num_orientations}). "
            f"JSON이 다른 설정으로 생성되었을 수 있습니다."
        )
    R = rots[orientation_idx]
    # abs(R) @ shape: 90° 축 회전 후 바운딩박스 크기 계산
    rotated = np.abs(R) @ np.array(shape, dtype=float)
    return tuple(int(round(v)) for v in rotated)


# ── JSON 파일 탐색 ─────────────────────────────────────────────────────────────

def find_json(arg=None):
    """
    CLI 인수 또는 기본 탐색으로 JSON 파일 경로를 반환합니다.

    우선순위:
    1. arg가 JSON 파일 → 직접 사용
    2. arg가 디렉토리 → 그 안의 packed_result.json
    3. arg가 이름 문자열 → results/<arg>/packed_result.json
    4. arg 없음 → results/*/packed_result.json 중 가장 최근 파일
                  없으면 results/packed_result.json (구형 경로 폴백)
    """
    if arg:
        p = Path(arg)
        if p.suffix == ".json":
            return p
        if p.is_dir():
            j = p / "packed_result.json"
            if j.exists():
                return j
            raise FileNotFoundError(f"디렉토리에 packed_result.json 없음: {p}")
        # 이름으로 탐색
        j = _RESULTS_DIR / p.name / "packed_result.json"
        if j.exists():
            return j
        raise FileNotFoundError(f"결과 없음: {j}")

    # 자동 탐색: results/*/packed_result.json (최근 순)
    candidates = sorted(
        _RESULTS_DIR.glob("*/packed_result.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    # 구형 results/packed_result.json도 후보에 포함
    old_path = _RESULTS_DIR / "packed_result.json"
    if old_path.exists():
        candidates.append(old_path)

    if not candidates:
        raise FileNotFoundError(
            f"JSON 파일 없음. 먼저 run.py를 실행해 결과를 생성하세요.\n"
            f"  python experiments/packing/run.py <space_mesh>"
        )

    if len(candidates) > 1:
        print(f"[탐색] {len(candidates)}개 결과 발견. 가장 최근 파일 사용:")
        for c in candidates[:5]:
            print(f"  {c}")
        if len(candidates) > 5:
            print(f"  ... 외 {len(candidates)-5}개")

    return candidates[0]


def _is_url(value: str | None) -> bool:
    return bool(value and (value.startswith("http://") or value.startswith("https://")))


def _fetch_json(url: str, timeout: float = 30.0) -> Any:
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset("utf-8")
            return json.loads(response.read().decode(charset))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} 응답: {url}\n{detail}") from e
    except URLError as e:
        raise RuntimeError(f"URL 조회 실패: {url}\n{e}") from e


def load_result(source: str | None = None, server: str | None = None, job: str | None = None):
    if server or job:
        if not (server and job):
            raise ValueError("--server와 --job은 함께 지정해야 합니다.")
        url = f"{server.rstrip('/')}/jobs/{job}/result"
        return _fetch_json(url), url

    if _is_url(source):
        return _fetch_json(str(source)), str(source)

    json_path = find_json(source)
    with open(json_path, encoding="utf-8") as f:
        return json.load(f), str(json_path)


def _fallback_space_name(source_name: str | None) -> str:
    if source_name and not _is_url(source_name):
        return Path(source_name).parent.name
    return "remote_result"


# ── Plotly 트레이스 헬퍼 ──────────────────────────────────────────────────────

def _to_hex(c):
    """RGBA 튜플 또는 hex 문자열을 hex 문자열로 변환."""
    if isinstance(c, str):
        return c
    r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _box_mesh(x0, y0, z0, dx, dy, dz, color, name, opacity, show_legend):
    import plotly.graph_objects as go

    x1, y1, z1 = x0+dx, y0+dy, z0+dz
    vx = [x0,x1,x1,x0,x0,x1,x1,x0]
    vy = [y0,y0,y1,y1,y0,y0,y1,y1]
    vz = [z0,z0,z0,z0,z1,z1,z1,z1]
    i = [7,0,0,0,4,4,6,1,4,0,3,6]
    j = [3,4,1,2,5,6,5,0,0,3,6,3]
    k = [0,7,5,1,4,7,1,2,5,7,7,7]
    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, legendgroup=name,
        showlegend=show_legend, flatshading=True,
    )


def _wireframe(x0, y0, z0, dx, dy, dz, color, name, width=1):
    import plotly.graph_objects as go

    x1, y1, z1 = x0+dx, y0+dy, z0+dz
    pts = [
        (x0,y0,z0),(x1,y0,z0),None, (x1,y0,z0),(x1,y1,z0),None,
        (x1,y1,z0),(x0,y1,z0),None, (x0,y1,z0),(x0,y0,z0),None,
        (x0,y0,z1),(x1,y0,z1),None, (x1,y0,z1),(x1,y1,z1),None,
        (x1,y1,z1),(x0,y1,z1),None, (x0,y1,z1),(x0,y0,z1),None,
        (x0,y0,z0),(x0,y0,z1),None, (x1,y0,z0),(x1,y0,z1),None,
        (x1,y1,z0),(x1,y1,z1),None, (x0,y1,z0),(x0,y1,z1),None,
    ]
    xs = [p[0] if p else None for p in pts]
    ys = [p[1] if p else None for p in pts]
    zs = [p[2] if p else None for p in pts]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color=color, width=width),
        name=name, legendgroup=name,
        showlegend=False, hoverinfo='skip',
    )


def _container_mesh_trace(verts_vox, faces):
    """
    실제 메시 형상을 반투명 Mesh3d로 렌더링합니다.

    verts_vox : (N, 3) ndarray  — 복셀 좌표계 정점
    faces     : (M, 3) ndarray  — 삼각형 인덱스
    """
    import plotly.graph_objects as go

    v = np.asarray(verts_vox, dtype=float)
    f = np.asarray(faces, dtype=int)

    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color='lightsteelblue',
        opacity=0.12,
        flatshading=False,
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2,
        ),
        lightposition=dict(x=1, y=2, z=3),
        name='Container',
        legendgroup='container',
        showlegend=True,
        hovertemplate='Container<extra></extra>',
    )


def _container_walls_trace(walls):
    """구형 JSON(container_walls)에 대한 폴백 렌더러 — 복셀 점군."""
    import plotly.graph_objects as go

    w = np.asarray(walls, dtype=float)
    step = max(1, len(w) // 5000)
    w = w[::step]
    return go.Scatter3d(
        x=w[:, 0], y=w[:, 1], z=w[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.15),
        name='Container (walls)',
        showlegend=True,
    )


# ── 메인 시각화 ───────────────────────────────────────────────────────────────

def visualize_data(data: dict[str, Any], source_name: str | None = None):
    import plotly.graph_objects as go

    tray_size       = data["tray_size"]
    meta            = data["meta"]
    placements      = data["placements"]
    colors          = [_to_hex(c) for c in data["colors"]]
    pitch           = data.get("pitch", 1.0)
    voxel_origin    = np.array(data.get("voxel_origin", [0.0, 0.0, 0.0]))
    num_orientations = data.get("num_orientations", 6)
    space_name      = data.get("space_name", _fallback_space_name(source_name))

    traces = []
    seen_types = set()
    out_of_bounds = 0

    # ── 컨테이너 형상 렌더링 ──────────────────────────────────────────────────
    if "container_mesh_vertices" in data and "container_mesh_faces" in data:
        # world 좌표 → voxel 좌표 변환: vox = (world - voxel_origin) / pitch
        verts_world = np.array(data["container_mesh_vertices"], dtype=float)
        faces       = np.array(data["container_mesh_faces"], dtype=int)
        verts_vox   = (verts_world - voxel_origin) / pitch

        traces.append(_container_mesh_trace(verts_vox, faces))
        print(f"[컨테이너] 메시 렌더링: {len(verts_world)}개 정점, {len(faces)}개 삼각형")

    elif "container_walls" in data:
        walls = data["container_walls"]
        if walls:
            traces.append(_container_walls_trace(walls))
            print(f"[컨테이너] 폴백 점군 렌더링: {len(walls)}개 복셀")
    else:
        print("[경고] JSON에 컨테이너 형상 데이터 없음 (container_mesh_vertices / container_walls 키 없음)")

    # ── 배치 아이템 렌더링 ────────────────────────────────────────────────────
    placed = [p for p in placements if p["success"]]
    failed = [p for p in placements if not p["success"]]
    for p in placed:
        m         = meta[p["item_index"]]
        type_name = m["type_name"]
        type_idx  = m["type_idx"]
        rshape    = get_rotated_shape(m["shape"], p["orientation_index"], num_orientations)
        pos       = p["position"]
        end       = [pos[i] + rshape[i] for i in range(3)]

        if any(end[i] > tray_size[i] for i in range(3)):
            out_of_bounds += 1

        color          = colors[type_idx % len(colors)]
        first_of_type  = type_name not in seen_types
        seen_types.add(type_name)

        traces.append(_box_mesh(
            *pos, *rshape,
            color=color, name=type_name,
            opacity=0.95, show_legend=first_of_type,
        ))
        traces.append(_wireframe(*pos, *rshape, color='#222222', name=type_name, width=2))

    # ── 배치 실패 아이템 렌더링 (고유 색상, tray 외부에 표시) ─────────────────
    # 타입별로 그룹화: 같은 타입은 z축으로 쌓고, 타입마다 x 오프셋 부여
    from collections import defaultdict as _defaultdict

    _GAP        = 2
    _base_x     = tray_size[0] + _GAP
    _type_groups = _defaultdict(list)
    for p in failed:
        m = meta[p["item_index"]]
        _type_groups[m["type_name"]].append((p, m))

    _cum_x = _base_x
    for _type_name in sorted(_type_groups):
        _group       = _type_groups[_type_name]
        _group_width = max(tuple(m["shape"])[0] for _, m in _group)
        _cum_z       = 0

        for p, m in _group:
            type_name = m["type_name"]
            type_idx  = m["type_idx"]
            shape     = tuple(m["shape"])

            fx, fy, fz = _cum_x, 0, _cum_z
            _cum_z += shape[2] + _GAP

            color         = colors[type_idx % len(colors)]
            label         = f"[실패] {type_name}"
            first_of_type = label not in seen_types
            seen_types.add(label)
            traces.append(_box_mesh(
                fx, fy, fz, *shape,
                color=color, name=label,
                opacity=0.40, show_legend=first_of_type,
            ))
            traces.append(_wireframe(fx, fy, fz, *shape, color=color, name=label))

        _cum_x += _group_width + _GAP

    # ── 채움률 계산 ───────────────────────────────────────────────────────────
    placed_vol = sum(int(np.prod(meta[p["item_index"]]["shape"])) for p in placed)
    free_vol   = data.get("container_free_volume", int(np.prod(tray_size)))
    fill_rate  = placed_vol / free_vol if free_vol > 0 else 0.0

    fail_str = f"  |  실패 {len(failed)}개" if failed else ""
    status = "✓ 전부 tray 안" if out_of_bounds == 0 else f"⚠ 범위 초과 {out_of_bounds}개"
    title = (
        f"{space_name}  |  "
        f"배치 {len(placed)}/{len(placements)}개{fail_str}  |  "
        f"채움률 {fill_rate:.1%}  |  "
        f"tray {tray_size}  |  {status}"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (voxel)",
            yaxis_title="Y (voxel)",
            zaxis_title="Z (voxel)",
            aspectmode="data",
            # 어두운 배경 → 반투명 컨테이너 내부가 잘 보임
            bgcolor='rgb(20, 20, 30)',
            xaxis=dict(showgrid=True, gridcolor='rgba(80,80,80,0.4)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(80,80,80,0.4)'),
            zaxis=dict(showgrid=True, gridcolor='rgba(80,80,80,0.4)'),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgb(30, 30, 40)',
        font=dict(color='white'),
    )
    fig.show()

    # ── 콘솔 검증 출력 ────────────────────────────────────────────────────────
    if failed:
        from collections import Counter
        fail_types = Counter(meta[p["item_index"]]["type_name"] for p in failed)
        print(f"\n[실패] 배치 실패 아이템 {len(failed)}개:")
        for type_name, count in fail_types.most_common():
            shape = next(m["shape"] for m in meta if m["type_name"] == type_name)
            print(f"  - {type_name} {shape}  × {count}개")

    print(f"\n[검증] 배치 {len(placed)}개 중 tray 범위 초과: {out_of_bounds}개")
    if "container_free_volume" in data:
        print(f"[채움률] {placed_vol:,} vox 배치 / {free_vol:,} vox 내부공간 = {fill_rate:.1%}")
    else:
        print(f"[채움률] {placed_vol:,} vox 배치 / {free_vol:,} vox 바운딩박스 = {fill_rate:.1%}")

    # 아이템 간 겹침 검사
    boxes = []
    for p in placed:
        rshape = get_rotated_shape(
            meta[p["item_index"]]["shape"], p["orientation_index"], num_orientations
        )
        boxes.append((p["position"], rshape, p["item_index"]))

    overlap_count = 0
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            p1, s1, idx1 = boxes[a]
            p2, s2, idx2 = boxes[b]
            if all(p1[i] < p2[i] + s2[i] and p2[i] < p1[i] + s1[i] for i in range(3)):
                overlap_count += 1
                print(f"  ⚠ 겹침: item {idx1} {p1}+{s1}  ↔  item {idx2} {p2}+{s2}")
    if overlap_count == 0:
        print("[검증] 아이템 간 겹침 없음 ✓")
    else:
        print(f"[검증] 겹치는 쌍: {overlap_count}개 ⚠")


def visualize(json_path: Path):
    data, source_name = load_result(str(json_path))
    visualize_data(data, source_name)


def main():
    parser = argparse.ArgumentParser(description="패킹 결과 JSON을 Plotly로 시각화합니다.")
    parser.add_argument(
        "source",
        nargs="?",
        help="로컬 JSON/결과 디렉토리/space 이름 또는 http(s) 결과 URL",
    )
    parser.add_argument("--server", help="Packing API 서버 주소. 예: http://server:8000")
    parser.add_argument("--job", help="조회할 job id")
    args = parser.parse_args()

    data, source_name = load_result(args.source, args.server, args.job)
    print(f"[읽기] {source_name}")
    visualize_data(data, source_name)


if __name__ == "__main__":
    main()
