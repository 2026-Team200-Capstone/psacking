#!/usr/bin/env python3
"""
패킹 결과 Plotly 인터랙티브 시각화
=====================================
packed_result.json을 읽어 브라우저에서 3D로 시각화합니다.
spectral_packer 없이 plotly만 있으면 실행 가능합니다.

사용법:
    python experiments/container/visualize.py
    python experiments/container/visualize.py path/to/packed_result.json
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE        = Path(__file__).parent
DEFAULT_JSON = _HERE / "results" / "packed_result.json"


# ── 회전 적용 ─────────────────────────────────────────────────────────────────

# get_orientations(item, 6) 순서와 일치하는 회전 행렬
# 0:identity, 1:RX, 2:RX², 3:RX³, 4:RY, 5:RY³
_ROT6 = [
    [[1, 0, 0], [0,  1,  0], [0,  0,  1]],  # 0: I
    [[1, 0, 0], [0,  0, -1], [0,  1,  0]],  # 1: RX
    [[1, 0, 0], [0, -1,  0], [0,  0, -1]],  # 2: RX²
    [[1, 0, 0], [0,  0,  1], [0, -1,  0]],  # 3: RX³
    [[0, 0, 1], [0,  1,  0], [-1, 0,  0]],  # 4: RY
    [[0, 0,-1], [0,  1,  0], [1,  0,  0]],  # 5: RY³
]


def get_rotated_shape(shape, orientation_idx):
    R = np.array(_ROT6[orientation_idx], dtype=float)
    rotated = np.abs(R) @ np.array(shape, dtype=float)
    return tuple(int(round(v)) for v in rotated)


# ── Plotly 박스 트레이스 ───────────────────────────────────────────────────────

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


def _wireframe(x0, y0, z0, dx, dy, dz, color, name, width=2):
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


# ── 메인 시각화 ───────────────────────────────────────────────────────────────

def _to_hex(c):
    """RGBA 튜플 또는 hex 문자열을 hex 문자열로 변환."""
    if isinstance(c, str):
        return c
    r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def visualize(json_path: Path):
    import plotly.graph_objects as go

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    tray_size  = data["tray_size"]
    meta       = data["meta"]
    placements = data["placements"]
    colors     = [_to_hex(c) for c in data["colors"]]

    traces = []
    seen_types = set()
    out_of_bounds = 0

    # 컨테이너 벽 (장애물 복셀) — 5000개로 다운샘플링
    walls = data.get("container_walls", [])
    if walls:
        walls = np.array(walls)
        step = max(1, len(walls) // 5000)
        w = walls[::step]
        traces.append(go.Scatter3d(
            x=w[:, 0], y=w[:, 1], z=w[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='lightgray', opacity=0.15),
            name='Container (walls)',
            showlegend=True,
        ))

    # 컨테이너 경계
    traces.append(_wireframe(0, 0, 0, *tray_size, color="black", name="Container bbox", width=4))
    traces.append(go.Scatter3d(
        x=[None], y=[None], z=[None], mode='lines',
        line=dict(color='black', width=4),
        name='Container bbox', legendgroup='Container bbox', showlegend=True,
    ))

    placed = [p for p in placements if p["success"]]
    for p in placed:
        m = meta[p["item_index"]]
        type_name = m["type_name"]
        type_idx  = m["type_idx"]
        rshape    = get_rotated_shape(m["shape"], p["orientation_index"])
        pos       = p["position"]
        end       = [pos[i] + rshape[i] for i in range(3)]

        if any(end[i] > tray_size[i] for i in range(3)):
            out_of_bounds += 1

        color = colors[type_idx % len(colors)]
        first_of_type = type_name not in seen_types
        seen_types.add(type_name)

        traces.append(_box_mesh(
            *pos, *rshape,
            color=color, name=type_name,
            opacity=0.55, show_legend=first_of_type,
        ))
        traces.append(_wireframe(*pos, *rshape, color=color, name=type_name, width=1))

    placed_vol = sum(int(np.prod(meta[p["item_index"]]["shape"])) for p in placed)
    fill_rate  = placed_vol / int(np.prod(tray_size))

    status = "✓ 전부 tray 안" if out_of_bounds == 0 else f"⚠ 범위 초과 {out_of_bounds}개"
    title = (
        f"3D Bin Packing  |  "
        f"배치 {len(placed)}/{len(placements)}개  |  "
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
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.show()
    print(f"\n[검증] 배치 {len(placed)}개 중 tray 범위 초과: {out_of_bounds}개")

    # 아이템 간 겹침 검사
    boxes = []
    for p in placed:
        rshape = get_rotated_shape(meta[p["item_index"]]["shape"], p["orientation_index"])
        pos = p["position"]
        boxes.append((pos, rshape, p["item_index"]))

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


def main():
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일 없음: {json_path}")
    print(f"[읽기] {json_path}")
    visualize(json_path)


if __name__ == "__main__":
    main()
