"""Unit tests for spectral_packer/support.py (support constraint helpers).

These tests deliberately load ``support.py`` directly from its file path,
bypassing the ``spectral_packer`` package ``__init__`` — the package import
requires the compiled ``_core`` CUDA module, which is not available on
machines without the Docker/GPU build. This keeps the support logic
testable everywhere.

Run standalone:  python tests/test_support.py
Or with pytest:  pytest tests/test_support.py
"""

import importlib.util
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "psacking_support", _ROOT / "spectral_packer" / "support.py"
)
support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(support)


# ── 브루트포스 기준 구현 ──────────────────────────────────────────────
def brute_force_counts(occ, item, ground_at_z0=True):
    """support_ratio_grid와 동일한 의미의 지지 복셀 수를 삼중 루프로 계산.

    FFT 구현과 같은 규약: shifted_occ는 occ와 같은 크기(그 밖은 0)이고,
    counts[q] = sum_v shifted_occ[q + v] * D[v].
    """
    contact = support.downward_facing_mask(item)
    shifted = np.zeros(occ.shape, dtype=np.int64)
    shifted[:, :, 1:] = (occ[:, :, :-1] != 0)
    if ground_at_z0:
        shifted[:, :, 0] = 1

    counts = np.zeros(occ.shape, dtype=np.int64)
    contacts = np.argwhere(contact)
    for q in np.ndindex(occ.shape):
        total = 0
        for v in contacts:
            w = (q[0] + v[0], q[1] + v[1], q[2] + v[2])
            if (
                w[0] < occ.shape[0]
                and w[1] < occ.shape[1]
                and w[2] < occ.shape[2]
            ):
                total += shifted[w]
        counts[q] = total
    return counts, int(contact.sum())


# ── downward_facing_mask ─────────────────────────────────────────────
def test_mask_solid_box():
    item = np.ones((2, 3, 2), dtype=np.int32)
    mask = support.downward_facing_mask(item)
    assert mask[:, :, 0].all(), "바닥층은 전부 접촉면이어야 함"
    assert not mask[:, :, 1].any(), "위층은 접촉면이 아니어야 함"
    assert mask.sum() == 6


def test_mask_overhang():
    # ㄱ자: x=0 기둥(z=0..1) + x=1 처마(z=1만) → 처마 밑면도 접촉면
    item = np.zeros((2, 1, 2), dtype=np.int32)
    item[0, 0, :] = 1
    item[1, 0, 1] = 1
    mask = support.downward_facing_mask(item)
    assert mask[0, 0, 0] and mask[1, 0, 1]
    assert not mask[0, 0, 1]
    assert mask.sum() == 2


# ── support_ratio_grid ───────────────────────────────────────────────
def test_ratio_box_on_floor():
    occ = np.zeros((5, 5, 4), dtype=np.int32)
    item = np.ones((2, 2, 1), dtype=np.int32)
    ratio, n = support.support_ratio_grid(occ, item, ground_at_z0=True)
    assert n == 4
    assert ratio[0, 0, 0] == 1.0, "바닥 위 배치는 완전 지지"
    assert ratio[3, 3, 0] == 1.0


def test_ratio_floating():
    occ = np.zeros((5, 5, 4), dtype=np.int32)
    item = np.ones((2, 2, 1), dtype=np.int32)
    ratio, _ = support.support_ratio_grid(occ, item, ground_at_z0=True)
    assert ratio[0, 0, 2] == 0.0, "공중 배치는 지지 0"


def test_ratio_half_overhang():
    # z=0에 2x2 받침 블록, 아이템 4x2x1을 z=1에 절반 걸침 → 4/8 = 0.5
    occ = np.zeros((6, 4, 4), dtype=np.int32)
    occ[0:2, 0:2, 0] = 1
    item = np.ones((4, 2, 1), dtype=np.int32)
    ratio, n = support.support_ratio_grid(occ, item, ground_at_z0=False)
    assert n == 8
    assert abs(ratio[0, 0, 1] - 0.5) < 1e-12


def test_ratio_ground_flag_off():
    occ = np.zeros((4, 4, 3), dtype=np.int32)
    item = np.ones((2, 2, 1), dtype=np.int32)
    ratio, _ = support.support_ratio_grid(occ, item, ground_at_z0=False)
    assert ratio[0, 0, 0] == 0.0, "ground_at_z0=False면 바닥 지지 없음"


def test_ratio_empty_item():
    occ = np.zeros((4, 4, 3), dtype=np.int32)
    item = np.zeros((2, 2, 1), dtype=np.int32)
    ratio, n = support.support_ratio_grid(occ, item)
    assert n == 0 and not ratio.any()


def test_ratio_matches_brute_force():
    rng = np.random.default_rng(42)
    for trial in range(4):
        occ = (rng.random((7, 6, 5)) < 0.35).astype(np.int32)
        item = (rng.random((3, 2, 2)) < 0.6).astype(np.int32)
        if item.sum() == 0:
            item[0, 0, 0] = 1
        for ground in (True, False):
            ratio, n = support.support_ratio_grid(occ, item, ground_at_z0=ground)
            ref_counts, ref_n = brute_force_counts(occ, item, ground_at_z0=ground)
            assert n == ref_n
            got = np.rint(ratio * n).astype(np.int64)
            assert np.array_equal(got, ref_counts), (
                f"trial={trial} ground={ground}: FFT 상관과 브루트포스 불일치"
            )


def test_ratio_item_with_trailing_padding():
    # 내용은 2x2x1인데 배열은 4x4x3 (뒤쪽 패딩) — 원점 규약이 유지돼야 함
    occ = np.zeros((6, 6, 4), dtype=np.int32)
    item = np.zeros((4, 4, 3), dtype=np.int32)
    item[0:2, 0:2, 0] = 1
    ratio, n = support.support_ratio_grid(occ, item, ground_at_z0=True)
    ref_counts, ref_n = brute_force_counts(occ, item, ground_at_z0=True)
    assert n == ref_n == 4
    assert np.array_equal(np.rint(ratio * n).astype(np.int64), ref_counts)
    assert ratio[0, 0, 0] == 1.0


# ── com_supported ────────────────────────────────────────────────────
def test_com_full_floor_support():
    occ = np.zeros((5, 5, 3), dtype=np.int32)
    item = np.ones((3, 3, 1), dtype=np.int32)
    assert support.com_supported(item, occ, (0, 0, 0), ground_at_z0=True)


def test_com_single_corner_fails():
    # 3x3 아이템이 귀퉁이 복셀 1개 위에만 얹힘 → COM(1,1)이 지지 영역 밖
    occ = np.zeros((5, 5, 3), dtype=np.int32)
    occ[0, 0, 0] = 1
    item = np.ones((3, 3, 1), dtype=np.int32)
    assert not support.com_supported(item, occ, (0, 0, 1), ground_at_z0=False)


def test_com_two_diagonal_corners_straddle():
    # 대각 두 점 지지 → straddle 폴백으로 통과 (문서화된 v1 근사 동작)
    occ = np.zeros((5, 5, 3), dtype=np.int32)
    occ[0, 0, 0] = 1
    occ[2, 2, 0] = 1
    item = np.ones((3, 3, 1), dtype=np.int32)
    assert support.com_supported(item, occ, (0, 0, 1), ground_at_z0=False)


def test_com_triangle_hull():
    # 세 지지점 (0,0),(0,1),(1,0) → COM(1,1)은 삼각형 밖 → 실패
    occ = np.zeros((5, 5, 3), dtype=np.int32)
    occ[0, 0, 0] = 1
    occ[0, 1, 0] = 1
    occ[1, 0, 0] = 1
    item = np.ones((3, 3, 1), dtype=np.int32)
    assert not support.com_supported(item, occ, (0, 0, 1), ground_at_z0=False)

    # 네 귀퉁이 지지 → COM이 hull 안 → 통과
    occ2 = np.zeros((5, 5, 3), dtype=np.int32)
    for x, y in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        occ2[x, y, 0] = 1
    assert support.com_supported(item, occ2, (0, 0, 1), ground_at_z0=False)


def test_com_no_support_fails():
    occ = np.zeros((5, 5, 3), dtype=np.int32)
    item = np.ones((2, 2, 1), dtype=np.int32)
    assert not support.com_supported(item, occ, (0, 0, 1), ground_at_z0=False)


# ── 단독 실행 러너 ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
