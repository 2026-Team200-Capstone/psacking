#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# 시스템 Python에 직접 설치
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
python3 -m pip install networkx Pillow

# _core.so 심볼릭 링크 (Blender Python이 editable install의 .so를 찾을 수 있도록)
CORE_SO=$(find /workspace/build -name "_core*.so" | head -1)
if [[ -n "${CORE_SO}" ]]; then
    ln -sf "${CORE_SO}" /workspace/spectral_packer/$(basename "${CORE_SO}")
fi

# Blender 내장 Python 경로 동적 탐색 (ENV 무시하고 직접 찾기)
unset BLENDER_PYTHON
BLENDER_PYTHON=$(find /opt/blender -name "python3*" -type f | grep "/bin/" | head -1)

if [[ -n "${BLENDER_PYTHON:-}" ]]; then
    SYSTEM_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    BLENDER_SITE=$($BLENDER_PYTHON -c "import site; print(site.getsitepackages()[0])")
    echo "${SYSTEM_SITE}" > "${BLENDER_SITE}/system_site.pth"
    echo "[entrypoint] Blender Python: ${BLENDER_PYTHON}"
    echo "[entrypoint] system_site.pth → ${BLENDER_SITE}"
fi

exec "$@"
