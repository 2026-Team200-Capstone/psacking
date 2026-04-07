FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    libfftw3-dev \
    wget \
    xz-utils \
    libx11-6 \
    libxi6 \
    libxxf86vm1 \
    libxrender1 \
    libgl1 \
    libxkbcommon0 \
    libsm6 \
    libice6 \
 && rm -rf /var/lib/apt/lists/*

# Blender 3.6 LTS 설치 (Python 3.10 — Ubuntu 22.04 기본 Python과 동일)
ARG BLENDER_VERSION=3.6.0
RUN wget -q https://download.blender.org/release/Blender3.6/blender-${BLENDER_VERSION}-linux-x64.tar.xz -O /tmp/blender.tar.xz \
 && tar -xf /tmp/blender.tar.xz -C /opt \
 && mv /opt/blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
 && rm /tmp/blender.tar.xz

ENV PATH="/opt/blender:${PATH}"

WORKDIR /workspace

COPY docker/entrypoint.sh /usr/local/bin/project-entrypoint
RUN chmod +x /usr/local/bin/project-entrypoint

ENTRYPOINT ["project-entrypoint"]
CMD ["bash"]
