# SageMaker training image for this repository. Build from the repository root:
#   DOCKER_BUILDKIT=1 docker build -f aws/Dockerfile -t <image> .
# Both Python environments are built by scripts/setup_envs.sh (pinned OpenPI + LIBERO); the CPU unit tests and the
# frozen-module check run during the build. No weights, data, .env or keys go into the image.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs python3 python3-pip build-essential \
      libegl1 libgl1 libgles2 libglvnd0 libglx0 libopengl0 libosmesa6 \
      libglib2.0-0 libglfw3 libglew2.2 ffmpeg patchelf \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir uv && uv --version
WORKDIR /opt/vla
COPY . /opt/vla
RUN printf '%s\n' \
      'OPENPI_DATA_HOME=/opt/build-assets/openpi' \
      'LIBERO_DATA_ROOT=/opt/build-assets/libero' \
      'CLVLA_OUTPUT_ROOT=/opt/build-assets/outputs' \
      'CLVLA_PROCESSED_ROOT=/opt/build-assets/processed' \
      'HF_LEROBOT_HOME=/opt/build-assets/lerobot' \
      'UV_CACHE_DIR=/opt/uv-cache' \
      'UV_PYTHON_INSTALL_DIR=/opt/uv-python' \
      'HF_HOME=/opt/build-assets/hf' > .env \
    && bash scripts/setup_envs.sh \
    && sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 \
    && scripts/py-policy -m pytest tests/unit -q -p no:cacheprovider \
    && rm -rf /opt/uv-cache /opt/build-assets .env \
    && chmod +x aws/*.sh
ENTRYPOINT ["/opt/vla/aws/entrypoint.sh"]
