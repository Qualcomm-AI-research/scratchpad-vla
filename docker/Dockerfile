# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MANISKILL_REF=v0.5.3
ARG CLEVRSKILLS_REPO=https://github.com/Qualcomm-AI-research/ClevrSkills.git
ARG CLEVRSKILLS_REF=main
ARG USER_UID=1000
ARG USER_GID=1000

ENV NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MANISKILL2_ROOT=/ManiSkill2 \
    CLEVRSKILLS_ROOT=/ClevrSkills \
    PYTHON_BIN=python3 \
    HOME=/home/scratchpad \
    HF_HOME=/workspace/hf_cache \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace/scratchpad-vla:/ManiSkill2:/ClevrSkills

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        less \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglvnd-dev \
        libjpeg-dev \
        libpng-dev \
        libsm6 \
        libvulkan1 \
        libxext6 \
        libxrender-dev \
        nano \
        openssh-client \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-setuptools \
        rsync \
        sudo \
        tmux \
        vim \
        wget \
        x11vnc \
        xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" scratchpad && \
    useradd \
        --uid "${USER_UID}" \
        --gid "${USER_GID}" \
        --create-home \
        --shell /bin/bash \
        scratchpad

RUN curl --fail --show-error --location https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools==72.1.0 wheel && \
    python3 -m pip install --no-cache-dir numpy==1.23.5 && \
    python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124

RUN git clone --depth 1 --branch "${MANISKILL_REF}" https://github.com/haosulab/ManiSkill.git /ManiSkill2 && \
    python3 -m pip install --no-cache-dir --requirement /ManiSkill2/requirements.txt && \
    python3 -m pip install --no-cache-dir --no-deps --editable /ManiSkill2 && \
    rm -rf /ManiSkill2/.git

RUN git clone --depth 1 --branch "${CLEVRSKILLS_REF}" "${CLEVRSKILLS_REPO}" /ClevrSkills && \
    rm -rf /ClevrSkills/.git

COPY docker/requirements.txt /tmp/scratchpad-requirements.txt
RUN python3 -m pip install --no-cache-dir --requirement /tmp/scratchpad-requirements.txt && \
    git config --global --add safe.directory '*'

# Vulkan/EGL configuration needed by ManiSkill/SAPIEN when using NVIDIA GPUs in Docker.
COPY docker/deps/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/deps/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY docker/deps/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/entrypoint.sh /usr/local/bin/scratchpad-entrypoint.sh
RUN chmod 644 \
        /usr/share/vulkan/icd.d/nvidia_icd.json \
        /etc/vulkan/implicit_layer.d/nvidia_layers.json \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    chmod 755 /usr/local/bin/scratchpad-entrypoint.sh

RUN mkdir -p /workspace/hf_cache /workspace/scratchpad-vla && \
    chown -R scratchpad:scratchpad \
        /home/scratchpad \
        /workspace \
        /ManiSkill2 \
        /ClevrSkills
WORKDIR /workspace/scratchpad-vla
COPY --chown=scratchpad:scratchpad . /workspace/scratchpad-vla/

ENV USER=scratchpad \
    LOGNAME=scratchpad
USER scratchpad
ENTRYPOINT ["/usr/local/bin/scratchpad-entrypoint.sh"]
CMD ["/bin/bash"]
