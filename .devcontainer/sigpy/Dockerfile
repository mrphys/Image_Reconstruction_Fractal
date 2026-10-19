# TF-free devcontainer for the sigpy backend (numpy on CPU, cupy on GPU).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo git git-lfs python3 python3-pip python3-venv ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

COPY requirements-sigpy.txt /tmp/requirements-sigpy.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements-sigpy.txt

# synthmr picks the backend from this variable (default would be tf).
ENV SYNTHMR_BACKEND=sigpy
