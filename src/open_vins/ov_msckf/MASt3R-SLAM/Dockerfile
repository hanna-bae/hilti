FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws

# 1) OS deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential cmake pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

    
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# 2) venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -U pip setuptools wheel

# 3) PyTorch (CUDA 12.4 wheel)
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

RUN pip install \
    numpy pyzmq opencv-python-headless


CMD ["bash"]
