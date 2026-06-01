# ============================================================
# Dockerfile - bge-m3 GPU Inference for Job Matching
# Base: NVIDIA CUDA 12.8 + Python 3.12
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install Python 3.12 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    python3.12-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Working directory
WORKDIR /app

# Install pip dependencies first (cached layer)
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.docker.txt

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Source code + data are mounted as volumes via docker-compose.yml

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "src/app.py"]
