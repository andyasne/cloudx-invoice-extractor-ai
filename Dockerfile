# Cloudx Invoice AI - Dockerfile
# Multi-stage build for efficient image size

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Training image
FROM base as training

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for training on CPU, modify for GPU)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Install package in development mode
RUN pip install -e .

# Default command for training
CMD ["python", "train.py"]


# Stage 3: API inference image
FROM base as api

# Copy requirements (lighter for API)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy only necessary files for API
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_api.py .
COPY setup.py .

# Install package
RUN pip install -e .

# Create directory for model checkpoints
RUN mkdir -p /app/models/checkpoints

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command for API
CMD ["python", "run_api.py", "--checkpoint", "/app/models/checkpoints/best_model.ckpt"]


# Stage 4: GPU training image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu-training

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Default command
CMD ["python", "train.py"]
