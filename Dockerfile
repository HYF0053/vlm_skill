# Use NVIDIA CUDA 13.0 with cuDNN base image on Ubuntu 22.04
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# Set environment variables for non-interactive apt-get and UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Install system dependencies required for OpenCV, Git, and Building packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip to the latest version, and install build tools
RUN pip install --no-cache-dir --upgrade pip ninja wheel setuptools

# Install PyTorch 2.9 for CUDA 13.0
# Note: The index-url might vary slightly depending on the official PyTorch wheel naming for CUDA 13.x
RUN pip install --no-cache-dir torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Set up working directory
WORKDIR /app

# Copy requirements and install third-party standard libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Disable the 'yolo' CLI to enforce Python-API usage
RUN echo 'alias yolo="echo '\''CLI 已禁用, 請使用 Python 腳本 (create_config.py/vision_model_trainer.py)'\''"' >> /etc/bash.bashrc

# Expose ports for connecting to the model or UI services
# 7860 is for Gradio, and 63574 was identified as a web service port
EXPOSE 7860
EXPOSE 63574
