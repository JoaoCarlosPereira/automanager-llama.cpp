# Automanager Llama.cpp

A lightweight FastAPI-based web manager for llama.cpp servers.

## Features
- **Auto-Discovery:** Automatically scans for .gguf models in /media/docker/models.
- **GPU Optimization:** 
  - Priority support for RTX 3090.
  - Configurable Tensor Split (default 95% on primary GPU).
  - Flash Attention and MLock enabled by default.
- **OpenAI Compatible:** Serves models via an OpenAI-compliant API.
- **Systemd Integration:** Runs as a background service on Ubuntu.

## Components
- `gemma_manager.py`: The main FastAPI application that provides the web UI and controls the `llama-server` processes.
- `start_gemma.sh`: A helper shell script to manually start a specific Gemma model with high-performance settings.

## Installation
Ensure you have llama.cpp compiled with CUDA support and fastapi, uvicorn, psutil installed in your Python environment.

## Usage
Run the manager:
python gemma_manager.py

Access the UI at http://localhost:8000.
