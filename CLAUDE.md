# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Run the web manager: `python gemma_manager.py`
- Manually start the llama-server: `./start_gemma.sh`

## Architecture & Structure
- **Core Logic**: `gemma_manager.py` is a FastAPI application that acts as a control plane for `llama-server`.
- **Process Management**: It manages `llama-server` instances using `subprocess.Popen` and `pkill`.
- **Hardware Integration**: Uses `nvidia-smi` and `llama-server --help` to detect GPUs and VRAM for tensor split optimization.
- **Frontend**: A single-page dashboard built with Tailwind CSS, embedded directly within the FastAPI routes.
- **Model Discovery**: Recursively scans `/media/docker/models` for `.gguf` files.
- **Logging**: 
    - Manager logs: `/root/manager.log`
    - Server logs: `/root/gemma_server.log` (streamed via `/logs` endpoint).
