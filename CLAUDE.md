# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Run the web manager: `python llama_manager.py`
- Manually start the llama-server: `./start_llama.sh`
- Manage as systemd service: `systemctl restart llama-manager.service` / `systemctl status llama-manager.service`
- View service logs: `journalctl -u llama-manager.service -f`

## Development Rules
For detailed technical guidelines, coding standards, and mandatory workflows, please refer to the [rules.md](rules.md) file.

## Architecture & Structure
- **Core Logic**: `llama_manager.py` is a FastAPI application that acts as a control plane for `llama-server`.
- **Process Management**: It manages `llama-server` instances using `subprocess.Popen` and `pkill -9`.
- **Hardware Integration**: Uses `nvidia-smi` and `llama-server --help` to detect GPUs and VRAM for tensor split optimization.
- **Frontend**: A single-page dashboard built with Tailwind CSS, embedded directly within the FastAPI routes (`/` endpoint).
- **Model Discovery**: Recursively scans `/media/docker/models` for `.gguf` files.
- **Logging**: 
    - Manager logs: `/root/manager.log`
    - Server logs: `/root/llama_server.log` (streamed via `/logs` SSE endpoint).

## Key File Paths
- Application: `/root/automanager-llama.cpp/llama_manager.py`
- Config file: `/root/automanager_config.json` (stores `default_model`)
- Models directory: `/media/docker/models/`
- Systemd service: `/etc/systemd/system/llama-manager.service`

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI (HTML) |
| GET | `/status` | llama-server running status + model name |
| GET | `/metrics` | CPU, RAM, GPU utilization/temp/power/VRAM |
| GET | `/models` | List all `.gguf` models with paths |
| GET | `/downloads` | Track active download progress |
| GET | `/downloads` | POST with `{url}` to start a download |
| GET | `/logs` | SSE stream of llama-server log output |
| GET | `/config` | Current configuration |
| POST | `/start` | Start llama-server with `{path, gpu_weights, context_size}` |
| POST | `/stop` | Kill running llama-server |
| POST | `/set_default` | Set default model with `{path: string|null}` |

## Server Configuration
- Dynamic IP: The application detects the local IP automatically.
- llama-server binds to `0.0.0.0:8085`
- Manager (FastAPI) binds to `0.0.0.0:8000`
- Default context size: 65536 tokens
