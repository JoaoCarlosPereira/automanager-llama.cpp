# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Run the web manager: `python llama_manager.py`
- Manually start the llama-server: `./start_llama.sh`
- Manage as systemd service: `systemctl restart llama-manager.service` / `systemctl status llama-manager.service`
- View service logs: `journalctl -u llama-manager.service -f`
- Quick-Install: `sudo bash installer/setup.sh`

## Development Rules
For detailed technical guidelines, coding standards, and mandatory workflows, please refer to the [rules.md](rules.md) file.

## Architecture & Structure
- **Core Logic**: `llama_manager.py` is a FastAPI application that acts as a control plane for `llama-server`.
- **Process Management**: It manages `llama-server` instances using `subprocess.Popen` and `pkill -9`.
- **Hardware Integration**: Uses `nvidia-smi` and `llama-server --help` to detect GPUs and VRAM for tensor split optimization.
- **Frontend**: A single-page dashboard built with Tailwind CSS, embedded directly within the FastAPI routes (`/` endpoint). JS modules live in `static/js/`.
- **Path Configuration**: `paths.py` loads install paths from `paths.json` (created from `paths.json.example` on first install).
- **Model Discovery**: Recursively scans `MODELS_DIR` from `paths.json` for `.gguf` files.
- **Logging**: 
    - Manager logs: `{logs_dir}/manager.log`
    - Server logs: `{logs_dir}/server.log` (streamed via `/logs` SSE endpoint)

## Key File Paths
- Application: `llama_manager.py` (repo root)
- Path config: `paths.json` (gitignored; defaults in `paths.json.example`)
- Config file: `data/automanager_config.json` relative to install root (stores `default_model`, auth, per-model settings)
- Models directory: `data/models/` relative to install root
- Logs directory: `logs/` relative to install root
- Systemd service: `/etc/systemd/system/llama-manager.service`
- Legacy layout (auto-detected under `/root` with existing dirs): `/media/docker/models`, `/root/automanager_config.json`

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI (HTML) |
| GET | `/status` | llama-server running status + model name |
| GET | `/metrics` | CPU, RAM, GPU utilization/temp/power/VRAM |
| GET | `/models` | List all `.gguf` models with paths |
| GET | `/downloads` | Track active download progress |
| POST | `/downloads` | Start download with `{url}` |
| GET | `/logs` | SSE stream of llama-server log output |
| GET | `/config` | Current configuration |
| POST | `/start` | Start llama-server with `{path, gpu_weights, context_size}` |
| POST | `/stop` | Kill running llama-server |
| POST | `/set_default` | Set default model with `{path: string|null}` |
| POST | `/models/dir` | Update models directory path |

## Server Configuration
- Dynamic IP: The application detects the local IP automatically.
- llama-server binds to `0.0.0.0:8085`
- Manager (FastAPI) binds to `0.0.0.0:8000`
- Default context size: 65536 tokens
