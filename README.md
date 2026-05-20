# 🚀 Automanager Llama.cpp

**A FastAPI control plane for orchestrating `llama-server` on NVIDIA multi-GPU Linux hosts.**

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

> **Alpha software.** Tuned for dedicated GPU servers (multi-GPU NVIDIA). APIs and defaults may change.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

---

## Quick Start

1. **Clone** the repository on your Linux server:

   ```bash
   git clone <repository-url> automanager-llama.cpp
   cd automanager-llama.cpp
   ```

2. **Install** with the Quick-Install script (requires root, Ubuntu/Debian, NVIDIA drivers, and `llama-server` on `PATH`):

   ```bash
   sudo bash installer/setup.sh
   ```

3. **Open the dashboard** at `http://<server-ip>:8000/` (default credentials are set on first run; change the admin password after login).

---

## Features

| Feature | Description | Status |
|--------|-------------|--------|
| **Multi-GPU orchestration** | Select GPUs, tune tensor split, and map `CUDA_VISIBLE_DEVICES` to `llama-server` device IDs | Stable |
| **OOM self-healing** | Watches server logs; on OOM, reduces primary GPU weight and retries until stable or critical failure | Stable |
| **Model library** | Recursive `.gguf` scan, rename, delete, default model, per-model settings | Stable |
| **URL downloads** | Download models into the models tree with progress tracking | Stable |
| **Live metrics** | CPU, RAM, and per-GPU utilization, temperature, power, VRAM via `nvidia-smi` | Stable |
| **Status lifecycle** | OFFLINE → STARTING → ONLINE → (REALOCANDO) → STOPPING; UI dims metrics when offline | Stable |
| **Log streaming** | SSE console tail of `llama-server` output | Stable |
| **Session + API key auth** | Cookie sessions and Bearer token for API clients | Stable |
| **Quick-Install** | Idempotent `installer/setup.sh` (venv, systemd, health check) | Stable |
| **Modular codebase** | Domain logic split across focused modules (`config_manager`, `log_manager`, orchestration in `llama_manager.py`) | In progress |

Default inference settings include **Flash Attention**, **mlock**, full GPU layer offload (`-ngl 99`), and a default context window of **65536** tokens (`DEFAULT_CONTEXT_SIZE`).

---

## Architecture

Automanager is a **control plane** (FastAPI on port **8000**) that manages a single **`llama-server`** child process (port **8085**). The dashboard is embedded HTML/JS served from `GET /`.

```
┌─────────────┐     HTTP :8000      ┌──────────────────────────────────────┐
│   Browser   │ ──────────────────► │  llama_manager.py (FastAPI)          │
│  Dashboard  │ ◄── SSE /status ─── │  Routes · auth · UI · DI wiring      │
└─────────────┘                     └───────────┬──────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
           ┌────────────────┐          ┌─────────────────┐          ┌─────────────────┐
           │ config_manager │          │  GPUDetector /  │          │ ProcessManager  │
           │ TokenManager   │          │  GPU weights    │          │ OOMWatchdog     │
           │ AuthManager    │          │  nvidia-smi     │          │ subprocess      │
           └────────┬───────┘          └────────┬────────┘          └────────┬────────┘
                    │                           │                            │
                    │                           │                            ▼
                    │                           │                   ┌─────────────────┐
                    │                           │                   │  llama-server   │
                    │                           └──────────────────►│  :8085 / GPUs   │
                    │                                               └─────────────────┘
                    ▼
           /root/automanager_config.json
           /media/docker/models/*.gguf
```

**Modular layout (post-refactor):**

| Module | Responsibility |
|--------|----------------|
| `llama_manager.py` | FastAPI app, routes, `ProcessManager`, `GPUDetector`, `ModelScanner`, embedded UI |
| `config_manager.py` | JSON config, API keys, admin auth |
| `log_manager.py` | Manager logging setup, server log paths, SSE streaming |
| `installer/setup.sh` | Quick-Install: deps, venv, systemd, health check |

Target end state (ADR-001): further extraction into `gpu_manager.py`, `process_manager.py`, and `ui_renderer.py` with `llama_manager.py` as a thin composition root.

---

## API Reference

Base URL: `http://<host>:8000`. Most endpoints require a valid **session cookie** or **`Authorization: Bearer <api-key>`** (from `GET /api/key`). `GET /logs` streams without auth for the embedded console.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard UI (HTML) |
| `GET` | `/status` | `llama-server` running state and loaded model |
| `GET` | `/metrics` | CPU, RAM, GPU utilization / temp / power / VRAM |
| `GET` | `/models` | List `.gguf` models under `MODELS_DIR` |
| `GET` | `/downloads` | Active download progress |
| `POST` | `/downloads` | Start download with `{ "url": "...", "filename": "optional" }` |
| `GET` | `/logs` | SSE stream of server log output |
| `GET` | `/config` | Current configuration (password hash omitted) |
| `POST` | `/start` | Start server: `{ "path", "gpu_weights", "context_size", "mmproj_path?", "split_mode?" }` |
| `POST` | `/stop` | Stop running `llama-server` |
| `POST` | `/set_default` | Set default model: `{ "path": string \| null }` |
| `POST` | `/rename` | Rename model file: `{ "path", "new_name" }` |
| `POST` | `/delete` | Delete model file: `{ "path" }` |
| `POST` | `/api/auth/login` | Session login: `{ "username", "password" }` |
| `POST` | `/api/auth/logout` | End session |
| `POST` | `/api/auth/change-password` | Change admin password |
| `GET` | `/api/key` | Return or create API key |
| `POST` | `/api/key/renew` | Rotate API key |

**Ports:** Manager `0.0.0.0:8000` · `llama-server` `0.0.0.0:8085`

---

## Hardware Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Linux — Ubuntu 22.04+ or Debian 11+ (Quick-Install target) |
| **GPU** | One or more NVIDIA GPUs with working drivers |
| **`nvidia-smi`** | Must run successfully and list at least one GPU |
| **`llama-server`** | Pre-built binary on `PATH` (not installed by setup script) |
| **RAM / VRAM** | Depends on model and `context_size`; multi-GPU setups use tensor split |
| **Disk** | Space for `.gguf` models under `MODELS_DIR` (default `/media/docker/models`) |

CUDA toolkit installation is assumed if your `llama-server` build requires it; the installer does not install drivers or CUDA.

---

## Installation

### Prerequisites

- Ubuntu 22.04+ or Debian 11+
- `sudo` / root access
- NVIDIA drivers + `nvidia-smi`
- `llama-server` on `PATH`
- Git clone of this repository

### Quick-Install (recommended)

From the repository root:

```bash
sudo bash installer/setup.sh
```

The script will:

1. Verify Ubuntu/Debian and root privileges  
2. Install `python3`, `python3-venv`, `python3-pip`, `curl`, `git`, `lsb-release`  
3. Warn if `llama-server` is missing  
4. Require at least one NVIDIA GPU  
5. Create or reuse `.venv/` and `pip install -r requirements.txt`  
6. Create `logs/` and ensure `/root/` exists for config  
7. Install and enable `llama-manager.service`  
8. Run `curl http://localhost:8000/status` and print the dashboard URL  

The script is **idempotent**: safe to re-run; it refreshes dependencies, rewrites the unit file, and restarts the service.

### Manual install

```bash
cd automanager-llama.cpp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs
python llama_manager.py
```

### systemd (manual)

Example unit (Quick-Install writes a similar file to `/etc/systemd/system/llama-manager.service`):

```ini
[Service]
WorkingDirectory=/path/to/automanager-llama.cpp
ExecStart=/path/to/automanager-llama.cpp/.venv/bin/python llama_manager.py
User=root
Restart=on-failure
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-manager.service
journalctl -u llama-manager.service -f
```

---

## Configuration

| Item | Default / path |
|------|----------------|
| **Manager port** | `8000` |
| **Server port** | `8085` |
| **Default context** | `65536` tokens (`DEFAULT_CONTEXT_SIZE`) |
| **Models directory** | `/media/docker/models` (`MODELS_DIR`) |
| **Main config** | `/root/automanager_config.json` (default model, per-model GPU/context settings, auth) |
| **Manager log** | `/root/manager.log` (fallback: `./manager.log`) |
| **Server log** | `/root/llama_server.log` (SSE `/logs`; project `logs/` used as refactor lands) |
| **Project logs dir** | `./logs/` (created by installer and at runtime) |

Change `MODELS_DIR` and paths in `llama_manager.py` constants if your layout differs.

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| **Health check failed after install** | `systemctl status llama-manager.service` and `journalctl -u llama-manager.service -n 50` |
| **FALHA CRÍTICA / OOM** | Model + `context_size` exceeds total VRAM on selected GPUs; reduce context or use a smaller quant |
| **No models listed** | `MODELS_DIR` exists and contains `.gguf` files; permissions for the service user |
| **`llama-server` not found** | Install binary and ensure `PATH` in the systemd unit includes its location |
| **GPU metrics empty** | `nvidia-smi` works as the service user; driver mismatch |
| **401 on API calls** | Log in via dashboard or pass `Authorization: Bearer <key>` from `GET /api/key` |
| **Logs not updating in UI** | Verify server log path exists and is writable (`/root/llama_server.log`) |

---

## Development

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python llama_manager.py
```

### Tests

```bash
pip install -r requirements.txt
pytest
```

Unit tests cover config/token helpers, GPU scanning, and OOM watchdog behavior; integration tests exercise auth and API routes with mocks.

### Project layout

```
automanager-llama.cpp/
├── llama_manager.py      # FastAPI app, routes, core services
├── config_manager.py     # Config, tokens, auth
├── log_manager.py        # Logging and SSE
├── installer/setup.sh    # Quick-Install
├── requirements.txt
├── tests/
├── design/               # Static design reference (not served by default)
└── start_llama.sh        # Example manual llama-server launch
```

Coding standards and workflows: see [rules.md](rules.md). Agent-oriented notes: [CLAUDE.md](CLAUDE.md).

### Contributing

1. Fork and branch from `main`.  
2. Keep changes focused; match existing style in `llama_manager.py`.  
3. Run `pytest` before opening a PR.  
4. Update README/API tables if you add or change endpoints.

---

## License

Copyright 2026 Automanager Llama.cpp contributors.

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the full text.
