# 🚀 Automanager Llama.cpp

**Languages:** [English](README.md) · [Português (BR)](README.pt-BR.md)

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
- [Hybrid platform integrations](#hybrid-platform-integrations)
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
   git clone https://github.com/JoaoCarlosPereira/automanager-llama.cpp.git automanager-llama.cpp
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
| **Hybrid platform models** | Detect Codex, Claude Code, and Google Antigravity CLI tools and expose them through the same dashboard/API flow | MVP |
| **Quick-Install** | Idempotent `installer/setup.sh` (venv, systemd, health check) | Stable |
| **Modular codebase** | Domain logic in focused modules; `llama_manager.py` is the composition root (routes + UI) | Stable |

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

**Modular layout:**

| Module | Responsibility |
|--------|----------------|
| `llama_manager.py` | FastAPI app, routes, auth wiring, embedded dashboard UI |
| `config_manager.py` | JSON config, API keys, admin auth |
| `gpu_manager.py` | GPU detection, tensor split, `CUDA_VISIBLE_DEVICES` |
| `process_manager.py` | `llama-server` subprocess, OOM watchdog |
| `model_manager.py` | Model scan, rename/delete, URL downloads |
| `platform_manager.py` | Startup detection and CLIProxyAPI sidecar lifecycle for platform integrations |
| `proxy_router.py` | Smart proxy backend selection, sticky sessions, and local/platform routing |
| `log_manager.py` | Rotating logs under `logs/`, SSE streaming |
| `schemas.py` | Pydantic request/response models |
| `paths.py` | Install path resolution (`paths.json`) |
| `installer/setup.sh` | Quick-Install: deps, venv, systemd, health check |
| `installer/uninstall.sh` | Remove systemd service and venv; `--purge` removes config/logs |

---

## Hybrid platform integrations

AutoManager can show supported subscription-backed CLI tools beside local
`.gguf` models. The MVP supports Codex, Claude Code, and Google Antigravity.

- Detection runs once when AutoManager starts. Restart AutoManager after you
  install, remove, or move one of the supported tools.
- AutoManager does not collect provider credentials, API keys, or platform
  logins. It uses the authentication state that already exists in the
  installed CLI tools.
- Starting a platform card starts a shared local CLIProxyAPI sidecar. Active
  platform integrations appear in `/status`, and their real sidecar model IDs
  pass through `/v1/models`.
- AutoManager includes a **Smart Proxy Mode**. When enabled, you set a "primary"
  model (local or platform). Clients requesting the primary model are routed
  to it. Concurrent requests (or requests matching a specific load-balancing strategy)
  are dynamically routed to other "proxy-eligible" secondary models (local
  or platform).
- You can configure a **Default Proxy Model** for each platform integration.
  When a platform backend acts as a secondary backend for the proxy (answering
  a request originally intended for the primary model), it will automatically
  use this default model ID to fulfill the request.
- You can also map custom names to any model using the UI's **Alias** feature,
  bypassing strict client-side model name validation.
- If a supported CLI is detected but CLIProxyAPI is missing, the card stays
  visible with a not-ready reason instead of disappearing.

---

## API Reference

Base URL: `http://<host>:8000`. Most endpoints require a valid **session cookie** or **`Authorization: Bearer <api-key>`** (from `GET /api/key`). `GET /logs` streams without auth for the embedded console.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard UI (HTML) |
| `GET` | `/status` | Local and platform runtime state |
| `GET` | `/metrics` | CPU, RAM, GPU utilization / temp / power / VRAM |
| `GET` | `/models` | List `.gguf` models and platform integration cards |
| `GET` | `/downloads` | Active download progress |
| `POST` | `/downloads` | Start download with `{ "url": "...", "filename": "optional" }` |
| `GET` | `/logs` | SSE stream of server log output |
| `GET` | `/config` | Current configuration (password hash omitted) |
| `POST` | `/start` | Start server: `{ "path", "gpu_weights", "context_size", "mmproj_path?", "split_mode?" }` |
| `POST` | `/stop` | Stop running `llama-server` |
| `POST` | `/platforms/{backend_id}/start` | Start a platform integration and the shared sidecar |
| `POST` | `/platforms/{backend_id}/stop` | Stop a platform integration; stops the sidecar when no platform remains active |
| `POST` | `/models/proxy` | Update proxy eligibility for a local `model_path` or platform `backend_id` |
| `POST` | `/proxy/config` | Update smart proxy settings, including `primary_model_path` or `primary_backend_id` |
| `GET` | `/v1/models` | OpenAI-compatible model list from local servers and active platform sidecar |
| `*` | `/v1/{path}` | OpenAI-compatible forwarding to local servers or the platform sidecar |
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
| **Disk** | Space for `.gguf` models under `models_dir` from `paths.json` (default `data/models/`) |

CUDA toolkit installation is assumed if your `llama-server` build requires it; the installer does not install drivers or CUDA.

---

## Installation

### Prerequisites

- Ubuntu 22.04+ or Debian 11+
- Python 3.11+
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

1. Verify Ubuntu/Debian, root privileges, and Python 3.11+  
2. Install `python3`, `python3-venv`, `python3-pip`, `python3-dev`, `curl`, `git`, `lsb-release`  
3. Warn if `llama-server` is missing  
4. Require at least one NVIDIA GPU  
5. Create `paths.json` from `paths.json.example` when missing  
6. Create or refresh `.venv/` and `pip install -r requirements.txt`  
7. Create configured directories (`data/models`, `data/`, `logs/`) via `paths.py`  
8. Install and enable `llama-manager.service`  
9. Run `curl http://localhost:8000/` (public dashboard) and print the dashboard URL  

The script is **idempotent**: safe to re-run; it refreshes dependencies, rewrites the unit file, and restarts the service.

### Manual install

```bash
cd automanager-llama.cpp
cp paths.json.example paths.json   # skip if paths.json already exists
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from paths import ensure_directories; ensure_directories()"
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
| **Path config** | `paths.json` at install root (from `paths.json.example`; gitignored) |
| **Models directory** | `data/models/` (relative to install root; editable via dashboard or `POST /models/dir`) |
| **Main config** | `data/automanager_config.json` (default model, per-model GPU/context settings, auth) |
| **Manager log** | `logs/manager.log` |
| **Server log** | `logs/server.log` |
| **Log rotation** | 10 MB per file, 3 backups (`RotatingFileHandler`) |

Edit `paths.json` to use absolute paths or a legacy layout. Installs under `/root` with existing `/media/docker/models` or `/root/automanager_config.json` auto-select legacy defaults via `paths.py`.

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| **Health check failed after install** | `systemctl status llama-manager.service` and `journalctl -u llama-manager.service -n 50` |
| **FALHA CRÍTICA / OOM** | Model + `context_size` exceeds total VRAM on selected GPUs; reduce context or use a smaller quant |
| **No models listed** | `models_dir` from `paths.json` exists and contains `.gguf` files; permissions for the service user |
| **`llama-server` not found** | Install binary and ensure `PATH` in the systemd unit includes its location |
| **GPU metrics empty** | `nvidia-smi` works as the service user; driver mismatch |
| **401 on API calls** | Log in via dashboard or pass `Authorization: Bearer <key>` from `GET /api/key` |
| **Logs not updating in UI** | Verify `logs/server.log` (or path from `paths.json`) exists and is writable |

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
pip install -r requirements-dev.txt
pytest
```

Unit tests cover config/token helpers, GPU scanning, and OOM watchdog behavior; integration tests exercise auth and API routes with mocks.

### Project layout

```
automanager-llama.cpp/
├── llama_manager.py      # FastAPI app, routes, UI
├── config_manager.py     # Config, tokens, auth
├── gpu_manager.py        # GPU detection and tensor split
├── process_manager.py    # Subprocess + OOM watchdog
├── model_manager.py      # Models and downloads
├── log_manager.py        # Logging and SSE
├── schemas.py            # Pydantic models
├── paths.py              # Install path resolution (paths.json)
├── paths.json.example    # Default path template for new installs
├── installer/setup.sh      # Quick-Install
├── installer/uninstall.sh  # Remove service/venv (--purge for config/logs)
├── static/js/            # Dashboard assets (e.g. Pac-Man background)
├── logs/                 # Runtime logs (gitignored)
├── requirements.txt
├── requirements-dev.txt  # Prod deps + pytest/httpx for development
├── tests/
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
