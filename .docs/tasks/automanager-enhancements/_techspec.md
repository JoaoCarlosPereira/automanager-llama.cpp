# Technical Specification: Automanager Llama.cpp Enhancements

**Slug:** automanager-enhancements  
**Date:** 2026-05-19  
**Status:** Draft — Pending Approval  
**Based on:** [PRD](.docs/tasks/automanager-enhancements/_prd.md)  
**Project:** Automanager Llama.cpp  
**Language:** English

---

## 1. Executive Summary

This specification defines the technical implementation of 6 enhancements to the Automanager Llama.cpp, a FastAPI-based control plane for `llama-server` instance orchestration. The current codebase is a single 2408-line monolithic file (`llama_manager.py`) with ~15% test coverage, unused dependencies, and no modular separation of concerns.

**Primary trade-off:** The modular refactoring approach (ADR-001) increases initial effort (Phase 1 extracts 7 modules from the monolith before any feature work begins) but enables independent testing, lower coupling, and safer incremental feature delivery. The alternative — direct feature implementation in the monolith — would be faster short-term but risks integration breakage across features that touch overlapping code (e.g., GPU enforcement and status accuracy both modify process management logic).

**Scope:** 6 features delivered across 4 phases:
- Phase 1: Modular refactoring (structural, no behavior change)
- Phase 2: GPU strict enforcement + log rotation + status accuracy
- Phase 3: Design overlay + Quick-Install + README redesign
- Phase 4: Validation & testing

---

## 2. System Architecture

### 2.1 Current Architecture

```
llama_manager.py (2408 lines, single file)
├── Constants & Config (lines 1-46)
├── Pydantic Models (lines 70-107)
├── ConfigManager (lines 115-167)
├── TokenManager (lines 175-204)
├── AuthManager (lines 212-271)
├── GPUDetector (lines 279-365)
├── ProcessManager (lines 374-577)
├── OOMWatchdog (lines 585-739)
├── ModelScanner (lines 747-872)
├── DownloadManager (lines 880-959)
├── SSEStreamer (lines 968-990)
├── FastAPI App + Routes (lines 997-1231)
├── index() + _build_html() (lines 1235-2312)
├── startup_event() (lines 2320-2369)
└── Entry point (lines 2406-2407)
```

**Current data flow:**
```
Browser ──HTTP──> FastAPI (llama_manager.py)
                        │
                        ├── ConfigManager ── JSON (/root/automanager_config.json)
                        ├── GPUDetector ── nvidia-smi + llama-server --help
                        ├── ProcessManager ── subprocess.Popen ── llama-server
                        ├── OOMWatchdog ── thread ── SERVER_LOG_PATH polling
                        ├── ModelScanner ── MODELS_DIR glob
                        ├── DownloadManager ── HTTP downloads
                        └── SSEStreamer ── SERVER_LOG_PATH tail ── Browser SSE
```

### 2.2 Target Architecture (Post-Refactoring)

```
llama_manager.py (~400 lines)          ← FastAPI app, routes, dependency injection
├── gpu_manager.py (~250 lines)        ← GPUDetector, GPU weight calculation, tensor split
├── log_manager.py (~180 lines)        ← Log rotation, SSE streaming, log path management
├── process_manager.py (~280 lines)    ← ProcessManager, OOMWatchdog, process lifecycle
├── ui_renderer.py (~600 lines)        ← index(), _build_html(), CSS/JS injection
├── config_manager.py (~120 lines)     ← ConfigManager, TokenManager, AuthManager
├── model_manager.py (~180 lines)      ← ModelScanner, DownloadManager
├── installer/
│   └── setup.sh (~150 lines)          ← Quick-Install script
└── LICENSE                            ← Apache 2.0

Data flow:
Browser ──HTTP──> FastAPI (llama_manager.py)
                        │
                        ├── config_manager ── JSON config
                        ├── gpu_manager ── nvidia-smi + CUDA_VISIBLE_DEVICES
                        ├── process_manager ── llama-server + OOMWatchdog
                        ├── log_manager ── logs/ directory + SSE
                        ├── model_manager ── models directory + HTTP downloads
                        └── ui_renderer ── HTML template (with Pac-Man canvas)
```

### 2.3 Module Dependency Graph

```
llama_manager.py
  ├── gpu_manager (depends on: config_manager)
  ├── log_manager (depends on: config_manager)
  ├── process_manager (depends on: gpu_manager, log_manager, config_manager)
  ├── ui_renderer (depends on: gpu_manager, config_manager, process_manager)
  ├── config_manager (no dependencies)
  ├── model_manager (depends on: config_manager, log_manager)

Internal:
gpu_manager ──→ process_manager ──→ log_manager
  │                        │
  └────────────────────────┘
(OOMWatchdog in process_manager reads from log_manager paths)
```

---

## 3. Module Structure & Interfaces

### 3.1 `config_manager.py`

Extracted from lines 115-204 of current `llama_manager.py`. No behavioral changes.

```python
class ConfigManager:
    def __init__(self, config_path: str = CONFIG_PATH) -> None
    def load(self) -> dict
    def save(self, data: dict) -> None
    def get_model_settings(self, model_path: str) -> dict
    def update_model_settings(self, model_path: str, settings: dict) -> None
    def get_default_model(self) -> Optional[str]
    def set_default_model(self, path: Optional[str]) -> None

class TokenManager:
    def __init__(self, config_manager: ConfigManager) -> None
    def generate(self) -> str
    def validate(self, key: str) -> bool
    def get_or_create(self) -> str
    def renew(self) -> str

class AuthManager:
    def __init__(self, config_manager: ConfigManager) -> None
    def authenticate(self, username: str, password: str) -> bool
    def verify_session(self, session_token: str) -> bool
    def create_session(self, username: str) -> str
    def logout(self, session_token: str) -> None
    def change_password(self, session_token: str, new_password: str) -> bool
```

**Changes:** None. Pure extraction.

### 3.2 `gpu_manager.py`

Extracts GPU detection, metrics, and tensor split logic from lines 279-511.

```python
class GPUInfo:
    index: int
    name: str
    vram: int  # MiB

class GPUWeight:
    index: int
    weight: float
    name: str
    active: bool = True
    is_main: bool = False

class GPUManager:
    def __init__(self, llama_server_bin: str = "llama-server") -> None
    def detect_gpus(self) -> List[GPUInfo]
    def get_metrics(self) -> Dict[str, Any]
    def compute_tensor_split(self, gpu_weights: List[GPUWeight], all_gpus: List[GPUInfo]) -> List[float]
    def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]
    def validate_gpu_weights(self, gpu_weights: List[GPUWeight]) -> tuple[bool, str]
```

**New methods:**
- `compute_tensor_split()`: Filters inactive GPUs, normalizes weights proportionally, returns clean split array.
- `get_visible_devices()`: Returns `CUDA_VISIBLE_DEVICES` string (e.g., `"0,1"`) or `None` if no active GPUs.
- `validate_gpu_weights()`: Returns `(True, "")` if valid, `(False, "error message")` if no active GPUs or weights don't sum correctly.

### 3.3 `log_manager.py`

Extracts logging configuration and SSE streaming from lines 50-63, 545-561, 968-990.

```python
class LogManager:
    def __init__(self, project_root: str, server_log_path: str, manager_log_path: str) -> None
    def setup_manager_logging(self) -> logging.Logger
    def get_server_log_path(self) -> str
    def rotate_server_log(self) -> None
    def clear_server_log(self) -> None
    def stream_logs(self) -> StreamingResponse
```

**New behavior:**
- `setup_manager_logging()`: Keeps existing `basicConfig()` AND adds `RotatingFileHandler` for `logs/manager.log`.
- `get_server_log_path()`: Returns the project-local path (`logs/server.log`).
- `rotate_server_log()`: Triggers rotation (called by ProcessManager on stop).
- `stream_logs()`: SSE stream reads from project-local path instead of `/root/llama_server.log`.

### 3.4 `process_manager.py`

Extracts ProcessManager and OOMWatchdog from lines 374-739.

```python
class ProcessManager:
    def __init__(self, gpu_manager: GPUManager, log_manager: LogManager, config_manager: ConfigManager) -> None
    def start(self, model_path: str, gpu_weights: List[GPUWeight], context_size: int, split_mode: str, mmproj_path: Optional[str] = None) -> dict
    def stop(self) -> bool
    def get_status(self) -> dict

class OOMWatchdog(threading.Thread):
    def __init__(self, process_manager: ProcessManager, log_manager: LogManager, config_manager: ConfigManager) -> None
    def run(self) -> None
    def stop_watchdog(self) -> None
```

**Key changes in `start()`:**
- Uses `gpu_manager.get_visible_devices()` to set `CUDA_VISIBLE_DEVICES` env var.
- Uses `gpu_manager.compute_tensor_split()` for clean split array.
- Calls `log_manager.get_server_log_path()` instead of hardcoded path.

### 3.5 `ui_renderer.py`

Extracts HTML generation from lines 1235-2312.

```python
class UIRenderer:
    def __init__(self, model_scanner, gpu_detector, config_manager, process_manager, token_manager, local_ip: str, is_authenticated: bool) -> None
    def render_dashboard(self) -> str
    def inject_pacman_canvas(self, html: str) -> str
    def extract_design_gradients(self) -> str
```

**Key changes:**
- `render_dashboard()`: Produces HTML with initial OFFLINE state. All status elements check process state before showing active indicators.
- `inject_pacman_canvas()`: Wraps dashboard content with `<canvas id="pacman-background">` at z-index 0.
- `extract_design_gradients()`: Returns CSS custom properties from `design/css/styles.css` for Tailwind override.

### 3.6 `model_manager.py`

Extracts ModelScanner and DownloadManager from lines 747-959. No behavioral changes.

### 3.7 Entry Point (`llama_manager.py`)

Slimmed from 2408 lines to ~400 lines.

```python
# Imports
from config_manager import ConfigManager, TokenManager, AuthManager
from gpu_manager import GPUManager
from log_manager import LogManager
from process_manager import ProcessManager, OOMWatchdog
from model_manager import ModelScanner, DownloadManager
from ui_renderer import UIRenderer

# App setup
app = FastAPI(...)

# Service initialization (injected dependencies)
config = ConfigManager()
token_mgr = TokenManager(config)
auth_mgr = AuthManager(config)
gpu_mgr = GPUManager()
log_mgr = LogManager(project_root, server_log, manager_log)
proc_mgr = ProcessManager(gpu_mgr, log_mgr, config)
oom_wd = OOMWatchdog(proc_mgr, log_mgr, config)
model_scan = ModelScanner(config)
dl_mgr = DownloadManager()

# Route functions reference injected services
@app.get("/status")
async def get_status():
    return proc_mgr.get_status()
```

---

## 4. Data Models

### 4.1 GPUWeight (existing, unchanged)

```python
class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True
    is_main: bool = False
```

### 4.2 GPUInfo (new)

```python
@dataclass
class GPUInfo:
    index: int        # GPU index as reported by nvidia-smi
    name: str         # GPU model name
    vram: int         # Total VRAM in MiB
```

### 4.3 Log Configuration (new constants)

```python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SERVER_LOG_PATH = os.path.join(LOGS_DIR, "server.log")
MANAGER_LOG_PATH = os.path.join(LOGS_DIR, "manager.log")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 3
```

### 4.4 Config Schema (unchanged)

No changes to the JSON config file structure. All new behavior is in module logic, not data format.

---

## 5. Implementation Details

### 5.1 Feature 1: Strict GPU Enforcement

**File:** `gpu_manager.py` + `process_manager.py`

**Implementation:**

```python
# gpu_manager.py — compute_tensor_split
def compute_tensor_split(self, gpu_weights: List[GPUWeight], all_gpus: List[GPUInfo]) -> List[float]:
    active = [w for w in gpu_weights if w.active and w.weight > 0]
    if not active:
        return []
    
    # Normalize weights proportionally among active GPUs
    total = sum(w.weight for w in active)
    return [round(w.weight / total, 4) for w in active]

# gpu_manager.py — get_visible_devices
def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]:
    active = [w for w in gpu_weights if w.active and w.weight > 0]
    if not active:
        return None
    return ",".join(str(w.index) for w in active)
```

**In `ProcessManager.start()`:**

```python
# Before: builds split for ALL detected GPUs
# After: uses GPUManager for clean split + CUDA_VISIBLE_DEVICES
visible_devices = self.gpu_manager.get_visible_devices(gpu_weights)
if not visible_devices:
    raise ValueError("No active GPUs selected. Enable at least one GPU.")

env["CUDA_VISIBLE_DEVICES"] = visible_devices

split = self.gpu_manager.compute_tensor_split(gpu_weights, all_gpus)
cmd.extend(["--tensor-split", ",".join(split)])
```

### 5.2 Feature 2: Log Rotation

**File:** `log_manager.py`

```python
def setup_manager_logging(self) -> logging.Logger:
    logger = logging.getLogger("automanager")
    
    # Existing basicConfig (preserves original behavior)
    # ... (unchanged)
    
    # Add RotatingFileHandler for project-local logs
    os.makedirs(self._logs_dir, exist_ok=True)
    
    rfh = RotatingFileHandler(
        self._manager_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
    )
    rfh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(rfh)
    
    return logger
```

**SSEStreamer update:** Reads from `log_manager.get_server_log_path()` instead of hardcoded path.

### 5.3 Feature 3: Design Overlay

**File:** `ui_renderer.py`

**Pac-Man Canvas injection:**

```python
def inject_pacman_canvas(self, html: str) -> str:
    canvas = '''<canvas id="pacman-background" aria-hidden="true" style="
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0; opacity: 0.35; pointer-events: none;
    "></canvas>'''
    
    # Inject canvas before </body>
    html = html.replace('</body>', f'{canvas}\n<script src="js/pacman_bg.js"></script></body>')
    return html
```

**Gradient CSS extraction:** Key values from `design/css/styles.css` injected as Tailwind config:

```css
/* From design/css/styles.css */
:root {
    --bs-primary: #1e30f3;
    --bs-secondary: #e21e80;
}
/* Injected into Tailwind config within llama_manager.py */
tailwind.config = {
    theme: {
        extend: {
            colors: {
                primary: '#1e30f3',
                secondary: '#e21e80',
            },
            backgroundImage: {
                'gradient-primary': 'linear-gradient(135deg, #1e30f3 0%, #e21e80 100%)',
            }
        }
    }
}
```

**Files to copy from `design/`:**
- `design/js/scripts.js` → `static/js/pacman_bg.js` (extract canvas animation only)
- `design/css/styles.css` → extract gradient/typography variables only

### 5.4 Feature 4: Status OFFLINE

**File:** `ui_renderer.py` + inline JavaScript

**Initial state in HTML:**

```html
<!-- Before any API call -->
<div id="status-badge" class="status-offline">
    <span class="dot dot-slate"></span>
    <span>OFFLINE</span>
</div>
```

**JS initialization (modified `initDashboard`):**

```javascript
function initDashboard() {
    // Initial state is OFFLINE — shown by default in HTML
    // First API call will update to actual state
    updateStatus();
    updateMetrics();
    updateDownloads();
    updateModels();
}

function updateStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            const badge = document.getElementById('status-badge');
            if (data.running) {
                badge.className = 'status-online';
                // ... update to ONLINE state
            } else {
                badge.className = 'status-offline';
                // ... ensure OFFLINE state is displayed
            }
        })
        .catch(() => {
            // Server unreachable — keep OFFLINE (or show warning)
            console.warn('Status endpoint unreachable');
        });
}
```

**GPU metrics dimming when OFFLINE:**

```javascript
function updateMetrics() {
    fetch('/metrics')
        .then(r => r.json())
        .then(data => {
            const meters = document.querySelectorAll('.metric-bar');
            if (!currentRunning) {
                meters.forEach(m => m.classList.add('dimmed'));
            } else {
                meters.forEach(m => m.classList.remove('dimmed'));
                // ... update values
            }
        });
}
```

### 5.5 Feature 5: Quick-Install Script

**File:** `installer/setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Step 1: OS Detection
if [[ ! -f /etc/os-release ]]; then
    log_error "Unsupported OS. This script requires a Linux distribution."
    exit 1
fi

DISTRO=$(. /etc/os-release && echo "$ID")
VERSION=$(. /etc/os-release && echo "$VERSION_ID")

if [[ "$DISTRO" != "ubuntu" && "$DISTRO" != "debian" ]]; then
    log_error "Unsupported distribution: $DISTRO. Expected Ubuntu or Debian."
    exit 1
fi

log_info "Detected: $DISTRO $VERSION"

# Step 2: Sudo verification
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)."
    exit 1
fi

# Step 3: Install system dependencies
log_info "Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl git \
    lsb-release

# Step 4: Verify llama-server
if ! command -v llama-server &> /dev/null; then
    log_warn "llama-server not found in PATH."
    log_warn "Please ensure llama-server binary is installed and accessible."
    log_warn "Download from: https://github.com/ggml-org/llama.cpp/releases"
fi

# Step 5: Verify NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ $GPU_COUNT -eq 0 ]]; then
    log_error "No NVIDIA GPUs detected."
    exit 1
fi
log_info "Detected $GPU_COUNT NVIDIA GPU(s)"

# Step 6: Create virtual environment
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    log_warn "Virtual environment already exists at $VENV_DIR. Removing..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"
log_info "Python dependencies installed"

# Step 7: Create directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p /root  # config directory

# Step 8: Generate systemd service
SERVICE_FILE="/etc/systemd/system/llama-manager.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Automanager Llama.cpp
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/llama_manager.py
Restart=on-failure
RestartSec=5
Environment=PATH=$VENV_DIR/bin:/usr/local/cuda/bin:$PATH
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable llama-manager.service
systemctl start llama-manager.service
log_info "Systemd service configured and started"

# Step 9: Health check
sleep 3
if curl -sf http://localhost:8000/status > /dev/null 2>&1; then
    log_info "Health check passed! Dashboard available at:"
    curl -s http://localhost:8000/status | python3 -m json.tool
    log_info "Automanager Llama.cpp installed successfully."
else
    log_warn "Health check failed. Check service status:"
    log_warn "  systemctl status llama-manager.service"
    log_warn "  journalctl -u llama-manager.service -f"
fi
```

### 5.6 Feature 6: README Redesign

**File:** `README.md`

12-section structure with markdown anchors, TOC, and accurate information. See PRD Section 4 Feature 6 for full spec.

---

## 6. API Changes

### 6.1 No Breaking Changes

All existing API endpoints retain their existing signatures and response formats:

| Endpoint | Change | Type |
|----------|--------|------|
| `GET /status` | No change (still returns `running`, `model`, `config`) | None |
| `POST /start` | No change (same `StartRequest` schema) | None |
| `POST /stop` | No change | None |
| `GET /metrics` | No change (same response structure) | None |
| `GET /models` | No change | None |
| `GET /logs` | SSE stream reads from project-local path instead of `/root/` | Transparent |
| `POST /downloads` | No change | None |
| `GET /downloads` | No change | None |
| `GET /` | HTML structure changes (canvas injection, gradient CSS, OFFLINE initial state) | Non-breaking (UI only) |

### 6.2 New Internal Constants

```python
# log_manager.py
LOGS_DIR = "/path/to/project/logs"
SERVER_LOG_PATH = os.path.join(LOGS_DIR, "server.log")
MANAGER_LOG_PATH = os.path.join(LOGS_DIR, "manager.log")
MAX_LOG_SIZE = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3
```

These replace the hardcoded `/root/llama_server.log` and `/root/manager.log` in internal logic but are not exposed in any API response.

---

## 7. Testing Strategy

### 7.1 Existing Tests (to be preserved and updated)

| Test File | Coverage | Action |
|-----------|----------|--------|
| `test_config_token.py` | 16 tests | Move to `tests/unit/test_config_manager.py` (ConfigManager + TokenManager extracted) |
| `test_gpu_scanner.py` | 8 tests | Move to `tests/unit/test_gpu_manager.py`, add tests for new GPUManager methods |
| `test_oom_watchdog.py` | 9 tests | Move to `tests/unit/test_oom_watchdog.py` in new module context |
| `test_api_endpoints.py` | 17 tests | Update imports to use extracted modules |

### 7.2 New Tests Required

| Test File | Tests to Add | Depends On |
|-----------|-------------|------------|
| `tests/unit/test_gpu_manager_new.py` | `compute_tensor_split` (5 tests), `get_visible_devices` (4 tests), `validate_gpu_weights` (3 tests) | Phase 1 |
| `tests/unit/test_log_manager.py` | `setup_manager_logging` (2 tests), `rotate_server_log` (2 tests), `stream_logs` path (1 test) | Phase 1 |
| `tests/unit/test_process_manager_new.py` | `start` with CUDA_VISIBLE_DEVICES (3 tests), `start` with no active GPUs (1 test) | Phase 2 |
| `tests/unit/test_ui_renderer_new.py` | `inject_pacman_canvas` (1 test), `render_dashboard` OFFLINE state (2 tests) | Phase 3 |
| `tests/unit/test_setup.sh` | Shellcheck validation (1 test), OS detection (2 tests), idempotency (1 test) | Phase 3 |

### 7.3 Integration Tests

| Test | Description | Depends On |
|------|-------------|------------|
| `start_model_with_cuda_visible` | Verify `CUDA_VISIBLE_DEVICES` env var is set correctly on `POST /start` | Phase 2 |
| `log_files_created` | Verify `logs/` directory and files created after startup | Phase 2 |
| `log_rotation_trigger` | Verify file rotation when log exceeds 10MB | Phase 2 |
| `status_offline_initial` | Verify page loads with OFFLINE state | Phase 3 |
| `pacman_canvas_injected` | Verify canvas element present in rendered HTML | Phase 3 |

---

## 8. Migration Plan

### 8.1 Phase 1: Modular Foundation

**Goal:** Extract monolith into modules with zero behavior change.

**Steps:**
1. Create `config_manager.py` — extract ConfigManager, TokenManager, AuthManager (lines 115-271)
2. Create `gpu_manager.py` — extract GPUDetector + GPUInfo dataclass (lines 279-365)
3. Create `log_manager.py` — extract LogManager class wrapping logging config + SSEStreamer (lines 50-63, 968-990)
4. Create `process_manager.py` — extract ProcessManager + OOMWatchdog (lines 374-739)
5. Create `model_manager.py` — extract ModelScanner + DownloadManager (lines 747-959)
6. Create `ui_renderer.py` — extract index() + _build_html() (lines 1235-2312)
7. Refactor `llama_manager.py` — update imports, inject dependencies, slim to ~400 lines
8. Run all existing tests — must pass

**Risk mitigation:** After each extraction, run `python3 -m py_compile` and verify the app starts. Commit each module separately for easy rollback.

### 8.2 Phase 2: Core Features

**Goal:** Implement GPU enforcement, log rotation, status accuracy.

**Steps:**
1. Add `compute_tensor_split()`, `get_visible_devices()`, `validate_gpu_weights()` to `gpu_manager.py`
2. Modify `ProcessManager.start()` to use GPUManager methods and set `CUDA_VISIBLE_DEVICES`
3. Add `RotatingFileHandler` to `log_manager.py`
4. Update `SERVER_LOG_PATH` constant to project-local `logs/server.log`
5. Update SSEStreamer to read from new path
6. Modify `initDashboard()` JS to show OFFLINE initially
7. Add status dimming logic for GPU meters when OFFLINE
8. Run tests (existing + new)

### 8.3 Phase 3: Visual & Deployment

**Steps:**
1. Copy Pac-Man canvas JS from `design/js/scripts.js` → `static/js/pacman_bg.js` (extract canvas animation only, remove portfolio-specific code)
2. Add canvas HTML + CSS to `ui_renderer.py`
3. Extract gradient CSS variables and inject as Tailwind config
4. Create `installer/setup.sh`
5. Redesign `README.md`
6. Add `LICENSE` file (Apache 2.0)
7. Full test suite run

### 8.4 Phase 4: Validation

**Steps:**
1. End-to-end testing on target hardware
2. Quick-Install script test on clean Ubuntu 22.04 VM
3. README accuracy audit
4. Final code review

---

## 9. Development Sequencing (Build Order)

| Step | Action | Depends On |
|------|--------|-----------|
| 1 | Create `config_manager.py` (extract ConfigManager, TokenManager, AuthManager) | — |
| 2 | Create `gpu_manager.py` (extract GPUDetector, add GPUInfo dataclass) | 1 (for config dependency) |
| 3 | Create `log_manager.py` (LogManager + SSE streamer) | — |
| 4 | Create `process_manager.py` (ProcessManager, OOMWatchdog) | 2, 3 (depends on GPU + log) |
| 5 | Create `model_manager.py` (ModelScanner, DownloadManager) | 1 |
| 6 | Create `ui_renderer.py` (index, _build_html) | 2, 4 |
| 7 | Refactor `llama_manager.py` (update imports, inject services, slim to ~400 lines) | 1-6 |
| 8 | Run existing tests, fix any breakage | 7 |
| 9 | Add GPUManager methods: compute_tensor_split, get_visible_devices, validate_gpu_weights | 8 |
| 10 | Modify ProcessManager.start() for CUDA_VISIBLE_DEVICES | 9 |
| 11 | Add RotatingFileHandler to log_manager.py | 3 |
| 12 | Update log paths (SERVER_LOG_PATH → project-local) | 11 |
| 13 | Update SSEStreamer to read from project-local path | 12 |
| 14 | Implement OFFLINE initial state in JS | 7 |
| 15 | Add GPU meter dimming logic | 14 |
| 16 | Copy and adapt Pac-Man canvas JS | 7 |
| 17 | Inject canvas + gradient CSS into UI | 16 |
| 18 | Create `installer/setup.sh` | — |
| 19 | Redesign README.md | 18 |
| 20 | Add LICENSE file (Apache 2.0) | — |
| 21 | Full test suite + E2E validation | 15, 17, 19 |

---

## 10. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Module extraction breaks existing imports | Medium | High | Commit after each module extraction; run py_compile after each step |
| CUDA_VISIBLE_DEVICES changes cause GPU detection failure | Medium | High | Test on actual multi-GPU hardware; keep fallback to existing behavior |
| RotatingFileHandler conflicts with existing basicConfig | Low | Medium | Use same logger name; verify no duplicate log entries |
| Pac-Man canvas JS conflicts with dashboard JS | Low | Low | Canvas JS is self-contained; uses separate canvas element; respects prefers-reduced-motion |
| setup.sh fails on non-standard Ubuntu | Medium | Medium | Support Ubuntu 22.04+ and Debian 11+; clear error messages for unsupported versions |
| README inaccuracies after implementation | Low | Low | README created in Phase 3 after all features are implemented |

---

## 11. Open Questions

| # | Question | Decision Needed By |
|---|----------|-------------------|
| O1 | Should setup.sh backup existing config files before overwriting? | Phase 3 implementation |
| O2 | Should the Pac-Man canvas throttle to 30fps explicitly, or rely on browser requestAnimationFrame defaults? | Phase 3 — canvas JS extraction |
| O3 | Is dual-write log approach (system paths + project logs) permanent or should system paths be deprecated after a transition period? | Phase 2 — log design decision |
| O4 | Are there additional hardware configurations beyond RTX 3090 + Tesla P100 that need testing? | Phase 4 — validation scope |
| O5 | Should the default admin password ("admin") be changed or require a forced change on first login? | Post-release enhancement (out of scope) |
