#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ "${EUID:-0}" -ne 0 ]]; then
  log_error "Run as root: sudo ./installer/setup.sh"
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  PYTHON_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  log_error "Python 3.11+ required (found ${PYTHON_VERSION})."
  exit 1
fi

apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev curl git lsb-release

if ! command -v llama-server &>/dev/null; then
  log_warn "llama-server not found in PATH."
  log_warn "Install from: https://github.com/ggml-org/llama.cpp/releases"
fi

if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Install NVIDIA drivers first."
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if [[ "${GPU_COUNT}" -eq 0 ]]; then
  log_error "No NVIDIA GPUs detected."
  exit 1
fi
log_info "Detected ${GPU_COUNT} NVIDIA GPU(s)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

# shellcheck source=platform_tools.sh
source "${SCRIPT_DIR}/platform_tools.sh"
install_platform_tools || log_warn "Continuing setup without full platform tool support."
PATHS_FILE="${PROJECT_DIR}/paths.json"
PATHS_EXAMPLE="${PROJECT_DIR}/paths.json.example"

if [[ ! -f "${PATHS_FILE}" ]]; then
  if [[ -f "${PATHS_EXAMPLE}" ]]; then
    cp "${PATHS_EXAMPLE}" "${PATHS_FILE}"
    log_info "Created ${PATHS_FILE} from paths.json.example"
  else
    log_error "Missing ${PATHS_EXAMPLE}. Cannot configure install paths."
    exit 1
  fi
else
  log_info "Using existing path configuration: ${PATHS_FILE}"
fi

if [[ -d "${VENV_DIR}" ]]; then
  log_warn "Removing existing venv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"
log_info "Python dependencies installed"

log_info "Creating configured directories..."
(cd "${PROJECT_DIR}" && "${VENV_DIR}/bin/python" - <<'PY'
from paths import ensure_directories

paths = ensure_directories()
print(f"  models_dir  -> {paths.models_dir}")
print(f"  config_file -> {paths.config_file}")
print(f"  logs_dir    -> {paths.logs_dir}")
PY
)

SERVICE_FILE="/etc/systemd/system/llama-manager.service"
cat >"${SERVICE_FILE}" <<EOF
[Unit]
Description=Automanager Llama.cpp
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/llama_manager.py
ExecStop=/bin/sh -c '/usr/bin/pkill -9 -f llama-server 2>/dev/null || true'
TimeoutStopSec=30
Restart=on-failure
RestartSec=5
Environment=PATH=${VENV_DIR}/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
Environment=HF_HOME=${PROJECT_DIR}/data/huggingface_cache

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable llama-manager.service
systemctl restart llama-manager.service
log_info "Systemd service configured"

MANAGER_PORT=8000
SERVER_PORT=8085

LOCAL_IP="$(
  cd "${PROJECT_DIR}" && "${VENV_DIR}/bin/python" - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 1))
    print(s.getsockname()[0])
except Exception:
    print("127.0.0.1")
finally:
    s.close()
PY
)"

sleep 3
if curl -sf "http://localhost:${MANAGER_PORT}/" >/dev/null 2>&1; then
  log_info "Health check passed."
else
  log_warn "Health check failed. Run: journalctl -u llama-manager.service -f"
fi

(cd "${PROJECT_DIR}" && HOME=/root "${VENV_DIR}/bin/python" - <<'PY'
from platform_manager import PlatformIntegrationManager
from config_manager import ConfigManager
from paths import CONFIG_PATH

manager = PlatformIntegrationManager(ConfigManager(CONFIG_PATH))
for item in manager.catalog():
    status = item.get("status")
    reason = item.get("reason") or ""
    path = item.get("executable_path") or "-"
    print(f"  {item['display_name']}: {status} ({path})")
    if reason:
        print(f"    reason: {reason}")
cliproxy = manager.cliproxy_detection
print(f"  CLIProxyAPI: {'detected' if cliproxy.detected else 'missing'} ({cliproxy.path or '-'})")
PY
) 2>/dev/null || log_warn "Could not print platform detection summary."

# Print Ollama Cloud status (non-CLI platform, HTTP backend)
log_info "  Ollama Cloud: disponivel (plataforma HTTP sem CLI)"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Automanager Llama.cpp instalado${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Dashboard:  ${YELLOW}http://${LOCAL_IP}:${MANAGER_PORT}/${NC}"
echo -e "  API proxy:  ${YELLOW}http://${LOCAL_IP}:${MANAGER_PORT}/v1/${NC}"
echo -e "  llama-server (interno): porta ${SERVER_PORT}"
echo ""
echo -e "  Login padrao: ${YELLOW}admin / admin${NC}"
echo -e "  Altere a senha no primeiro acesso."
echo ""
echo -e "  Plataformas hibridas: Codex, Antigravity (agy) e Claude Code"
echo -e "  sao instaladas pelo setup. Autentique cada CLI antes de usar."
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

log_info "Installation complete."
log_info "Edit ${PATHS_FILE} to change models, config, or logs locations, then rerun setup or restart the service."
log_info "To uninstall later: sudo bash installer/uninstall.sh"
