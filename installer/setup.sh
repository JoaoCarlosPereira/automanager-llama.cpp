#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ ! -f /etc/os-release ]]; then
  log_error "Unsupported OS. Ubuntu or Debian required."
  exit 1
fi

# shellcheck source=/dev/null
source /etc/os-release
if [[ "${ID}" != "ubuntu" && "${ID}" != "debian" ]]; then
  log_error "Unsupported distribution: ${ID}. Expected Ubuntu or Debian."
  exit 1
fi

if [[ "${EUID:-0}" -ne 0 ]]; then
  log_error "Run as root: sudo ./installer/setup.sh"
  exit 1
fi

log_info "Detected: ${ID} ${VERSION_ID:-unknown}"

apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv curl git lsb-release

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

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p /root

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
Restart=on-failure
RestartSec=5
Environment=PATH=${VENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable llama-manager.service
systemctl restart llama-manager.service
log_info "Systemd service configured"

sleep 3
if curl -sf http://localhost:8000/status >/dev/null 2>&1; then
  log_info "Health check passed."
  log_info "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/"
else
  log_warn "Health check failed. Run: journalctl -u llama-manager.service -f"
fi

log_info "Installation complete."
