#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PURGE_DATA=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage: sudo bash installer/uninstall.sh [options]

Options:
  --purge   Remove local config, logs, and paths.json (models are kept)
  --yes     Skip confirmation prompt
  -h, --help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge)
      PURGE_DATA=1
      shift
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

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
  log_error "Run as root: sudo bash installer/uninstall.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PATHS_FILE="${PROJECT_DIR}/paths.json"
SERVICE_NAME="llama-manager.service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"

CONFIG_FILE="${PROJECT_DIR}/data/automanager_config.json"
LOGS_DIR="${PROJECT_DIR}/logs"
MODELS_DIR="${PROJECT_DIR}/data/models"

if [[ -f "${PATHS_FILE}" ]] && command -v python3 &>/dev/null; then
  export PROJECT_DIR
  read -r CONFIG_FILE LOGS_DIR MODELS_DIR < <(
    cd "${PROJECT_DIR}" && python3 - <<'PY'
import json
import os

install_root = os.environ["PROJECT_DIR"]
paths_file = os.path.join(install_root, "paths.json")
defaults = {
    "config_file": "data/automanager_config.json",
    "logs_dir": "logs",
    "models_dir": "data/models",
}

def resolve(value: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(install_root, expanded))

entries = defaults.copy()
if os.path.isfile(paths_file):
    try:
        with open(paths_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            for key in defaults:
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    entries[key] = value.strip()
    except (OSError, json.JSONDecodeError):
        pass

print(resolve(entries["config_file"]), resolve(entries["logs_dir"]), resolve(entries["models_dir"]))
PY
  )
fi

echo ""
echo -e "${YELLOW}Automanager Llama.cpp - desinstalacao${NC}"
echo ""
echo "  Projeto:      ${PROJECT_DIR}"
echo "  Servico:      ${SERVICE_NAME}"
echo "  Virtualenv:   ${VENV_DIR}"
if [[ "${PURGE_DATA}" -eq 1 ]]; then
  echo "  Config:       ${CONFIG_FILE}"
  echo "  Logs:         ${LOGS_DIR}"
  echo "  paths.json:   ${PATHS_FILE}"
fi
echo "  Modelos:      ${MODELS_DIR} (mantidos)"
echo ""

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  read -r -p "Continuar com a desinstalacao? [y/N] " reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    log_info "Desinstalacao cancelada."
    exit 0
  fi
fi

if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
  log_info "Stopping ${SERVICE_NAME}..."
  systemctl stop "${SERVICE_NAME}" || true
fi

if systemctl is-enabled --quiet "${SERVICE_NAME}" 2>/dev/null; then
  log_info "Disabling ${SERVICE_NAME}..."
  systemctl disable "${SERVICE_NAME}" || true
fi

# If the service is still running, let systemd handle graceful shutdown.
# Avoid global pkill — it can kill unrelated llama-server instances on the host.
if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
  log_info "Stopping ${SERVICE_NAME} (service already running)..."
  systemctl stop "${SERVICE_NAME}" || true
fi

if [[ -f "${SERVICE_FILE}" ]]; then
  rm -f "${SERVICE_FILE}"
  log_info "Removed ${SERVICE_FILE}"
fi

systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true

if [[ -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
  log_info "Removed virtualenv at ${VENV_DIR}"
else
  log_warn "Virtualenv not found at ${VENV_DIR}"
fi

if [[ "${PURGE_DATA}" -eq 1 ]]; then
  _purge_paths=()

  # --- helper: validate a path for safe deletion ---
  _validate_purge_path() {
    local p="$1" label="$2"
    # Reject empty / bare /
    if [[ -z "${p}" || "${p}" == "/" ]]; then
      log_error "Purge target '${label}' is empty or '/'. Aborting to prevent data loss."
      return 1
    fi
    # Resolve real path (handle symlinks)
    if [[ ! -e "${p}" ]]; then
      log_warn "Purge target '${label}' does not exist: ${p}"
      return 1
    fi
    local real
    real="$(realpath "${p}" 2>/dev/null)" || {
      log_error "Cannot resolve purge target '${label}': ${p}"
      return 1
    }
    # Must be under PROJECT_DIR (or its children)
    if [[ "${real}" != "${PROJECT_DIR}" && "${real}" != "${PROJECT_DIR}/"* ]]; then
      log_error "Purge target '${label}' resolves outside project: ${real}"
      log_error "  (PROJECT_DIR=${PROJECT_DIR})"
      log_error "External paths are rejected for --purge safety."
      return 1
    fi
    # For directories, refuse if it looks like /usr /etc /home /opt /var /tmp /boot /srv /root
    if [[ -d "${real}" ]]; then
      case "${real}" in
        /usr|/usr/*|/etc|/etc/*|/home|/home/*|/opt|/opt/*|\
/var|/var/*|/tmp|/tmp/*|/boot|/boot/*|/srv|/srv/*|\
/root|/root/*)
          log_error "Purge target '${label}' is a dangerous directory: ${real}"
          return 1
          ;;
      esac
    fi
    _purge_paths+=("${real}")
    log_info "  Purge-validated '${label}': ${real}"
    return 0
  }

  # --- config file ---
  if [[ -n "${CONFIG_FILE:-}" ]]; then
    _validate_purge_path "${CONFIG_FILE}" "config" || { PURGE_DATA=0; }
  fi

  # --- logs directory ---
  if [[ -n "${LOGS_DIR:-}" ]]; then
    _validate_purge_path "${LOGS_DIR}" "logs" || { PURGE_DATA=0; }
  fi

  # --- paths.json (file only) ---
  if [[ -n "${PATHS_FILE:-}" ]]; then
    _validate_purge_path "${PATHS_FILE}" "paths.json" || { PURGE_DATA=0; }
  fi

  if [[ "${PURGE_DATA}" -eq 1 ]]; then
    # Delete all validated paths
    for _pp in "${_purge_paths[@]}"; do
      if [[ -f "${_pp}" ]]; then
        rm -f "${_pp}"
        log_info "Removed ${_pp}"
      elif [[ -d "${_pp}" ]]; then
        rm -rf "${_pp}"
        log_info "Removed directory ${_pp}"
      fi
    done
  fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Automanager Llama.cpp desinstalado${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
if [[ "${PURGE_DATA}" -eq 1 ]]; then
  echo -e "  Config, logs e paths.json foram removidos."
else
  echo -e "  Config, logs e paths.json foram mantidos."
  echo -e "  Para remover tambem: ${YELLOW}sudo bash installer/uninstall.sh --purge --yes${NC}"
fi
echo -e "  Modelos mantidos em: ${YELLOW}${MODELS_DIR}${NC}"
echo -e "  Codigo-fonte mantido em: ${YELLOW}${PROJECT_DIR}${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

log_info "Uninstall complete."
