#!/usr/bin/env bash
# Hybrid platform tool installation helpers for Automanager setup.
# Source this file from setup.sh after log_* helpers are defined.

detect_linux_arch() {
  local machine
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *)
      log_error "Unsupported CPU architecture for platform tools: ${machine}"
      return 1
      ;;
  esac
}

codex_asset_suffix() {
  local arch="$1"
  case "${arch}" in
    amd64) echo "x86_64-unknown-linux-musl" ;;
    aarch64) echo "aarch64-unknown-linux-musl" ;;
    *) return 1 ;;
  esac
}

is_executable_file() {
  [[ -n "${1:-}" && -f "$1" && -x "$1" ]]
}

resolve_platform_command() {
  local cmd="$1"
  local candidate=""

  if command -v "${cmd}" &>/dev/null; then
    command -v "${cmd}"
    return 0
  fi

  for candidate in \
    "${HOME:-/root}/.local/bin/${cmd}" \
    "/usr/local/bin/${cmd}" \
    "/usr/bin/${cmd}"; do
    if is_executable_file "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  if [[ "${cmd}" == "codex" ]]; then
    candidate="${HOME:-/root}/.codex/packages/standalone/current/bin/codex"
    if is_executable_file "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  fi

  return 1
}

install_cliproxyapi() {
  local arch install_dir tmp_dir version tag asset archive binary

  if resolve_platform_command cli-proxy-api &>/dev/null \
    || resolve_platform_command CLIProxyAPI &>/dev/null; then
    log_info "CLIProxyAPI already installed: $(resolve_platform_command cli-proxy-api || resolve_platform_command CLIProxyAPI)"
    return 0
  fi

  arch="$(detect_linux_arch)" || return 1
  install_dir="/usr/local/bin"
  (
    tmp_dir="$(mktemp -d)"
    version="$(
      curl -fsSL https://api.github.com/repos/router-for-me/CLIProxyAPI/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'
    )"
    tag="${version#v}"
    asset="CLIProxyAPI_${tag}_linux_${arch}.tar.gz"

    log_info "Downloading CLIProxyAPI ${version} (${arch})..."
    curl -fsSL -o "${tmp_dir}/${asset}" \
      "https://github.com/router-for-me/CLIProxyAPI/releases/download/${version}/${asset}"
    tar -xzf "${tmp_dir}/${asset}" -C "${tmp_dir}"
    binary="${tmp_dir}/cli-proxy-api"
    if ! is_executable_file "${binary}"; then
      log_error "CLIProxyAPI archive did not contain cli-proxy-api"
      exit 1
    fi

    install -m 0755 "${binary}" "${install_dir}/cli-proxy-api"
    ln -sf "${install_dir}/cli-proxy-api" "${install_dir}/CLIProxyAPI"
    log_info "Installed CLIProxyAPI to ${install_dir}/cli-proxy-api"
  )
}

install_codex() {
  local arch suffix asset tag tmp_dir binary target

  if resolve_platform_command codex &>/dev/null; then
    log_info "Codex CLI already installed: $(resolve_platform_command codex)"
    return 0
  fi

  arch="$(detect_linux_arch)" || return 1
  suffix="$(codex_asset_suffix "${arch}")" || return 1
  (
    tmp_dir="$(mktemp -d)"
    tag="$(
      curl -fsSL https://api.github.com/repos/openai/codex/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'
    )"
    asset="codex-${suffix}.tar.gz"

    log_info "Downloading Codex CLI (${tag}, ${arch})..."
    curl -fsSL -o "${tmp_dir}/${asset}" \
      "https://github.com/openai/codex/releases/download/${tag}/${asset}"
    tar -xzf "${tmp_dir}/${asset}" -C "${tmp_dir}"
    binary="${tmp_dir}/codex-${suffix}"
    target="/usr/local/bin/codex"
    if ! is_executable_file "${binary}"; then
      log_error "Codex archive did not contain expected binary codex-${suffix}"
      exit 1
    fi

    install -m 0755 "${binary}" "${target}"
    log_info "Installed Codex CLI to ${target}"
  )
}

install_antigravity_cli() {
  if resolve_platform_command agy &>/dev/null; then
    log_info "Google Antigravity CLI already installed: $(resolve_platform_command agy)"
    return 0
  fi

  log_info "Installing Google Antigravity CLI (agy)..."
  if ! curl -fsSL https://antigravity.google/cli/install.sh | bash; then
    log_warn "Google Antigravity CLI installation failed."
    return 1
  fi

  if resolve_platform_command agy &>/dev/null; then
    log_info "Google Antigravity CLI installed: $(resolve_platform_command agy)"
  else
    log_warn "Google Antigravity CLI install script finished, but agy was not found."
    return 1
  fi
}

install_claude_code() {
  if resolve_platform_command claude &>/dev/null; then
    log_info "Claude Code already installed: $(resolve_platform_command claude)"
    return 0
  fi

  log_info "Installing Claude Code..."
  if ! curl -fsSL https://claude.ai/install.sh | bash; then
    log_warn "Claude Code installation failed."
    return 1
  fi

  if resolve_platform_command claude &>/dev/null; then
    log_info "Claude Code installed: $(resolve_platform_command claude)"
  else
    log_warn "Claude Code install script finished, but claude was not found."
    return 1
  fi
}

install_httpx_deps() {
  local venv_path="${INSTALL_DIR:-/opt/automanager}/venv"
  local pip_path="${venv_path}/bin/pip"
  local python_path="${venv_path}/bin/python"

  if [[ ! -f "${pip_path}" ]]; then
    log_warn "Python virtualenv not found at ${venv_path}, skipping httpx check."
    return 0
  fi

  if "${python_path}" -c "import httpx" 2>/dev/null; then
    log_info "httpx is already installed in the virtualenv."
    return 0
  fi

  log_info "httpx not found in virtualenv — installing..."
  if "${pip_path}" install --upgrade pip &>/dev/null \
    && "${pip_path}" install "httpx>=0.27.0" 2>&1 | tail -1; then
    log_info "httpx installed successfully."
  else
    log_error "Failed to install httpx in ${venv_path}."
    return 1
  fi
}

verify_platform_tools() {
  local label path
  local -a required=(cli-proxy-api codex agy)
  local -a optional=(claude)

  log_info "Platform tool detection summary:"
  for label in "${required[@]}"; do
    if path="$(resolve_platform_command "${label}")"; then
      log_info "  ${label}: ${path}"
    else
      log_warn "  ${label}: not found"
    fi
  done
  for label in "${optional[@]}"; do
    if path="$(resolve_platform_command "${label}")"; then
      log_info "  ${label}: ${path}"
    else
      log_warn "  ${label}: not found (optional)"
    fi
  done
}

install_platform_tools() {
  local failed=0

  log_info "Installing hybrid platform dependencies..."
  install_httpx_deps || failed=1
  install_cliproxyapi || failed=1
  install_codex || failed=1
  install_antigravity_cli || failed=1
  install_claude_code || true
  verify_platform_tools

  if [[ "${failed}" -ne 0 ]]; then
    log_warn "Some required platform tools could not be installed."
    log_warn "Hybrid platform cards may stay unavailable until you install them manually."
    return 1
  fi

  log_info "Required platform tools are installed."
  log_warn "Authenticate each CLI separately before starting platform cards in the dashboard."
  return 0
}
