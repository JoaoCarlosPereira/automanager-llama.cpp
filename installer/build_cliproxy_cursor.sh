#!/usr/bin/env bash
# Build CLIProxyAPI with the Cursor agent executor and install cli-proxy-api.
set -euo pipefail

PIN="fd48ea6840f5572deb53aeb5657740937ac9daaa"
GO_VERSION="1.26.4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${ROOT}/patches/cliproxy-cursor.patch"
TARGET="${CLIPROXY_INSTALL_PATH:-/usr/local/bin/cli-proxy-api}"

log_info() { printf '[cliproxy-cursor] %s\n' "$*"; }
log_error() { printf '[cliproxy-cursor] ERROR: %s\n' "$*" >&2; }

if [[ ! -f "${PATCH}" ]]; then
  log_error "Patch not found: ${PATCH}"
  exit 1
fi

go_is_new_enough() {
  local raw version major rest minor
  if ! command -v go >/dev/null 2>&1; then
    return 1
  fi
  raw="$(go env GOVERSION 2>/dev/null || true)"
  if [[ -z "${raw}" ]]; then
    raw="$(go version 2>/dev/null | awk '{print $3}')"
  fi
  version="${raw#go}"
  major="${version%%.*}"
  rest="${version#*.}"
  minor="${rest%%.*}"
  [[ "${major}" =~ ^[0-9]+$ && "${minor}" =~ ^[0-9]+$ ]] || return 1
  [[ "${major}" -gt 1 || ( "${major}" -eq 1 && "${minor}" -ge 26 ) ]]
}

if ! go_is_new_enough; then
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|amd64) goarch="amd64" ;;
    aarch64|arm64) goarch="arm64" ;;
    *)
      log_error "Unsupported architecture for the Go toolchain: ${arch}"
      exit 1
      ;;
  esac
  toolchain_dir="${TMPDIR:-/tmp}/go-${GO_VERSION}"
  if [[ ! -x "${toolchain_dir}/go/bin/go" ]]; then
    tarball="go${GO_VERSION}.linux-${goarch}.tar.gz"
    tmp_tarball="$(mktemp)"
    log_info "Downloading Go ${GO_VERSION}..."
    curl -fsSL -o "${tmp_tarball}" "https://go.dev/dl/${tarball}"
    mkdir -p "${toolchain_dir}"
    tar -C "${toolchain_dir}" -xzf "${tmp_tarball}"
    rm -f "${tmp_tarball}"
  fi
  export PATH="${toolchain_dir}/go/bin:${PATH}"
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/src"
log_info "Fetching CLIProxyAPI ${PIN}..."
git -C "${work}/src" init -q
git -C "${work}/src" remote add origin https://github.com/router-for-me/CLIProxyAPI.git
git -C "${work}/src" fetch --depth 1 origin "${PIN}"
git -C "${work}/src" checkout --detach FETCH_HEAD
git -C "${work}/src" apply "${PATCH}"

log_info "Building cli-proxy-api..."
(
  cd "${work}/src"
  CGO_ENABLED=0 go build -o "${work}/cli-proxy-api" ./cmd/server
)
install -m 0755 "${work}/cli-proxy-api" "${TARGET}"
link_dir="$(dirname "${TARGET}")"
ln -sfn "${TARGET}" "${link_dir}/CLIProxyAPI"
log_info "Installed ${TARGET}"
