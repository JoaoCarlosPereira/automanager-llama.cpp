#!/usr/bin/env bash
# Install the official llama.cpp llama-server binary used by AutoManager.
# Source this file from setup.sh after log_* helpers and PROJECT_DIR exist.
#
# Runtime evidence:
#   paths.json.example            -> "llama_server_bin": "bin/llama-server"
#   llama_server_bin.py           -> discovers bin/llama-server and llama.cpp builds
#   gpu_manager.py                -> llama-server and llama-gguf in the same directory
#   installer/setup.sh            -> https://github.com/ggml-org/llama.cpp/releases
# AutoManager does not execute a llama.cpp source checkout. This script installs
# the published Ubuntu binary (and the matching CUDA runtime archive) instead of
# cloning and compiling the tree.

LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_MANIFEST_NAME=".automanager-managed-files"

llama_cpp_release_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x64" ;;
    aarch64|arm64) echo "arm64" ;;
    *)
      log_error "Unsupported CPU architecture for llama.cpp binaries: $(uname -m)"
      return 1
      ;;
  esac
}

detect_driver_cuda_version() {
  local line
  line="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
  printf '%s' "${line}"
}

llama_cpp_dest() {
  printf '%s/bin' "${PROJECT_DIR}"
}

llama_cpp_manifest() {
  printf '%s/%s' "$(llama_cpp_dest)" "${LLAMA_CPP_MANIFEST_NAME}"
}

llama_server_already_available() {
  local configured=""

  if command -v llama-server &>/dev/null; then
    return 0
  fi
  if [[ -f "$(llama_cpp_dest)/llama-server" && -x "$(llama_cpp_dest)/llama-server" ]]; then
    return 0
  fi
  if [[ ! -f "${PROJECT_DIR}/paths.json" ]]; then
    return 1
  fi

  configured="$(
    PROJECT_DIR="${PROJECT_DIR}" python3 - <<'PY'
import json
import os

root = os.environ["PROJECT_DIR"]
path = os.path.join(root, "paths.json")
try:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
if not isinstance(raw, dict):
    raise SystemExit(0)
value = raw.get("llama_server_bin")
if not isinstance(value, str) or not value.strip():
    raise SystemExit(0)
expanded = os.path.expanduser(value.strip())
if not os.path.isabs(expanded):
    expanded = os.path.normpath(os.path.join(root, expanded))
else:
    expanded = os.path.normpath(expanded)
print(expanded)
PY
  )"
  [[ -n "${configured}" && -f "${configured}" && -x "${configured}" ]]
}

remove_managed_llama_cpp() {
  local dest manifest dest_real rel target real
  dest="$(llama_cpp_dest)"
  manifest="$(llama_cpp_manifest)"
  if [[ ! -f "${manifest}" ]]; then
    return 0
  fi
  if [[ ! -d "${dest}" ]]; then
    rm -f "${manifest}"
    return 0
  fi
  dest_real="$(realpath "${dest}")"
  while IFS= read -r rel || [[ -n "${rel}" ]]; do
    rel="${rel%%$'\r'}"
    [[ -z "${rel}" || "${rel}" == \#* ]] && continue
    case "${rel}" in
      /*|..|../*|*/..|*/../*|.|./*)
        log_warn "Ignoring unsafe llama.cpp manifest entry: ${rel}"
        continue
        ;;
    esac
    target="${dest}/${rel}"
    if [[ -L "${target}" || -f "${target}" ]]; then
      real="$(realpath -m "${target}")"
      case "${real}" in
        "${dest_real}"/*)
          rm -f "${target}"
          ;;
        *)
          log_warn "Skipping llama.cpp manifest path outside bin/: ${real}"
          ;;
      esac
    fi
  done < "${manifest}"
  rm -f "${manifest}"
  rmdir "${dest}" 2>/dev/null || true
  log_info "Removed managed llama.cpp files from ${dest}"
}

_copy_flat_entries() {
  local src_dir="$1"
  local dest="$2"
  local list_file="$3"
  local entry base

  [[ -d "${src_dir}" ]] || return 0
  while IFS= read -r -d '' entry; do
    base="$(basename "${entry}")"
    [[ -n "${base}" && "${base}" != "." && "${base}" != ".." ]] || continue
    cp -a "${entry}" "${dest}/${base}"
    printf '%s\n' "${base}" >> "${list_file}"
  done < <(find "${src_dir}" -maxdepth 1 \( -type f -o -type l \) -print0)
}

_stage_server_archive() {
  local archive="$1"
  local dest="$2"
  local list_file="$3"
  local tmp server src_dir

  tmp="$(mktemp -d)"
  tar -xzf "${archive}" -C "${tmp}"
  server=""
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] || continue
    if [[ "${candidate}" == */bin/llama-server ]]; then
      server="${candidate}"
      break
    fi
    if [[ -z "${server}" ]]; then
      server="${candidate}"
    fi
  done < <(find "${tmp}" -type f -name 'llama-server')
  if [[ -z "${server}" ]]; then
    rm -rf "${tmp}"
    log_error "llama.cpp archive did not contain llama-server"
    return 1
  fi
  src_dir="$(dirname "${server}")"
  _copy_flat_entries "${src_dir}" "${dest}" "${list_file}"
  if [[ -d "${src_dir}/lib" ]]; then
    _copy_flat_entries "${src_dir}/lib" "${dest}" "${list_file}"
  fi
  rm -rf "${tmp}"
}

_stage_cudart_archive() {
  local archive="$1"
  local dest="$2"
  local list_file="$3"
  local tmp lib src_dir

  tmp="$(mktemp -d)"
  tar -xzf "${archive}" -C "${tmp}"
  lib="$(find "${tmp}" \( -type f -o -type l \) -name 'libcudart.so*' -print -quit || true)"
  if [[ -z "${lib}" ]]; then
    rm -rf "${tmp}"
    log_warn "CUDA runtime archive did not contain libcudart; continuing with the server binary."
    return 0
  fi
  src_dir="$(dirname "${lib}")"
  _copy_flat_entries "${src_dir}" "${dest}" "${list_file}"
  rm -rf "${tmp}"
}

install_llama_cpp() {
  local dest manifest arch cuda_ver work releases_json selection
  local tag asset cudart variant url list_file name

  dest="$(llama_cpp_dest)"
  manifest="$(llama_cpp_manifest)"

  if [[ "${LLAMA_CPP_UPDATE:-}" == "1" && -f "${manifest}" ]]; then
    log_info "LLAMA_CPP_UPDATE=1 — refreshing managed llama.cpp binaries."
  elif [[ -e "${dest}/llama-server" && ! -f "${manifest}" ]]; then
    log_warn "bin/llama-server exists and was not installed by this setup. Leaving it untouched."
    return 0
  elif llama_server_already_available; then
    log_info "llama-server already available. Not downloading another copy."
    return 0
  fi

  arch="$(llama_cpp_release_arch)" || return 1
  cuda_ver="$(detect_driver_cuda_version)"
  if [[ -z "${cuda_ver}" ]]; then
    log_warn "Could not read the CUDA version from nvidia-smi. Selecting the oldest published CUDA build."
  else
    log_info "Driver CUDA version reported by nvidia-smi: ${cuda_ver}"
  fi

  work="$(mktemp -d)"
  releases_json="${work}/releases.json"
  log_info "Resolving llama.cpp binary from ${LLAMA_CPP_REPO} releases..."
  if ! curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    -H "User-Agent: automanager-setup" \
    -o "${releases_json}" \
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"; then
    log_error "Could not list llama.cpp releases."
    rm -rf "${work}"
    return 1
  fi

  local release_py="${SCRIPT_DIR:-}/llama_cpp_release.py"
  if [[ ! -f "${release_py}" ]]; then
    release_py="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llama_cpp_release.py"
  fi
  if ! selection="$(
    python3 "${release_py}" --arch "${arch}" --cuda "${cuda_ver}" < "${releases_json}"
  )"; then
    log_error "No llama.cpp Ubuntu binary matched this machine."
    rm -rf "${work}"
    return 1
  fi
  IFS=$'\t' read -r tag asset cudart variant <<<"${selection}"
  if [[ -z "${tag}" || -z "${asset}" ]]; then
    log_error "llama.cpp release selection was empty."
    rm -rf "${work}"
    return 1
  fi
  log_info "Selected llama.cpp ${tag} (${variant}): ${asset}"

  url="${LLAMA_CPP_REPO}/releases/download/${tag}/${asset}"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${work}/llama.tar.gz" "${url}"; then
    log_error "Download failed: ${url}"
    rm -rf "${work}"
    return 1
  fi

  local stage
  stage="${work}/stage"
  mkdir -p "${stage}"
  list_file="${work}/files.list"
  : > "${list_file}"
  if ! _stage_server_archive "${work}/llama.tar.gz" "${stage}" "${list_file}"; then
    rm -rf "${work}"
    return 1
  fi

  if [[ -n "${cudart}" ]]; then
    url="${LLAMA_CPP_REPO}/releases/download/${tag}/${cudart}"
    log_info "Downloading CUDA runtime ${cudart}"
    if curl -fL --retry 3 --retry-delay 2 -o "${work}/cudart.tar.gz" "${url}"; then
      if ! _stage_cudart_archive "${work}/cudart.tar.gz" "${stage}" "${list_file}"; then
        rm -rf "${work}"
        return 1
      fi
    else
      log_warn "CUDA runtime download failed. The llama-server binary was still staged."
    fi
  fi

  if [[ ! -f "${stage}/llama-server" ]]; then
    log_error "Staged llama.cpp archive has no llama-server."
    rm -rf "${work}"
    return 1
  fi
  if [[ ! -f "${stage}/llama-gguf" ]]; then
    log_warn "This llama.cpp release did not include llama-gguf. Layer detection will use llama-server only."
  fi

  if [[ -f "${manifest}" ]]; then
    remove_managed_llama_cpp
  fi
  mkdir -p "${dest}"

  while IFS= read -r name; do
    [[ -n "${name}" ]] || continue
    cp -a "${stage}/${name}" "${dest}/${name}"
  done < <(sort -u "${list_file}")

  local bin_name
  for bin_name in llama-server llama-cli llama-gguf llama-quantize llama-mtmd-cli; do
    if [[ -f "${dest}/${bin_name}" ]]; then
      chmod 0755 "${dest}/${bin_name}"
    fi
  done

  {
    printf '# %s %s %s\n' "${LLAMA_CPP_REPO}" "${tag}" "${variant}"
    sort -u "${list_file}"
  } > "${manifest}"
  rm -rf "${work}"
  log_info "Installed llama-server to ${dest}/llama-server"
  log_info "Source tree was not cloned. AutoManager runs this binary, not a llama.cpp checkout."
}
