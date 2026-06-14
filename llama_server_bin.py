"""Discover llama-server executable on the host."""

from __future__ import annotations

import glob
import json
import logging
import os
import platform
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from paths import INSTALL_ROOT, PATHS_FILE, _resolve_path

logger = logging.getLogger("automanager")

_CACHED: Optional[str] = None
_HELP_CACHE: Optional[str] = None
_BINS_LIST_CACHE: Optional[List[Dict[str, Any]]] = None
_IS_WINDOWS = platform.system() == "Windows"
_EXECUTABLE_NAMES = (
    ("llama-server.exe", "llama-server")
    if _IS_WINDOWS
    else ("llama-server",)
)


def _is_executable(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    if _IS_WINDOWS:
        lower = path.lower()
        if lower.endswith((".exe", ".bat", ".cmd")):
            return True
    return os.access(path, os.X_OK)


def _read_paths_json_llama_bin(
    install_root: str,
    paths_file: str,
) -> Optional[str]:
    if not os.path.isfile(paths_file):
        return None
    try:
        with open(paths_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("llama_server_bin")
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_path(install_root, value.strip())


def _search_roots(install_root: str) -> List[str]:
    root = os.path.normpath(install_root)
    parent = os.path.dirname(root)
    grandparent = os.path.dirname(parent)
    home = os.path.expanduser("~")
    roots: List[str] = []
    for candidate in (root, parent, grandparent, home):
        norm = os.path.normpath(candidate)
        if norm and norm not in roots:
            roots.append(norm)
    return roots


def _relative_candidate_dirs() -> List[str]:
    dirs = [
        "bin",
        "build/bin",
        "build/bin/Release",
        "build/bin/Debug",
        "llama.cpp/build/bin",
        "llama.cpp/build/bin/Release",
        "llama.cpp/build/bin/Debug",
        os.path.join("..", "llama.cpp", "build", "bin"),
        os.path.join("..", "llama.cpp", "build", "bin", "Release"),
        os.path.join("..", "..", "llama.cpp", "build", "bin"),
    ]
    if not _IS_WINDOWS:
        dirs.extend(
            [
                "/usr/local/bin",
                "/usr/bin",
                "/opt/llama.cpp/bin",
            ]
        )
    return dirs


def _iter_candidates(install_root: str) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for base in _search_roots(install_root):
        for rel in _relative_candidate_dirs():
            if os.path.isabs(rel):
                for name in _EXECUTABLE_NAMES:
                    path = os.path.normpath(os.path.join(rel, name))
                    if path not in seen:
                        seen.add(path)
                        ordered.append(path)
                continue
            for name in _EXECUTABLE_NAMES:
                path = os.path.normpath(os.path.join(base, rel, name))
                if path not in seen:
                    seen.add(path)
                    ordered.append(path)
    return ordered


def _glob_build_bins(search_roots: List[str]) -> List[str]:
    """Find llama-server under */build/bin in search roots (one level deep)."""
    found: List[str] = []
    seen: set[str] = set()
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        patterns: List[str] = []
        for name in _EXECUTABLE_NAMES:
            patterns.extend(
                [
                    os.path.join(root, "*", "build", "bin", name),
                    os.path.join(root, "*", "build", "bin", "Release", name),
                    os.path.join(root, "*", "build", "bin", "Debug", name),
                ]
            )
        for pattern in patterns:
            for path in glob.glob(pattern):
                norm = os.path.normpath(path)
                if norm not in seen and _is_executable(norm):
                    seen.add(norm)
                    found.append(norm)
    return found


def _collect_all_candidate_paths(
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> List[str]:
    """Return unique executable llama-server paths in priority order."""
    root = install_root or INSTALL_ROOT
    pf = paths_file or PATHS_FILE
    ordered: List[str] = []
    seen: set[str] = set()

    def add(path: Optional[str]) -> None:
        if not path:
            return
        norm = os.path.normpath(path)
        if norm in seen:
            return
        if _is_executable(norm):
            seen.add(norm)
            ordered.append(norm)

    env_value = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if env_value:
        add(env_value)

    configured = _read_paths_json_llama_bin(root, pf)
    add(configured)

    for name in _EXECUTABLE_NAMES:
        add(shutil.which(name))

    for candidate in _iter_candidates(root):
        add(candidate)

    for candidate in _glob_build_bins(_search_roots(root)):
        add(candidate)

    return ordered


def get_bin_version_info(bin_path: str) -> Dict[str, Optional[str]]:
    """Parse ``llama-server --version`` output."""
    try:
        proc = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return {"build": None, "commit": None, "label": os.path.basename(bin_path)}

    build = None
    commit = None
    match = re.search(r"version:\s*(\d+)\s*\(([a-f0-9]+)\)", output, re.IGNORECASE)
    if match:
        build = match.group(1)
        commit = match.group(2)

    parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(bin_path))))
    if parent in ("bin", "Release", "Debug"):
        parent = os.path.basename(os.path.dirname(bin_path))
    if not parent or parent == "bin":
        parent = "llama-server"

    if build and commit:
        label = f"{parent} · build {build} ({commit[:7]})"
    elif build:
        label = f"{parent} · build {build}"
    else:
        label = parent

    return {"build": build, "commit": commit, "label": label}


def is_turboquant_bin(bin_path: str) -> bool:
    """True when *bin_path* supports TurboQuant+ KV types (turbo2/3/4)."""
    if not bin_path or not _is_executable(bin_path):
        return False
    help_text = get_llama_server_help(bin_path)
    return "turbo2" in help_text and "turbo3" in help_text and "turbo4" in help_text


def turboquant_cache_types() -> List[str]:
    """Return turbo KV cache type tokens supported by TurboQuant+ builds."""
    return ["turbo2", "turbo3", "turbo4"]


def validate_turboquant_cache_types(
    cache_type_k: str,
    cache_type_v: str,
    bin_path: str,
) -> Optional[str]:
    """Return an error message when turbo KV types mismatch the selected binary."""
    turbo_types = set(turboquant_cache_types())
    uses_turbo = cache_type_k in turbo_types or cache_type_v in turbo_types
    if uses_turbo and not is_turboquant_bin(bin_path):
        return (
            "Tipos de cache turbo2/turbo3/turbo4 exigem o binário llama-cpp-turboquant. "
            "Selecione a versão TurboQuant+ ou ajuste os tipos de cache."
        )
    return None


def list_llama_server_bins(
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover all llama-server binaries and return metadata for the UI."""
    global _BINS_LIST_CACHE
    if _BINS_LIST_CACHE is not None and install_root is None and paths_file is None:
        default_path = get_llama_server_bin()
        return {
            "bins": _BINS_LIST_CACHE,
            "default": default_path if _is_executable(default_path) else (
                _BINS_LIST_CACHE[0]["path"] if _BINS_LIST_CACHE else None
            ),
        }

    paths = _collect_all_candidate_paths(install_root, paths_file)
    default = resolve_llama_server_bin(install_root, paths_file)
    if default and default not in paths:
        paths.insert(0, default)

    bins: List[Dict[str, Any]] = []
    for path in paths:
        info = get_bin_version_info(path)
        bins.append(
            {
                "path": path,
                "label": info["label"],
                "build": info["build"],
                "commit": info["commit"],
                "is_default": path == default,
                "is_turboquant": is_turboquant_bin(path),
            }
        )

    if install_root is None and paths_file is None:
        _BINS_LIST_CACHE = bins

    return {
        "bins": bins,
        "default": default or (bins[0]["path"] if bins else None),
    }


def resolve_llama_server_bin(
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> Optional[str]:
    """Return absolute path to llama-server when found, else None."""
    root = install_root or INSTALL_ROOT
    pf = paths_file or PATHS_FILE

    env_value = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if env_value and _is_executable(env_value):
        return os.path.normpath(env_value)

    configured = _read_paths_json_llama_bin(root, pf)
    if configured and _is_executable(configured):
        return os.path.normpath(configured)
    if configured:
        logger.warning(
            "llama_server_bin configurado mas inacessivel: %s", configured
        )

    for name in _EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found and _is_executable(found):
            return os.path.normpath(found)

    for candidate in _iter_candidates(root):
        if _is_executable(candidate):
            return candidate

    return None


def get_llama_server_bin() -> str:
    """Return cached llama-server path, resolving on first call."""
    global _CACHED
    if _CACHED is None:
        resolved = resolve_llama_server_bin()
        _CACHED = resolved or (
            "llama-server.exe" if _IS_WINDOWS else "llama-server"
        )
        if resolved:
            logger.info("llama-server encontrado: %s", resolved)
        else:
            logger.warning(
                "llama-server nao encontrado automaticamente; "
                "configure LLAMA_SERVER_BIN ou paths.json (llama_server_bin)"
            )
    return _CACHED


def reset_llama_server_bin_cache() -> None:
    """Clear cached resolution (tests)."""
    global _CACHED, _HELP_CACHE, _BINS_LIST_CACHE
    _CACHED = None
    _HELP_CACHE = None
    _BINS_LIST_CACHE = None


def get_llama_server_help(bin_path: Optional[str] = None) -> str:
    """Return ``llama-server --help`` output (empty on failure)."""
    global _HELP_CACHE
    target = bin_path or get_llama_server_bin()
    if bin_path is None and _HELP_CACHE is not None:
        return _HELP_CACHE
    try:
        proc = subprocess.run(
            [target, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        help_text = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        help_text = ""
    if bin_path is None:
        _HELP_CACHE = help_text
    return help_text


def supports_cli_flag(flag: str, bin_path: Optional[str] = None) -> bool:
    """True when ``flag`` appears in ``llama-server --help``."""
    return flag in get_llama_server_help(bin_path)


def llama_server_available() -> bool:
    """True when a runnable llama-server binary was resolved."""
    return _is_executable(get_llama_server_bin())
