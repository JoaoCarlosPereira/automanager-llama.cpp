"""Discover llama-server executable on the host."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from typing import List, Optional

from paths import INSTALL_ROOT, PATHS_FILE, _resolve_path

logger = logging.getLogger("automanager")

_CACHED: Optional[str] = None
_HELP_CACHE: Optional[str] = None
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
    global _CACHED, _HELP_CACHE
    _CACHED = None
    _HELP_CACHE = None


def get_llama_server_help() -> str:
    """Return cached ``llama-server --help`` output (empty on failure)."""
    global _HELP_CACHE
    if _HELP_CACHE is not None:
        return _HELP_CACHE
    try:
        import subprocess

        proc = subprocess.run(
            [get_llama_server_bin(), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        _HELP_CACHE = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        _HELP_CACHE = ""
    return _HELP_CACHE


def supports_cli_flag(flag: str) -> bool:
    """True when ``flag`` appears in ``llama-server --help``."""
    return flag in get_llama_server_help()


def llama_server_available() -> bool:
    """True when a runnable llama-server binary was resolved."""
    return _is_executable(get_llama_server_bin())
