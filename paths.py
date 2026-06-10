"""Install-root path configuration loaded from paths.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

INSTALL_ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS_FILE = os.path.join(INSTALL_ROOT, "paths.json")

DEFAULT_PATH_ENTRIES: Dict[str, str] = {
    "models_dir": "data/models",
    "config_file": "data/automanager_config.json",
    "logs_dir": "logs",
}


def _default_entries(install_root: str) -> Dict[str, str]:
    return dict(DEFAULT_PATH_ENTRIES)


@dataclass(frozen=True)
class InstallPaths:
    install_root: str
    models_dir: str
    config_file: str
    logs_dir: str

    @property
    def manager_log(self) -> str:
        return os.path.join(self.logs_dir, "manager.log")

    @property
    def server_log(self) -> str:
        return os.path.join(self.logs_dir, "server.log")


def _resolve_path(install_root: str, value: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(install_root, expanded))


def _load_entries(paths_file: str, install_root: str) -> Dict[str, str]:
    if not os.path.isfile(paths_file):
        return _default_entries(install_root)
    try:
        with open(paths_file, "r", encoding="utf-8") as handle:
            raw: Any = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _default_entries(install_root)
    if not isinstance(raw, dict):
        return _default_entries(install_root)
    merged = _default_entries(install_root)
    for key in DEFAULT_PATH_ENTRIES:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    return merged


def get_paths(
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> InstallPaths:
    root = install_root or INSTALL_ROOT
    entries = _load_entries(paths_file or PATHS_FILE, root)
    return InstallPaths(
        install_root=root,
        models_dir=_resolve_path(root, entries["models_dir"]),
        config_file=_resolve_path(root, entries["config_file"]),
        logs_dir=_resolve_path(root, entries["logs_dir"]),
    )


def ensure_directories(
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> InstallPaths:
    paths = get_paths(install_root=install_root, paths_file=paths_file)
    for directory in (
        paths.models_dir,
        paths.logs_dir,
        os.path.dirname(paths.config_file),
    ):
        if directory:
            os.makedirs(directory, exist_ok=True)
    return paths


def _save_entries(entries: Dict[str, str], paths_file: str) -> None:
    directory = os.path.dirname(paths_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = paths_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    os.replace(tmp_path, paths_file)


def update_models_dir(
    models_dir: str,
    install_root: Optional[str] = None,
    paths_file: Optional[str] = None,
) -> InstallPaths:
    value = models_dir.strip()
    if not value or "\0" in value:
        raise ValueError("models_dir inválido")
    root = install_root or INSTALL_ROOT
    pf = paths_file or PATHS_FILE
    entries = _load_entries(pf, root)
    entries["models_dir"] = value
    resolved = _resolve_path(root, value)
    os.makedirs(resolved, exist_ok=True)
    _save_entries(entries, pf)
    return get_paths(root, pf)


def reload_module_paths() -> InstallPaths:
    global _paths, MODELS_DIR, CONFIG_PATH, LOGS_DIR, MANAGER_LOG_PATH, SERVER_LOG_PATH
    _paths = get_paths()
    MODELS_DIR = _paths.models_dir
    CONFIG_PATH = _paths.config_file
    LOGS_DIR = _paths.logs_dir
    MANAGER_LOG_PATH = _paths.manager_log
    SERVER_LOG_PATH = _paths.server_log
    return _paths


# Module-level defaults used by the rest of the application.
_paths = get_paths()
MODELS_DIR = _paths.models_dir
CONFIG_PATH = _paths.config_file
LOGS_DIR = _paths.logs_dir
MANAGER_LOG_PATH = _paths.manager_log
SERVER_LOG_PATH = _paths.server_log
