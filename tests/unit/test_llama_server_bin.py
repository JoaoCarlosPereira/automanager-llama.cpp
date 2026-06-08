"""Unit tests for llama-server binary discovery."""

import json
import os
import platform
from unittest.mock import patch

import llama_server_bin as lsb


def _touch_executable(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"")
    if platform.system() != "Windows":
        os.chmod(path, 0o755)


def test_resolve_from_paths_json(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    bin_dir = install_root / "custom" / "bin"
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_path = bin_dir / exe_name
    _touch_executable(str(bin_path))

    paths_file = install_root / "paths.json"
    paths_file.write_text(
        json.dumps({"llama_server_bin": "custom/bin/" + exe_name}),
        encoding="utf-8",
    )

    lsb.reset_llama_server_bin_cache()
    with patch.dict(os.environ, {}, clear=True):
        resolved = lsb.resolve_llama_server_bin(
            install_root=str(install_root),
            paths_file=str(paths_file),
        )

    assert resolved == str(bin_path.resolve())


def test_resolve_from_env_var(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_path = tmp_path / exe_name
    _touch_executable(str(bin_path))

    lsb.reset_llama_server_bin_cache()
    with patch.dict(os.environ, {"LLAMA_SERVER_BIN": str(bin_path)}, clear=True):
        resolved = lsb.resolve_llama_server_bin(
            install_root=str(install_root),
            paths_file=str(install_root / "missing.json"),
        )

    assert resolved == str(bin_path.resolve())


def test_resolve_from_candidate_relative_to_install_root(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_path = install_root / "llama.cpp" / "build" / "bin" / exe_name
    _touch_executable(str(bin_path))

    lsb.reset_llama_server_bin_cache()
    with patch.dict(os.environ, {}, clear=True), patch(
        "llama_server_bin.shutil.which", return_value=None
    ):
        resolved = lsb.resolve_llama_server_bin(
            install_root=str(install_root),
            paths_file=str(install_root / "missing.json"),
        )

    assert resolved == str(bin_path.resolve())


def test_resolve_returns_none_when_missing(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()

    lsb.reset_llama_server_bin_cache()
    with patch.dict(os.environ, {}, clear=True), patch(
        "llama_server_bin.shutil.which", return_value=None
    ):
        resolved = lsb.resolve_llama_server_bin(
            install_root=str(install_root),
            paths_file=str(install_root / "missing.json"),
        )

    assert resolved is None
