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
    ), patch(
        "llama_server_bin._search_roots", return_value=[str(install_root)]
    ), patch(
        "llama_server_bin._glob_build_bins", return_value=[]
    ):
        resolved = lsb.resolve_llama_server_bin(
            install_root=str(install_root),
            paths_file=str(install_root / "missing.json"),
        )

    assert resolved is None


def test_list_llama_server_bins_deduplicates(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_a = install_root / "llama.cpp" / "build" / "bin" / exe_name
    bin_b = install_root / "llama-cpp-alt" / "build" / "bin" / exe_name
    _touch_executable(str(bin_a))
    _touch_executable(str(bin_b))

    lsb.reset_llama_server_bin_cache()
    with patch.dict(os.environ, {}, clear=True), patch(
        "llama_server_bin.shutil.which", return_value=None
    ), patch(
        "llama_server_bin.get_bin_version_info",
        side_effect=lambda path: {"build": "1", "commit": "abc", "label": os.path.basename(path)},
    ):
        result = lsb.list_llama_server_bins(
            install_root=str(install_root),
            paths_file=str(install_root / "missing.json"),
        )

    paths = [item["path"] for item in result["bins"]]
    assert str(bin_a.resolve()) in paths
    assert str(bin_b.resolve()) in paths
    assert len(paths) == len(set(paths))


def test_get_bin_version_info_parses_output():
    with patch("llama_server_bin.subprocess.run") as run:
        run.return_value.stdout = "version: 9554 (d403f00ec)\n"
        run.return_value.stderr = ""
        info = lsb.get_bin_version_info("/tmp/llama-server")
    assert info["build"] == "9554"
    assert info["commit"] == "d403f00ec"


def test_get_bin_version_info_parses_dev_output():
    with patch("llama_server_bin.subprocess.run") as run:
        run.return_value.stdout = "version: 0.3.0-dev (build 10665, commit e9b087580)\n"
        run.return_value.stderr = ""
        info = lsb.get_bin_version_info("/tmp/llama-server")
    assert info["build"] == "10665"
    assert info["commit"] == "e9b087580"


def test_is_turboquant_bin_detects_turbo_help(tmp_path):
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_path = tmp_path / exe_name
    _touch_executable(str(bin_path))

    with patch("llama_server_bin.get_llama_server_help") as help_mock:
        help_mock.return_value = "allowed values: turbo2, turbo3, turbo4"
        assert lsb.is_turboquant_bin(str(bin_path)) is True
        help_mock.return_value = "allowed values: f16, q8_0, q4_0"
        assert lsb.is_turboquant_bin(str(bin_path)) is False


def test_validate_turboquant_cache_types_requires_turbo_bin(tmp_path):
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    bin_path = tmp_path / exe_name
    _touch_executable(str(bin_path))

    with patch("llama_server_bin.is_turboquant_bin", return_value=False):
        err = lsb.validate_turboquant_cache_types("q8_0", "turbo3", str(bin_path))
        assert err is not None

    with patch("llama_server_bin.is_turboquant_bin", return_value=True):
        err = lsb.validate_turboquant_cache_types("q8_0", "turbo3", str(bin_path))
        assert err is None
