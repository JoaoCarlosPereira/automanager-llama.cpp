"""Installer coverage for the official llama.cpp binary and managed uninstall."""

import json
import os
import stat
import subprocess
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INSTALLER = os.path.join(REPO_ROOT, "installer")


def _load_selector():
    import importlib.util

    path = os.path.join(INSTALLER, "llama_cpp_release.py")
    spec = importlib.util.spec_from_file_location("llama_cpp_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release(tag, names):
    return {"tag_name": tag, "assets": [{"name": name} for name in names]}


def _sample_releases():
    tag = "b11307"
    names = [
        f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        f"llama-{tag}-bin-ubuntu-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-arm64.tar.gz",
        f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        f"cudart-llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"cudart-llama-{tag}-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
        "nightly-tag.txt",
    ]
    return [
        _release("v0.5.0", ["nightly-tag.txt"]),
        _release(tag, names),
    ]


class TestLlamaCppAssetSelection:
    def test_skips_release_without_binaries_and_matches_driver(self):
        module = _load_selector()
        selected = module.select_llama_cpp_assets(
            _sample_releases(), arch="x64", cuda_version="12.8"
        )
        assert selected["tag"] == "b11307"
        assert selected["asset"] == "llama-b11307-bin-ubuntu-cuda-12.8-x64.tar.gz"
        assert selected["cudart"] == "cudart-llama-b11307-bin-ubuntu-cuda-12.8-x64.tar.gz"
        assert selected["variant"] == "cuda-12.8"
        assert selected["repo"] == "https://github.com/ggml-org/llama.cpp"

    def test_newer_driver_picks_newer_cuda_build(self):
        module = _load_selector()
        selected = module.select_llama_cpp_assets(
            _sample_releases(), arch="x64", cuda_version="13.4"
        )
        assert selected["variant"] == "cuda-13.4"
        assert "cuda-13.4-x64" in selected["asset"]

    def test_old_driver_falls_back_to_cpu_archive(self):
        module = _load_selector()
        selected = module.select_llama_cpp_assets(
            _sample_releases(), arch="x64", cuda_version="11.7"
        )
        assert selected["variant"] == "cpu"
        assert selected["asset"] == "llama-b11307-bin-ubuntu-x64.tar.gz"
        assert selected["cudart"] == ""

    def test_unknown_cuda_picks_oldest_cuda_build(self):
        module = _load_selector()
        selected = module.select_llama_cpp_assets(
            _sample_releases(), arch="x64", cuda_version=None
        )
        assert selected["variant"] == "cuda-12.8"

    def test_arm64_does_not_select_x64_archive(self):
        module = _load_selector()
        selected = module.select_llama_cpp_assets(
            _sample_releases(), arch="arm64", cuda_version="13.4"
        )
        assert selected["asset"].endswith("-arm64.tar.gz")
        assert "x64" not in selected["asset"]

    def test_cli_prints_tsv(self):
        script = os.path.join(INSTALLER, "llama_cpp_release.py")
        proc = subprocess.run(
            ["python3", script, "--arch", "x64", "--cuda", "12.8"],
            input=json.dumps(_sample_releases()),
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        tag, asset, cudart, variant = proc.stdout.rstrip("\n").split("\t")
        assert tag == "b11307"
        assert asset.startswith("llama-b11307-bin-ubuntu-cuda-12.8")
        assert cudart.startswith("cudart-llama-b11307")
        assert variant == "cuda-12.8"

    def test_cli_fails_when_nothing_matches(self):
        script = os.path.join(INSTALLER, "llama_cpp_release.py")
        proc = subprocess.run(
            ["python3", script, "--arch", "x64", "--cuda", "12.8"],
            input="[]",
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 1


class TestInstallerScripts:
    @staticmethod
    def _read(name):
        with open(os.path.join(INSTALLER, name), encoding="utf-8") as handle:
            return handle.read()

    def test_shell_scripts_parse(self):
        for name in ("setup.sh", "uninstall.sh", "llama_cpp.sh", "platform_tools.sh"):
            proc = subprocess.run(
                ["bash", "-n", os.path.join(INSTALLER, name)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr

    def test_setup_installs_requirements_and_llama_cpp(self):
        setup = self._read("setup.sh")
        assert "Skipping venv creation" not in setup
        pip_at = setup.index('pip install -r "${PROJECT_DIR}/requirements.txt"')
        llama_at = setup.index("install_llama_cpp")
        assert pip_at < llama_at
        assert 'SERVICE_NAME="llama-manager.service"' in setup
        assert setup.index('SERVICE_NAME="llama-manager.service"') < setup.index(
            "${SERVICE_NAME}"
        )
        assert "https://github.com/ggml-org/llama.cpp" in self._read("llama_cpp.sh")
        assert "libgomp1" in setup
        assert "libnuma1" in setup

    def test_uninstall_removes_managed_binaries_and_purges_hf_cache(self):
        uninstall = self._read("uninstall.sh")
        assert "remove_managed_llama_cpp" in uninstall
        assert "huggingface_cache" in uninstall
        assert "data/models" in uninstall


def _run_bash(script, env):
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc.stdout


class TestManagedLlamaCppRemoval:
    def test_remove_managed_files_keeps_unlisted_files(self, tmp_path):
        project = tmp_path / "automanager"
        bindir = project / "bin"
        bindir.mkdir(parents=True)
        managed = bindir / "llama-server"
        managed.write_text("managed\n", encoding="utf-8")
        managed.chmod(managed.stat().st_mode | stat.S_IEXEC)
        extra = bindir / "custom-note.txt"
        extra.write_text("keep\n", encoding="utf-8")
        outside = tmp_path / "secret.txt"
        outside.write_text("secret\n", encoding="utf-8")
        (bindir / ".automanager-managed-files").write_text(
            textwrap.dedent(
                """\
                # https://github.com/ggml-org/llama.cpp b1 cuda-12.8
                llama-server
                ../secret.txt
                /etc/passwd
                """
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PROJECT_DIR"] = str(project)
        env["PATH"] = "/usr/bin:/bin"
        _run_bash(
            textwrap.dedent(
                """\
                set -euo pipefail
                log_info() { echo "INFO $*"; }
                log_warn() { echo "WARN $*"; }
                log_error() { echo "ERROR $*"; }
                source installer/llama_cpp.sh
                remove_managed_llama_cpp
                """
            ),
            env,
        )
        assert not managed.exists()
        assert extra.read_text(encoding="utf-8") == "keep\n"
        assert outside.read_text(encoding="utf-8") == "secret\n"
        assert not (bindir / ".automanager-managed-files").exists()

    def test_existing_unmanaged_binary_is_not_replaced(self, tmp_path):
        project = tmp_path / "automanager"
        bindir = project / "bin"
        bindir.mkdir(parents=True)
        binary = bindir / "llama-server"
        binary.write_text("custom-build\n", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        env = os.environ.copy()
        env["PROJECT_DIR"] = str(project)
        env["SCRIPT_DIR"] = INSTALLER
        env["PATH"] = "/usr/bin:/bin"
        env.pop("LLAMA_CPP_UPDATE", None)
        stdout = _run_bash(
            textwrap.dedent(
                """\
                set -euo pipefail
                log_info() { echo "INFO $*"; }
                log_warn() { echo "WARN $*"; }
                log_error() { echo "ERROR $*"; }
                source installer/llama_cpp.sh
                install_llama_cpp
                """
            ),
            env,
        )
        assert "Leaving it untouched" in stdout
        assert binary.read_text(encoding="utf-8") == "custom-build\n"
        assert not (bindir / ".automanager-managed-files").exists()
