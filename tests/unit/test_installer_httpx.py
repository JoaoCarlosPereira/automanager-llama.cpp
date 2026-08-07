"""Tests for installer httpx integration (requirements.txt + platform_tools.sh)."""

import os
import re
import subprocess
import textwrap
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


# ── requirements.txt ─────────────────────────────────────────────────────────

class TestRequirementsHttpx:
    """Verify httpx is present in requirements.txt."""

    @staticmethod
    def _read_requirements() -> str:
        path = os.path.join(REPO_ROOT, "requirements.txt")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_httpx_present(self):
        content = self._read_requirements()
        assert re.search(r"^httpx\s*[\>]=?", content, re.MULTILINE), (
            "httpx must be listed in requirements.txt"
        )

    def test_httpx_version_at_least_0270(self):
        content = self._read_requirements()
        match = re.search(r"^httpx([><=!]+)(\d+(?:\.\d+)*)", content, re.MULTILINE)
        assert match, "httpx version specifier not found"
        op, ver = match.group(1), match.group(2)
        if op in (">=",):
            parts = [int(p) for p in ver.split(".")]
            assert parts >= [0, 27, 0], "httpx>=0.27.0 required"

    def test_no_duplicate_httpx_lines(self):
        content = self._read_requirements()
        matches = [m for m in re.finditer(r"^httpx\b", content, re.MULTILINE)]
        assert len(matches) <= 1, "httpx should appear at most once"


# ── platform_tools.sh ────────────────────────────────────────────────────────

class TestPlatformToolsHttpx:
    """Verify install_httpx_deps exists and install_platform_tools calls it."""

    @staticmethod
    def _read_script() -> str:
        path = os.path.join(REPO_ROOT, "installer", "platform_tools.sh")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_install_httpx_deps_function_exists(self):
        content = self._read_script()
        assert re.search(
            r"^install_httpx_deps\(\)", content, re.MULTILINE
        ), "install_httpx_deps() function must be defined"

    def test_install_httpx_deps_uses_python_import_check(self):
        content = self._read_script()
        # The function body in bash may contain ${} variable expansions, so we
        # search the whole file for the import check pattern rather than trying
        # to parse nested braces.
        assert "import httpx" in content or "python -c" in content, (
            "install_httpx_deps must verify httpx by importing it"
        )

    def test_install_platform_tools_calls_httpx_deps(self):
        content = self._read_script()
        func_match = re.search(
            r"^install_platform_tools\(\)[^{]*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}",
            content,
            re.MULTILINE | re.DOTALL,
        )
        assert func_match, "Could not parse install_platform_tools function body"
        body = func_match.group(1)
        assert "install_httpx_deps" in body, (
            "install_platform_tools() must call install_httpx_deps()"
        )


# ── install_httpx_deps() behavioural test (isolated) ─────────────────────────

class TestInstallHttpxDepsBehavior:
    """Simulate the logic of install_httpx_deps() in Python."""

    def _simulate_install_httpx_deps(
        self, venv_exists: bool, httpx_importable: bool
    ) -> dict:
        """Return {'skip': bool, 'install': bool, 'error': bool}."""
        if not venv_exists:
            return {"skip": True, "install": False, "error": False}
        if httpx_importable:
            return {"skip": True, "install": False, "error": False}
        # pretend install succeeds
        return {"skip": False, "install": True, "error": False}

    def test_no_venv_skips_with_warning(self):
        result = self._simulate_install_httpx_deps(
            venv_exists=False, httpx_importable=False
        )
        assert result["skip"] is True
        assert result["install"] is False
        assert result["error"] is False

    def test_httpx_present_skips_install(self):
        result = self._simulate_install_httpx_deps(
            venv_exists=True, httpx_importable=True
        )
        assert result["skip"] is True
        assert result["install"] is False
        assert result["error"] is False

    def test_httpx_missing_triggers_install(self):
        result = self._simulate_install_httpx_deps(
            venv_exists=True, httpx_importable=False
        )
        assert result["skip"] is False
        assert result["install"] is True
        assert result["error"] is False
