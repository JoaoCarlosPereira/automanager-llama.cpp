import json
import time
from pathlib import Path

import pytest

from cliproxy_auth import (
    CLIProxyAuthManager,
    ensure_runtime_config,
    list_provider_auth_status,
    parse_login_output,
)


class FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        if not self._lines:
            return ""
        return self._lines.pop(0)


class FakeProcess:
    def __init__(self, lines, return_code=0):
        self.stdout = FakeStdout(lines)
        self.stdin = type("Stdin", (), {"write": lambda self, data: None, "flush": lambda self: None})()
        self._return_code = return_code
        self._polled = False

    def poll(self):
        if not self._lines_remaining() and self._polled:
            return self._return_code
        return None

    def _lines_remaining(self):
        return bool(self.stdout._lines)

    def wait(self, timeout=None):
        self._polled = True
        return self._return_code

    def terminate(self):
        self._return_code = -15

    def kill(self):
        self._return_code = -9


class FakePlatformManager:
    cliproxy_detection = type(
        "Detection",
        (),
        {"detected": True, "path": "/usr/local/bin/cli-proxy-api", "reason": None},
    )()


def test_parse_device_login_output():
    text = """
Starting Codex device authentication...
Codex device URL: https://auth.openai.com/codex/device
Codex device code: W2F1-HUYX5
Waiting for Codex authentication callback...
"""
    parsed = parse_login_output(text)
    assert parsed["auth_url"] == "https://auth.openai.com/codex/device"
    assert parsed["device_code"] == "W2F1-HUYX5"


def test_parse_oauth_login_output():
    text = """
Visit the following URL to continue authentication:
https://claude.ai/oauth/authorize?client_id=test
Waiting for Claude authentication callback...
"""
    parsed = parse_login_output(text)
    assert parsed["auth_url"].startswith("https://claude.ai/oauth/authorize")
    assert parsed["needs_callback"] is True


def test_parse_codex_oauth_open_browser_output():
    text = """
Opening browser for Codex authentication
Attempting to open URL in browser: https://auth.openai.com/oauth/authorize?client_id=test
Waiting for Codex authentication callback...
Paste the Codex callback URL (or press Enter to keep waiting):
"""
    parsed = parse_login_output(text)
    assert parsed["auth_url"].startswith("https://auth.openai.com/oauth/authorize")
    assert parsed["needs_callback"] is True
    assert "Paste the Codex callback URL" in (parsed["callback_hint"] or "")


def test_list_provider_auth_status_detects_codex_account(tmp_path):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "codex-user@example.com-plus.json").write_text("{}", encoding="utf-8")

    statuses = list_provider_auth_status(tmp_path)
    assert statuses["codex"]["authenticated"] is True
    assert statuses["codex"]["accounts"] == ["codex-user@example.com-plus.json"]
    assert statuses["claude"]["authenticated"] is False


def test_start_login_creates_session_and_marks_waiting(tmp_path):
    ensure_runtime_config(tmp_path)
    manager = CLIProxyAuthManager(
        FakePlatformManager(),
        runtime_dir=tmp_path,
        popen_factory=lambda *args, **kwargs: FakeProcess(
            [
                "Codex device URL: https://auth.openai.com/codex/device\n",
                "Codex device code: ABCD-1234\n",
            ],
            return_code=0,
        ),
    )

    session = manager.start_login("codex")
    deadline = time.time() + 2
    while time.time() < deadline:
        current = manager.get_session(session["id"])
        if current and current.get("device_code"):
            break
        time.sleep(0.05)

    current = manager.get_session(session["id"])
    assert current is not None
    assert current["device_code"] == "ABCD-1234"
    assert current["auth_url"] == "https://auth.openai.com/codex/device"


def test_start_login_marks_completed_when_auth_file_appears(tmp_path):
    ensure_runtime_config(tmp_path)
    auth_dir = tmp_path / "auth"

    def popen_factory(*args, **kwargs):
        (auth_dir / "codex-user@example.com-plus.json").write_text(
            json.dumps({"ok": True}),
            encoding="utf-8",
        )
        return FakeProcess([], return_code=0)

    manager = CLIProxyAuthManager(
        FakePlatformManager(),
        runtime_dir=tmp_path,
        popen_factory=popen_factory,
    )
    session = manager.start_login("codex")
    deadline = time.time() + 2
    while time.time() < deadline:
        current = manager.get_session(session["id"])
        if current and current["status"] == "completed":
            break
        time.sleep(0.05)

    current = manager.get_session(session["id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["accounts"]
