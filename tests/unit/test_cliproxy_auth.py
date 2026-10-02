import json
import time
from pathlib import Path

import pytest

from cliproxy_auth import (
    CLIProxyAuthManager,
    ensure_runtime_config,
    list_provider_auth_status,
    parse_login_output,
    set_provider_account_order,
    sync_antigravity_from_cli,
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


def test_list_provider_auth_status_detects_cursor_account(tmp_path):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "cursor-dev@example.com.json").write_text(
        '{"type":"cursor"}', encoding="utf-8"
    )

    statuses = list_provider_auth_status(tmp_path)
    assert statuses["cursor"]["authenticated"] is True
    assert statuses["cursor"]["accounts"] == ["cursor-dev@example.com.json"]
    assert statuses["cursor"]["default_method"] == "oauth"


def test_start_login_cursor_uses_cli_flag(tmp_path):
    ensure_runtime_config(tmp_path)
    captured = {}

    def factory(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProcess(["https://cursor.com/login?user=dev\n"], return_code=0)

    manager = CLIProxyAuthManager(
        FakePlatformManager(),
        runtime_dir=tmp_path,
        popen_factory=factory,
    )
    view = manager.start_login("cursor")
    deadline = time.time() + 2
    while time.time() < deadline and not captured.get("cmd"):
        time.sleep(0.01)

    assert captured["cmd"][1:3] == ["-cursor-login", "-no-browser"]
    assert view["provider"] == "cursor"


def test_account_order_prefers_higher_priority_and_persists(tmp_path):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "claude-financeiro.json").write_text(
        json.dumps({"type": "claude", "access_token": "keep-me", "priority": 1}),
        encoding="utf-8",
    )
    (auth_dir / "claude-joao.json").write_text(
        json.dumps({"type": "claude", "access_token": "also-keep"}),
        encoding="utf-8",
    )

    status = list_provider_auth_status(tmp_path)["claude"]
    assert status["accounts"] == ["claude-financeiro.json", "claude-joao.json"]
    assert status["account_details"][0]["priority"] == 1

    updated = set_provider_account_order(
        "claude",
        ["claude-joao.json", "claude-financeiro.json"],
        tmp_path,
    )
    assert updated["accounts"] == ["claude-joao.json", "claude-financeiro.json"]
    first = json.loads((auth_dir / "claude-joao.json").read_text(encoding="utf-8"))
    second = json.loads((auth_dir / "claude-financeiro.json").read_text(encoding="utf-8"))
    assert first["priority"] == 2
    assert first["access_token"] == "also-keep"
    assert second["priority"] == 1
    assert second["access_token"] == "keep-me"


def test_account_order_rejects_unknown_account(tmp_path):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "claude-financeiro.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        set_provider_account_order(
            "claude",
            ["claude-financeiro.json", "../secrets.json"],
            tmp_path,
        )


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


def test_sync_antigravity_from_cli_missing(tmp_path):
    missing_file = tmp_path / "does-not-exist"
    res = sync_antigravity_from_cli(runtime_dir=tmp_path, cli_token_path=missing_file)
    assert res is None


def test_sync_antigravity_from_cli_success(tmp_path):
    import base64

    # Build dummy JWT id_token with email
    payload = json.dumps({"email": "test-dev@example.com"}).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
    fake_jwt = f"header.{b64_payload}.signature"

    token_file = tmp_path / "cli_token"
    token_file.write_text(
        json.dumps({
            "token": {
                "access_token": "ya29.fake-token",
                "refresh_token": "1//fake-refresh",
                "token_type": "Bearer",
                "expiry": "2026-10-01T17:00:00Z",
            },
            "id_token": fake_jwt,
            "auth_method": "oauth",
        }),
        encoding="utf-8",
    )

    dest = sync_antigravity_from_cli(runtime_dir=tmp_path, cli_token_path=token_file)
    assert dest is not None
    assert dest.name == "antigravity-test-dev@example.com.json"
    assert dest.is_file()

    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved["type"] == "antigravity"
    assert saved["email"] == "test-dev@example.com"
    assert saved["access_token"] == "ya29.fake-token"
    assert saved["refresh_token"] == "1//fake-refresh"
    assert saved["project_id"] == "aicode-consumers"
    assert saved["expired"] == "2026-10-01T17:00:00Z"
    assert saved["disabled"] is False

    saved["priority"] = 2
    saved["project_id"] = "custom-project"
    dest.write_text(json.dumps(saved), encoding="utf-8")
    sync_antigravity_from_cli(runtime_dir=tmp_path, cli_token_path=token_file)
    kept = json.loads(dest.read_text(encoding="utf-8"))
    assert kept["priority"] == 2
    assert kept["project_id"] == "custom-project"
    assert kept["type"] == "antigravity"
    assert kept["access_token"] == "ya29.fake-token"


def test_forward_http_callback(monkeypatch):
    import http.server
    import threading

    received_path = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            received_path.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Foauth-callback"
    callback_url = f"http://localhost:{port}/oauth-callback?code=testcode123&state=teststate456"

    CLIProxyAuthManager._forward_http_callback(callback_url, auth_url)
    t.join(timeout=2)
    server.server_close()

    assert len(received_path) == 1
    assert "code=testcode123" in received_path[0]
    assert "state=teststate456" in received_path[0]

