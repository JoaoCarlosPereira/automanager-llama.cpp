"""HTML contract tests for the dashboard SPA served at GET /."""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import llama_manager
from llama_manager import app

FAKE_IP = "192.168.50.10"
FAKE_API_TOKEN = "sk-test-html-contract-token-32chars"


class FakeAuthManager:
    valid_session = "session-html-contract"

    def verify_session(self, session_token):
        return session_token == self.valid_session


@pytest.fixture
def mock_index_deps(monkeypatch):
    """Mocks services used by GET / so tests run without GPU or disk."""
    monkeypatch.setattr(llama_manager, "get_local_ip", lambda: FAKE_IP)

    token_manager = MagicMock()
    token_manager.get_or_create.return_value = FAKE_API_TOKEN
    monkeypatch.setattr(llama_manager, "token_manager", token_manager)

    model_scanner = MagicMock()
    model_scanner.scan.return_value = {
        "models": [
            {
                "path": "/tmp/models/test.gguf",
                "name": "test.gguf",
                "dir": "models",
                "last_config": None,
                "mmproj_candidates": [],
                "auto_mmproj": None,
            }
        ],
        "projectors": [],
    }
    monkeypatch.setattr(llama_manager, "model_scanner", model_scanner)

    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "NVIDIA Test GPU", "vram": 24000}
    ]
    monkeypatch.setattr(llama_manager, "gpu_manager", gpu_manager)

    config_manager = MagicMock()
    config_manager.load.return_value = {"default_model": None, "model_configs": {}}
    monkeypatch.setattr(llama_manager, "config_manager", config_manager)

    process_manager = MagicMock()
    process_manager.get_status.return_value = {"running": False}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)

    auth_manager = FakeAuthManager()
    monkeypatch.setattr(llama_manager, "auth_manager", auth_manager)
    return auth_manager


@pytest.fixture
def client(mock_index_deps):
    return TestClient(app)


@pytest.fixture
def html(client):
    response = client.get("/")
    assert response.status_code == 200
    return response.text


def test_html_contains_login_overlay(html):
    assert 'id="login-overlay"' in html
    assert 'id="login-form"' in html
    assert 'id="login-username"' in html
    assert 'id="login-password"' in html
    assert 'onsubmit="handleLogin(event)"' in html


def test_html_contains_status_badge(html):
    assert 'id="status-badge"' in html
    assert "OFFLINE" in html


def test_html_contains_metrics_panel(html):
    assert 'id="metrics-panel"' in html
    assert 'id="cpu-val"' in html
    assert 'id="cpu-bar"' in html
    assert 'id="ram-val"' in html
    assert 'id="ram-bar"' in html


def test_html_contains_gpu_table(html):
    assert 'id="gpu-table-body"' in html
    assert "gpu-row" in html
    assert "gpu-weight" in html
    assert "gpu-checkbox" in html
    assert "gpu-pin" in html


def test_html_contains_model_list(html):
    assert 'id="model-list-container"' in html
    assert "model-item-container" in html


def test_html_contains_log_terminal(html):
    assert 'id="log-box"' in html
    assert "Limpar" in html


def test_html_contains_pacman_canvas(html):
    assert 'id="pacman-background"' in html
    assert 'aria-hidden="true"' in html


def test_html_contains_active_model_card(html):
    assert 'id="active-card"' in html
    assert 'id="active-model-name"' in html
    assert 'id="uptime-val"' in html


def test_html_serves_external_js_scripts(html):
    for script in ("auth.js", "models.js", "metrics.js", "gpu.js", "index.js"):
        assert f'type="module" src="/static/js/{script}"' in html


def test_html_contains_api_token(html):
    assert 'id="api-token"' in html
    assert FAKE_API_TOKEN in html
    assert 'id="api-link"' in html


def test_html_injects_ip(html):
    assert f'id="display-ip"' in html
    assert FAKE_IP in html
    assert f'window.fixedIp = "{FAKE_IP}"' in html
    assert 'id="chat-link"' in html


def test_html_contains_default_model_checkbox(html):
    assert "model-default-checkbox" in html


def test_html_contains_context_size_select(html):
    assert 'id="context-size"' in html
    assert 'id="context-size-custom"' in html


def test_html_contains_parallel_slots_input(html):
    assert 'id="parallel-slots"' in html


def test_html_contains_batch_size_select(html):
    assert 'id="batch-size"' in html


def test_html_contains_mmproj_select(html):
    assert 'id="mmproj-path"' in html


def test_html_contains_split_mode_select(html):
    assert 'id="split-mode"' in html


def test_html_contains_auto_balance_toggle(html):
    assert 'id="auto-balance-toggle"' in html
    assert 'id="auto-balance-badge"' in html


def test_html_contains_auto_balance_cancel_btn(html):
    assert 'id="auto-balance-cancel-btn"' in html


def test_html_contains_auto_balance_capacity_alert(html):
    assert 'id="auto-balance-capacity-alert"' in html


def test_html_contains_download_url_input(html):
    assert 'id="download-url"' in html


def test_html_contains_download_status(html):
    assert 'id="download-status"' in html


def test_html_contains_password_change_section(html):
    assert 'id="current-password"' in html
    assert 'id="new-password"' in html
    assert 'id="password-change-status"' in html


def test_login_overlay_visible_when_unauthenticated(client):
    html = client.get("/").text
    assert 'id="login-overlay"' in html
    assert 'style="display: flex;"' in html
    assert 'id="dashboard"' in html
    assert 'style="display: none;"' in html


def test_login_overlay_hidden_when_authenticated(client):
    client.cookies.set("session_token", FakeAuthManager.valid_session)
    html = client.get("/").text
    assert 'id="login-overlay"' in html
    assert 'style="display: none;"' in html
    assert 'style="display: block;"' in html
