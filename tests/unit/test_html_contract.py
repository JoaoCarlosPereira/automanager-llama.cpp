"""HTML contract tests for the dashboard SPA served at GET /.

Asserts the contract of the multi-model tabbed UI (4.0.0 refactor): a sidebar
"Biblioteca", a fixed host-metrics panel, a tab bar/container, and a
``<template id="model-tab-template">`` carrying the per-model controls (the
``tab-*`` classes cloned into each open tab).
"""

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

    def check_auth_cookie(self, request):
        return request.cookies.get("session_token") == self.valid_session

    def check_auth(self, request):
        return True


def _inline_display(html, element_id):
    """Return the inline ``display:`` value of the element with the given id."""
    start = html.index(f'id="{element_id}"')
    segment = html[start:start + 500]
    marker = "style=\"display: "
    j = segment.index(marker) + len(marker)
    return segment[j:segment.index(";", j)]


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
        "storage": {
            "path": "/tmp/models",
            "used_gb": 0.0,
            "total_gb": 100.0,
        },
    }
    monkeypatch.setattr(llama_manager, "model_scanner", model_scanner)

    cpu_info_mock = MagicMock()
    cpu_info_mock.name = "Test CPU"
    cpu_info_mock.ram_total_mb = 16384
    cpu_info_mock.ram_used_mb = 4096
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "NVIDIA Test GPU", "vram": 24000}
    ]
    gpu_manager.detect_cpu_info.return_value = cpu_info_mock
    monkeypatch.setattr(llama_manager, "gpu_manager", gpu_manager)

    config_manager = MagicMock()
    config_manager.get_config.return_value = {"default_model": None, "model_configs": {}}
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


# ── Login / shell ─────────────────────────────────────────────────────────


def test_html_contains_login_overlay(html):
    assert 'id="login-overlay"' in html
    assert 'id="login-form"' in html
    assert 'id="login-username"' in html
    assert 'id="login-password"' in html
    assert 'onsubmit="handleLogin(event)"' in html


def test_html_contains_status_badge(html):
    assert 'id="status-badge"' in html
    assert "OFFLINE" in html


def test_html_contains_sidebar_shell(html):
    """The collapsible 'Biblioteca' sidebar and its toggle."""
    assert 'id="sidebar"' in html
    assert 'id="sidebar-toggle"' in html
    assert "Biblioteca" in html


# ── Host metrics panel ────────────────────────────────────────────────────


def test_html_contains_metrics_panel(html):
    assert 'id="metrics-panel"' in html
    assert 'id="cpu-val"' in html
    assert 'id="cpu-bar"' in html
    assert 'id="ram-val"' in html
    assert 'id="ram-bar"' in html


def test_html_contains_mini_gpu_metrics(html):
    """GPU cards are rendered dynamically into the mini-gpu-metrics strip."""
    assert 'id="mini-gpu-metrics"' in html


# ── Multi-model tab system ────────────────────────────────────────────────


def test_html_contains_tab_system(html):
    assert 'id="tab-bar"' in html
    assert 'id="tabs-container"' in html
    assert 'id="no-tab-content"' in html
    assert "Arquitetura Multi-Modelo" in html


def test_html_contains_model_tab_template(html):
    assert 'id="model-tab-template"' in html
    assert "model-tab-name" in html
    assert "model-tab-path" in html
    assert "tab-status-badge" in html
    assert "tab-actions" in html


def test_template_contains_engine_params(html):
    """Per-model engine parameters live as tab-* controls in the template."""
    assert "tab-context-size" in html
    assert "tab-context-size-custom" in html
    assert "tab-parallel-slots" in html
    assert "tab-batch-size" in html
    assert "tab-ubatch-size" in html
    assert "tab-cache-type-k" in html
    assert "tab-cache-type-v" in html
    assert "tab-threads" in html
    assert "tab-split-mode" in html


def test_template_contains_feature_toggles(html):
    assert "tab-thinking-toggle" in html
    assert "tab-mtp-toggle" in html
    assert "tab-numa-toggle" in html
    assert "tab-auto-balance-toggle" in html


def test_template_contains_gpu_weights_body(html):
    """GPU/CPU weight rows are injected into the per-tab table body by JS."""
    assert "tab-gpu-table-body" in html


def test_template_contains_calibration_controls(html):
    assert "tab-smart-calibrate-btn" in html
    assert "tab-reset-defaults-btn" in html
    assert "tab-proposed-config" in html
    assert "tab-apply-config-btn" in html
    assert "tab-discard-config-btn" in html


def test_template_contains_log_box(html):
    assert "tab-log-box" in html
    assert "tab-clear-logs-btn" in html


# ── Model library (sidebar) ───────────────────────────────────────────────


def test_html_contains_model_list(html):
    assert 'id="model-list-container"' in html
    assert "model-item-container" in html
    assert 'id="model-count"' in html


def test_html_contains_default_model_toggle(html):
    """Each library item carries an auto-start (default) checkbox."""
    assert "setDefaultModel(this," in html


# ── Downloads ─────────────────────────────────────────────────────────────


def test_html_contains_download_section(html):
    assert 'id="download-url"' in html
    assert 'id="download-list"' in html


# ── Vision import ─────────────────────────────────────────────────────────


def test_html_contains_vision_import_modal(html):
    assert 'id="vision-import-modal"' in html
    assert 'id="vision-import-url"' in html
    assert "submitVisionImport" in html


def test_html_does_not_contain_global_mmproj_select(html):
    assert 'id="mmproj-path"' not in html


# ── Version update modal ──────────────────────────────────────────────────


def test_html_contains_version_update_modal(html):
    assert 'id="version-update-modal"' in html
    assert 'id="version-commits-list"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert "dismissVersionModal()" in html


# ── API token / IP / config ───────────────────────────────────────────────


def test_html_contains_api_token(html):
    assert 'id="api-token"' in html
    assert FAKE_API_TOKEN in html


def test_html_injects_ip(html):
    assert FAKE_IP in html
    assert f'window.fixedIp = "{FAKE_IP}"' in html


def test_html_contains_models_dir_config(html):
    assert 'id="models-dir-input"' in html
    assert 'id="repo-storage"' in html


def test_html_contains_password_change_section(html):
    assert 'id="current-password"' in html
    assert 'id="new-password"' in html
    assert 'id="password-change-status"' in html


# ── Assets / regressions ──────────────────────────────────────────────────


def test_html_serves_dashboard_js(html):
    assert 'type="module" src="/static/js/index.js?v=' in html
    assert 'src="/static/js/pacman_bg.js?v=' not in html


def test_html_does_not_contain_pacman_canvas(html):
    assert 'id="pacman-background"' not in html


# ── Auth-gated visibility ─────────────────────────────────────────────────


def test_login_overlay_visible_when_unauthenticated(client):
    html = client.get("/").text
    assert _inline_display(html, "login-overlay") == "flex"
    assert _inline_display(html, "dashboard") == "none"


def test_login_overlay_hidden_when_authenticated(client):
    client.cookies.set("session_token", FakeAuthManager.valid_session)
    html = client.get("/").text
    assert _inline_display(html, "login-overlay") == "none"
    assert _inline_display(html, "dashboard") == "flex"
