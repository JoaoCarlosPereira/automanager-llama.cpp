import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import llama_manager
from llama_manager import app


class FakeAuthManager:
    """Small auth double that exercises route behavior without real config I/O."""

    valid_session = "session-ok"

    def __init__(self, allow_session=True, allow_api_key=True):
        self.allow_session = allow_session
        self.allow_api_key = allow_api_key
        self.logged_out = []

    def authenticate(self, username, password):
        if username == "admin" and password == "admin":
            return self.valid_session
        return None

    def verify_session(self, session_token):
        return self.allow_session and session_token == self.valid_session

    def verify_api_key(self, credentials):
        return self.allow_api_key and credentials.credentials == "sk-valid-token"

    def logout(self, session_token):
        self.logged_out.append(session_token)

    def change_password(self, old_password, new_password):
        return old_password == "admin" and bool(new_password)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llama_manager, "auth_manager", FakeAuthManager())
    return TestClient(app)


@pytest.fixture
def authenticated_client(client):
    client.cookies.set("session_token", FakeAuthManager.valid_session)
    return client


def test_login_success_sets_session_cookie(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.cookies.get("session_token") == FakeAuthManager.valid_session

    set_cookie = response.headers["set-cookie"]
    assert "session_token=session-ok" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie


def test_login_failure_returns_401_without_session_cookie(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciais invalidas"
    assert client.cookies.get("session_token") is None
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("POST", "/api/auth/change-password", {"username": "old", "password": "new"}),
        ("GET", "/status", None),
        (
            "POST",
            "/start",
            {
                "path": "/media/docker/models/model.gguf",
                "gpu_weights": [
                    {"index": 0, "weight": 100.0, "name": "GPU-0", "active": True}
                ],
                "context_size": 4096,
            },
        ),
        ("POST", "/stop", None),
        ("GET", "/metrics", None),
        ("GET", "/models", None),
        (
            "POST",
            "/rename",
            {"path": "/media/docker/models/model.gguf", "new_name": "renamed"},
        ),
        ("POST", "/delete", {"path": "/media/docker/models/model.gguf"}),
        ("POST", "/downloads", {"url": "https://example.com/model.gguf"}),
        ("GET", "/downloads", None),
        ("GET", "/api/key", None),
        ("POST", "/api/key/renew", None),
        ("GET", "/config", None),
        ("POST", "/set_default", {"path": "/media/docker/models/model.gguf"}),
    ],
)
def test_protected_endpoints_return_401_without_auth(
    monkeypatch, method, path, json_body
):
    monkeypatch.setattr(
        llama_manager,
        "auth_manager",
        FakeAuthManager(allow_session=False, allow_api_key=False),
    )
    client = TestClient(app)

    kwargs = {"json": json_body} if json_body is not None else {}
    response = client.request(method, path, **kwargs)

    assert response.status_code == 401


def test_api_key_endpoint_returns_current_key_when_authenticated(
    monkeypatch, authenticated_client
):
    token_manager = MagicMock()
    token_manager.get_or_create.return_value = "sk-current"
    monkeypatch.setattr(llama_manager, "token_manager", token_manager)

    response = authenticated_client.get("/api/key")

    assert response.status_code == 200
    assert response.json() == {"key": "sk-current"}
    token_manager.get_or_create.assert_called_once_with()


def test_api_key_renew_endpoint_returns_new_key_when_authenticated(
    monkeypatch, authenticated_client
):
    token_manager = MagicMock()
    token_manager.renew.return_value = "sk-renewed"
    monkeypatch.setattr(llama_manager, "token_manager", token_manager)

    response = authenticated_client.post("/api/key/renew")

    assert response.status_code == 200
    assert response.json() == {"key": "sk-renewed"}
    token_manager.renew.assert_called_once_with()


def test_status_endpoint_uses_process_manager_mock(monkeypatch, authenticated_client):
    process_manager = MagicMock()
    process_manager.get_status.return_value = {
        "running": True,
        "pid": 1234,
        "model": "model.gguf",
        "recovery": {"active": False, "failed": False, "message": ""},
    }
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)

    response = authenticated_client.get("/status")

    assert response.status_code == 200
    assert response.json() == process_manager.get_status.return_value
    process_manager.get_status.assert_called_once_with()


def test_metrics_endpoint_uses_gpu_detector_mock(monkeypatch, authenticated_client):
    gpu_detector = MagicMock()
    gpu_detector.get_metrics.return_value = {
        "cpu": 12.5,
        "ram": 45.0,
        "gpus": [{"index": 0, "utilization": 7, "memory_used": 1024}],
    }
    monkeypatch.setattr(llama_manager, "gpu_detector", gpu_detector)

    response = authenticated_client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == gpu_detector.get_metrics.return_value
    gpu_detector.get_metrics.assert_called_once_with()


def test_models_endpoint_uses_model_scanner_mock(monkeypatch, authenticated_client):
    model_scanner = MagicMock()
    model_scanner.scan.return_value = {
        "models": [
            {
                "path": "/media/docker/models/model.gguf",
                "name": "model.gguf",
                "dir": ".",
                "last_config": None,
                "mmproj_candidates": [],
                "auto_mmproj": None,
            }
        ],
        "projectors": [],
    }
    monkeypatch.setattr(llama_manager, "model_scanner", model_scanner)

    response = authenticated_client.get("/models")

    assert response.status_code == 200
    assert response.json() == model_scanner.scan.return_value
    model_scanner.scan.assert_called_once_with()
