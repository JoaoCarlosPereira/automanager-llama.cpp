import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import llama_manager
from gpu_manager import ALL_GPU_LAYERS
from llama_manager import app
from schemas import GPUWeight

MIXED_GPU_CPU_PAYLOAD = {
    "path": "/media/docker/models/model.gguf",
    "gpu_weights": [
        {
            "index": 0,
            "weight": 70.0,
            "name": "GPU-0",
            "active": True,
            "device": "gpu",
        },
        {
            "index": -1,
            "weight": 30.0,
            "name": "CPU",
            "active": True,
            "device": "cpu",
        },
    ],
    "context_size": 4096,
    "total_layers": 32,
}


class FakeAuthManager:
    """Small auth double that exercises route behavior without real config I/O."""

    valid_session = "session-ok"

    def __init__(self, allow_session=True, allow_api_key=True):
        self.allow_session = allow_session
        self.allow_api_key = allow_api_key
        self.logged_out = []

    def authenticate(self, username, password):
        if username == "admin" and password == "admin":
            return {"token": self.valid_session, "force_password_change": False}
        return None

    def verify_session(self, session_token):
        if session_token in self.logged_out:
            return False
        return self.allow_session and session_token == self.valid_session

    def verify_api_key(self, credentials):
        return self.allow_api_key and credentials.credentials == "sk-valid-token"

    def logout(self, session_token):
        self.logged_out.append(session_token)

    def change_password(self, old_password, new_password):
        return old_password == "admin" and bool(new_password)

    def check_auth(self, request=None) -> bool:
        if request is None:
            return False
        session_token = request.cookies.get("session_token")
        if session_token and self.verify_session(session_token):
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            from fastapi.security import HTTPAuthorizationCredentials

            creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth_header[7:].strip(),
            )
            return self.verify_api_key(creds)
        return False


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
    assert response.json()["status"] == "ok"
    assert "force_password_change" in response.json()
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


def test_logout_clears_session_cookie(client, authenticated_client):
    response = authenticated_client.post("/api/auth/logout")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    status = authenticated_client.get("/status")
    assert status.status_code == 401


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


def test_change_password_endpoint_uses_auth_manager(authenticated_client):
    response = authenticated_client.post(
        "/api/auth/change-password",
        json={"username": "admin", "password": "new-password"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_change_password_endpoint_rejects_wrong_current_password(authenticated_client):
    response = authenticated_client.post(
        "/api/auth/change-password",
        json={"username": "wrong", "password": "new-password"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Senha atual incorreta"


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
        "cpu_name": "Test CPU",
        "ram": 45.0,
        "ram_total_mb": 16384,
        "ram_used_mb": 7340,
        "gpus": [{"index": 0, "utilization": 7, "memory_used": 1024}],
    }
    monkeypatch.setattr(llama_manager, "gpu_manager", gpu_detector)

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


def test_set_models_dir_updates_scanner_and_returns_models(
    monkeypatch, authenticated_client, tmp_path
):
    new_models_dir = tmp_path / "new-models"
    scan_payload = {
        "models": [],
        "projectors": [],
        "storage": {
            "path": str(new_models_dir),
            "used_gb": 0.0,
            "total_gb": 10.0,
        },
    }

    model_scanner = MagicMock()
    model_scanner.scan.return_value = scan_payload
    download_mgr = MagicMock()
    monkeypatch.setattr(llama_manager, "model_scanner", model_scanner)
    monkeypatch.setattr(llama_manager, "download_mgr", download_mgr)

    from paths import InstallPaths

    def fake_update(value):
        model_scanner.models_dir = str(new_models_dir)
        download_mgr.models_dir = str(new_models_dir)
        return InstallPaths(
            install_root=str(tmp_path),
            models_dir=str(new_models_dir),
            config_file=str(tmp_path / "config.json"),
            logs_dir=str(tmp_path / "logs"),
        )

    monkeypatch.setattr(llama_manager, "update_models_dir", fake_update)
    monkeypatch.setattr(llama_manager, "reload_module_paths", lambda: fake_update(""))

    response = authenticated_client.post(
        "/models/dir",
        json={"models_dir": str(new_models_dir)},
    )

    assert response.status_code == 200
    assert response.json() == scan_payload
    assert model_scanner.models_dir == str(new_models_dir)
    assert download_mgr.models_dir == str(new_models_dir)
    model_scanner.scan.assert_called_once_with()


def test_start_auto_balance_takes_priority_over_manual_override(
    monkeypatch, authenticated_client
):
    process_manager = MagicMock()
    process_manager.start_auto_balance.return_value = {
        "message": "Auto-balance em andamento",
        "probing": True,
    }
    process_manager.start.return_value = {"message": "Started"}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", MagicMock())

    response = authenticated_client.post(
        "/start",
        json={
            "path": "/media/docker/models/model.gguf",
            "gpu_weights": [
                {"index": 0, "weight": 100.0, "name": "GPU-0", "active": True}
            ],
            "context_size": 4096,
            "auto_balance": True,
            "manual_gpu_override": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["probing"] is True
    process_manager.start_auto_balance.assert_called_once()
    process_manager.start.assert_not_called()


def test_start_auto_balance_clears_pinned_flags(monkeypatch, authenticated_client):
    """POST /start com auto_balance clássico zera pinned GPU antes de delegar."""
    process_manager = MagicMock()
    process_manager.start_auto_balance.return_value = {"probing": True}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", MagicMock())

    response = authenticated_client.post(
        "/start",
        json={
            "path": "/media/docker/models/model.gguf",
            "gpu_weights": [
                {"index": 0, "weight": 70.0, "name": "GPU-0",
                 "active": True, "is_main": True, "pinned": True},
                {"index": 1, "weight": 30.0, "name": "GPU-1",
                 "active": True, "pinned": True},
            ],
            "context_size": 4096,
            "auto_balance": True,
        },
    )

    assert response.status_code == 200
    sent_request = process_manager.start_auto_balance.call_args.args[0]
    assert all(w.pinned is False for w in sent_request.gpu_weights)


def test_start_smart_calibration_preserves_pinned_flags(monkeypatch, authenticated_client):
    """POST /start com smart_calibration mantém pinned GPU do usuário."""
    process_manager = MagicMock()
    process_manager.start_auto_balance.return_value = {"probing": True}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", MagicMock())

    response = authenticated_client.post(
        "/start",
        json={
            "path": "/media/docker/models/model.gguf",
            "gpu_weights": [
                {"index": 0, "weight": 70.0, "name": "GPU-0",
                 "active": True, "is_main": True, "pinned": True},
                {"index": 1, "weight": 30.0, "name": "GPU-1",
                 "active": True, "pinned": False},
            ],
            "context_size": 4096,
            "auto_balance": True,
            "smart_calibration": True,
            "pinned_fields": {"cache_type": True, "threads": False},
        },
    )

    assert response.status_code == 200
    sent_request = process_manager.start_auto_balance.call_args.args[0]
    assert sent_request.gpu_weights[0].pinned is True
    assert sent_request.gpu_weights[1].pinned is False
    assert sent_request.pinned_fields == {"cache_type": True, "threads": False}


def test_start_persists_gpu_pinned_and_pinned_fields(monkeypatch, authenticated_client):
    """POST /start normal salva pinned em gpu_weights e pinned_fields."""
    process_manager = MagicMock()
    process_manager.start.return_value = {"status": "started", "port": 8085}
    config_manager = MagicMock()
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", config_manager)

    gpu_weights = [
        {"index": 0, "weight": 60.0, "name": "GPU-0",
         "active": True, "is_main": True, "pinned": True},
        {"index": 1, "weight": 40.0, "name": "GPU-1",
         "active": True, "pinned": False},
    ]
    pinned_fields = {"cache_type": True, "batch_size": True}

    response = authenticated_client.post(
        "/start",
        json={
            "path": "/media/docker/models/model.gguf",
            "gpu_weights": gpu_weights,
            "context_size": 4096,
            "pinned_fields": pinned_fields,
            "auto_balance_profile": True,
        },
    )

    assert response.status_code == 200
    saved = config_manager.update_model_settings.call_args.args[1]
    assert saved["gpu_weights"][0]["pinned"] is True
    assert saved["gpu_weights"][1]["pinned"] is False
    assert saved["pinned_fields"] == pinned_fields
    assert saved["auto_balance_profile"] is True


# ── CPU offload integration ───────────────────────────────────────────────


def test_metrics_endpoint_returns_cpu_offload_fields(
    monkeypatch, authenticated_client
):
    """GET /metrics exposes cpu_name and RAM fields used by the CPU offload UI."""
    gpu_detector = MagicMock()
    gpu_detector.get_metrics.return_value = {
        "cpu": 12.5,
        "cpu_name": "AMD Ryzen 9 7950X",
        "ram": 45.0,
        "ram_total_mb": 32768,
        "ram_used_mb": 14745,
        "gpus": [{"index": 0, "utilization": 7, "memory_used": 1024}],
    }
    monkeypatch.setattr(llama_manager, "gpu_manager", gpu_detector)

    response = authenticated_client.get("/metrics")
    payload = response.json()

    assert response.status_code == 200
    assert isinstance(payload["cpu_name"], str) and payload["cpu_name"]
    assert isinstance(payload["ram_total_mb"], int) and payload["ram_total_mb"] > 0
    assert isinstance(payload["ram_used_mb"], int) and payload["ram_used_mb"] >= 0
    assert payload["ram_used_mb"] <= payload["ram_total_mb"]
    gpu_detector.get_metrics.assert_called_once_with()


def test_start_accepts_cpu_weight_in_gpu_weights(
    monkeypatch, authenticated_client
):
    """POST /start accepts device=cpu with index=-1 alongside GPU weights."""
    process_manager = MagicMock()
    process_manager.start.return_value = {"message": "Started", "pid": 1234}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", MagicMock())
    monkeypatch.setattr(
        llama_manager.gpu_manager, "detect_model_layers", lambda _path: 32
    )

    response = authenticated_client.post("/start", json=MIXED_GPU_CPU_PAYLOAD)

    assert response.status_code == 200
    process_manager.start.assert_called_once()
    gpu_weights = process_manager.start.call_args.kwargs["gpu_weights"]
    cpu_entry = next(w for w in gpu_weights if w.device == "cpu")
    assert cpu_entry.index == -1
    assert cpu_entry.weight == 30.0
    assert cpu_entry.active is True


def test_app_gpu_manager_validate_weights_accepts_any_cpu_weight():
    """CPU weight has no cap — only sum validation matters."""
    weights = [
        GPUWeight(index=0, weight=20.0, name="GPU-0", device="gpu"),
        GPUWeight(index=-1, weight=80.0, name="CPU", device="cpu"),
    ]

    ok, msg = llama_manager.gpu_manager.validate_gpu_weights(weights)

    assert ok is True
    assert msg == ""


def test_app_gpu_manager_validate_weights_accepts_cpu_at_70_percent():
    weights = [
        GPUWeight(index=0, weight=30.0, name="GPU-0", device="gpu"),
        GPUWeight(index=-1, weight=70.0, name="CPU", device="cpu"),
    ]

    ok, msg = llama_manager.gpu_manager.validate_gpu_weights(weights)

    assert ok is True
    assert msg == ""


def test_app_gpu_manager_compute_n_gpu_layers_mixed_gpu_cpu():
    """Wired GPUManager computes GPU layer count from GPU weights only."""
    weights = [
        GPUWeight(index=0, weight=70.0, name="GPU-0", device="gpu"),
        GPUWeight(index=-1, weight=30.0, name="CPU", device="cpu"),
    ]

    n_gpu_layers = llama_manager.gpu_manager.compute_n_gpu_layers(
        weights, total_layers=32
    )

    assert n_gpu_layers == 22


@pytest.mark.parametrize(
    ("gpu_weight", "cpu_weight", "total_layers", "expected_ngl"),
    [
        (100.0, 0.0, 32, ALL_GPU_LAYERS),
        (50.0, 50.0, 32, 16),
        (70.0, 30.0, 80, 56),
        (0.0, 100.0, 32, 0),
    ],
)
def test_app_gpu_manager_compute_n_gpu_layers_parametrized(
    gpu_weight, cpu_weight, total_layers, expected_ngl
):
    weights = [
        GPUWeight(index=0, weight=gpu_weight, name="GPU-0", device="gpu"),
        GPUWeight(index=-1, weight=cpu_weight, name="CPU", device="cpu"),
    ]

    assert (
        llama_manager.gpu_manager.compute_n_gpu_layers(weights, total_layers)
        == expected_ngl
    )


def test_start_forwards_total_layers_for_mixed_gpu_cpu_weights(
    monkeypatch, authenticated_client
):
    """POST /start passes total_layers to ProcessManager for dynamic -ngl."""
    process_manager = MagicMock()
    process_manager.start.return_value = {"message": "Started"}
    monkeypatch.setattr(llama_manager, "process_manager", process_manager)
    monkeypatch.setattr(llama_manager, "config_manager", MagicMock())

    response = authenticated_client.post("/start", json=MIXED_GPU_CPU_PAYLOAD)

    assert response.status_code == 200
    assert process_manager.start.call_args.kwargs["total_layers"] == 32


def test_version_check_requires_auth(client):
    response = client.get("/api/system/version-check")
    assert response.status_code == 401


def test_version_check_returns_payload_when_authenticated(monkeypatch, authenticated_client):
    from version_manager import VersionCheckResult, VersionCommit as VmCommit

    fake_result = VersionCheckResult(
        status="ok",
        update_available=True,
        current_ref="abc1234",
        remote_ref="def5678",
        branch="main",
        commits=[
            VmCommit(
                sha="fullsha1",
                message="feat: version alert",
                author="Dev",
                date="2026-06-07T12:00:00-03:00",
            )
        ],
    )
    monkeypatch.setattr(llama_manager, "check_for_updates", lambda _root: fake_result)

    response = authenticated_client.get("/api/system/version-check")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["update_available"] is True
    assert body["current_ref"] == "abc1234"
    assert body["remote_ref"] == "def5678"
    assert body["branch"] == "main"
    assert len(body["commits"]) == 1
    assert body["commits"][0]["message"] == "feat: version alert"


def test_version_check_unavailable_status(monkeypatch, authenticated_client):
    from version_manager import VersionCheckResult

    monkeypatch.setattr(
        llama_manager,
        "check_for_updates",
        lambda _root: VersionCheckResult(status="unavailable"),
    )

    response = authenticated_client.get("/api/system/version-check")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["update_available"] is False


def test_version_check_error_status(monkeypatch, authenticated_client):
    from version_manager import VersionCheckResult

    monkeypatch.setattr(
        llama_manager,
        "check_for_updates",
        lambda _root: VersionCheckResult(
            status="error",
            error_message="git fetch falhou",
        ),
    )

    response = authenticated_client.get("/api/system/version-check")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_message"] == "git fetch falhou"
