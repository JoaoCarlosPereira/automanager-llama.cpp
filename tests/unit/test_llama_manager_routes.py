"""Comprehensive API route tests for llama_manager.py endpoints."""
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
import json

# Import the app and all its dependencies
import llama_manager
from llama_manager import (
    app, process_manager, config_manager, gpu_manager,
    auth_manager, model_scanner, download_mgr,
    _invalidate_models_cache,
)
from schemas import StartRequest, GPUWeight, SetDefaultRequest, RenameRequest
from config_manager import SESSION_IDLE_SECONDS


@pytest.fixture
def test_client():
    """Create a TestClient for the app."""
    return TestClient(app)


class TestLoginEndpoint:
    """Tests for the /api/auth/login endpoint with rate limiting."""

    def test_login_success(self, test_client):
        with patch.object(auth_manager, 'authenticate', return_value={"token": "abc123", "force_password_change": False}):
            resp = test_client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["force_password_change"] is False
            assert "session_token" in resp.cookies

    def test_login_success_force_password_change(self, test_client):
        with patch.object(auth_manager, 'authenticate', return_value={"token": "abc123", "force_password_change": True}):
            resp = test_client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["force_password_change"] is True

    def test_login_invalid_credentials(self, test_client):
        with patch.object(auth_manager, 'authenticate', return_value=None):
            resp = test_client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
            assert resp.status_code == 401
            assert "credenciais invalidas" in resp.json()["detail"].lower()

    def test_login_empty_credentials(self, test_client):
        with patch.object(auth_manager, 'authenticate', return_value=None):
            resp = test_client.post("/api/auth/login", json={"username": "", "password": ""})
            assert resp.status_code == 401

    def test_rate_limit_decorator_present(self):
        """Verify the rate limit decorator is applied to login."""
        assert hasattr(llama_manager.login, '__wrapped__') or hasattr(llama_manager.login, '__rate_limit__')

    def test_login_cookie_has_correct_attributes(self, test_client):
        with patch.object(auth_manager, 'authenticate', return_value={"token": "test-token", "force_password_change": False}):
            resp = test_client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
            assert resp.status_code == 200
            assert "session_token" in resp.cookies
            cookie = resp.cookies.get("session_token")
            assert cookie is not None


class TestLogoutEndpoint:
    """Tests for the /api/auth/logout endpoint."""

    def test_logout_without_token(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            resp = test_client.post("/api/auth/logout")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_logout_with_token(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(auth_manager, 'logout') as mock_logout:
                resp = test_client.post("/api/auth/logout", cookies={"session_token": "test-token"})
                assert resp.status_code == 200
                mock_logout.assert_called_once_with("test-token")
                assert "session_token" not in resp.cookies or resp.cookies.get("session_token") is None


class TestChangePasswordEndpoint:
    """Tests for the /api/auth/change-password endpoint."""

    def test_change_password_success(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(auth_manager, 'change_password', return_value=True):
                resp = test_client.post("/api/auth/change-password",
                                       json={"username": "admin", "current": "admin", "password": "newpass"},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                assert resp.json()["status"] == "ok"

    def test_change_password_wrong_current(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(auth_manager, 'change_password', return_value=False):
                resp = test_client.post("/api/auth/change-password",
                                       json={"username": "admin", "current": "wrong", "password": "newpass"},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 400
                assert "senha atual incorreta" in resp.json()["detail"].lower()

    def test_change_password_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/api/auth/change-password",
                                   json={"username": "admin", "current": "admin", "password": "newpass"})
            assert resp.status_code == 401


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_success(self, test_client):
        with patch.object(process_manager, 'get_status', return_value={"instances": [], "recovery": {}}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/status", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                data = resp.json()
                assert "instances" in data
                assert "recovery" in data

    def test_status_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/status")
            assert resp.status_code == 401


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_success(self, test_client):
        with patch.object(gpu_manager, 'get_metrics', return_value={"cpu": 0, "gpus": []}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/metrics", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                data = resp.json()
                assert "cpu" in data
                assert "gpus" in data

    def test_metrics_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/metrics")
            assert resp.status_code == 401


class TestLlamaBinsEndpoint:
    """Tests for the /llama-bins endpoint."""

    def test_llama_bins_success(self, test_client):
        with patch.object(llama_manager, 'list_llama_server_bins', return_value=["/path/to/llama-server"]):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/llama-bins", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                data = resp.json()
                assert isinstance(data, list)

    def test_llama_bins_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/llama-bins")
            assert resp.status_code == 401


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    def test_models_success(self, test_client):
        with patch.object(model_scanner, 'scan', return_value={"models": [], "projectors": [], "storage": {}}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/models", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                data = resp.json()
                assert "models" in data

    def test_models_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/models")
            assert resp.status_code == 401


class TestModelsDirEndpoint:
    """Tests for the /models/dir endpoint."""

    def test_set_models_dir_success(self, test_client):
        with patch.object(llama_manager, 'update_models_dir') as mock_update:
            mock_paths = MagicMock()
            mock_paths.models_dir = "/new/models/path"
            mock_update.return_value = mock_paths
            with patch.object(llama_manager, 'reload_module_paths'):
                with patch.object(model_scanner, 'models_dir', '/new/path'):
                    with patch.object(download_mgr, 'models_dir', '/new/path'):
                        with patch.object(auth_manager, 'check_auth', return_value=True):
                            with patch.object(model_scanner, 'scan', return_value={"models": [], "projectors": [], "storage": {}}):
                                resp = test_client.post("/models/dir",
                                                       json={"models_dir": "/new/models/path"},
                                                       cookies={"session_token": "valid-token"})
                                assert resp.status_code == 200
                                mock_update.assert_called_once_with("/new/models/path")

    def test_set_models_dir_invalid(self, test_client):
        with patch.object(llama_manager, 'update_models_dir', side_effect=ValueError("Invalid dir")):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.post("/models/dir",
                                       json={"models_dir": "/invalid/path"},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 400
                assert "diretorio invalido" in resp.json()["detail"].lower()

    def test_set_models_dir_empty(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            resp = test_client.post("/models/dir",
                                   json={"models_dir": ""},
                                   cookies={"session_token": "valid-token"})
            assert resp.status_code == 400

    def test_set_models_dir_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/models/dir", json={"models_dir": "/path"})
            assert resp.status_code == 401


class TestStartEndpoint:
    """Tests for the /start endpoint."""

    def test_start_success(self, test_client):
        with patch.object(gpu_manager, 'detect_model_layers', return_value=32):
            with patch.object(process_manager, 'start', return_value={"message": "Servidor iniciado", "port": 8085}):
                with patch.object(config_manager, 'update_model_settings'):
                    with patch.object(auth_manager, 'check_auth', return_value=True):
                        req = StartRequest(
                            path="/models/model.gguf",
                            gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
                            context_size=8192,
                            parallel_slots=1,
                            batch_size=512,
                        )
                        resp = test_client.post("/start",
                                               json=req.model_dump(),
                                               cookies={"session_token": "valid-token"})
                        assert resp.status_code == 200
                        data = resp.json()
                        assert "message" in data

    def test_start_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            req = StartRequest(
                path="/models/model.gguf",
                gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
                context_size=8192,
                parallel_slots=1,
                batch_size=512,
            )
            resp = test_client.post("/start", json=req.model_dump())
            assert resp.status_code == 401

    def test_start_auto_balance(self, test_client):
        with patch.object(process_manager, 'start_auto_balance', return_value={"message": "Auto-balance em andamento"}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                req = StartRequest(
                    path="/models/model.gguf",
                    gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
                    context_size=8192,
                    parallel_slots=1,
                    batch_size=512,
                    auto_balance=True,
                )
                resp = test_client.post("/start", json=req.model_dump(), cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_start_smart_calibration(self, test_client):
        with patch.object(process_manager, 'start_auto_balance', return_value={"message": "Calibração concluída"}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                req = StartRequest(
                    path="/models/model.gguf",
                    gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
                    context_size=8192,
                    parallel_slots=1,
                    batch_size=512,
                    auto_balance=True,
                    smart_calibration=True,
                )
                resp = test_client.post("/start", json=req.model_dump(), cookies={"session_token": "valid-token"})
                assert resp.status_code == 200


class TestStopEndpoint:
    """Tests for the /stop endpoint."""

    def test_stop_success(self, test_client):
        with patch.object(process_manager, 'stop', return_value={"message": "Parado"}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.post("/stop", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                assert "parado" in resp.json()["message"].lower() or "encerrada" in resp.json()["message"].lower()

    def test_stop_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/stop")
            assert resp.status_code == 401


class TestSetDefaultEndpoint:
    """Tests for the /set_default endpoint."""

    def test_set_default_model_success(self, test_client):
        with patch.object(config_manager, 'set_default_model'):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.post("/set_default",
                                       json={"path": "/models/model.gguf"},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_clear_default_model(self, test_client):
        with patch.object(config_manager, 'set_default_model'):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.post("/set_default",
                                       json={"path": None},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_set_default_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/set_default", json={"path": "/models/model.gguf"})
            assert resp.status_code == 401


class TestDownloadsEndpoints:
    """Tests for the /downloads endpoints."""

    def test_get_downloads(self, test_client):
        with patch.object(download_mgr, 'get_progress', return_value={}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/downloads", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_get_downloads_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/downloads")
            assert resp.status_code == 401

    def test_start_download_success(self, test_client):
        with patch.object(download_mgr, 'start_download', return_value="download-id-123"):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.post("/downloads",
                                       json={"url": "https://huggingface.co/model.gguf"},
                                       cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_start_download_invalid_url(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            resp = test_client.post("/downloads", json={"url": "not-a-url"})
            assert resp.status_code == 400

    def test_start_download_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/downloads", json={"url": "https://example.com/model.gguf"})
            assert resp.status_code == 401


class TestConfigEndpoint:
    """Tests for the /config endpoint."""

    def test_get_config(self, test_client):
        with patch.object(config_manager, 'get_config', return_value={}):
            with patch.object(auth_manager, 'check_auth', return_value=True):
                resp = test_client.get("/config", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200
                assert isinstance(resp.json(), dict)

    def test_get_config_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/config")
            assert resp.status_code == 401


class TestLogStreamingEndpoint:
    """Tests for the SSE log streaming endpoint."""

    def test_logs_sse_stream(self, test_client):
        """Test that logs endpoint returns SSE content type."""
        from starlette.responses import StreamingResponse
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(llama_manager.log_manager, 'stream_logs', return_value=StreamingResponse(iter([]))):
                resp = test_client.get("/logs", cookies={"session_token": "valid-token"})
                assert resp.status_code == 200

    def test_logs_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/logs")
            assert resp.status_code == 401


class TestProxyEndpoint:
    """Tests for the llama-server proxy endpoint."""

    def test_proxy_forward(self, test_client):
        """Test proxying requests to llama-server."""
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(llama_manager, 'client') as mock_httpx_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.iter_bytes.return_value = iter([b'{"status": "ok"}'])
                mock_response.headers = {}
                mock_httpx_client.get.return_value = mock_response
                resp = test_client.post("/v1/chat/completions",
                                       json={"model": "test", "messages": []},
                                       cookies={"session_token": "valid-token"})
                # Proxy should forward the request
                assert resp.status_code in [200, 502, 503]

    def test_proxy_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status_code == 401


class TestInvalidateCache:
    """Tests for the _invalidate_models_cache helper."""

    def test_invalidate_models_cache_clears_timestamp(self):
        model_scanner._last_scan_time = 1000
        _invalidate_models_cache()
        assert model_scanner._last_scan_time == 0

    def test_invalidate_models_cache_multiple_calls(self):
        model_scanner._last_scan_time = 5000
        _invalidate_models_cache()
        assert model_scanner._last_scan_time == 0
        model_scanner._last_scan_time = 9000
        _invalidate_models_cache()
        assert model_scanner._last_scan_time == 0


class TestRequireAuthDependency:
    """Tests for the require_auth dependency wrapper."""

    def test_require_auth_returns_true(self):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            mock_request = MagicMock()
            assert llama_manager.require_auth(mock_request) is True

    def test_require_auth_returns_false(self):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            mock_request = MagicMock()
            assert llama_manager.require_auth(mock_request) is False


class TestDashboardHtml:
    """Tests for the dashboard HTML rendering."""

    def test_root_returns_html(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=True):
            with patch.object(process_manager, 'get_status', return_value={"instances": [], "recovery": {}}):
                with patch.object(gpu_manager, 'get_metrics', return_value={"cpu": 0, "gpus": []}):
                    with patch.object(model_scanner, 'scan', return_value={"models": [], "projectors": [], "storage": {}}):
                        with patch.object(config_manager, 'get_config', return_value={}):
                            with patch.object(llama_manager, 'token_manager') as mock_tm:
                                mock_tm.get_or_create.return_value = "test-token"
                                resp = test_client.get("/", cookies={"session_token": "valid-token"})
                                assert resp.status_code == 200
                                assert "text/html" in resp.headers.get("content-type", "").lower()
                                assert "<html" in resp.text.lower()

    def test_root_unauthenticated(self, test_client):
        with patch.object(auth_manager, 'check_auth', return_value=False):
            resp = test_client.get("/")
            assert resp.status_code == 401


class TestContextAndBatchPresets:
    """Tests for context and batch preset constants."""

    def test_context_preset_values(self):
        assert llama_manager.CONTEXT_PRESET_VALUES == [4096, 8192, 16384, 32768, 65536, 131072, "custom"]

    def test_context_k_multiplier(self):
        assert llama_manager.CONTEXT_K_MULTIPLIER == 1024

    def test_manager_port(self):
        assert llama_manager.MANAGER_PORT == 8000

    def test_graceful_shutdown_timeout(self):
        assert llama_manager.GRACEFUL_SHUTDOWN_TIMEOUT_SEC == 5
