import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import llama_manager
from config_manager import ConfigManager
from llama_manager import app, auth_manager, config_manager
from platform_manager import platform_model_listing_id

client = TestClient(app)

PLATFORM_INST = {
    "port": 9100,
    "backend_type": "platform",
    "model": "Codex",
    "provider": "antigravity",
}
GEMINI_LISTING = platform_model_listing_id("gemini-3.1-pro-low", "antigravity")


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[llama_manager.require_api_token] = lambda: True
    app.dependency_overrides[auth_manager.check_auth] = lambda: True
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def alias_cfg(tmp_path, monkeypatch):
    cfg = ConfigManager(config_path=str(tmp_path / "cfg.json"))
    monkeypatch.setattr(llama_manager, "config_manager", cfg)
    cfg.set_model_alias("gpt-4o", "gemini-3.1-pro-low")
    return cfg


def test_config_resolve_model_alias(alias_cfg):
    assert alias_cfg.resolve_model_alias("gpt-4o") == "gemini-3.1-pro-low"
    assert alias_cfg.resolve_model_alias("other") == "other"


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_v1_models_includes_aliases(mock_get, mock_status, alias_cfg):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    mock_get.return_value = resp

    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert GEMINI_LISTING in ids
    assert "gpt-4o" in ids
    alias_entry = next(m for m in response.json()["data"] if m["id"] == "gpt-4o")
    assert alias_entry["owned_by"] == "llamacpp"
    assert "meta" in alias_entry


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_v1_models_platform_entries_use_opaque_cursor_slug(mock_get, mock_status, alias_cfg):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    mock_get.return_value = resp

    response = client.get("/v1/models")
    assert response.status_code == 200
    entry = next(m for m in response.json()["data"] if m["id"] == GEMINI_LISTING)
    assert entry["owned_by"] == "llamacpp"
    assert entry["meta"]["root_model"] == "gemini-3.1-pro-low"
    assert "n_ctx" in entry["meta"]


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.post", new_callable=AsyncMock)
def test_chat_resolves_opaque_platform_listing(mock_post, mock_get, mock_status, alias_cfg):
    mock_status.return_value = {
        "instances": [
            {"port": 8085, "backend_type": "local", "model": "Qwen3.6-35B.gguf", "model_path": "/m/Qwen.gguf"},
            PLATFORM_INST,
        ],
    }
    models_resp = MagicMock(spec=httpx.Response)
    models_resp.raise_for_status.return_value = None
    models_resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    mock_get.return_value = models_resp
    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"Content-Type": "application/json"})
    upstream.content = json.dumps({"id": "x", "model": "gemini-3.1-pro-low"}).encode()
    mock_post.return_value = upstream

    response = client.post(
        "/v1/chat/completions",
        json={"model": GEMINI_LISTING, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    sent = json.loads(mock_post.call_args.kwargs["content"])
    assert sent["model"] == "gemini-3.1-pro-low"
    assert response.json()["model"] == GEMINI_LISTING


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.post", new_callable=AsyncMock)
def test_chat_resolves_alias_before_forward(mock_post, mock_get, mock_status, alias_cfg):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    models_resp = MagicMock(spec=httpx.Response)
    models_resp.raise_for_status.return_value = None
    models_resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    mock_get.return_value = models_resp
    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"Content-Type": "application/json"})
    upstream.content = json.dumps({"id": "x", "model": "gemini-3.1-pro-low"}).encode()
    mock_post.return_value = upstream

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    sent = json.loads(mock_post.call_args.kwargs["content"])
    assert sent["model"] == "gemini-3.1-pro-low"
    assert response.json()["model"] == "gpt-4o"


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_v1_models_by_id_returns_listing_entry(mock_get, mock_status, alias_cfg):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    mock_get.return_value = resp

    response = client.get(f"/v1/models/{GEMINI_LISTING}")
    assert response.status_code == 200
    assert response.json()["id"] == GEMINI_LISTING
    assert response.json()["owned_by"] == "llamacpp"
