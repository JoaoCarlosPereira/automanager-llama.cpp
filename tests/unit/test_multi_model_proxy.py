import pytest
import json
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from llama_manager import app, process_manager, auth_manager

client = TestClient(app)

@pytest.fixture
def mock_instances():
    return [
        {
            "port": 8085,
            "model": "model_a.gguf",
            "model_path": "/path/to/model_a.gguf"
        },
        {
            "port": 8086,
            "model": "model_b.gguf",
            "model_path": "/path/to/model_b.gguf"
        }
    ]

async def mock_aiter(items):
    for item in items:
        yield item

@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[auth_manager.check_auth] = lambda: True
    yield
    app.dependency_overrides.clear()

@patch("llama_manager.process_manager.get_status")
@patch("llama_manager.client.request")
@patch("llama_manager.client.post")
@patch("llama_manager.client.stream")
def test_openai_proxy_routing(mock_stream, mock_post, mock_request, mock_get_status, mock_instances):
    # Setup mocks
    mock_get_status.return_value = {"instances": mock_instances}
    
    # Mock successful response
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = httpx.Headers({"Content-Type": "application/json"})
    mock_resp.aiter_bytes = MagicMock(return_value=mock_aiter([b'{"id": "chatcmpl-123"}']))
    mock_post.return_value = mock_resp

    # 1. Test routing to default (8085) when no model is specified
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl-123"}
    # Check that it called port 8085
    args, kwargs = mock_post.call_args
    assert "8085" in args[0]

    # 2. Test routing to specific model (8086)
    mock_post.reset_mock()
    response = client.post("/v1/chat/completions", json={"model": "model_b.gguf", "messages": []})
    assert response.status_code == 200
    args, kwargs = mock_post.call_args
    assert "8086" in args[0]

    # 3. Test model not found
    response = client.post("/v1/chat/completions", json={"model": "non_existent.gguf"})
    assert response.status_code == 404
    assert "nao esta carregado" in response.json()["detail"].lower()

def _models_resp(model_id):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"object": "list", "data": [{"id": model_id, "object": "model"}]}
    return resp

@patch("llama_manager.process_manager.get_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_openai_proxy_models_aggregates_all_instances(mock_get, mock_get_status, mock_instances):
    mock_get_status.return_value = {"instances": mock_instances}

    async def fake_get(url, **kwargs):
        if "8085" in url:
            return _models_resp("/path/to/model_a.gguf")
        return _models_resp("/path/to/model_b.gguf")

    mock_get.side_effect = fake_get

    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = [m["id"] for m in payload["data"]]
    assert sorted(ids) == ["/path/to/model_a.gguf", "/path/to/model_b.gguf"]

@patch("llama_manager.process_manager.get_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_openai_proxy_models_skips_unreachable_instance(mock_get, mock_get_status, mock_instances):
    mock_get_status.return_value = {"instances": mock_instances}

    async def fake_get(url, **kwargs):
        if "8085" in url:
            raise httpx.ConnectError("connection refused")
        return _models_resp("/path/to/model_b.gguf")

    mock_get.side_effect = fake_get

    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert ids == ["/path/to/model_b.gguf"]

@patch("llama_manager.process_manager.get_status")
def test_openai_proxy_no_instances(mock_get_status):
    mock_get_status.return_value = {"instances": []}
    response = client.post("/v1/chat/completions", json={"model": "any"})
    assert response.status_code == 503
    assert "No model loaded" in response.json()["detail"] or "Nenhum modelo carregado" in response.json()["detail"]
