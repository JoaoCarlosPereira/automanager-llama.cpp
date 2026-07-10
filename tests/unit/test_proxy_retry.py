import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import llama_manager
from llama_manager import app, auth_manager
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
def override_auth(monkeypatch):
    app.dependency_overrides[llama_manager.require_api_token] = lambda: True
    app.dependency_overrides[auth_manager.check_auth] = lambda: True
    monkeypatch.setattr(
        llama_manager.config_manager,
        "get_smart_proxy_settings",
        lambda: {"enabled": False},
    )
    yield
    app.dependency_overrides.clear()


def _models_resp():
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "object": "list",
        "data": [{"id": "gemini-3.1-pro-low", "object": "model", "owned_by": "antigravity"}],
    }
    return resp


def _chat_resp(model: str = "gemini-3.1-pro-low"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.headers = httpx.Headers({"Content-Type": "application/json"})
    resp.content = json.dumps({"id": "x", "model": model}).encode()
    return resp


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.post", new_callable=AsyncMock)
def test_platform_chat_retries_transient_502(mock_post, mock_get, mock_status):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    mock_get.return_value = _models_resp()
    err = MagicMock(spec=httpx.Response)
    err.status_code = 502
    err.headers = httpx.Headers({})
    err.content = b'{"error":"busy"}'
    mock_post.side_effect = [err, err, _chat_resp()]

    response = client.post(
        "/v1/chat/completions",
        json={"model": GEMINI_LISTING, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert mock_post.call_count == 3
    assert response.json()["model"] == "gemini-3.1-pro-low"


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.post", new_callable=AsyncMock)
def test_platform_chat_returns_502_after_retries_exhausted(mock_post, mock_get, mock_status):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    mock_get.return_value = _models_resp()
    err = MagicMock(spec=httpx.Response)
    err.status_code = 502
    err.headers = httpx.Headers({})
    err.content = b'{"error":"busy"}'
    mock_post.return_value = err

    response = client.post(
        "/v1/chat/completions",
        json={"model": GEMINI_LISTING, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    assert mock_post.call_count == llama_manager._PROXY_MAX_ATTEMPTS


@patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.send", new_callable=AsyncMock)
def test_platform_stream_buffers_and_retries_incomplete_body(
    mock_send, mock_get, mock_status, mock_sleep
):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    mock_get.return_value = _models_resp()

    incomplete = MagicMock(spec=httpx.Response)
    incomplete.status_code = 200
    incomplete.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_incomplete():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    incomplete.aiter_bytes = aiter_incomplete
    incomplete.aclose = AsyncMock()

    complete = MagicMock(spec=httpx.Response)
    complete.status_code = 200
    complete.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_complete():
        yield b'data: {"model":"x","choices":[]}\n\ndata: [DONE]\n\n'

    complete.aiter_bytes = aiter_complete
    complete.aclose = AsyncMock()
    mock_send.side_effect = [incomplete, complete]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": GEMINI_LISTING,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert mock_send.call_count == 2
    assert mock_sleep.await_count == 1
    assert b"data: [DONE]" in response.content


@patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.send", new_callable=AsyncMock)
def test_platform_stream_retries_http_502_before_client(mock_send, mock_get, mock_status, mock_sleep):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    mock_get.return_value = _models_resp()

    err = MagicMock(spec=httpx.Response)
    err.status_code = 502
    err.headers = httpx.Headers({})
    err.aread = AsyncMock(return_value=b"")
    err.aclose = AsyncMock()

    ok = MagicMock(spec=httpx.Response)
    ok.status_code = 200
    ok.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_bytes():
        yield b'data: {"model":"x"}\n\ndata: [DONE]\n\n'

    ok.aiter_bytes = aiter_bytes
    ok.aclose = AsyncMock()
    mock_send.side_effect = [err, ok]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": GEMINI_LISTING,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert mock_send.call_count == 2
    assert mock_sleep.await_count == 1


@patch("llama_manager._hybrid_status")
@patch("llama_manager.client.get", new_callable=AsyncMock)
@patch("llama_manager.client.send", new_callable=AsyncMock)
def test_platform_stream_replays_raw_chunks_unchanged(mock_send, mock_get, mock_status):
    mock_status.return_value = {"instances": [PLATFORM_INST]}
    mock_get.return_value = _models_resp()

    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_bytes():
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    upstream.aiter_bytes = aiter_bytes
    upstream.aclose = AsyncMock()
    mock_send.return_value = upstream

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": GEMINI_LISTING,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert response.content == (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
