"""Tests for llama-server Web UI reverse proxy (/ui/{port}/)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import llama_manager
from llama_manager import (
    _filter_proxy_headers,
    _inject_ui_base_tag,
    app,
)


@pytest.fixture
def test_client():
    return TestClient(app)


def test_filter_proxy_headers_strips_content_encoding():
    headers = {
        "content-type": "application/javascript",
        "content-encoding": "gzip",
        "content-length": "12345",
        "cache-control": "public",
    }
    filtered = _filter_proxy_headers(headers)
    assert filtered == {
        "content-type": "application/javascript",
        "cache-control": "public",
    }


def test_inject_ui_base_tag_inserts_base_href():
    html = "<!doctype html><html><head><title>Chat</title></head><body></body></html>"
    out = _inject_ui_base_tag(html, 8085)
    assert '<base href="/ui/8085/">' in out
    assert out.index("<base href=") < out.index("<title>")


def test_inject_ui_base_tag_rewrites_legacy_base_literal():
    html = "<head></head><script>base: new URL('.', location).pathname.slice(0, -1)</script>"
    out = _inject_ui_base_tag(html, 8090)
    assert 'base: "/ui/8090"' in out


@patch("llama_manager.client.request", new_callable=AsyncMock)
def test_ui_proxy_asset_omits_content_encoding(mock_request, test_client):
    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = 200
    upstream.headers = httpx.Headers(
        {
            "content-type": "application/javascript",
            "content-encoding": "gzip",
            "content-length": "999",
        }
    )

    async def _iter():
        yield b"var chat = true;"

    upstream.aiter_bytes = MagicMock(return_value=_iter())
    mock_request.return_value = upstream

    resp = test_client.get("/ui/8085/_app/immutable/bundle.js")

    assert resp.status_code == 200
    assert resp.content == b"var chat = true;"
    assert resp.headers.get("content-encoding") is None
    assert resp.headers.get("content-length") is None


@patch("llama_manager.client.get", new_callable=AsyncMock)
def test_ui_proxy_index_injects_base_tag(mock_get, test_client):
    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = 200
    upstream.text = "<!doctype html><html lang=\"en\">\n\t<head>\n\t\t<title>UI</title>\n\t</head><body></body></html>"
    mock_get.return_value = upstream

    resp = test_client.get("/ui/8085/")

    assert resp.status_code == 200
    assert '<base href="/ui/8085/">' in resp.text
