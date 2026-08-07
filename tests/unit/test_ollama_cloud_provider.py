"""Unit tests for OllamaCloudProvider — direct HTTP with /v1."""

import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from platform_ollama_cloud import (
    OllamaCloudAccount,
    OllamaCloudHTTPError,
    OllamaCloudProvider,
)


@pytest.fixture
def account() -> OllamaCloudAccount:
    return OllamaCloudAccount(
        id="acc-001",
        api_key="sk-test-key-123",
        label="Test Account",
        status="available",
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_account(self, account):
        provider = OllamaCloudProvider(account)
        assert provider.account is account

    def test_creates_httpx_client(self, account):
        provider = OllamaCloudProvider(account)
        assert provider._client is not None
        assert isinstance(provider._client, httpx.AsyncClient)

    def test_base_url(self, account):
        provider = OllamaCloudProvider(account)
        assert str(provider._client.base_url) == "https://ollama.com/v1/"

    def test_timeout(self, account):
        provider = OllamaCloudProvider(account)
        assert provider._client.timeout.connect == 30.0
        assert provider._client.timeout.read == 30.0
        assert provider._client.timeout.write == 30.0
        assert provider._client.timeout.pool == 30.0

    def test_authorization_header(self, account):
        provider = OllamaCloudProvider(account)
        assert "Authorization" in provider._client.headers
        assert provider._client.headers["Authorization"] == "Bearer sk-test-key-123"


# ---------------------------------------------------------------------------
# chat_completion (success)
# ---------------------------------------------------------------------------

class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_sends_post_with_bearer_token(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "Hello"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            resp = await provider.chat_completion(
                model="gpt-oss-20b",
                messages=[{"role": "user", "content": "Hi"}],
            )

            mock_client.post.assert_called_once_with(
                "/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )
            assert resp is mock_response
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_includes_tools_when_provided(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "chatcmpl-2", "choices": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            tools = [{"type": "function", "function": {"name": "greet"}}]
            await provider.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "say hi"}],
                tools=tools,
            )

            call_json = mock_client.post.call_args[1]["json"]
            assert call_json["tools"] == tools
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_stream_true_sets_stream_in_payload(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "chatcmpl-3", "choices": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            await provider.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )

            call_json = mock_client.post.call_args[1]["json"]
            assert call_json["stream"] is True
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_raises_ollama_cloud_error_on_4xx(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.url = "https://ollama.com/v1/chat/completions"
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            with pytest.raises(OllamaCloudHTTPError) as exc_info:
                await provider.chat_completion(
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
            assert exc_info.value.status_code == 400
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_raises_ollama_cloud_error_on_5xx(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.url = "https://ollama.com/v1/chat/completions"
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            with pytest.raises(OllamaCloudHTTPError) as exc_info:
                await provider.chat_completion(
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
            assert exc_info.value.status_code == 500
            await mock_client.aclose()


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_model_list(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1", "display_name": "Model 1"},
                {"id": "model-2", "display_name": "Model 2"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            models = await provider.list_models()

            assert len(models) == 2
            assert models[0]["id"] == "model-1"
            assert models[1]["id"] == "model-2"
            mock_client.get.assert_called_once_with("/models")
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_models(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            models = await provider.list_models()
            assert models == []
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_raises_on_error_response(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.url = "https://ollama.com/v1/models"
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            with pytest.raises(OllamaCloudHTTPError):
                await provider.list_models()
            await mock_client.aclose()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            result = await provider.health_check()
            assert result is True
            mock_client.get.assert_called_once_with("/models")
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, account):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("connection failed")
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            result = await provider.health_check()
            assert result is False
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self, account):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadTimeout("timeout")
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            result = await provider.health_check()
            assert result is False
            await mock_client.aclose()

    @pytest.mark.asyncio
    async def test_returns_true_on_2xx(self, account):
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            result = await provider.health_check()
            assert result is True
            await mock_client.aclose()


# ---------------------------------------------------------------------------
# parse_error
# ---------------------------------------------------------------------------

class TestParseError:
    def _make_response(
        self,
        status_code: int = 429,
        headers: Dict[str, str] | None = None,
        json_body: Dict[str, Any] | None = None,
        text: str = "",
        url: str = "https://ollama.com/v1/chat/completions",
    ) -> httpx.Response:
        """Helper to build a fake httpx.Response."""
        request = httpx.Request("POST", url)
        if json_body is not None and not text:
            import json as _json
            text = _json.dumps(json_body)
            json_body = None  # prevent httpx from receiving both
        return httpx.Response(
            status_code=status_code,
            headers=headers or {},
            json=json_body,
            text=text,
            request=request,
        )

    def test_retry_after_header(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(
            status_code=429,
            headers={"retry-after": "120.5"},
        )
        retry_after, msg = provider.parse_error(resp)
        assert retry_after == 120.5
        assert "429" in msg

    def test_rate_limit_reset_header(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        future_ts = time.time() + 45.0
        resp = self._make_response(
            status_code=429,
            headers={"ratelimit-reset": str(future_ts)},
        )
        retry_after, _ = provider.parse_error(resp)
        assert retry_after is not None
        assert 40 < retry_after < 50  # approximate 45s

    def test_retry_after_takes_precedence(self):
        """Retry-After header should be used before RateLimit-Reset."""
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        future_ts = time.time() + 100.0
        resp = self._make_response(
            status_code=429,
            headers={
                "retry-after": "10.0",
                "ratelimit-reset": str(future_ts),
            },
        )
        retry_after, _ = provider.parse_error(resp)
        assert retry_after == 10.0

    def test_no_retry_headers_returns_none(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(status_code=500)
        retry_after, msg = provider.parse_error(resp)
        assert retry_after is None
        assert "500" in msg

    def test_error_message_with_json_detail(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(
            status_code=429,
            json_body={
                "error": {"message": "Rate limit exceeded. Please retry after 120 seconds."}
            },
        )
        _, msg = provider.parse_error(resp)
        assert "Rate limit exceeded" in msg

    def test_error_message_with_top_level_message(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(
            status_code=500,
            json_body={"message": "Internal server error occurred"},
        )
        _, msg = provider.parse_error(resp)
        assert "Internal server error occurred" in msg

    def test_error_message_when_json_decode_fails(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        request = httpx.Request("POST", "https://ollama.com/v1/chat/completions")
        resp = httpx.Response(
            status_code=502,
            headers={},
            text="Bad Gateway",
            request=request,
        )
        retry_after, msg = provider.parse_error(resp)
        assert retry_after is None
        assert "502" in msg

    def test_invalid_retry_after_header_treated_as_none(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(
            status_code=429,
            headers={"retry-after": "not_a_number"},
        )
        retry_after, _ = provider.parse_error(resp)
        assert retry_after is None

    def test_invalid_rate_limit_reset_header_treated_as_none(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(
            status_code=429,
            headers={"ratelimit-reset": "not_a_timestamp"},
        )
        retry_after, _ = provider.parse_error(resp)
        assert retry_after is None

    def test_rate_limit_reset_in_past_returns_zero_min(self):
        """When RateLimit-Reset is in the past, retry_after should be near 0."""
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        past_ts = time.time() - 10.0
        resp = self._make_response(
            status_code=429,
            headers={"ratelimit-reset": str(past_ts)},
        )
        retry_after, _ = provider.parse_error(resp)
        assert retry_after is not None
        assert retry_after >= 0  # should be small positive or near zero

    def test_4xx_error_in_message(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(status_code=400)
        _, msg = provider.parse_error(resp)
        assert "400" in msg

    def test_5xx_error_in_message(self):
        provider = OllamaCloudProvider(
            OllamaCloudAccount(id="x", api_key="y")
        )
        resp = self._make_response(status_code=503)
        _, msg = provider.parse_error(resp)
        assert "503" in msg


# ---------------------------------------------------------------------------
# OllamaCloudHTTPError
# ---------------------------------------------------------------------------

class TestOllamaCloudHTTPError:
    def test_constructor_sets_fields(self):
        err = OllamaCloudHTTPError(429, (120.5, "Rate limit exceeded"))
        assert err.status_code == 429
        assert err.retry_after == 120.5
        assert err.message == "Rate limit exceeded"

    def test_str_representation(self):
        err = OllamaCloudHTTPError(500, (None, "Server error"))
        assert "500" in str(err)
        assert "Server error" in str(err)


# ---------------------------------------------------------------------------
# close / __del__
# ---------------------------------------------------------------------------

class TestClose:
    @pytest.mark.asyncio
    async def test_closes_client(self, account):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            provider = OllamaCloudProvider(account)
            await provider.close()

            mock_client.aclose.assert_called_once()
