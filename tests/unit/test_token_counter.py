"""Tests for exact-on-demand request token budgeting."""

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from token_counter import HybridTokenCounter


def _instance(port=8085, model="model.gguf"):
    return {
        "port": port,
        "model": model,
        "model_path": f"/models/{model}",
        "backend_type": "local",
        "start_time": 123.0,
    }


def _response(payload, status=200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status
    response.raise_for_status.side_effect = (
        None if status < 400 else httpx.HTTPStatusError(
            "failed",
            request=httpx.Request("POST", "http://local"),
            response=httpx.Response(status),
        )
    )
    response.json.return_value = payload
    return response


@pytest.mark.asyncio
async def test_small_request_uses_fast_path_without_http():
    client = MagicMock()
    client.post = AsyncMock()
    counter = HybridTokenCounter(client)
    body = {"messages": [{"role": "user", "content": "hello"}]}

    budget = await counter.count(
        body, [_instance()], {}, exact_required=False
    )

    assert budget.source == "conservative"
    assert budget.required_context > budget.prompt_tokens
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_count_skips_stopped_local_instances():
    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        _response({"prompt": "prompt"}),
        _response({"tokens": [1, 2, 3]}),
    ])
    counter = HybridTokenCounter(client)
    stopped = {**_instance(8086, "stopped.gguf"), "status": "stopped"}

    budget = await counter.count(
        {"messages": [{"role": "user", "content": "hello"}]},
        [_instance(), stopped],
        {},
        exact_required=True,
    )

    assert budget.source == "exact"
    assert budget.exact_backends == 1
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_exact_count_uses_template_and_model_tokenizer():
    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        _response({"prompt": "<user>hello</user>"}),
        _response({"tokens": list(range(123))}),
    ])
    counter = HybridTokenCounter(client)
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 300,
    }

    budget = await counter.count(
        body,
        [_instance()],
        {"Authorization": "Bearer secret", "x-ignore": "value"},
        exact_required=True,
    )

    assert budget.source == "exact"
    assert budget.prompt_tokens == 123
    assert budget.output_tokens == 300
    assert budget.required_context == 123 + 300 + 512 + 256
    assert client.post.await_count == 2
    template_call, tokenize_call = client.post.await_args_list
    assert template_call.args[0].endswith("/apply-template")
    assert tokenize_call.args[0].endswith("/tokenize")
    assert template_call.kwargs["headers"] == {
        "Authorization": "Bearer secret",
        "content-type": "application/json",
    }


@pytest.mark.asyncio
async def test_exact_count_converts_custom_tools_only_in_template_copy():
    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        _response({"prompt": "<tool>ApplyPatch</tool>"}),
        _response({"tokens": list(range(321))}),
    ])
    counter = HybridTokenCounter(client)
    body = {
        "messages": [{"role": "user", "content": "change it"}],
        "tools": [{
            "type": "custom",
            "name": "ApplyPatch",
            "description": "Apply a patch",
            "format": {"type": "grammar", "syntax": "lark"},
        }],
    }

    budget = await counter.count(body, [_instance()], {}, exact_required=True)

    assert budget.source == "exact"
    assert budget.prompt_tokens == 321
    template_call = client.post.await_args_list[0]
    assert template_call.kwargs["json"]["tools"][0]["type"] == "function"
    assert body["tools"][0]["type"] == "custom"


@pytest.mark.asyncio
async def test_exact_count_is_cached_without_storing_tokens_again():
    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        _response({"prompt": "prompt"}),
        _response({"tokens": [1, 2, 3]}),
    ])
    counter = HybridTokenCounter(client)
    body = {"messages": [{"role": "user", "content": "same"}]}

    first = await counter.count(body, [_instance()], {}, exact_required=True)
    second = await counter.count(body, [_instance()], {}, exact_required=True)

    assert first.source == "exact"
    assert second.source == "cached_exact"
    assert second.prompt_tokens == 3
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_any_exact_failure_keeps_safe_upper_bound():
    client = MagicMock()

    async def post(url, **kwargs):
        if "8086" in url:
            raise httpx.ConnectError("offline")
        if url.endswith("/apply-template"):
            return _response({"prompt": "prompt"})
        return _response({"tokens": [1, 2, 3]})

    client.post = AsyncMock(side_effect=post)
    counter = HybridTokenCounter(client)
    body = {"messages": [{"role": "user", "content": "x" * 100}]}

    budget = await counter.count(
        body,
        [_instance(), _instance(8086, "other.gguf")],
        {},
        exact_required=True,
    )

    assert budget.source == "conservative"
    assert budget.prompt_tokens == counter.conservative_prompt_upper_bound(body)
    assert budget.prompt_tokens > 3


def test_output_and_media_reserves_are_part_of_required_context():
    body = {
        "max_completion_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }],
    }

    assert HybridTokenCounter.output_reserve(body) == 4096
    assert HybridTokenCounter.media_reserve(body) == 1024
    assert HybridTokenCounter.conservative_required_context(body) > 4096 + 1024


def test_fast_upper_bound_has_negligible_cost_for_large_payload():
    body = {"messages": [{"role": "user", "content": "abcd" * 100_000}]}

    started = time.perf_counter()
    result = HybridTokenCounter.conservative_required_context(body)
    duration_ms = (time.perf_counter() - started) * 1000

    assert result > 400_000
    assert duration_ms < 100
