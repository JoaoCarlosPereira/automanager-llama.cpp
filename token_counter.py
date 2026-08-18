"""Fast, conservative token budgeting with exact llama-server fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import httpx

from context_optimizer import count_media_inputs
from request_normalizer import normalize_custom_tools_for_local


DEFAULT_OUTPUT_RESERVE = 2048
PROTOCOL_OVERHEAD = 512
SAFETY_MARGIN = 256
CONSERVATIVE_TEMPLATE_ALLOWANCE = 1024
IMAGE_TOKEN_RESERVE = 1024


@dataclass(frozen=True)
class RequestTokenBudget:
    prompt_tokens: int
    output_tokens: int
    overhead_tokens: int
    media_tokens: int
    required_context: int
    source: str
    duration_ms: float
    exact_backends: int = 0


class HybridTokenCounter:
    """Counts once per request and only asks llama-server near a limit.

    The cheap upper bound is deliberately based on UTF-8 bytes, not chars/4:
    byte-fallback tokenizers cannot produce more ordinary text tokens than
    input bytes.  Template/protocol/media allowances cover content introduced
    outside the serialized request.  Near a context boundary the loaded
    model's own chat template and vocabulary provide the exact text count.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        cache_size: int = 256,
        exact_timeout_seconds: float = 5.0,
    ) -> None:
        self._client = client
        self._cache_size = max(1, int(cache_size))
        self._exact_timeout = max(0.1, float(exact_timeout_seconds))
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    @staticmethod
    def _serialized_payload(body: Dict[str, Any]) -> bytes:
        try:
            return json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return str(body).encode("utf-8", errors="replace")

    @classmethod
    def conservative_prompt_upper_bound(cls, body: Dict[str, Any]) -> int:
        messages = body.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        return (
            len(cls._serialized_payload(body))
            + CONSERVATIVE_TEMPLATE_ALLOWANCE
            + message_count * 32
        )

    @staticmethod
    def output_reserve(body: Dict[str, Any]) -> int:
        for field in ("max_completion_tokens", "max_tokens"):
            value = body.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return DEFAULT_OUTPUT_RESERVE

    @staticmethod
    def media_reserve(body: Dict[str, Any]) -> int:
        count = count_media_inputs(body.get("messages"))
        count += count_media_inputs(body.get("input"))
        return count * IMAGE_TOKEN_RESERVE

    @classmethod
    def conservative_required_context(cls, body: Dict[str, Any]) -> int:
        return (
            cls.conservative_prompt_upper_bound(body)
            + cls.output_reserve(body)
            + cls.media_reserve(body)
            + PROTOCOL_OVERHEAD
            + SAFETY_MARGIN
        )

    @staticmethod
    def _local_instances(instances: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [
            instance
            for instance in instances
            if instance.get("backend_type", "local") != "platform"
            and instance.get("status", "running") == "running"
            and isinstance(instance.get("port"), int)
        ]

    @classmethod
    def _cache_key(cls, body: Dict[str, Any], instance: Dict[str, Any]) -> str:
        identity = {
            "backend_id": instance.get("backend_id"),
            "model": instance.get("model"),
            "model_path": instance.get("model_path"),
            "port": instance.get("port"),
            "start_time": instance.get("start_time"),
        }
        digest = hashlib.sha256()
        digest.update(cls._serialized_payload(body))
        digest.update(b"\x1f")
        digest.update(cls._serialized_payload(identity))
        return digest.hexdigest()

    async def _cache_get(self, key: str) -> Optional[int]:
        async with self._cache_lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    async def _cache_put(self, key: str, value: int) -> None:
        async with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    async def _exact_local_count(
        self,
        body: Dict[str, Any],
        instance: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> tuple[int, bool]:
        key = self._cache_key(body, instance)
        cached = await self._cache_get(key)
        if cached is not None:
            return cached, True

        port = int(instance["port"])
        template_body, _ = normalize_custom_tools_for_local(body)
        template_body = dict(template_body)
        template_body["model"] = instance.get("model") or body.get("model")
        request_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() in {"authorization", "content-type"}
        }
        request_headers.setdefault("content-type", "application/json")
        timeout = httpx.Timeout(self._exact_timeout)

        template_response = await self._client.post(
            f"http://127.0.0.1:{port}/apply-template",
            json=template_body,
            headers=request_headers,
            timeout=timeout,
        )
        template_response.raise_for_status()
        prompt = template_response.json().get("prompt")
        if prompt is None:
            raise ValueError("llama-server /apply-template returned no prompt")

        tokenize_response = await self._client.post(
            f"http://127.0.0.1:{port}/tokenize",
            json={
                "content": prompt,
                "add_special": True,
                "parse_special": True,
                "with_pieces": False,
            },
            headers=request_headers,
            timeout=timeout,
        )
        tokenize_response.raise_for_status()
        tokens = tokenize_response.json().get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("llama-server /tokenize returned invalid tokens")
        count = len(tokens)
        await self._cache_put(key, count)
        return count, False

    async def count(
        self,
        body: Dict[str, Any],
        instances: Iterable[Dict[str, Any]],
        headers: Mapping[str, str],
        *,
        exact_required: bool,
    ) -> RequestTokenBudget:
        started = time.perf_counter()
        upper_bound = self.conservative_prompt_upper_bound(body)
        prompt_tokens = upper_bound
        source = "conservative"
        exact_backends = 0

        local_instances = self._local_instances(instances)
        if exact_required and local_instances:
            results = await asyncio.gather(
                *(
                    self._exact_local_count(body, instance, headers)
                    for instance in local_instances
                ),
                return_exceptions=True,
            )
            exact_counts = []
            all_exact = True
            all_cached = True
            for result in results:
                if isinstance(result, Exception):
                    all_exact = False
                    all_cached = False
                    continue
                count, cached = result
                exact_counts.append(count)
                exact_backends += 1
                all_cached = all_cached and cached

            # One failed tokenizer is enough to retain the safe upper bound:
            # routing may later select that backend during overflow/failover.
            if all_exact and exact_counts:
                prompt_tokens = max(exact_counts)
                source = "cached_exact" if all_cached else "exact"

        output_tokens = self.output_reserve(body)
        media_tokens = self.media_reserve(body)
        overhead_tokens = PROTOCOL_OVERHEAD + SAFETY_MARGIN
        required_context = (
            prompt_tokens + output_tokens + media_tokens + overhead_tokens
        )
        return RequestTokenBudget(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            overhead_tokens=overhead_tokens,
            media_tokens=media_tokens,
            required_context=required_context,
            source=source,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            exact_backends=exact_backends,
        )
