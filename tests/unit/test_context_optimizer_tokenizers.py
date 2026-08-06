"""Testes do registro de tokenizers e estimador conservador (Task 3)."""

import asyncio
import time
import pytest
from context_optimizer import (
    ConservativeEstimator,
    TokenCountSource,
    TokenizerRegistry,
)


class DummyTokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def encode(self, text):
        return [1] * (len(text) * self.multiplier)


@pytest.mark.asyncio
async def test_resolution_order_model_exact_over_family():
    mappings = {
        "models": {"gpt-4o": "org/gpt4o-tok"},
        "families": {"gpt-4o": "org/family-tok"},
    }

    def fake_fetcher(ident, rev):
        return DummyTokenizer(multiplier=1)

    reg = TokenizerRegistry(fetcher=fake_fetcher)

    res = await reg.get_count("hello", model_name="gpt-4o", family_name="gpt-4o", configured_mappings=mappings)
    assert res.source == TokenCountSource.ESTIMATED
    assert res.tokenizer_ref == "org/gpt4o-tok"

    await asyncio.sleep(0.05)
    res2 = await reg.get_count("hello", model_name="gpt-4o", family_name="gpt-4o", configured_mappings=mappings)
    assert res2.source == TokenCountSource.EXACT_MODEL
    assert res2.tokenizer_ref == "org/gpt4o-tok"


@pytest.mark.asyncio
async def test_resolution_order_family_when_model_missing():
    mappings = {
        "models": {},
        "families": {"llama": "org/llama-tok"},
    }

    def fake_fetcher(ident, rev):
        return DummyTokenizer(multiplier=2)

    reg = TokenizerRegistry(fetcher=fake_fetcher)

    res = await reg.get_count("hi", family_name="llama", configured_mappings=mappings)
    assert res.source == TokenCountSource.ESTIMATED
    assert res.tokenizer_ref == "org/llama-tok"

    await asyncio.sleep(0.05)
    res2 = await reg.get_count("hi", family_name="llama", configured_mappings=mappings)
    assert res2.source == TokenCountSource.FAMILY
    assert res2.tokenizer_ref == "org/llama-tok"


@pytest.mark.asyncio
async def test_estimator_used_when_unmapped():
    reg = TokenizerRegistry()
    res = await reg.get_count({"messages": [{"role": "user", "content": "olá"}]})
    assert res.source == TokenCountSource.ESTIMATED
    assert res.tokenizer_ref is None
    assert res.tokens > 0


@pytest.mark.asyncio
async def test_deduplicated_concurrent_downloads():
    call_count = 0

    def fake_slow_fetcher(ident, rev):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return DummyTokenizer()

    reg = TokenizerRegistry(fetcher=fake_slow_fetcher)
    mappings = {"models": {"m1": "org/slow-tok"}}

    tasks = [
        reg.get_count("test payload", model_name="m1", configured_mappings=mappings)
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks)

    for r in results:
        assert r.source == TokenCountSource.ESTIMATED

    await asyncio.sleep(0.2)

    res_cached = await reg.get_count("test payload", model_name="m1", configured_mappings=mappings)
    assert res_cached.source == TokenCountSource.EXACT_MODEL
    assert call_count == 1


@pytest.mark.asyncio
async def test_download_failure_applies_backoff():
    fetch_attempts = 0

    def failing_fetcher(ident, rev):
        nonlocal fetch_attempts
        fetch_attempts += 1
        raise RuntimeError("network down")

    reg = TokenizerRegistry(fetcher=failing_fetcher)
    mappings = {"models": {"m1": "org/bad-tok"}}

    res1 = await reg.get_count("payload", model_name="m1", configured_mappings=mappings)
    assert res1.source == TokenCountSource.ESTIMATED

    await asyncio.sleep(0.05)
    assert fetch_attempts == 1

    res2 = await reg.get_count("payload", model_name="m1", configured_mappings=mappings)
    assert res2.source == TokenCountSource.ESTIMATED
    await asyncio.sleep(0.05)
    assert fetch_attempts == 1


@pytest.mark.asyncio
async def test_heavy_tokenization_does_not_block_event_loop():
    def heavy_fetcher(ident, rev):
        class HeavyTokenizer:
            def encode(self, text):
                time.sleep(0.15)
                return [1] * len(text)
        return HeavyTokenizer()

    reg = TokenizerRegistry(fetcher=heavy_fetcher)
    mappings = {"models": {"m1": "org/heavy-tok"}}

    await reg.get_count("payload", model_name="m1", configured_mappings=mappings)
    await asyncio.sleep(0.05)

    sentinel_ran = False

    async def sentinel():
        nonlocal sentinel_ran
        await asyncio.sleep(0.02)
        sentinel_ran = True

    task_tok = reg.get_count("long payload text", model_name="m1", configured_mappings=mappings)
    task_sentinel = sentinel()

    start = time.time()
    await asyncio.gather(task_tok, task_sentinel)
    elapsed = time.time() - start

    assert sentinel_ran is True
    assert elapsed >= 0.15
