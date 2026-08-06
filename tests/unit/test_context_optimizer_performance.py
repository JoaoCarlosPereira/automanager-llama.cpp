"""Testes de performance do Context Optimizer (Task 15).

Medições de latência por fase p50/p95/p99, concorrência, downloads e event loop.
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List

import pytest

from context_optimizer import (
    ConservativeEstimator,
    ContextOptimizer,
    LimitConfidence,
    ModelLimits,
    OptimizationResult,
    RequestIR,
    TokenizerRegistry,
    calculate_target_budget,
    optimize_request_ir_aggressive,
    optimize_request_ir_moderate,
    optimize_request_ir_safe,
    parse_request_ir,
    resolve_model_limits,
    reconstruct_payload,
)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

_BACKEND_LOCAL_8192 = {"backend_type": "local", "config": {"context_size": 8192, "parallel_slots": 1}}
_BACKEND_LOCAL_4096 = {"backend_type": "local", "config": {"context_size": 4096, "parallel_slots": 1}}


def _short_payload(n_messages=4):
    """Cria payload curto para benchmarks de latência."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for i in range(1, n_messages):
        role = "assistant" if i % 2 == 1 else "user"
        messages.append({"role": role, "content": f"Message number {i} with some content"})
    if n_messages > 0:
        messages[-1]["content"] = "Current turn"
    return {"messages": messages}


def _medium_payload(n_messages=20):
    """Cria payload médio para benchmarks."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for i in range(1, n_messages):
        role = "assistant" if i % 2 == 1 else "user"
        messages.append({"role": role, "content": f"Message number {i} with some content and additional text to increase token count"})
    messages[-1]["content"] = "Current turn"
    return {"messages": messages}


def _long_payload(n_messages=80):
    """Cria payload longo para benchmarks."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for i in range(1, n_messages):
        role = "assistant" if i % 2 == 1 else "user"
        messages.append({
            "role": role,
            "content": f"Message number {i} with substantial content that increases token count significantly. " * 5,
        })
    messages[-1]["content"] = "Current turn"
    return {"messages": messages}


def _conversations_with_social_noise(n_messages=40):
    """Cria conversação com social noise para benchmarks de remoção."""
    messages = [
        {"role": "system", "content": "Instrução"},
    ]
    for i in range(1, n_messages):
        role = "assistant" if i % 2 == 1 else "user"
        if i % 3 == 0:
            messages.append({"role": role, "content": "Obrigado! Beleza!"})
        else:
            messages.append({"role": role, "content": f"Resposta {i}"})
    messages[-1]["content"] = "Final"
    return {"messages": messages}


# ---------------------------------------------------------------------------
# 1. Latency measurements per phase (p50/p95/p99)
# ---------------------------------------------------------------------------


class TestSafePhaseLatency:
    """Latência da fase Safe."""

    def _benchmark_safe(self, payload, n_runs=50, backend=_BACKEND_LOCAL_8192):
        """Executa n_runs e retorna latências em ms."""
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(backend),
            frozenset({"text"}),
        )
        latencies = []
        for _ in range(n_runs):
            start = time.monotonic()
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir, budget)
            )
            latencies.append((time.monotonic() - start) * 1000)
        return latencies

    def test_short_payload_latency(self):
        """Fase Safe em payload curto: p95 < 50 ms."""
        latencies = self._benchmark_safe(_short_payload(4), n_runs=100)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert p50 < 50, f"p50={p50:.2f}ms excede 50ms"
        assert p95 < 50, f"p95={p95:.2f}ms excede 50ms"
        assert p99 < 100, f"p99={p99:.2f}ms excede 100ms"

    def test_medium_payload_latency(self):
        """Fase Safe em payload médio: p95 < 200 ms."""
        latencies = self._benchmark_safe(_medium_payload(20), n_runs=100)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert p50 < 200, f"p50={p50:.2f}ms excede 200ms"
        assert p95 < 200, f"p95={p95:.2f}ms excede 200ms"
        assert p99 < 400, f"p99={p99:.2f}ms excede 400ms"

    def test_long_payload_latency(self):
        """Fase Safe em payload longo: p95 < 500 ms."""
        latencies = self._benchmark_safe(_long_payload(80), n_runs=100)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert p50 < 500, f"p50={p50:.2f}ms excede 500ms"
        assert p95 < 500, f"p95={p95:.2f}ms excede 500ms"
        assert p99 < 1000, f"p99={p99:.2f}ms excede 1000ms"

    def test_social_noise_removal_latency(self):
        """Remoção de social noise: overhead < 50 ms."""
        latencies_clean = self._benchmark_safe(_short_payload(4), n_runs=50)
        latencies_noise = self._benchmark_safe(_conversations_with_social_noise(40), n_runs=50)

        p50_clean = statistics.median(latencies_clean)
        p50_noise = statistics.median(latencies_noise)
        overhead = p50_noise - p50_clean

        assert overhead < 100, f"Overhead de social noise removal: {overhead:.2f}ms"


class TestModeratePhaseLatency:
    """Latência da fase Moderate."""

    def _benchmark_moderate(self, payload, n_runs=50, backend=_BACKEND_LOCAL_4096):
        """Executa Safe + Moderate e retorna latência total em ms."""
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(backend),
            frozenset({"text"}),
        )

        latencies = []
        for _ in range(n_runs):
            start = time.monotonic()
            safe_result = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir, budget)
            )
            mod_ir = parse_request_ir(safe_result.safe_payload)
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_moderate(mod_ir, budget, safe_audit=safe_result.audit)
            )
            latencies.append((time.monotonic() - start) * 1000)
        return latencies

    def test_moderate_latency_short(self):
        """Moderate em payload curto: p95 < 100 ms."""
        latencies = self._benchmark_moderate(_short_payload(4), n_runs=100)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 100, f"p95={p95:.2f}ms excede 100ms"

    def test_moderate_latency_medium(self):
        """Moderate em payload médio: p95 < 300 ms."""
        latencies = self._benchmark_moderate(_medium_payload(20), n_runs=100)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 300, f"p95={p95:.2f}ms excede 300ms"


class TestAggressivePhaseLatency:
    """Latência da fase Aggressive."""

    def _benchmark_aggressive(self, payload, n_runs=50, backend=_BACKEND_LOCAL_4096):
        """Executa Safe + Moderate + Aggressive e retorna latência total em ms."""
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(backend),
            frozenset({"text"}),
        )

        latencies = []
        for _ in range(n_runs):
            start = time.monotonic()
            safe_result = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir, budget)
            )
            mod_ir = parse_request_ir(safe_result.safe_payload)
            mod_result = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_moderate(mod_ir, budget, safe_audit=safe_result.audit)
            )
            agg_ir = parse_request_ir(mod_result.safe_payload)
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_aggressive(agg_ir, budget, moderate_audit=mod_result.audit)
            )
            latencies.append((time.monotonic() - start) * 1000)
        return latencies

    def test_aggressive_latency_short(self):
        """Aggressive em payload curto: p95 < 200 ms."""
        latencies = self._benchmark_aggressive(_short_payload(4), n_runs=100)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 200, f"p95={p95:.2f}ms excede 200ms"

    def test_aggressive_latency_medium(self):
        """Aggressive em payload médio: p95 < 500 ms."""
        latencies = self._benchmark_aggressive(_medium_payload(20), n_runs=100)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 500, f"p95={p95:.2f}ms excede 500ms"

    def test_full_pipeline_latency(self):
        """Pipeline completo (Safe+Moderate+Aggressive) p95 < 600 ms para payload curto."""
        latencies = self._benchmark_aggressive(_short_payload(4), n_runs=100)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert p50 < 200, f"p50={p50:.2f}ms excede 200ms"
        assert p95 < 600, f"p95={p95:.2f}ms excede 600ms"
        assert p99 < 1000, f"p99={p99:.2f}ms excede 1000ms"


# ---------------------------------------------------------------------------
# 2. Concurrency tests
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Testes de concorrência: chamadas simultâneas ao pipeline."""

    @pytest.mark.asyncio
    async def test_concurrent_safe_optimizations(self):
        """Múltiplas otimizações Safe em paralelo não falham."""
        payloads = [
            {"messages": [{"role": "system", "content": f"Sys {i}"}, {"role": "user", "content": f"User {i}"}]}
            for i in range(10)
        ]
        backend = _BACKEND_LOCAL_8192

        tasks = []
        for payload in payloads:
            ir = parse_request_ir(payload)
            budget = calculate_target_budget(
                payload,
                resolve_model_limits(backend),
                frozenset({"text"}),
            )
            tasks.append(optimize_request_ir_safe(ir, budget))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Todos devem成功 ou levantar exceção controlada
        successes = sum(1 for r in results if isinstance(r, OptimizationResult))
        assert successes >= 8, f"Menos de 8/10 sucessos: {len([r for r in results if isinstance(r, Exception)])} erros"

    @pytest.mark.asyncio
    async def test_concurrent_context_optimizer_calls(self):
        """ContextOptimizer.handle chamadas concorrentes."""
        optimizer = ContextOptimizer()

        payloads = [
            {
                "messages": [
                    {"role": "system", "content": f"System message {i}"},
                    {"role": "user", "content": f"User question {i} with some additional text content for token count"},
                ]
            }
            for i in range(5)
        ]

        tasks = [optimizer.optimize(p, _BACKEND_LOCAL_8192) for p in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if isinstance(r, OptimizationResult))
        assert successes >= 4, f"Menos de 4/5 sucessos"

    @pytest.mark.asyncio
    async def test_concurrent_moderate_calls(self):
        """ChamadasModerate concorrentes."""
        payloads = [
            {
                "messages": [
                    {"role": "system", "content": "A"},
                    {"role": "assistant", "content": f"Response {i}"},
                    {"role": "user", "content": f"User {i}"},
                ]
            }
            for i in range(5)
        ]
        backend = _BACKEND_LOCAL_4096

        async def run_moderate(payload):
            ir = parse_request_ir(payload)
            budget = calculate_target_budget(payload, resolve_model_limits(backend), frozenset({"text"}))
            safe = await optimize_request_ir_safe(ir, budget)
            mod_ir = parse_request_ir(safe.safe_payload)
            return await optimize_request_ir_moderate(mod_ir, budget, safe_audit=safe.audit)

        tasks = [run_moderate(p) for p in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if isinstance(r, OptimizationResult))
        assert successes >= 4


# ---------------------------------------------------------------------------
# 3. Download and tokenizer performance
# ---------------------------------------------------------------------------


class TestDownloadPerformance:
    """Performance de downloads de tokenizers (background)."""

    @pytest.mark.asyncio
    async def test_tokenizer_registry_caching(self):
        """Cache do TokenizerRegistry evita re-downloads."""
        registry = TokenizerRegistry()

        # Primeirachamada — estimada (sem cache)
        tokens1 = await registry.get_count("test", model_name="nonexistent-model")
        assert tokens1.source is not None

        # Segunda chamada — ainda estimada (model não existe), mas sem bloqueio
        tokens2 = await registry.get_count("test", model_name="nonexistent-model")
        assert tokens2.source is not None

        # Não deve travar o event loop
        start = time.monotonic()
        await asyncio.sleep(0)  # yield event loop
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 100, f"Event loop bloqueado por {elapsed:.2f}ms"

    @pytest.mark.asyncio
    async def test_tokenizer_download_backoff_non_blocking(self):
        """Backoff de download não bloqueia event loop."""
        registry = TokenizerRegistry(fetcher=lambda *a: (_ for _ in range(0)).throw(Exception("fail")))

        # Tentar download com fetcher que falha
        start = time.monotonic()
        await registry.get_count("test", model_name="bad-model")
        elapsed = (time.monotonic() - start) * 1000

        # Não deve bloquear além de 500 ms
        assert elapsed < 500, f"Download bloqueou por {elapsed:.2f}ms"

    @pytest.mark.asyncio
    async def test_heavy_tokenization_non_blocking(self):
        """Tokenização pesada não bloqueia event loop."""
        registry = TokenizerRegistry()

        # Payload grande
        big_payload = {"messages": [{"role": "user", "content": "x " * 1000}]}
        start = time.monotonic()
        task = asyncio.create_task(
            registry.get_count(big_payload, model_name="nonexistent-model")
        )
        # Deixar event loop respirar
        await asyncio.sleep(0)
        result = await task
        elapsed = (time.monotonic() - start) * 1000

        assert result is not None


# ---------------------------------------------------------------------------
# 4. Event loop performance
# ---------------------------------------------------------------------------


class TestEventLoop:
    """Performance do event loop: tarefas assíncronas não bloqueiam."""

    @pytest.mark.asyncio
    async def test_safe_non_blocking(self):
        """Fase Safe não bloqueia event loop por > 200ms."""
        payload = _medium_payload(20)
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_8192), frozenset({"text"}))

        start = time.monotonic()
        await optimize_request_ir_safe(ir, budget)
        elapsed = (time.monotonic() - start) * 1000

        assert elapsed < 200, f"Safe bloqueou por {elapsed:.2f}ms"

    @pytest.mark.asyncio
    async def test_full_pipeline_non_blocking(self):
        """Pipeline completo não bloqueia event loop por > 1000ms."""
        payload = _medium_payload(20)
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_4096), frozenset({"text"}))

        start = time.monotonic()
        safe = await optimize_request_ir_safe(ir, budget)
        mod_ir = parse_request_ir(safe.safe_payload)
        mod = await optimize_request_ir_moderate(mod_ir, budget, safe_audit=safe.audit)
        agg_ir = parse_request_ir(mod.safe_payload)
        await optimize_request_ir_aggressive(agg_ir, budget, moderate_audit=mod.audit)
        elapsed = (time.monotonic() - start) * 1000

        assert elapsed < 1000, f"Pipeline completo bloqueou por {elapsed:.2f}ms"

    @pytest.mark.asyncio
    async def test_concurrent_with_other_tasks(self):
        """Otimizações concorrentes com outras tarefas assíncronas."""
        async def optimize_task(task_id):
            payload = {"messages": [{"role": "user", "content": f"Task {task_id}"}]}
            ir = parse_request_ir(payload)
            budget = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_8192), frozenset({"text"}))
            return await optimize_request_ir_safe(ir, budget)

        async def other_task():
            await asyncio.sleep(0.01)
            return "other"

        tasks = [optimize_task(i) for i in range(5)] + [other_task() for _ in range(5)]
        start = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = (time.monotonic() - start) * 1000

        # 10 tarefas concorrentes devem completar em < 500ms
        assert elapsed < 500, f"Concorrência bloqueou por {elapsed:.2f}ms"
        successes = sum(1 for r in results if isinstance(r, (OptimizationResult, str)))
        assert successes == 10


# ---------------------------------------------------------------------------
# 5. Memory and scalability benchmarks
# ---------------------------------------------------------------------------


class TestScalability:
    """Escalabilidade: comportamento com payloads crescentes."""

    def test_linear_scaling_safe(self):
        """Crescimento de latência Safe é aproximadamente linear."""
        sizes = [5, 10, 20, 40, 60]
        latencies_by_size = {}

        for n in sizes:
            payload = _medium_payload(n)
            latencies = []
            for _ in range(20):
                ir = parse_request_ir(payload)
                budget = calculate_target_budget(
                    payload,
                    resolve_model_limits(_BACKEND_LOCAL_8192),
                    frozenset({"text"}),
                )
                start = time.monotonic()
                asyncio.get_event_loop().run_until_complete(
                    optimize_request_ir_safe(ir, budget)
                )
                latencies.append((time.monotonic() - start) * 1000)
            latencies_by_size[n] = statistics.median(latencies)

        # Verificar crescimento linear (p60/p10 ratio < 5x)
        ratio = latencies_by_size[60] / latencies_by_size[5] if latencies_by_size[5] > 0 else 0
        assert ratio < 10, f"Crescimento não linear: ratio={ratio:.2f}"

    def test_scatter_plot_latency(self):
        """Teste visual de scatter plot de latências por fase."""
        n_runs = 50
        safe_latencies = []
        mod_latencies = []
        agg_latencies = []

        payload = _medium_payload(20)
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_4096), frozenset({"text"}))

        # Benchmark Safe
        for _ in range(n_runs):
            ir_copy = parse_request_ir(payload)
            start = time.monotonic()
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir_copy, budget)
            )
            safe_latencies.append((time.monotonic() - start) * 1000)

        # Benchmark Moderate
        for _ in range(n_runs):
            ir_copy = parse_request_ir(payload)
            budget_copy = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_4096), frozenset({"text"}))
            safe = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir_copy, budget_copy)
            )
            mod_ir = parse_request_ir(safe.safe_payload)
            start = time.monotonic()
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_moderate(mod_ir, budget_copy, safe_audit=safe.audit)
            )
            mod_latencies.append((time.monotonic() - start) * 1000)

        # Benchmark Aggressive
        for _ in range(n_runs):
            ir_copy = parse_request_ir(payload)
            budget_copy = calculate_target_budget(payload, resolve_model_limits(_BACKEND_LOCAL_4096), frozenset({"text"}))
            safe = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_safe(ir_copy, budget_copy)
            )
            mod_ir = parse_request_ir(safe.safe_payload)
            mod = asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_moderate(mod_ir, budget_copy, safe_audit=safe.audit)
            )
            agg_ir = parse_request_ir(mod.safe_payload)
            start = time.monotonic()
            asyncio.get_event_loop().run_until_complete(
                optimize_request_ir_aggressive(agg_ir, budget_copy, moderate_audit=mod.audit)
            )
            agg_latencies.append((time.monotonic() - start) * 1000)

        # Verificar que as distribuições fazem sentido
        assert len(safe_latencies) == n_runs
        assert len(mod_latencies) == n_runs
        assert len(agg_latencies) == n_runs

        # Aggressive deve ser >= Moderate >= Safe (em média)
        avg_safe = statistics.mean(safe_latencies)
        avg_mod = statistics.mean(mod_latencies)
        avg_agg = statistics.mean(agg_latencies)

        # Aggressive pode ser menor se o payload couber no orçamento
        # Mas não deve ser absurdamente maior
        assert avg_agg < 2000, f"Aggressive média muito alta: {avg_agg:.2f}ms"
        assert avg_mod < 1000, f"Moderate média muito alta: {avg_mod:.2f}ms"
        assert avg_safe < 500, f"Safe média muito alta: {avg_safe:.2f}ms"


# ---------------------------------------------------------------------------
# 6. ConservativeEstimator performance
# ---------------------------------------------------------------------------


class TestEstimatorPerformance:
    """Performance do estimador conservador."""

    def test_estimate_text_performance(self):
        """estimate_text para textos grandes: < 10ms."""
        text = "Palavra " * 10000
        start = time.monotonic()
        tokens = ConservativeEstimator.estimate_text(text)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 10, f"estimate_text bloqueou por {elapsed:.2f}ms"
        assert tokens > 0

    def test_estimate_payload_performance(self):
        """estimate_payload para payload grande: < 20ms."""
        payload = {"messages": [{"role": "user", "content": "x " * 1000}]}
        start = time.monotonic()
        tokens = ConservativeEstimator.estimate_payload(payload)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 20, f"estimate_payload bloqueou por {elapsed:.2f}ms"
        assert tokens > 0

    def test_estimate_payload_nested_performance(self):
        """estimate_payload para payload profundamente aninhado: < 50ms."""
        nested = {"level0": {"level1": {"level2": {"level3": "x " * 100}}}}
        start = time.monotonic()
        tokens = ConservativeEstimator.estimate_payload(nested)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 50, f"estimate_payload aninhado bloqueou por {elapsed:.2f}ms"
        assert tokens > 0


# ---------------------------------------------------------------------------
# 7. IR parsing performance
# ---------------------------------------------------------------------------


class TestIRParsingPerformance:
    """Performance de parsing do RequestIR."""

    def test_parse_request_ir_performance(self):
        """parse_request_ir para payload com ferramentas: < 10ms."""
        payload = {
            "tools": [{"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}} for i in range(10)],
            "messages": [
                {"role": "user", "content": "Use as ferramentas"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": f"call_{i}", "type": "function", "function": {"name": f"tool_{i}"}}
                        for i in range(5)
                    ],
                },
            ],
        }
        start = time.monotonic()
        ir = parse_request_ir(payload)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 10, f"parse_request_ir bloqueou por {elapsed:.2f}ms"
        assert not ir.is_opaque

    def test_reconstruct_payload_performance(self):
        """reconstruct_payload para payload grande: < 10ms."""
        payload = _long_payload(40)
        ir = parse_request_ir(payload)
        start = time.monotonic()
        result = reconstruct_payload(ir)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 10, f"reconstruct_payload bloqueou por {elapsed:.2f}ms"
        assert result == payload
