"""Testes do ProxyRouter (tasks 02, 03 e 05 — afinidade, sticky, seleção, SSE)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config_manager import ConfigManager, normalize_model_path
from proxy_router import (
    ProxyError,
    ProxyRouter,
    RouteDecision,
    RoutePlan,
    StaleRoutePlan,
    StickySession,
    rewrite_json_model,
    rewrite_sse_stream,
)

MAIN_PATH = "models/main.gguf"
AUX0_PATH = "models/aux0.gguf"
AUX1_PATH = "models/aux1.gguf"


def make_instance(port, model_path, ctx=65536, slots=1, gpu_name="NVIDIA RTX 3090",
                  gpu_index=0):
    return {
        "port": port,
        "status": "running",
        "model": model_path.split("/")[-1],
        "model_path": model_path,
        "config": {
            "context_size": ctx,
            "parallel_slots": slots,
            "gpu_weights": [
                {
                    "index": gpu_index,
                    "weight": 1.0,
                    "name": gpu_name,
                    "active": True,
                    "is_main": True,
                    "device": "gpu",
                }
            ],
        },
    }


def make_platform_instance(
    port=9100,
    backend_id="platform:codex",
    model="Codex",
    provider="codex",
):
    return {
        "port": port,
        "status": "running",
        "model": model,
        "model_path": None,
        "backend_id": backend_id,
        "backend_type": "platform",
        "provider": provider,
        "config": {
            "backend_id": backend_id,
            "backend_type": "platform",
            "provider": provider,
            "proxy_eligible": True,
            "max_parallel_requests": 1,
        },
    }


DEFAULT_INSTANCES = [
    make_instance(8085, MAIN_PATH, gpu_name="NVIDIA RTX 3090", gpu_index=0),
    make_instance(8086, AUX0_PATH, gpu_name="Tesla P100", gpu_index=1),
    make_instance(8087, AUX1_PATH, gpu_name="Tesla P100", gpu_index=2),
]


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def status_holder():
    return {"instances": list(DEFAULT_INSTANCES)}


@pytest.fixture
def proxy_config(tmp_config_manager: ConfigManager):
    tmp_config_manager.update_smart_proxy_settings(
        {"enabled": True, "primary_model_path": MAIN_PATH, "max_wait_seconds": 1}
    )
    return tmp_config_manager


@pytest.fixture
def router(proxy_config, status_holder, tmp_path, clock):
    return ProxyRouter(
        get_status=lambda: status_holder,
        config_manager=proxy_config,
        sessions_path=tmp_path / "proxy_sessions.json",
        now=clock,
    )


def body_with(tag=None, system="Voce ajuda.", user="Oi", model="main.gguf",
              metadata=None):
    system_content = f"[AGENT:{tag}] {system}" if tag else system
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user},
        ],
    }
    if metadata:
        body["metadata"] = metadata
    return body


async def resolve(router, *, headers=None, body=None, ip="10.0.0.1",
                  ua="cursor/1.0", dry_run=False):
    return await router.resolve(
        headers=headers or {},
        body=body or body_with(),
        client_ip=ip,
        user_agent=ua,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Extração de afinidade (task 02)
# ---------------------------------------------------------------------------

class TestAffinityExtraction:
    def test_session_header_wins_over_everything(self, router):
        key, tag = router.extract_affinity(
            {"X-Automanager-Session-Id": "abc", "x-automanager-agent-id": "zz"},
            body_with(tag="sql-reviewer", metadata={"session_id": "meta"}),
            "1.1.1.1", "ua",
        )
        assert key == "sid:abc"
        assert tag == "sql-reviewer"

    def test_metadata_beats_tag(self, router):
        key, tag = router.extract_affinity(
            {}, body_with(tag="sql-reviewer", metadata={"agent_id": "a1"}),
            "1.1.1.1", "ua",
        )
        assert key == "aid:a1"
        assert tag == "sql-reviewer"

    def test_tag_in_system_message(self, router):
        key, tag = router.extract_affinity(
            {}, body_with(tag="sql-reviewer"), "1.1.1.1", "ua"
        )
        assert tag == "sql-reviewer"
        assert key.startswith("agent:sql-reviewer:")

    def test_stable_hash_is_deterministic(self, router):
        k1, _ = router.extract_affinity({}, body_with(), "1.1.1.1", "cursor")
        k2, _ = router.extract_affinity({}, body_with(), "1.1.1.1", "cursor")
        assert k1 == k2
        assert k1.startswith("hash:")

    def test_user_agent_changes_hash(self, router):
        k1, _ = router.extract_affinity({}, body_with(), "1.1.1.1", "cursor")
        k2, _ = router.extract_affinity({}, body_with(), "1.1.1.1", "cline")
        assert k1 != k2

    def test_empty_or_invalid_tag_not_matched(self, router):
        body = body_with()
        body["messages"][0]["content"] = "[AGENT:] [AGENT:com espaco] oi"
        assert router.detect_tag(body) is None

    def test_system_message_searched_first(self, router):
        body = {
            "messages": [
                {"role": "user", "content": "[AGENT:user-tag] pergunta"},
                {"role": "system", "content": "[AGENT:sys-tag] instrucoes"},
            ]
        }
        assert router.detect_tag(body) == "sys-tag"

    def test_content_as_parts_list(self, router):
        body = {
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": "[AGENT:vision] oi"}]},
            ]
        }
        assert router.detect_tag(body) == "vision"


# ---------------------------------------------------------------------------
# Tabela sticky, TTL e persistência (task 02)
# ---------------------------------------------------------------------------

class TestStickySessions:
    @pytest.mark.asyncio
    async def test_incompatible_persisted_platform_session_is_dropped(
        self, router, proxy_config, status_holder, clock
    ):
        """Sessão antiga de plataforma não pode contaminar o principal local."""
        antigravity = make_platform_instance(
            port=8317,
            backend_id="platform:google-antigravity",
            provider="antigravity",
            model="Google Antigravity",
        )
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            antigravity,
        ]
        proxy_config.update_platform_settings(
            "platform:google-antigravity", {"proxy_eligible": True}
        )
        body = body_with(model="main.gguf")
        affinity_key, _ = router.extract_affinity({}, body, "127.0.0.1", "")
        router._sessions[affinity_key] = StickySession(
            affinity_key=affinity_key,
            backend_port=8317,
            backend_model_path="",
            external_model="main.gguf",
            internal_model="Google Antigravity",
            detected_tag=None,
            created_at=clock().isoformat(),
            last_used_at=clock().isoformat(),
            backend_id="platform:google-antigravity",
            backend_type="platform",
            provider="antigravity",
        )

        decision = await resolve(router, body=body)

        assert decision.backend_port == 8085
        assert decision.reason == "main_preference"

    @pytest.mark.asyncio
    async def test_ttl_expires_by_inactivity(self, router, clock):
        decision = await resolve(router, body=body_with(tag="a1"))
        await router.release(decision.backend_port)
        assert len(await router.sessions()) == 1
        clock.advance(minutes=181)
        assert await router.sessions() == []

    @pytest.mark.asyncio
    async def test_persistence_round_trip(
        self, router, proxy_config, status_holder, tmp_path, clock
    ):
        d1 = await resolve(router, body=body_with(tag="a1"))
        d2 = await resolve(router, body=body_with(tag="a2"))
        await router.release(d1.backend_port)
        await router.release(d2.backend_port)

        router2 = ProxyRouter(
            get_status=lambda: status_holder,
            config_manager=proxy_config,
            sessions_path=tmp_path / "proxy_sessions.json",
            now=clock,
        )
        sessions = {s.affinity_key: s for s in await router2.sessions()}
        assert d1.affinity_key in sessions
        assert sessions[d1.affinity_key].backend_port == d1.backend_port

    @pytest.mark.asyncio
    async def test_corrupted_sessions_file_starts_empty(
        self, proxy_config, status_holder, tmp_path, clock
    ):
        path = tmp_path / "proxy_sessions.json"
        path.write_text("{corrompido", encoding="utf-8")
        router = ProxyRouter(
            get_status=lambda: status_holder,
            config_manager=proxy_config,
            sessions_path=path,
            now=clock,
        )
        assert await router.sessions() == []

    @pytest.mark.asyncio
    async def test_rebind_by_model_path_after_port_change(
        self, router, status_holder
    ):
        decision = await resolve(router, body=body_with(tag="a1"))
        await router.release(decision.backend_port)
        old_port = decision.backend_port
        moved = [
            make_instance(9001 + i, inst["model_path"],
                          gpu_name="Tesla P100", gpu_index=i)
            for i, inst in enumerate(DEFAULT_INSTANCES)
        ]
        status_holder["instances"] = moved
        again = await resolve(router, body=body_with(tag="a1"))
        await router.release(again.backend_port)
        assert again.sticky_hit is True
        assert again.backend_port != old_port
        session = (await router.sessions())[0]
        assert session.backend_port == again.backend_port

    @pytest.mark.asyncio
    async def test_clear_sessions(self, router):
        d = await resolve(router, body=body_with(tag="a1"))
        await router.release(d.backend_port)
        assert await router.clear_sessions("inexistente") == 0
        assert await router.clear_sessions(d.affinity_key) == 1
        assert await router.clear_sessions() == 0

    @pytest.mark.asyncio
    async def test_expire_idle_returns_removed_count(self, router, clock):
        d = await resolve(router, body=body_with(tag="a1"))
        await router.release(d.backend_port)
        assert await router.expire_idle() == 0
        clock.advance(minutes=181)
        assert await router.expire_idle() == 1
        assert await router.sessions() == []

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self, router):
        d1 = await resolve(router, body=body_with(tag="a1"))
        d2 = await resolve(router, body=body_with(tag="a2"))
        await router.release(d1.backend_port)
        await router.release(d2.backend_port)
        assert await router.clear_sessions() == 2
        assert await router.sessions() == []


# ---------------------------------------------------------------------------
# Seleção de backend (task 03)
# ---------------------------------------------------------------------------

class TestSelection:
    @pytest.mark.asyncio
    async def test_main_without_tag_prefers_primary(self, router):
        decision = await resolve(router, body=body_with())
        await router.release(decision.backend_port)
        assert decision.backend_port == 8085
        assert decision.reason == "main_preference"
        assert decision.rewrite is False

    @pytest.mark.asyncio
    async def test_agent_main_tag_prefers_primary(self, router):
        decision = await resolve(router, body=body_with(tag="main"))
        await router.release(decision.backend_port)
        assert decision.backend_port == 8085

    @pytest.mark.asyncio
    async def test_subagent_prefers_primary_when_available(self, router):
        decision = await resolve(router, body=body_with(tag="sql-reviewer"))
        await router.release(decision.backend_port)
        assert decision.backend_port == 8085
        assert decision.reason == "subagent_main_preference"
        assert decision.rewrite is False

    @pytest.mark.asyncio
    async def test_subagent_overflows_when_primary_busy(self, router):
        d_main = await resolve(router, body=body_with())  # ocupa 8085
        decision = await resolve(router, body=body_with(tag="sql-reviewer"))
        assert decision.backend_port == 8086  # empate 8086/8087 -> menor porta
        assert decision.reason == "subagent_least_busy"
        assert decision.rewrite is True
        await router.release(d_main.backend_port)
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_measured_speed_orders_backends_after_primary(self, router):
        router.set_benchmark_results({
            f"local:{normalize_model_path(MAIN_PATH)}": 900.0,
            f"local:{normalize_model_path(AUX0_PATH)}": 700.0,
            f"local:{normalize_model_path(AUX1_PATH)}": 120.0,
        }, measured_at="2026-08-06T12:00:00+00:00")

        primary = await resolve(router, body=body_with())
        overflow = await resolve(router, body=body_with(tag="fastest"))

        assert primary.backend_port == 8085
        assert overflow.backend_port == 8087
        assert overflow.reason == "subagent_least_busy"
        await router.release(primary.backend_port)
        await router.release(overflow.backend_port)

    @pytest.mark.asyncio
    async def test_secondary_local_precedes_cloud(
        self, router, proxy_config, status_holder
    ):
        codex = make_platform_instance(backend_id="platform:codex")
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_instance(8086, AUX0_PATH),
            codex,
        ]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        router.set_benchmark_results({
            f"local:{normalize_model_path(AUX0_PATH)}": 10.0,
            "platform:codex": 1000.0,
        })

        primary = await resolve(router, body=body_with())
        overflow = await resolve(router, body=body_with(tag="local-first"))

        assert primary.backend_port == 8085
        assert overflow.backend_port == 8086
        assert overflow.backend_type == "local"
        await router.release(primary.backend_id)
        await router.release(overflow.backend_id)

    @pytest.mark.asyncio
    async def test_cloud_is_used_after_secondary_local_capacity(
        self, router, proxy_config, status_holder
    ):
        codex = make_platform_instance(backend_id="platform:codex")
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_instance(8086, AUX0_PATH),
            codex,
        ]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        primary = await resolve(router, body=body_with())
        local = await resolve(router, body=body_with(tag="local"))
        platform = await resolve(router, body=body_with(tag="cloud-after-local"))

        assert local.backend_port == 8086
        assert platform.backend_id == "platform:codex"
        for decision in (primary, platform, local):
            await router.release(decision.backend_id)

    @pytest.mark.asyncio
    async def test_secondary_local_with_free_slot_precedes_idle_cloud(
        self, router, proxy_config, status_holder
    ):
        codex = make_platform_instance(backend_id="platform:codex")
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_instance(8086, AUX0_PATH, slots=2),
            codex,
        ]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        proxy_config.update_model_settings(
            AUX0_PATH, {"max_parallel_requests": 2}
        )

        primary = await resolve(router, body=body_with())
        local_first = await resolve(router, body=body_with(tag="local-1"))
        local_second = await resolve(router, body=body_with(tag="local-2"))

        assert local_first.backend_port == 8086
        assert local_second.backend_port == 8086
        assert local_second.backend_type == "local"
        for decision in (primary, local_first, local_second):
            await router.release(decision.backend_id)

    def test_snapshot_is_sorted_by_measured_speed_after_primary(self, router):
        router.set_benchmark_results({
            f"local:{normalize_model_path(MAIN_PATH)}": 900.0,
            f"local:{normalize_model_path(AUX0_PATH)}": 700.0,
            f"local:{normalize_model_path(AUX1_PATH)}": 120.0,
        }, measured_at="2026-08-06T12:00:00+00:00")

        snapshot = router.backends_snapshot()

        assert [backend["port"] for backend in snapshot] == [8085, 8087, 8086]
        assert [backend["speed_rank"] for backend in snapshot] == [0, 1, 2]
        assert snapshot[1]["startup_latency_ms"] == 120.0
        assert snapshot[1]["benchmark_measured_at"] == (
            "2026-08-06T12:00:00+00:00"
        )

    @pytest.mark.asyncio
    async def test_sticky_hit_ignores_load(self, router):
        d1 = await resolve(router, body=body_with(tag="sql-reviewer"))
        await router.release(d1.backend_port)
        d2 = await resolve(router, body=body_with(tag="sql-reviewer"))
        await router.release(d2.backend_port)
        assert d2.backend_port == d1.backend_port
        assert d2.sticky_hit is True
        assert d2.reason == "sticky"

    @pytest.mark.asyncio
    async def test_sticky_session_updates_models_when_request_model_changes(
        self, router, tmp_path
    ):
        headers = {"x-automanager-session-id": "same-conversation"}
        first = await resolve(
            router, headers=headers, body=body_with(model="main.gguf")
        )
        await router.release(first.backend_id)

        second = await resolve(
            router, headers=headers, body=body_with(model="aux0.gguf")
        )
        await router.release(second.backend_id)

        session = (await router.sessions())[0]
        assert second.internal_model == "aux0.gguf"
        assert session.external_model == "aux0.gguf"
        assert session.internal_model == "aux0.gguf"

        persisted = json.loads(
            (tmp_path / "proxy_sessions.json").read_text(encoding="utf-8")
        )["sessions"][0]
        assert persisted["external_model"] == "aux0.gguf"
        assert persisted["internal_model"] == "aux0.gguf"

    @pytest.mark.asyncio
    async def test_context_constraint_excludes_small_secondary(
        self, router, status_holder
    ):
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH, ctx=131072),
            make_instance(8086, AUX0_PATH, ctx=8192),
        ]
        big_user = "x" * (9000 * 4)  # ~9k tokens estimados > 8192
        decision = await resolve(
            router, body=body_with(tag="sql-reviewer", user=big_user)
        )
        await router.release(decision.backend_port)
        assert decision.backend_port == 8085

    @pytest.mark.asyncio
    async def test_proxy_eligible_false_excluded(self, router, proxy_config):
        proxy_config.update_model_settings(AUX0_PATH, {"proxy_eligible": False})
        d_main = await resolve(router, body=body_with())
        decision = await resolve(router, body=body_with(tag="a1"))
        assert decision.backend_port == 8087
        await router.release(d_main.backend_port)
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_platform_secondary_requires_explicit_proxy_eligibility(
        self, router, proxy_config, status_holder
    ):
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_platform_instance(),
        ]
        d_main = await resolve(router, body=body_with())
        # Plataforma inelegível não recebe tráfego; o principal cresce.
        expanded = await resolve(router, body=body_with(tag="a1"))
        assert expanded.backend_port == 8085
        assert expanded.reason == "dynamic_capacity"
        await router.release(expanded.backend_port)

        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        decision = await resolve(router, body=body_with(tag="a2"))
        assert decision.backend_port == 9100
        assert decision.backend_id == "platform:codex"
        assert decision.backend_type == "platform"
        await router.release(d_main.backend_port)
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_platform_primary_routes_through_sidecar_when_eligible(
        self, router, proxy_config, status_holder
    ):
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_platform_instance(),
        ]
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        decision = await resolve(router, body=body_with(model="codex-pro"))
        assert decision.backend_port == 9100
        assert decision.internal_model == "codex-pro"
        assert decision.external_model == "codex-pro"
        assert decision.backend_type == "platform"
        assert decision.rewrite is False
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_dynamic_primary_and_failover_use_each_models_context_limit(
        self, router, proxy_config, status_holder, monkeypatch
    ):
        codex = make_platform_instance(
            port=8317, backend_id="platform:codex", provider="codex"
        )
        antigravity = make_platform_instance(
            port=8317,
            backend_id="platform:google-antigravity",
            model="Google Antigravity",
            provider="antigravity",
        )
        status_holder["instances"] = [codex, antigravity]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        proxy_config.update_platform_settings(
            "platform:google-antigravity",
            {
                "proxy_eligible": True,
                "default_model": "antigravity-31prolow.gguf",
            },
        )
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        router._requested_primary_resolver = (
            lambda instances, model: next(
                item for item in instances
                if item["provider"] == (
                    "antigravity" if model.startswith("antigravity-") else "codex"
                )
            )
        )
        limits = {
            ("codex", "codex-56luna.gguf"): 372_000,
            ("antigravity", "antigravity-31prolow.gguf"): 1_048_576,
        }
        router._context_limit_resolver = (
            lambda instance, model: limits.get((instance["provider"], model), 0)
        )
        monkeypatch.setattr(
            router, "estimate_prompt_tokens", lambda body: body["estimated_tokens"]
        )
        headers = {"X-Automanager-Session-Id": "cursor-context"}

        luna = await resolve(
            router,
            headers=headers,
            body={"model": "codex-56luna.gguf", "estimated_tokens": 300_000},
        )
        assert luna.backend_id == "platform:codex"
        await router.release(luna.backend_id)

        overflow = await resolve(
            router,
            headers=headers,
            body={"model": "codex-56luna.gguf", "estimated_tokens": 400_000},
        )
        assert overflow.backend_id == "platform:google-antigravity"
        assert overflow.internal_model == "antigravity-31prolow.gguf"
        assert overflow.reason == "reassign_context_limit"
        await router.release(overflow.backend_id)

        antigravity_primary = await resolve(
            router,
            headers={"X-Automanager-Session-Id": "antigravity-context"},
            body={
                "model": "antigravity-31prolow.gguf",
                "estimated_tokens": 900_000,
            },
        )
        assert antigravity_primary.backend_id == "platform:google-antigravity"
        await router.release(antigravity_primary.backend_id)

        with pytest.raises(ProxyError) as exc_info:
            await resolve(
                router,
                headers={"X-Automanager-Session-Id": "too-large"},
                body={
                    "model": "antigravity-31prolow.gguf",
                    "estimated_tokens": 1_000_000,
                },
            )
        assert exc_info.value.status_code == 413
        assert exc_info.value.code == "context_too_large"

    @pytest.mark.asyncio
    async def test_platform_primary_must_be_proxy_eligible(
        self, router, proxy_config, status_holder
    ):
        status_holder["instances"] = [make_platform_instance()]
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        with pytest.raises(ProxyError) as exc_info:
            await resolve(router, body=body_with(model="codex-pro"))
        assert exc_info.value.code == "backend_not_eligible"

    def test_backend_snapshot_includes_platform_identity(
        self, router, proxy_config, status_holder
    ):
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_platform_instance(),
        ]
        snapshot = router.backends_snapshot()
        platform = next(b for b in snapshot if b["backend_type"] == "platform")
        assert platform["backend_id"] == "platform:codex"
        assert platform["provider"] == "codex"
        assert platform["state"] == "not_eligible"

        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        enabled = next(
            b for b in router.backends_snapshot()
            if b["backend_id"] == "platform:codex"
        )
        assert enabled["state"] == "online"

    @pytest.mark.asyncio
    async def test_shared_sidecar_port_tracks_in_flight_per_backend_id(
        self, router, proxy_config, status_holder
    ):
        """Duas integrações no mesmo sidecar não devem compartilhar contador ocupado."""
        shared_port = 8317
        codex = make_platform_instance(
            port=shared_port, backend_id="platform:codex"
        )
        antigravity = make_platform_instance(
            port=shared_port,
            backend_id="platform:google-antigravity",
            model="Google Antigravity",
            provider="antigravity",
        )
        status_holder["instances"] = [codex, antigravity]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        proxy_config.update_platform_settings(
            "platform:google-antigravity", {"proxy_eligible": True}
        )
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:google-antigravity"}
        )
        decision = await resolve(
            router, body=body_with(model="antigravity-proagent.gguf")
        )
        by_id = {b["backend_id"]: b for b in router.backends_snapshot()}
        assert decision.backend_id == "platform:google-antigravity"
        assert by_id["platform:google-antigravity"]["in_flight"] == 1
        assert by_id["platform:google-antigravity"]["state"] == "busy"
        assert by_id["platform:codex"]["in_flight"] == 0
        assert by_id["platform:codex"]["state"] == "online"
        await router.release(decision.backend_id)

    @pytest.mark.asyncio
    async def test_platform_primary_overflow_prefers_secondary_local(
        self, router, proxy_config, status_holder
    ):
        """Mesmo com principal cloud, secundários locais evitam outro cloud."""
        shared_port = 8317
        antigravity = make_platform_instance(
            port=shared_port, backend_id="platform:google-antigravity",
            provider="antigravity",
        )
        codex = make_platform_instance(
            port=shared_port, backend_id="platform:codex", provider="codex",
        )
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_instance(8086, AUX0_PATH),
            antigravity,
            codex,
        ]
        proxy_config.update_platform_settings(
            "platform:google-antigravity", {"proxy_eligible": True}
        )
        proxy_config.update_platform_settings(
            "platform:codex",
            {"proxy_eligible": True, "default_model": "codex-default.gguf"},
        )
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:google-antigravity"}
        )
        d_main = await resolve(
            router, body=body_with(model="antigravity-proagent.gguf")
        )
        overflow = await resolve(
            router, body=body_with(tag="a1", model="antigravity-proagent.gguf")
        )
        assert d_main.backend_id == "platform:google-antigravity"
        assert overflow.backend_port == 8085
        assert overflow.backend_type == "local"
        assert overflow.reason == "subagent_least_busy"
        await router.release(d_main.backend_id)
        await router.release(overflow.backend_id)

    @pytest.mark.asyncio
    async def test_platform_secondary_uses_default_model(
        self, router, proxy_config, status_holder
    ):
        """Plataforma secundária encaminha para o default_model configurado."""
        codex = make_platform_instance(backend_id="platform:codex")
        status_holder["instances"] = [make_instance(8085, MAIN_PATH), codex]
        proxy_config.update_platform_settings(
            "platform:codex",
            {"proxy_eligible": True, "default_model": "codex-54mini.gguf"},
        )
        d_main = await resolve(router, body=body_with())
        overflow = await resolve(router, body=body_with(tag="a1"))
        assert overflow.backend_id == "platform:codex"
        assert overflow.internal_model == "codex-54mini.gguf"
        await router.release(d_main.backend_id)
        await router.release(overflow.backend_id)

    @pytest.mark.asyncio
    async def test_platform_secondary_without_default_falls_back(
        self, router, proxy_config, status_holder
    ):
        """Sem default_model, a plataforma secundária mantém o external_model."""
        codex = make_platform_instance(backend_id="platform:codex")
        status_holder["instances"] = [make_instance(8085, MAIN_PATH), codex]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        d_main = await resolve(router, body=body_with())
        overflow = await resolve(router, body=body_with(tag="a1"))
        assert overflow.backend_id == "platform:codex"
        assert overflow.internal_model == "main.gguf"
        await router.release(d_main.backend_id)
        await router.release(overflow.backend_id)

    @pytest.mark.asyncio
    async def test_disabled_backend_never_gets_new_sessions(self, router):
        router.set_backend_enabled(8086, False)
        d_main = await resolve(router, body=body_with())
        decision = await resolve(router, body=body_with(tag="a1"))
        assert decision.backend_port == 8087
        await router.release(d_main.backend_port)
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_reassign_once_when_backend_down(self, router, status_holder):
        d_main = await resolve(router, body=body_with())  # ocupa primary
        decision = await resolve(router, body=body_with(tag="a1"))
        await router.release(decision.backend_port)
        dead_port = decision.backend_port
        assert dead_port != 8085
        status_holder["instances"] = [
            inst for inst in DEFAULT_INSTANCES if inst["port"] != dead_port
        ]
        # Principal continua ocupado (d_main nao foi liberado): forca o
        # caminho de deteccao de backend caido em vez da volta ao principal.
        again = await resolve(router, body=body_with(tag="a1"))
        await router.release(again.backend_port)
        await router.release(d_main.backend_port)
        assert again.reason == "reassign_backend_down"
        assert again.backend_port != dead_port

    @pytest.mark.asyncio
    async def test_sticky_session_returns_to_primary_when_free(self, router):
        """Sessao sticky presa num secundario volta pro principal assim que
        ele libera, em vez de continuar la enquanto durar o TTL."""
        d_main = await resolve(router, body=body_with())  # ocupa principal
        decision = await resolve(router, body=body_with(tag="a1"))
        assert decision.backend_port != 8085
        await router.release(d_main.backend_port)  # principal libera
        await router.release(decision.backend_port)
        again = await resolve(router, body=body_with(tag="a1"))
        assert again.reason == "sticky_return_primary"
        assert again.backend_port == 8085
        await router.release(again.backend_port)

    @pytest.mark.asyncio
    async def test_concurrent_same_hash_branches_across_backends(self, router):
        """Subagentes sem tag e com prompts idênticos (mesma hash) em
        paralelo devem ramificar para backends livres, não enfileirar."""
        body = body_with()  # sem tag -> hash:
        d1 = await resolve(router, body=body)
        d2 = await resolve(router, body=body)  # base ocupada -> ramo #2
        d3 = await resolve(router, body=body)  # -> ramo #3
        ports = {d1.backend_port, d2.backend_port, d3.backend_port}
        assert ports == {8085, 8086, 8087}
        assert d1.affinity_key.startswith("hash:")
        assert d2.affinity_key == f"{d1.affinity_key}#2"
        assert d3.affinity_key == f"{d1.affinity_key}#3"
        assert d2.reason == "hash_branch"
        for d in (d1, d2, d3):
            await router.release(d.backend_port)

    @pytest.mark.asyncio
    async def test_hash_branch_is_sticky_on_reuse(self, router):
        body = body_with()
        d1 = await resolve(router, body=body)
        d2 = await resolve(router, body=body)
        await router.release(d2.backend_port)
        # Base segue ocupada; nova requisição concorrente reusa o ramo #2
        d2_again = await resolve(router, body=body)
        assert d2_again.affinity_key == d2.affinity_key
        assert d2_again.backend_port == d2.backend_port
        assert d2_again.sticky_hit is True
        assert d2_again.reason == "sticky_branch"
        await router.release(d1.backend_port)
        await router.release(d2_again.backend_port)

    @pytest.mark.asyncio
    async def test_tagged_sessions_overflow_to_free_backend(self, router, status_holder):
        """Afinidade explícita ocupada transborda para o próximo backend livre."""
        d1 = await resolve(router, body=body_with(tag="rock"))
        d2 = await resolve(router, body=body_with(tag="rock"))
        assert d2.backend_port != d1.backend_port
        assert d2.affinity_key == f"{d1.affinity_key}#2"
        assert d2.reason == "sticky_busy_overflow"
        assert router.in_flight(d1.backend_port) == 1
        assert router.in_flight(d2.backend_port) == 1
        await router.release(d1.backend_port)
        await router.release(d2.backend_port)

    @pytest.mark.asyncio
    async def test_tagged_overflow_branch_is_reused(self, router):
        d1 = await resolve(router, body=body_with(tag="rock"))
        d2 = await resolve(router, body=body_with(tag="rock"))
        await router.release(d2.backend_port)

        d2_again = await resolve(router, body=body_with(tag="rock"))

        assert d2_again.affinity_key == d2.affinity_key
        assert d2_again.backend_port == d2.backend_port
        assert d2_again.reason == "sticky_branch"
        await router.release(d1.backend_port)
        await router.release(d2_again.backend_port)

    @pytest.mark.asyncio
    async def test_tagged_session_uses_busy_backend_only_when_all_are_busy(
        self, router
    ):
        d1 = await resolve(router, body=body_with(tag="rock"))
        d2 = await resolve(router, body=body_with(tag="rock"))
        d3 = await resolve(router, body=body_with(tag="rock"))
        assert {d1.backend_port, d2.backend_port, d3.backend_port} == {
            8085, 8086, 8087
        }

        d4 = await resolve(router, body=body_with(tag="rock"))

        assert d4.backend_port == d1.backend_port
        assert d4.reason == "sticky_dynamic_capacity"
        for decision in (d1, d2, d3, d4):
            await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_untagged_new_session_overflows_when_primary_busy(
        self, router
    ):
        """PRD F7: sessão nova não espera — transborda para secundário livre."""
        first = await resolve(router, body=body_with())  # ocupa primary (max=1)
        second = await resolve(router, body=body_with(user="outra conversa"))
        assert second.backend_port in (8086, 8087)
        assert second.reason == "primary_busy_overflow"
        assert second.rewrite is True
        await router.release(first.backend_port)
        await router.release(second.backend_port)

    @pytest.mark.asyncio
    async def test_single_primary_grows_instead_of_timing_out(
        self, router, status_holder
    ):
        # Somente o principal online: capacidade base cresce sob pressão.
        status_holder["instances"] = [make_instance(8085, MAIN_PATH)]
        first = await resolve(router, body=body_with())  # ocupa primary (max=1)
        second = await resolve(router, body=body_with(user="outra conversa"))
        assert second.backend_port == 8085
        assert second.reason == "dynamic_capacity"
        assert router.in_flight(8085) == 2
        snapshot = router.backends_snapshot()[0]
        assert snapshot["max_parallel"] == 1
        assert snapshot["effective_parallel"] == 2
        assert snapshot["capacity_mode"] == "dynamic"
        await router.release(first.backend_port)
        await router.release(second.backend_port)

    @pytest.mark.asyncio
    async def test_dynamic_capacity_balances_and_prioritizes_primary(self, router):
        """Após preencher a base, cresce pela carga relativa; empate vai ao principal."""
        decisions = []
        for index in range(6):
            decisions.append(await resolve(
                router, body=body_with(user=f"conversa dinamica {index}")
            ))
        assert [d.backend_port for d in decisions] == [
            8085, 8086, 8087, 8085, 8086, 8087,
        ]
        assert decisions[3].reason == "dynamic_capacity"
        for decision in decisions:
            await router.release(decision.backend_id)

    @pytest.mark.asyncio
    async def test_slot_freed_during_wait_is_used(self, router, status_holder):
        status_holder["instances"] = [make_instance(8085, MAIN_PATH)]
        first = await resolve(router, body=body_with())

        async def free_soon():
            await asyncio.sleep(0.3)
            await router.release(first.backend_port)

        release_task = asyncio.create_task(free_soon())
        decision = await resolve(router, body=body_with(user="conversa 2"))
        await release_task
        assert decision.backend_port == 8085
        await router.release(decision.backend_port)

    @pytest.mark.asyncio
    async def test_dry_run_creates_nothing(self, router):
        decision = await resolve(router, body=body_with(tag="a1"), dry_run=True)
        assert isinstance(decision, RouteDecision)
        assert await router.sessions() == []
        assert router.in_flight(decision.backend_port) == 0

    @pytest.mark.asyncio
    async def test_no_primary_online_uses_redundant_backend(
        self, router, status_holder
    ):
        status_holder["instances"] = [make_instance(8086, AUX0_PATH)]
        decision = await resolve(router, body=body_with())
        assert decision.backend_port == 8086
        assert decision.internal_model == "aux0.gguf"
        assert decision.external_model == "main.gguf"
        assert decision.rewrite is True
        await router.release(decision.backend_id)

    @pytest.mark.asyncio
    async def test_backend_cooldown_skips_failed_primary_until_expiry(
        self, router, status_holder, clock
    ):
        primary_id = next(
            backend for backend in router.backends_snapshot()
            if backend["port"] == 8085
        )["backend_id"]
        await router.mark_backend_unavailable(
            primary_id, 300, reason="HTTP 429"
        )

        failover = await resolve(router, body=body_with())
        assert failover.backend_port != 8085
        assert next(
            backend for backend in router.backends_snapshot()
            if backend["port"] == 8085
        )["state"] == "cooldown"
        await router.release(failover.backend_id)

        clock.advance(seconds=301)
        recovered = await resolve(router, body=body_with(user="nova conversa"))
        assert recovered.backend_port == 8085
        await router.release(recovered.backend_id)

    @pytest.mark.asyncio
    async def test_max_parallel_respected_per_backend(
        self, router, proxy_config
    ):
        proxy_config.update_model_settings(MAIN_PATH, {"max_parallel_requests": 2})
        d1 = await resolve(router, body=body_with(user="c1"))
        d2 = await resolve(router, body=body_with(user="c2"))
        assert d1.backend_port == d2.backend_port == 8085
        assert router.in_flight(8085) == 2
        await router.release(8085)
        await router.release(8085)

    @pytest.mark.asyncio
    async def test_concurrent_resolves_stay_sticky_per_tag(
        self, router, proxy_config
    ):
        for path in (MAIN_PATH, AUX0_PATH, AUX1_PATH):
            proxy_config.update_model_settings(path, {"max_parallel_requests": 4})
        tags = ["a1", "a2", "a3"]
        decisions = await asyncio.gather(*[
            resolve(router, body=body_with(tag=tag))
            for tag in tags for _ in range(2)
        ])
        by_tag = {}
        for d in decisions:
            by_tag.setdefault(d.detected_tag, set()).add(d.backend_port)
            await router.release(d.backend_port)
        for tag, ports in by_tag.items():
            assert len(ports) == 1, f"tag {tag} usou mais de um backend: {ports}"

    @pytest.mark.asyncio
    async def test_admin_reassign_moves_session(self, router):
        """Reatribuir sempre prioriza o principal, mesmo que a sessao ja
        esteja fixada num secundario por o principal ter estado ocupado."""
        d_main = await resolve(router, body=body_with())  # ocupa o principal
        decision = await resolve(router, body=body_with(tag="a1"))
        assert decision.backend_port != 8085
        await router.release(d_main.backend_port)  # principal libera
        await router.release(decision.backend_port)
        new_decision = await router.reassign(decision.affinity_key)
        assert new_decision is not None
        assert new_decision.backend_port == 8085
        assert new_decision.reason == "reassign_admin"
        assert await router.reassign("nao-existe") is None
        await router.release(new_decision.backend_port)

    @pytest.mark.asyncio
    async def test_failover_prefers_local_then_cloud(
        self, router, proxy_config, status_holder
    ):
        """Após falha, locais elegíveis antecedem plataformas secundárias."""
        shared_port = 8317
        codex = make_platform_instance(
            port=shared_port, backend_id="platform:codex", provider="codex",
        )
        antigravity = make_platform_instance(
            port=shared_port,
            backend_id="platform:google-antigravity",
            model="Google Antigravity",
            provider="antigravity",
        )
        status_holder["instances"] = [
            make_instance(8085, MAIN_PATH),
            make_instance(8086, AUX0_PATH),
            codex,
            antigravity,
        ]
        proxy_config.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        proxy_config.update_platform_settings(
            "platform:google-antigravity",
            {"proxy_eligible": True, "default_model": "antigravity-31prolow.gguf"},
        )
        proxy_config.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        decision = await resolve(router, body=body_with(model="codex-56sol.gguf"))
        assert decision.backend_id == "platform:codex"
        await router.release(decision.backend_id)

        hop1 = await router.reassign(
            decision.affinity_key,
            exclude_backend_ids={"platform:codex"},
            reason="reassign_upstream_error",
        )
        assert hop1 is not None
        assert hop1.backend_type == "local"

        local_backend_ids = {
            router._backend_id(instance)
            for instance in status_holder["instances"]
            if instance.get("backend_type", "local") != "platform"
        }
        hop2 = await router.reassign(
            decision.affinity_key,
            exclude_backend_ids={"platform:codex", *local_backend_ids},
            reason="reassign_upstream_error",
        )
        assert hop2 is not None
        assert hop2.backend_id == "platform:google-antigravity"
        assert hop2.backend_type == "platform"
        assert hop2.internal_model == "antigravity-31prolow.gguf"

    @pytest.mark.asyncio
    async def test_release_accumulates_usage_tokens(self, router):
        decision = await resolve(router, body=body_with(tag="a1"))
        await router.release(
            decision.backend_port,
            affinity_key=decision.affinity_key,
            usage={"total_tokens": 123},
        )
        session = (await router.sessions())[0]
        assert session.tokens_processed == 123


# ---------------------------------------------------------------------------
# Reescrita SSE (task 05 — ADR-006)
# ---------------------------------------------------------------------------

async def _chunks(items):
    for item in items:
        yield item


async def collect_sse(chunks, external="main.gguf", usage_holder=None):
    out = b""
    async for piece in rewrite_sse_stream(_chunks(chunks), external, usage_holder):
        out += piece
    return out


class TestSseRewrite:
    @pytest.mark.asyncio
    async def test_whole_event_rewritten(self):
        event = b'data: {"model":"aux0.gguf","choices":[]}\n\n'
        out = await collect_sse([event])
        assert b'"model": "main.gguf"' in out or b'"model":"main.gguf"' in out
        assert b"aux0.gguf" not in out

    @pytest.mark.asyncio
    async def test_model_name_split_between_chunks(self):
        chunks = [b'data: {"model":"au', b'x0.gguf","choices":[]}\n\n']
        out = await collect_sse(chunks)
        assert b"aux0.gguf" not in out
        assert json.loads(out.split(b"data: ", 1)[1].split(b"\n", 1)[0])[
            "model"
        ] == "main.gguf"

    @pytest.mark.asyncio
    async def test_done_keepalive_and_blank_pass_untouched(self):
        chunks = [b"data: [DONE]\n", b": ping\n", b"\n"]
        out = await collect_sse(chunks)
        assert out == b"data: [DONE]\n: ping\n\n"

    @pytest.mark.asyncio
    async def test_invalid_json_fail_open(self):
        chunks = [b"data: {nao-e-json}\n"]
        assert await collect_sse(chunks) == b"data: {nao-e-json}\n"

    @pytest.mark.asyncio
    async def test_trailing_bytes_without_newline_are_flushed(self):
        chunks = [b'data: {"model":"aux0.gguf"}']
        out = await collect_sse(chunks)
        assert b"main.gguf" in out

    @pytest.mark.asyncio
    async def test_usage_captured_from_final_event(self):
        holder = {}
        chunks = [
            b'data: {"model":"aux0.gguf","usage":{"total_tokens":42}}\n',
            b"data: [DONE]\n",
        ]
        await collect_sse(chunks, usage_holder=holder)
        assert holder["usage"]["total_tokens"] == 42

    @pytest.mark.asyncio
    async def test_crlf_lines_preserved(self):
        chunks = [b'data: {"model":"aux0.gguf"}\r\n\r\n']
        out = await collect_sse(chunks)
        assert out.endswith(b"\r\n\r\n")
        assert b"main.gguf" in out

    def test_rewrite_json_model_non_stream(self):
        body = json.dumps(
            {"model": "aux0.gguf", "usage": {"total_tokens": 7}}
        ).encode()
        rewritten, usage = rewrite_json_model(body, "main.gguf")
        assert json.loads(rewritten)["model"] == "main.gguf"
        assert usage == {"total_tokens": 7}

    def test_rewrite_json_model_invalid_body_untouched(self):
        assert rewrite_json_model(b"<html>", "main.gguf") == (b"<html>", None)


# ---------------------------------------------------------------------------
# Separação plan_route / commit_route (Task 7)
# ---------------------------------------------------------------------------

class TestPlanAndCommit:
    @pytest.mark.asyncio
    async def test_required_context_includes_output_reserve(self, router):
        with pytest.raises(ProxyError) as exc_info:
            await router.plan_route(
                headers={},
                body=body_with(tag="reserved-output"),
                client_ip="127.0.0.1",
                user_agent="pytest",
                prompt_tokens=60_000,
                required_context=66_000,
                token_count_source="exact",
            )

        assert exc_info.value.status_code == 413
        assert exc_info.value.code == "context_too_large"

    @pytest.mark.asyncio
    async def test_exact_budget_is_preserved_in_decision(self, router):
        plan = await router.plan_route(
            headers={},
            body=body_with(tag="exact-budget"),
            client_ip="127.0.0.1",
            user_agent="pytest",
            prompt_tokens=50_000,
            required_context=53_536,
            token_count_source="exact",
        )

        assert plan.decision.prompt_tokens_estimated == 50_000
        assert plan.decision.required_context_tokens == 53_536
        assert plan.decision.token_count_source == "exact"

    @pytest.mark.asyncio
    async def test_commit_revalidates_required_context(
        self, router, status_holder
    ):
        plan = await router.plan_route(
            headers={},
            body=body_with(tag="shrink-after-plan"),
            client_ip="127.0.0.1",
            user_agent="pytest",
            prompt_tokens=40_000,
            required_context=50_000,
            token_count_source="exact",
        )
        planned = next(
            item for item in status_holder["instances"]
            if item["port"] == plan.decision.backend_port
        )
        planned["config"]["context_size"] = 49_000

        with pytest.raises(StaleRoutePlan):
            await router.commit_route(plan)

    @pytest.mark.asyncio
    async def test_plan_route_has_no_side_effects(self, router, tmp_path):
        body = body_with(tag="test-plan")
        plan = await router.plan_route(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        assert isinstance(plan, RoutePlan)
        assert plan.commit_token != ""
        # 1. plan_route() não altera sessões em memória
        assert await router.sessions() == []
        # 2. plan_route() não altera in_flight
        assert router.in_flight(plan.decision.backend_port) == 0
        # 3. plan_route() não salva arquivo de sessões
        sessions_file = tmp_path / "proxy_sessions.json"
        assert not sessions_file.exists()

    @pytest.mark.asyncio
    async def test_commit_route_applies_effects(self, router, tmp_path):
        body = body_with(tag="test-commit")
        plan = await router.plan_route(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        decision = await router.commit_route(plan)
        assert isinstance(decision, RouteDecision)
        assert decision.backend_port == plan.decision.backend_port
        # Efeitos aplicados: sessão criada e in_flight incrementado
        assert len(await router.sessions()) == 1
        assert router.in_flight(decision.backend_port) == 1

        await router.release(decision.backend_port)
        assert router.in_flight(decision.backend_port) == 0

    @pytest.mark.asyncio
    async def test_duplicate_commit_rejected(self, router):
        body = body_with(tag="test-dup")
        plan = await router.plan_route(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        decision = await router.commit_route(plan)
        await router.release(decision.backend_port)

        # Segundo commit com o mesmo plano deve falhar
        with pytest.raises(ProxyError) as exc_info:
            await router.commit_route(plan)
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "duplicate_commit"

    @pytest.mark.asyncio
    async def test_stale_plan_when_backend_removed(self, router, status_holder):
        body = body_with(tag="test-stale")
        plan = await router.plan_route(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        planned_port = plan.decision.backend_port

        # Remover o backend planejado antes do commit
        status_holder["instances"] = [
            inst for inst in status_holder["instances"] if inst["port"] != planned_port
        ]

        with pytest.raises(StaleRoutePlan) as exc_info:
            await router.commit_route(plan)
        assert exc_info.value.status_code == 503
        assert exc_info.value.code == "stale_route_plan"

    @pytest.mark.asyncio
    async def test_failure_before_commit_does_not_leak_capacity(self, router):
        body = body_with(tag="test-leak")
        plan = await router.plan_route(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        target_port = plan.decision.backend_port
        # Se ocorrer uma exceção na aplicação antes de commit_route, a capacidade não é vazada
        assert router.in_flight(target_port) == 0
        assert await router.sessions() == []

    @pytest.mark.asyncio
    async def test_resolve_wrapper_uses_plan_and_commit(self, router):
        body = body_with(tag="test-wrapper")
        decision = await router.resolve(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        assert isinstance(decision, RouteDecision)
        assert router.in_flight(decision.backend_port) == 1
        assert len(await router.sessions()) == 1


class TestPlanLargerWindow:
    @pytest.mark.asyncio
    async def test_plan_larger_window_is_side_effect_free(self, router):
        body = body_with(tag="test-pf-side-effect")
        plan = await router.plan_larger_window(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
            current_limit=32000,
            required_capabilities={"text"},
        )
        assert plan is not None
        assert router.in_flight(plan.decision.backend_port) == 0
        assert await router.sessions() == []

    @pytest.mark.asyncio
    async def test_plan_larger_window_returns_none_if_no_larger_candidate(self, router):
        body = body_with(tag="test-pf-none")
        plan = await router.plan_larger_window(
            headers={},
            body=body,
            client_ip="127.0.0.1",
            user_agent="pytest",
            current_limit=1000000,  # Maior que qualquer backend disponível
            required_capabilities={"text"},
        )
        assert plan is None
