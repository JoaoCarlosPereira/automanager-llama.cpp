"""Testes unitários do fallback de janela do Context Optimizer (Task 09)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config_manager import ConfigManager
from context_optimizer import (
    LimitConfidence,
    ModelLimits,
    RequiredCapabilities,
    derive_required_capabilities,
)
from proxy_router import ProxyRouter, RoutePlan


def make_instance(
    port,
    model_path,
    ctx=65536,
    slots=1,
    backend_id=None,
    backend_type="local",
    mmproj=None,
):
    inst = {
        "port": port,
        "status": "running",
        "model": model_path.split("/")[-1] if model_path else "model",
        "model_path": model_path,
        "backend_id": backend_id or f"local:{model_path}",
        "backend_type": backend_type,
        "config": {
            "context_size": ctx,
            "parallel_slots": slots,
            "gpu_weights": [
                {
                    "index": 0,
                    "weight": 1.0,
                    "name": "GPU",
                    "active": True,
                    "is_main": True,
                    "device": "gpu",
                }
            ],
        },
    }
    if mmproj:
        inst["config"]["mmproj_path"] = mmproj
    return inst


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


MAIN_PATH = "models/main.gguf"
AUX0_PATH = "models/aux0.gguf"


@pytest.fixture
def proxy_config(tmp_path):
    cfg = ConfigManager(str(tmp_path / "automanager_config.json"))
    cfg.update_smart_proxy_settings(
        {"enabled": True, "primary_model_path": MAIN_PATH, "max_wait_seconds": 1}
    )
    return cfg


@pytest.fixture
def status_holder():
    return {
        "instances": [
            make_instance(8085, MAIN_PATH, ctx=65536),
            make_instance(8086, AUX0_PATH, ctx=131072),
        ]
    }


@pytest.fixture
def router(proxy_config, status_holder, tmp_path):
    return ProxyRouter(
        get_status=lambda: status_holder,
        config_manager=proxy_config,
        sessions_path=tmp_path / "proxy_sessions.json",
    )


@pytest.mark.asyncio
async def test_principal_insuficiente_migra_para_candidato_maior(router):
    """Principal (64k) é insuficiente, migra para candidato maior (128k)."""
    headers = {}
    body = {
        "model": "main.gguf",
        "messages": [{"role": "user", "content": "x" * 200000}],
    }

    req_caps = derive_required_capabilities(body)
    plan = await router.plan_larger_window(
        headers=headers,
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=req_caps,
    )

    assert plan is not None
    assert plan.decision.backend_port == 8086
    assert plan.decision.reason == "fallback_larger_window"


@pytest.mark.asyncio
async def test_candidato_sem_visao_rejeitado_para_request_com_imagem(proxy_config, tmp_path):
    """Candidato sem visão é rejeitado para request com imagem."""
    status_holder = {
        "instances": [
            make_instance(8085, MAIN_PATH, ctx=65536, mmproj="model.mmproj"),
            make_instance(8086, AUX0_PATH, ctx=131072, mmproj=None),  # sem visão
        ]
    }
    router = ProxyRouter(
        get_status=lambda: status_holder,
        config_manager=proxy_config,
        sessions_path=tmp_path / "proxy_sessions.json",
    )

    body = {
        "model": "main.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Descreva"},
                    {"type": "image_url", "image_url": {"url": "http://img.jpg"}},
                ],
            }
        ],
    }

    req_caps = derive_required_capabilities(body)
    assert req_caps.vision is True

    plan = await router.plan_larger_window(
        headers={},
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=req_caps,
    )

    # 8086 não tem visão, então deve retornar None
    assert plan is None


@pytest.mark.asyncio
async def test_candidato_sem_tools_rejeitado_para_request_com_tools(proxy_config, tmp_path):
    """Candidato sem suporte a ferramentas é rejeitado."""
    status_holder = {
        "instances": [
            make_instance(8085, MAIN_PATH, ctx=65536),
            make_platform_instance(9100, "platform:codex", "Codex"),
        ]
    }
    router = ProxyRouter(
        get_status=lambda: status_holder,
        config_manager=proxy_config,
        sessions_path=tmp_path / "proxy_sessions.json",
    )

    def eval_cand(inst):
        if inst["port"] == 9100:
            return (
                ModelLimits(
                    context_tokens=1000000,
                    max_output_tokens=4096,
                    source="platform_catalog",
                    confidence=LimitConfidence.KNOWN_PROVIDER,
                ),
                frozenset({"text"}),  # sem tools!
            )
        return (
            ModelLimits(
                context_tokens=65536,
                max_output_tokens=None,
                source="local",
                confidence=LimitConfidence.KNOWN_LOCAL,
            ),
            frozenset({"text", "tools"}),
        )

    body = {
        "model": "main.gguf",
        "messages": [{"role": "user", "content": "Rode"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }

    req_caps = derive_required_capabilities(body)
    assert req_caps.tools is True

    plan = await router.plan_larger_window(
        headers={},
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=req_caps,
        candidate_evaluator=eval_cand,
    )

    assert plan is None


@pytest.mark.asyncio
async def test_capacidade_desconhecida_nao_presumida(proxy_config, tmp_path):
    """Candidato com janela desconhecida (UNKNOWN) é descartado do fallback."""
    status_holder = {
        "instances": [
            make_instance(8085, MAIN_PATH, ctx=65536),
            make_platform_instance(9100, "platform:unk", "Unknown"),
        ]
    }
    router = ProxyRouter(
        get_status=lambda: status_holder,
        config_manager=proxy_config,
        sessions_path=tmp_path / "proxy_sessions.json",
    )

    def eval_cand(inst):
        if inst["port"] == 9100:
            return (
                ModelLimits(
                    context_tokens=None,
                    max_output_tokens=None,
                    source="platform_catalog",
                    confidence=LimitConfidence.UNKNOWN,
                ),
                frozenset({"text", "tools"}),
            )
        return (
            ModelLimits(
                context_tokens=65536,
                max_output_tokens=None,
                source="local",
                confidence=LimitConfidence.KNOWN_LOCAL,
            ),
            frozenset({"text", "tools"}),
        )

    body = {"model": "main.gguf", "messages": [{"role": "user", "content": "teste"}]}

    plan = await router.plan_larger_window(
        headers={},
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=derive_required_capabilities(body),
        candidate_evaluator=eval_cand,
    )

    assert plan is None


@pytest.mark.asyncio
async def test_backend_inicial_nao_recebe_cooldown_nem_failed_ids(router):
    """Garante que planejar janela maior não chama cooldown nem marca backend indisponível."""
    body = {"model": "main.gguf", "messages": [{"role": "user", "content": "teste"}]}
    plan = await router.plan_larger_window(
        headers={},
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=derive_required_capabilities(body),
    )
    assert plan is not None
    assert router.is_backend_disabled(8085) is False
    assert router._backend_cooldown_until("local:models/main.gguf") is None


@pytest.mark.asyncio
async def test_sticky_registra_apenas_backend_commitado(router):
    """Sticky session só deve ser criada quando commit_route for invocado."""
    body = {"model": "main.gguf", "messages": [{"role": "user", "content": "teste"}]}
    headers = {"X-Automanager-Session-Id": "sess-fallback-1"}

    plan = await router.plan_larger_window(
        headers=headers,
        body=body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=derive_required_capabilities(body),
    )
    assert plan is not None

    # Durante o planejamento, a sessão sticky ainda NÃO existe
    sessions_before = await router.sessions()
    assert not any(s.affinity_key == "sid:sess-fallback-1" for s in sessions_before)

    # Após o commit, a sessão é atrelada ao backend de destino (8086)
    decision = await router.commit_route(plan)
    assert decision.backend_port == 8086

    sessions_after = await router.sessions()
    sess = next(s for s in sessions_after if s.affinity_key == "sid:sess-fallback-1")
    assert sess.backend_port == 8086


@pytest.mark.asyncio
async def test_request_subsequente_menor_retorna_ao_principal(router):
    """Request subsequente pequeno na mesma sessão sticky retorna ao backend principal sem hysteresis."""
    headers = {"X-Automanager-Session-Id": "sess-return-test"}
    large_body = {"model": "main.gguf", "messages": [{"role": "user", "content": "x" * 200000}]}

    # 1. Requisição grande planeja janela maior (8086) e commita
    plan = await router.plan_larger_window(
        headers=headers,
        body=large_body,
        client_ip="127.0.0.1",
        user_agent="pytest",
        current_limit=65536,
        required_capabilities=derive_required_capabilities(large_body),
    )
    assert plan.decision.backend_port == 8086
    await router.commit_route(plan)

    # 2. Requisição subsequente pequena (cabe no principal 8085)
    small_body = {"model": "main.gguf", "messages": [{"role": "user", "content": "oi"}]}
    decision = await router.resolve(
        headers=headers,
        body=small_body,
        client_ip="127.0.0.1",
        user_agent="pytest",
    )

    # Retorna ao principal (8085) imediatamente
    assert decision.backend_port == 8085
    assert decision.reason == "sticky_return_primary"
