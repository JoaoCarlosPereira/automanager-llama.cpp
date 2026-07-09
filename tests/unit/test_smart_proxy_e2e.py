"""Task 09 — logs [proxy] e cenário ponta a ponta do PRD.

Reproduz os critérios de aceite: 3 backends (3090 + 2× P100), main + 3
subagentes tagueados distribuídos e estáveis, fallback em queda com log,
transparência do model e regressão zero com o modo desligado.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import llama_manager
from tests.unit.test_smart_proxy_routes import (  # noqa: F401 (fixtures)
    AUX0_PATH,
    AUX1_PATH,
    MAIN_PATH,
    chat_body,
    client,
    make_instance,
    override_auth,
    smart_env,
)


def _echo_backend(mock_post):
    """client.post fake: responde com o model recebido e a porta na resposta."""

    async def fake_post(url, content=b"", headers=None, timeout=None, **kwargs):
        sent = json.loads(content)
        port = int(url.split(":")[2].split("/")[0])
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = httpx.Headers({"Content-Type": "application/json"})
        payload = {
            "id": f"cmpl-{port}",
            "model": sent["model"],
            "usage": {"total_tokens": 10},
        }
        body = json.dumps(payload).encode()
        resp.content = body

        async def aiter_bytes():
            yield body

        resp.aiter_bytes = aiter_bytes
        return resp

    mock_post.side_effect = fake_post


SUBAGENTS = ["delphi-auditor", "sql-reviewer", "test-writer"]


class TestPrdScenario:
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_main_and_subagents_distributed_and_stable(
        self, mock_post, smart_env
    ):
        _echo_backend(mock_post)
        assignments = {}
        for round_num in range(3):
            for tag in [None] + SUBAGENTS:
                body = chat_body(tag=tag)
                response = client.post("/v1/chat/completions", json=body)
                assert response.status_code == 200
                # Transparência: toda resposta mostra o modelo principal
                assert response.json()["model"] == "main.gguf"
                port = int(response.headers["x-automanager-backend"])
                key = tag or "main"
                if round_num == 0:
                    assignments[key] = port
                else:
                    assert assignments[key] == port, (
                        f"{key} trocou de backend na rodada {round_num}"
                    )
        # main no principal; subagentes também preferem o principal quando livre
        assert assignments["main"] == 8085
        assert assignments["delphi-auditor"] == 8085
        assert assignments["sql-reviewer"] == 8085
        assert assignments["test-writer"] == 8085
        assert set(assignments.values()) == {8085}

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_backend_down_reassigns_with_log(self, mock_post, smart_env, caplog):
        _echo_backend(mock_post)
        # Ocupa o primary para forçar o subagente ao secundário
        import asyncio
        asyncio.run(smart_env.router.acquire(8085))
        try:
            client.post("/v1/chat/completions", json=chat_body(tag="sql-reviewer"))
            sessions = smart_env.router._sessions
            old_port = list(sessions.values())[0].backend_port
            assert old_port != 8085
            # Backend da sessão sai do ar (removido do status)
            smart_env.holder["instances"] = [
                inst for inst in smart_env.holder["instances"]
                if inst["port"] != old_port
            ]
            with caplog.at_level(logging.INFO, logger="automanager"):
                response = client.post(
                    "/v1/chat/completions", json=chat_body(tag="sql-reviewer")
                )
            assert response.status_code == 200
            assert response.json()["model"] == "main.gguf"
            new_port = int(response.headers["x-automanager-backend"])
            assert new_port != old_port
            log_text = caplog.text
            assert f"[proxy] backend {old_port} unavailable" in log_text
            assert "reason=backend_down" in log_text
            assert f"old_backend={old_port}" in log_text
        finally:
            asyncio.run(smart_env.router.release(8085))

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_disabled_mode_full_legacy_regression(self, mock_post, smart_env):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        _echo_backend(mock_post)
        # Pedir o principal pelo nome roteia direto sem sticky/reescrita
        response = client.post(
            "/v1/chat/completions", json=chat_body(tag="sql-reviewer")
        )
        assert response.status_code == 200
        assert "x-automanager-backend" not in response.headers
        assert smart_env.router._sessions == {}


class TestProxyLogs:
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_route_log_contains_all_fields(self, mock_post, smart_env, caplog):
        _echo_backend(mock_post)
        with caplog.at_level(logging.INFO, logger="automanager"):
            client.post("/v1/chat/completions", json=chat_body(tag="sql-reviewer"))
        route_lines = [
            r.message for r in caplog.records if "[proxy] route" in r.message
        ]
        assert len(route_lines) == 1
        line = route_lines[0]
        for field in (
            "external_model=main.gguf", "internal_model=", "backend=", "gpu=",
            "affinity_key=agent:sql-reviewer:", "sticky_hit=False",
            "reason=subagent_main_preference", "stream=False", "prompt_tokens_estimated=",
        ):
            assert field in line, f"campo ausente no log de rota: {field}"

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_new_session_log_emitted_once(self, mock_post, smart_env, caplog):
        _echo_backend(mock_post)
        with caplog.at_level(logging.INFO, logger="automanager"):
            client.post("/v1/chat/completions", json=chat_body(tag="novo"))
            client.post("/v1/chat/completions", json=chat_body(tag="novo"))
        new_lines = [
            r.message for r in caplog.records
            if "new sticky session" in r.message
        ]
        assert len(new_lines) == 1
        assert "affinity_key=agent:novo:" in new_lines[0]
        assert "selected_backend=" in new_lines[0]
        assert "reason=" in new_lines[0]
        sticky_lines = [
            r.message for r in caplog.records
            if "sticky_hit=True" in r.message
        ]
        assert len(sticky_lines) == 1

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_logs_never_leak_message_content(self, mock_post, smart_env, caplog):
        _echo_backend(mock_post)
        sentinel = "SEGREDO-DO-USUARIO-XYZ123"
        with caplog.at_level(logging.DEBUG, logger="automanager"):
            client.post(
                "/v1/chat/completions",
                json=chat_body(tag="privado", user=sentinel),
            )
        assert sentinel not in caplog.text
