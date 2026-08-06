"""Testes de rota do Modo Proxy Inteligente (tasks 04, 05 e 06).

Segue o padrão de tests/unit/test_multi_model_proxy.py: TestClient(app),
dependency_overrides para auth e patch do cliente httpx compartilhado.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import llama_manager
from config_manager import ConfigManager
from llama_manager import app, auth_manager
from proxy_router import ProxyRouter
from platform_manager import platform_model_listing_id

client = TestClient(app)

MAIN_PATH = "/path/to/main.gguf"
AUX0_PATH = "/path/to/aux0.gguf"
AUX1_PATH = "/path/to/aux1.gguf"


def make_instance(port, model_path, ctx=65536, slots=1, gpu_name="NVIDIA RTX 3090",
                  gpu_index=0):
    return {
        "port": port,
        "status": "running",
        "model": model_path.rsplit("/", 1)[-1],
        "model_path": model_path,
        "config": {
            "context_size": ctx,
            "parallel_slots": slots,
            "gpu_weights": [{
                "index": gpu_index, "weight": 1.0, "name": gpu_name,
                "active": True, "is_main": True, "device": "gpu",
            }],
        },
    }


def make_platform_instance(port=9100, backend_id="platform:codex", model="Codex"):
    return {
        "port": port,
        "status": "running",
        "model": model,
        "model_path": None,
        "backend_id": backend_id,
        "backend_type": "platform",
        "provider": "codex",
        "config": {
            "backend_id": backend_id,
            "backend_type": "platform",
            "provider": "codex",
            "proxy_eligible": True,
            "max_parallel_requests": 1,
        },
    }


def default_instances():
    return [
        make_instance(8085, MAIN_PATH, gpu_name="NVIDIA RTX 3090", gpu_index=0),
        make_instance(8086, AUX0_PATH, gpu_name="Tesla P100", gpu_index=1),
        make_instance(8087, AUX1_PATH, gpu_name="Tesla P100", gpu_index=2),
    ]


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[llama_manager.require_auth] = lambda: True
    app.dependency_overrides[llama_manager.require_api_token] = lambda: True
    app.dependency_overrides[auth_manager.check_auth] = lambda: True
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def smart_env(tmp_path, monkeypatch):
    """Ambiente com proxy ligado: config isolada + router novo + 3 instâncias."""
    cfg = ConfigManager(str(tmp_path / "automanager_config.json"))
    cfg.update_smart_proxy_settings({
        "enabled": True,
        "primary_model_path": MAIN_PATH,
        "max_wait_seconds": 1,
    })
    holder = {"instances": default_instances()}
    router = ProxyRouter(
        get_status=lambda: holder,
        config_manager=cfg,
        sessions_path=tmp_path / "proxy_sessions.json",
        context_limit_resolver=llama_manager._platform_model_context_limit,
        requested_primary_resolver=llama_manager._requested_primary_instance,
    )
    monkeypatch.setattr(llama_manager, "config_manager", cfg)
    monkeypatch.setattr(llama_manager, "proxy_router", router)
    monkeypatch.setattr(
        llama_manager.process_manager, "get_status", lambda: holder
    )
    return SimpleNamespace(cfg=cfg, router=router, holder=holder)


def _mock_response(payload: dict, port: int = 0, status: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.headers = httpx.Headers({"Content-Type": "application/json"})
    content = json.dumps(payload).encode()
    resp.content = content
    resp.json.return_value = payload

    async def aiter_bytes():
        yield content

    resp.aiter_bytes = aiter_bytes
    return resp


def _models_response(model_ids):
    return _mock_response({
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "test"}
            for model_id in model_ids
        ],
    })


def chat_body(tag=None, user="Oi", model="main.gguf", stream=False):
    system = f"[AGENT:{tag}] Voce ajuda." if tag else "Voce ajuda."
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if stream:
        body["stream"] = True
    return body


# ---------------------------------------------------------------------------
# Task 04 — desvio /v1, filtro /v1/models, transparência não-stream
# ---------------------------------------------------------------------------

class TestSmartRouting:
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_primary_model_routed_with_internal_rewrite(
        self, mock_post, smart_env
    ):
        import asyncio
        asyncio.run(smart_env.router.acquire(8085))  # primary ocupado -> overflow
        try:
            mock_post.return_value = _mock_response(
                {"id": "cmpl-1", "model": "aux0.gguf"}
            )
            response = client.post(
                "/v1/chat/completions", json=chat_body(tag="sql-reviewer")
            )
            assert response.status_code == 200
            url = mock_post.call_args.args[0]
            assert "8086" in url  # secundário least-busy (empate -> menor porta aux)
            sent = json.loads(mock_post.call_args.kwargs["content"])
            assert sent["model"] == "aux0.gguf"  # model interno no encaminhamento
            # Transparência: resposta externa mostra o principal
            assert response.json()["model"] == "main.gguf"
            assert response.headers["x-automanager-backend"] == "8086"
            assert response.headers["x-automanager-backend-model"] == "aux0.gguf"
        finally:
            asyncio.run(smart_env.router.release(8085))

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_unknown_extra_fields_preserved(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        body = chat_body(tag="a1")
        body["min_p"] = 0.05
        body["extra_body"] = {"custom": 1}
        client.post("/v1/chat/completions", json=body)
        sent = json.loads(mock_post.call_args.kwargs["content"])
        assert sent["min_p"] == 0.05
        assert sent["extra_body"] == {"custom": 1}

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_requested_local_model_becomes_dynamic_primary(
        self, mock_post, smart_env
    ):
        mock_post.return_value = _mock_response({"id": "x", "model": "aux0.gguf"})
        response = client.post(
            "/v1/chat/completions", json=chat_body(model="aux0.gguf")
        )
        assert response.status_code == 200
        url = mock_post.call_args.args[0]
        assert "8086" in url
        # O modelo invocado vira principal, sem reescrita do nome externo.
        assert response.json()["model"] == "aux0.gguf"
        assert response.headers["x-automanager-backend"] == "8086"
        assert response.headers["x-automanager-backend-id"] == (
            "local:/path/to/aux0.gguf"
        )
        assert next(iter(smart_env.router._sessions.values())).backend_port == 8086

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_sticky_across_requests_through_handler(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        ports = []
        for _ in range(3):
            client.post("/v1/chat/completions", json=chat_body(tag="fixado"))
            ports.append(mock_post.call_args.args[0])
        assert len(set(ports)) == 1

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_lists_all_instances_when_proxy_enabled(self, mock_get, smart_env):
        """Clientes externos (Cursor) precisam ver todos os modelos em /v1/models."""

        def side_effect(url, *args, **kwargs):
            if "8085" in url:
                return _models_response(["main.gguf"])
            if "8086" in url:
                return _models_response(["aux0.gguf"])
            return _models_response(["aux1.gguf"])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        ids = [item["id"] for item in response.json()["data"]]
        assert ids == ["main.gguf", "aux0.gguf", "aux1.gguf"]

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_includes_aux_when_primary_offline(self, mock_get, smart_env):
        smart_env.holder["instances"] = [make_instance(8086, AUX0_PATH)]

        def side_effect(url, *args, **kwargs):
            if "8086" in url:
                return _models_response(["aux0.gguf"])
            return _models_response([])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["data"]] == ["aux0.gguf"]

    def test_v1_models_503_when_no_instances(self, smart_env):
        smart_env.holder["instances"] = []
        response = client.get("/v1/models")
        assert response.status_code == 503

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_primary_offline_fails_over_on_chat(self, mock_post, smart_env):
        smart_env.holder["instances"] = [make_instance(8086, AUX0_PATH)]
        mock_post.return_value = _mock_response(
            {"id": "x", "model": "aux0.gguf"}
        )
        response = client.post("/v1/chat/completions", json=chat_body())
        assert response.status_code == 200
        assert "8086" in mock_post.call_args.args[0]
        assert response.json()["model"] == "main.gguf"

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_in_flight_returns_to_zero_after_success_and_error(
        self, mock_post, smart_env
    ):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_request_error_reassigns_once_then_succeeds(
        self, mock_post, smart_env
    ):
        ok = _mock_response({"id": "x", "model": "m"})
        mock_post.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            ok,
        ]
        response = client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert response.status_code == 200
        urls = [c.args[0] for c in mock_post.call_args_list]
        assert len(urls) == 4
        assert urls[0] == urls[1] == urls[2]
        assert urls[2] != urls[3]

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_request_error_all_backends_exhausted(
        self, mock_post, smart_env
    ):
        mock_post.side_effect = httpx.ConnectError("refused")
        response = client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert response.status_code in (502, 503)
        assert response.json()["error"]["code"] in (
            "backend_unreachable", "no_backend",
        )
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_http_429_failovers_to_another_backend(
        self, mock_post, mock_sleep, smart_env
    ):
        limited = _mock_response({"error": "rate_limit"}, status=429)
        ok = _mock_response({"id": "x", "model": "m"})
        # 3 retries no mesmo backend + 1 sucesso no failover
        mock_post.side_effect = [limited, limited, limited, ok]
        response = client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert response.status_code == 200
        urls = [c.args[0] for c in mock_post.call_args_list]
        assert len(urls) == 4
        assert urls[0] == urls[1] == urls[2]
        assert urls[3] != urls[0]
        session = client.get("/proxy/sessions").json()[0]
        assert f":{session['backend_port']}/" in urls[3]

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_stream_http_502_failovers_before_client(
        self, mock_send, mock_sleep, smart_env
    ):
        err = MagicMock(spec=httpx.Response)
        err.status_code = 502
        err.headers = httpx.Headers({})
        err.aread = AsyncMock(return_value=b"")
        err.aclose = AsyncMock()
        ok = _sse_upstream([b'data: {"model":"m"}\n\ndata: [DONE]\n\n'])
        # 3 aberturas no primario + 1 sucesso no failover
        mock_send.side_effect = [err, err, err, ok]
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(tag="a1", stream=True),
        )
        assert response.status_code == 200
        assert mock_send.call_count == 4
        assert b"data: [DONE]" in response.content
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )


# ---------------------------------------------------------------------------
# Task 05 — streaming SSE ponta a ponta
# ---------------------------------------------------------------------------

def _sse_upstream(chunks):
    upstream = MagicMock()
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk

    upstream.aiter_bytes = aiter_bytes
    upstream.aclose = AsyncMock()
    return upstream


class TestSmartStreaming:
    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_stream_rewrites_model_and_keeps_done(self, mock_send, smart_env):
        import asyncio
        asyncio.run(smart_env.router.acquire(8085))  # primary ocupado -> overflow
        try:
            chunks = [
                b'data: {"model":"aux0.gguf","choices":[{"delta":{"content":"ol"}}]}\n\n',
                b'data: {"model":"au', b'x0.gguf","usage":{"total_tokens":9}}\n\n',
                b"data: [DONE]\n\n",
            ]
            mock_send.return_value = _sse_upstream(chunks)
            response = client.post(
                "/v1/chat/completions", json=chat_body(tag="sql-reviewer", stream=True)
            )
            assert response.status_code == 200
            body = response.content
            assert b"aux0.gguf" not in body
            assert body.count(b'"model": "main.gguf"') == 2
            assert b"data: [DONE]" in body
            assert response.headers["x-automanager-backend"] in ("8086", "8087")
            # usage do evento final alimenta tokens_processed da sessão
            sessions = smart_env.router._sessions
            assert list(sessions.values())[0].tokens_processed == 9
        finally:
            asyncio.run(smart_env.router.release(8085))

    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_stream_to_primary_passes_bytes_raw(self, mock_send, smart_env):
        chunks = [b'data: {"model":"main.gguf","choices":[]}\n\n', b"data: [DONE]\n\n"]
        mock_send.return_value = _sse_upstream(chunks)
        response = client.post(
            "/v1/chat/completions", json=chat_body(stream=True)
        )
        assert response.status_code == 200
        assert response.content == b"".join(chunks)
        assert response.headers["x-automanager-backend"] == "8085"

    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_stream_releases_slot_on_completion(self, mock_send, smart_env):
        mock_send.return_value = _sse_upstream([b"data: [DONE]\n\n"])
        client.post("/v1/chat/completions", json=chat_body(tag="a2", stream=True))
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )


# ---------------------------------------------------------------------------
# Task 06 — endpoints administrativos /proxy/* e /models/proxy
# ---------------------------------------------------------------------------

ADMIN_ROUTES = [
    ("POST", "/proxy/config", {"enabled": True}),
    ("POST", "/models/proxy", {"model_path": "m.gguf", "proxy_eligible": True}),
    ("GET", "/proxy/status", None),
    ("GET", "/proxy/backends", None),
    ("POST", "/proxy/backends/8086/enable", None),
    ("POST", "/proxy/backends/8086/disable", None),
    ("GET", "/proxy/sessions", None),
    ("DELETE", "/proxy/sessions", None),
    ("DELETE", "/proxy/sessions/algum", None),
    ("POST", "/proxy/sessions/algum/reassign", None),
    ("POST", "/proxy/resolve", {"model": "x", "messages": []}),
]


class TestAdminEndpoints:
    def test_all_admin_routes_require_auth(self, smart_env):
        app.dependency_overrides[llama_manager.require_auth] = lambda: False
        try:
            for method, url, body in ADMIN_ROUTES:
                response = client.request(method, url, json=body)
                assert response.status_code == 401, f"{method} {url}"
        finally:
            app.dependency_overrides[llama_manager.require_auth] = lambda: True

    def test_proxy_config_persists_and_status_reflects(self, smart_env):
        response = client.post(
            "/proxy/config",
            json={"enabled": True, "primary_model_path": MAIN_PATH},
        )
        assert response.status_code == 200
        status = client.get("/proxy/status").json()
        assert status["enabled"] is True
        assert status["exposed_model"] == "main.gguf"
        assert status["primary"]["port"] == 8085

    def test_proxy_config_accepts_platform_primary_backend(self, smart_env):
        response = client.post(
            "/proxy/config",
            json={"enabled": True, "primary_backend_id": "platform:codex"},
        )
        assert response.status_code == 200
        payload = response.json()["smart_proxy"]
        assert payload["primary_backend_id"] == "platform:codex"
        assert client.get("/proxy/status").json()["primary_backend_id"] == "platform:codex"

    def test_proxy_config_rejects_unknown_primary(self, smart_env):
        response = client.post(
            "/proxy/config", json={"primary_model_path": "/nao/existe.gguf"}
        )
        assert response.status_code == 400
        assert "desconhecido" in response.json()["detail"]

    def test_models_proxy_marks_not_eligible(self, smart_env):
        response = client.post(
            "/models/proxy",
            json={"model_path": AUX0_PATH, "proxy_eligible": False},
        )
        assert response.status_code == 200
        backends = client.get("/proxy/backends").json()
        aux0 = next(b for b in backends if b["port"] == 8086)
        assert aux0["state"] == "not_eligible"

    def test_models_proxy_accepts_platform_backend_id(self, smart_env):
        response = client.post(
            "/models/proxy",
            json={"backend_id": "platform:codex", "proxy_eligible": True},
        )
        assert response.status_code == 200
        config = smart_env.cfg.get_config()
        assert config["platform_configs"]["platform:codex"]["proxy_eligible"] is True
        assert "platform:codex" not in config.get("model_configs", {})

    def test_backends_snapshot_shape(self, smart_env):
        backends = client.get("/proxy/backends").json()
        assert len(backends) == 3
        primary = next(b for b in backends if b["port"] == 8085)
        assert primary["role"] == "primary"
        assert primary["backend_type"] == "local"
        assert primary["gpu"] == "NVIDIA RTX 3090 #0"
        assert primary["ctx_per_slot"] == 65536
        aux1 = next(b for b in backends if b["port"] == 8087)
        assert aux1["gpu"] == "Tesla P100 #2"
        assert aux1["role"] == "secondary"

    def test_disable_enable_backend_cycle(self, smart_env):
        client.post("/proxy/backends/8086/disable")
        resolve = client.post(
            "/proxy/resolve", json=chat_body(tag="novo-agente")
        ).json()
        assert resolve["selected_backend"] != 8086
        client.post("/proxy/backends/8086/enable")
        backends = client.get("/proxy/backends").json()
        assert next(b for b in backends if b["port"] == 8086)["state"] == "online"

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_session_delete_and_404(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        client.post("/v1/chat/completions", json=chat_body(tag="apagar"))
        sessions = client.get("/proxy/sessions").json()
        assert len(sessions) == 1
        key = sessions[0]["affinity_key"]
        assert client.delete("/proxy/sessions/inexistente").status_code == 404
        assert client.delete(f"/proxy/sessions/{key}").status_code == 200
        assert client.get("/proxy/sessions").json() == []

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_sessions_clear_all(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        client.post("/v1/chat/completions", json=chat_body(tag="a2"))
        assert len(client.get("/proxy/sessions").json()) == 2
        resp = client.delete("/proxy/sessions")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 2
        assert client.get("/proxy/sessions").json() == []

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_session_reassign_moves_backend(self, mock_post, smart_env):
        """Reatribuir sempre prioriza o principal — a sessao criada com o
        principal desabilitado deve voltar pra ele assim que reabilitado."""
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        client.post("/proxy/backends/8085/disable")
        client.post("/v1/chat/completions", json=chat_body(tag="mover"))
        session = client.get("/proxy/sessions").json()[0]
        assert session["backend_port"] != 8085
        client.post("/proxy/backends/8085/enable")
        response = client.post(
            f"/proxy/sessions/{session['affinity_key']}/reassign"
        )
        assert response.status_code == 200
        assert response.json()["backend_port"] == 8085
        assert (
            client.post("/proxy/sessions/nao-existe/reassign").status_code == 404
        )

    def test_resolve_contract_and_no_side_effects(self, smart_env):
        response = client.post(
            "/proxy/resolve", json=chat_body(tag="sql-reviewer")
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["proxy_enabled"] is True
        assert payload["external_model"] == "main.gguf"
        assert payload["detected_tag"] == "sql-reviewer"
        assert payload["affinity_key"].startswith("agent:sql-reviewer:")
        assert payload["selected_backend"] == 8085
        assert payload["internal_model"] == "main.gguf"
        assert payload["sticky_hit"] is False
        assert payload["reason"] == "subagent_main_preference"
        # dry_run: nenhuma sessão criada, nenhum contador alterado
        assert client.get("/proxy/sessions").json() == []
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )

    def test_resolve_when_disabled_reports_flag(self, smart_env):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        response = client.post("/proxy/resolve", json=chat_body())
        assert response.json() == {"proxy_enabled": False}

    def test_resolve_returns_sidecar_backend_for_platform_primary(self, smart_env):
        smart_env.holder["instances"] = [make_platform_instance()]
        smart_env.cfg.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        smart_env.cfg.update_smart_proxy_settings(
            {"enabled": True, "primary_backend_id": "platform:codex"}
        )
        response = client.post(
            "/proxy/resolve", json=chat_body(model="codex-pro")
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["selected_backend"] == 9100
        assert payload["backend_id"] == "platform:codex"
        assert payload["backend_type"] == "platform"
        assert payload["internal_model"] == "codex-pro"

    def test_requested_model_overrides_configured_primary(self, smart_env):
        first = client.post("/proxy/resolve", json=chat_body()).json()
        assert first["selected_backend"] == 8085
        client.post("/proxy/config", json={"primary_model_path": AUX1_PATH})
        second = client.post("/proxy/resolve", json=chat_body()).json()
        assert second["selected_backend"] == 8085
        dynamic = client.post(
            "/proxy/resolve", json=chat_body(model="aux1.gguf")
        ).json()
        assert dynamic["selected_backend"] == 8087
        assert dynamic["external_model"] == "aux1.gguf"


# ---------------------------------------------------------------------------
# Modo desligado — comportamento legado intacto (task 04)
# ---------------------------------------------------------------------------

class TestProxyDisabled:
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_disabled_mode_uses_legacy_flow(
        self, mock_post, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"Content-Type": "application/json"})
        mock_resp.content = b'{"id": "legacy", "model": "main.gguf"}'

        async def aiter_bytes():
            yield b'{"id": "legacy", "model": "main.gguf"}'

        mock_resp.aiter_bytes = aiter_bytes
        mock_post.return_value = mock_resp
        response = client.post("/v1/chat/completions", json=chat_body())
        assert response.status_code == 200
        assert response.json()["id"] == "legacy"
        assert "x-automanager-backend" not in response.headers
        assert smart_env.router._sessions == {}


class TestHybridV1Availability:
    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_operator_flow_start_platform_exposes_sidecar_model(
        self, mock_get, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        active = {"value": False}

        class FakePlatformManager:
            def start_backend(self, backend_id, sidecar):
                active["value"] = True
                return {
                    "backend_id": backend_id,
                    "backend_type": "platform",
                    "provider": "codex",
                    "active": True,
                    "status": "running",
                    "sidecar_port": 9100,
                }

            def runtime_states(self):
                return [{
                    "backend_id": "platform:codex",
                    "backend_type": "platform",
                    "provider": "codex",
                    "active": active["value"],
                    "status": "running" if active["value"] else "detected",
                    "sidecar_port": 9100 if active["value"] else None,
                }]

            def active_instances(self):
                return [make_platform_instance()] if active["value"] else []

            def catalog(self):
                return []

            def get(self, backend_id):
                return {"backend_id": backend_id}

        sidecar = MagicMock()
        sidecar.status.return_value = {"status": "running", "port": 9100}
        monkeypatch.setattr(llama_manager, "platform_manager", FakePlatformManager())
        monkeypatch.setattr(llama_manager, "cliproxy_sidecar", sidecar)

        start = client.post("/platforms/platform:codex/start")
        assert start.status_code == 200

        def side_effect(url, *args, **kwargs):
            if "9100" in url:
                return _models_response(["codex-pro"])
            return _models_response([])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["data"]] == [
            platform_model_listing_id("codex-pro", "codex")
        ]

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_returns_local_models_when_no_platform_active(
        self, mock_get, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(llama_manager.platform_manager, "active_instances", lambda: [])

        def side_effect(url, *args, **kwargs):
            if "8085" in url:
                return _models_response(["main.gguf"])
            if "8086" in url:
                return _models_response(["aux0.gguf"])
            return _models_response(["aux1.gguf"])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["data"]] == [
            "main.gguf",
            "aux0.gguf",
            "aux1.gguf",
        ]

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_merges_active_sidecar_models(
        self, mock_get, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(
            llama_manager.platform_manager,
            "active_instances",
            lambda: [make_platform_instance()],
        )

        def side_effect(url, *args, **kwargs):
            if "9100" in url:
                return _models_response(["codex-pro"])
            if "8085" in url:
                return _models_response(["main.gguf"])
            return _models_response([])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        ids = [item["id"] for item in response.json()["data"]]
        assert ids == ["main.gguf", platform_model_listing_id("codex-pro", "codex")]

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_reports_context_for_each_requested_model(
        self, mock_get, smart_env, monkeypatch
    ):
        codex = make_platform_instance(
            port=8317, backend_id="platform:codex", model="Codex"
        )
        antigravity = make_platform_instance(
            port=8317,
            backend_id="platform:google-antigravity",
            model="Google Antigravity",
        )
        antigravity["provider"] = "antigravity"
        antigravity["config"].update({
            "backend_id": "platform:google-antigravity",
            "provider": "antigravity",
        })
        smart_env.holder["instances"] = [codex, antigravity]
        monkeypatch.setattr(
            llama_manager,
            "_platform_model_catalog_cache",
            {
                "codex": {
                    "gpt-5.6-luna": {
                        "id": "gpt-5.6-luna",
                        "context_length": 372_000,
                        "max_completion_tokens": 128_000,
                    }
                },
                "antigravity": {
                    "gemini-3.1-pro-low": {
                        "id": "gemini-3.1-pro-low",
                        "inputTokenLimit": 1_048_576,
                        "outputTokenLimit": 65_535,
                    }
                },
            },
        )
        monkeypatch.setattr(
            llama_manager, "_platform_model_catalog_loaded_at", 10**18
        )
        mock_get.return_value = _models_response(
            ["gpt-5.6-luna", "gemini-3.1-pro-low"]
        )

        response = client.get("/v1/models")
        assert response.status_code == 200
        by_id = {item["id"]: item for item in response.json()["data"]}
        luna = by_id["codex-56luna.gguf"]
        antigravity_model = by_id["antigravity-31prolow.gguf"]
        assert luna["context_length"] == 372_000
        assert luna["meta"]["n_ctx"] == 372_000
        assert antigravity_model["context_length"] == 1_048_576
        assert antigravity_model["meta"]["n_ctx"] == 1_048_576

        detail = client.get("/v1/models/codex-56luna.gguf")
        assert detail.status_code == 200
        assert detail.json()["context_length"] == 372_000
        assert detail.json()["meta"]["n_ctx"] == 372_000

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_dedupes_local_and_sidecar_ids(
        self, mock_get, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(
            llama_manager.platform_manager,
            "active_instances",
            lambda: [make_platform_instance()],
        )

        def side_effect(url, *args, **kwargs):
            if "9100" in url:
                return _models_response(["main.gguf", "codex-pro"])
            if "8085" in url:
                return _models_response(["main.gguf"])
            return _models_response([])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        ids = [item["id"] for item in response.json()["data"]]
        assert ids == ["main.gguf", platform_model_listing_id("codex-pro", "codex")]

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_chat_with_sidecar_model_id_forwards_to_sidecar_port(
        self, mock_post, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(
            llama_manager.platform_manager,
            "active_instances",
            lambda: [make_platform_instance()],
        )
        codex_listing = platform_model_listing_id("codex-pro", "codex")
        mock_post.return_value = _mock_response({"id": "chatcmpl-1", "model": "codex-pro"})

        response = client.post(
            "/v1/chat/completions", json=chat_body(model="codex-pro")
        )

        assert response.status_code == 200
        assert "9100" in mock_post.call_args.args[0]
        assert response.json()["model"] == "codex-pro"

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_sidecar_models_failure_does_not_break_local_listing(
        self, mock_get, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(
            llama_manager.platform_manager,
            "active_instances",
            lambda: [make_platform_instance()],
        )

        def side_effect(url, *args, **kwargs):
            if "9100" in url:
                raise httpx.ConnectError("refused")
            if "8085" in url:
                return _models_response(["main.gguf"])
            return _models_response([])

        mock_get.side_effect = side_effect
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["data"]] == ["main.gguf"]

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_smart_proxy_platform_primary_forwards_to_sidecar(
        self, mock_post, mock_get, smart_env
    ):
        llama_manager.clear_platform_listing_registry()
        smart_env.holder["instances"] = [make_platform_instance()]
        smart_env.cfg.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        smart_env.cfg.update_smart_proxy_settings(
            {"enabled": True, "primary_backend_id": "platform:codex"}
        )
        mock_get.return_value = _models_response(["codex-pro"])
        mock_post.return_value = _mock_response({"id": "chatcmpl-1", "model": "codex-pro"})
        codex_listing = platform_model_listing_id("codex-pro", "codex")

        response = client.post(
            "/v1/chat/completions", json=chat_body(model=codex_listing)
        )

        assert response.status_code == 200
        assert "9100" in mock_post.call_args.args[0]
        sent = json.loads(mock_post.call_args.kwargs["content"])
        assert sent["model"] == "codex-pro"

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_platform_rate_limit_fails_over_to_configured_provider_default(
        self, mock_post, mock_get, smart_env
    ):
        """429 no Codex abre o circuito e migra para o modelo padrão Antigravity."""
        llama_manager.clear_platform_listing_registry()
        codex = make_platform_instance(
            port=8317, backend_id="platform:codex", model="Codex"
        )
        antigravity = make_platform_instance(
            port=8317,
            backend_id="platform:google-antigravity",
            model="Google Antigravity",
        )
        antigravity["provider"] = "antigravity"
        antigravity["config"].update({
            "backend_id": "platform:google-antigravity",
            "provider": "antigravity",
        })
        smart_env.holder["instances"] = [codex, antigravity]
        smart_env.cfg.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        smart_env.cfg.update_platform_settings(
            "platform:google-antigravity",
            {
                "proxy_eligible": True,
                "default_model": "antigravity-default",
            },
        )
        smart_env.cfg.update_smart_proxy_settings(
            {"enabled": True, "primary_backend_id": "platform:codex"}
        )
        mock_get.return_value = _models_response(
            ["codex-pro", "antigravity-default"]
        )
        limited = _mock_response({"error": "rate_limit"}, status=429)
        ok = _mock_response({"id": "chatcmpl-1", "model": "antigravity-default"})
        mock_post.side_effect = [limited, limited, limited, ok, ok]

        response = client.post(
            "/v1/chat/completions",
            json=chat_body(model=platform_model_listing_id("codex-pro", "codex")),
        )

        assert response.status_code == 200
        assert mock_post.call_count == llama_manager._PROXY_MAX_ATTEMPTS + 1
        sent_models = [
            json.loads(call.kwargs["content"])["model"]
            for call in mock_post.call_args_list
        ]
        assert sent_models[:3] == ["codex-pro"] * 3
        assert sent_models[3] == "antigravity-default"
        assert response.headers["x-automanager-backend-id"] == (
            "platform:google-antigravity"
        )
        codex = next(
            backend for backend in smart_env.router.backends_snapshot()
            if backend["backend_id"] == "platform:codex"
        )
        assert codex["state"] == "cooldown"

        # Uma nova requisição evita o Codex enquanto o circuito estiver aberto.
        second = client.post(
            "/v1/chat/completions",
            json=chat_body(model=platform_model_listing_id("codex-pro", "codex")),
        )
        assert second.status_code == 200
        assert mock_post.call_count == llama_manager._PROXY_MAX_ATTEMPTS + 2
        assert json.loads(mock_post.call_args.kwargs["content"])["model"] == (
            "antigravity-default"
        )

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_oversized_context_is_rejected_before_backend(self, mock_post, smart_env):
        body = chat_body(user="x" * (70_000 * 4))

        response = client.post("/v1/chat/completions", json=body)

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "context_too_large"
        mock_post.assert_not_called()

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_smart_proxy_rebuilds_stale_registry_on_alias_miss(
        self, mock_post, mock_get, smart_env
    ):
        """Registry montado com catálogo incompleto do sidecar (ex.: boot sem
        rede) não pode encaminhar o alias cru: reconstrói e resolve o modelo real."""
        llama_manager.clear_platform_listing_registry()
        # Snapshot velho: só um modelo não relacionado ficou registrado.
        llama_manager.register_platform_model_listings("gpt-5.4-mini", "codex")
        smart_env.holder["instances"] = [make_platform_instance()]
        smart_env.cfg.update_platform_settings(
            "platform:codex", {"proxy_eligible": True}
        )
        smart_env.cfg.update_smart_proxy_settings(
            {"enabled": True, "primary_backend_id": "platform:codex"}
        )
        # Catálogo atual do sidecar já tem o modelo que faltava.
        mock_get.return_value = _models_response(["gpt-5.4-mini", "gpt-5.6-sol"])
        mock_post.return_value = _mock_response(
            {"id": "chatcmpl-1", "model": "gpt-5.6-sol"}
        )
        sol_listing = platform_model_listing_id("gpt-5.6-sol", "codex")
        assert sol_listing == "codex-56sol.gguf"

        response = client.post(
            "/v1/chat/completions", json=chat_body(model=sol_listing)
        )

        assert response.status_code == 200
        assert "9100" in mock_post.call_args.args[0]
        sent = json.loads(mock_post.call_args.kwargs["content"])
        assert sent["model"] == "gpt-5.6-sol"
        mock_get.assert_called()

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_platform_chat_requires_api_token(
        self, mock_post, smart_env, monkeypatch
    ):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        monkeypatch.setattr(llama_manager.platform_manager, "runtime_states", lambda: [])
        monkeypatch.setattr(
            llama_manager.platform_manager,
            "active_instances",
            lambda: [make_platform_instance()],
        )
        app.dependency_overrides.pop(llama_manager.require_api_token, None)

        response = client.post(
            "/v1/chat/completions", json=chat_body(model="codex-pro")
        )
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Invalid API Key"
        mock_post.assert_not_called()

        app.dependency_overrides[llama_manager.require_api_token] = lambda: True


class TestStartupSpeedRanking:
    @pytest.mark.asyncio
    async def test_local_benchmark_waits_until_model_is_ready(self, monkeypatch):
        unavailable = httpx.Response(
            503, request=httpx.Request("GET", "http://127.0.0.1:8085/health")
        )
        ready = httpx.Response(
            200, request=httpx.Request("GET", "http://127.0.0.1:8085/health")
        )
        get = AsyncMock(side_effect=[unavailable, ready])
        sleep = AsyncMock()
        monkeypatch.setattr(llama_manager.client, "get", get)
        monkeypatch.setattr(llama_manager.asyncio, "sleep", sleep)

        result = await llama_manager._wait_proxy_benchmark_ready({
            "backend_type": "local",
            "backend_id": "local:8085",
            "port": 8085,
        })

        assert result is True
        assert get.await_count == 2
        sleep.assert_awaited_once_with(1)

    def test_cache_is_reused_only_for_same_model_fingerprint(self, tmp_path):
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"model-v1")
        targets = [{
            "key": f"local:{model_path}",
            "backend_id": "local:8085",
            "backend_type": "local",
            "provider": None,
            "port": 8085,
            "model": "model.gguf",
            "model_path": str(model_path),
        }]
        fingerprint = llama_manager._proxy_benchmark_fingerprint(targets)
        cache_path = tmp_path / "proxy_backend_benchmarks.json"
        payload = {
            "schema": llama_manager._PROXY_BENCHMARK_SCHEMA,
            "fingerprint": fingerprint,
            "measured_at": "2026-08-06T12:00:00+00:00",
            "latencies_ms": {f"local:{model_path}": 123.0},
        }

        llama_manager._save_proxy_benchmark_cache(payload, str(cache_path))

        assert llama_manager._load_proxy_benchmark_cache(
            fingerprint, str(cache_path)
        ) == payload
        assert llama_manager._load_proxy_benchmark_cache(
            "different", str(cache_path)
        ) is None

        model_path.write_bytes(b"model-v2-with-different-size")
        changed = llama_manager._proxy_benchmark_fingerprint(targets)
        assert changed != fingerprint
        assert llama_manager._load_proxy_benchmark_cache(
            changed, str(cache_path)
        ) is None

    def test_platform_default_model_changes_fingerprint(self):
        base = {
            "key": "platform:codex",
            "backend_id": "platform:codex",
            "backend_type": "platform",
            "provider": "codex",
            "port": 8317,
            "model_path": None,
        }

        sol = llama_manager._proxy_benchmark_fingerprint([
            {**base, "model": "gpt-5.6-sol"}
        ])
        luna = llama_manager._proxy_benchmark_fingerprint([
            {**base, "model": "gpt-5.6-luna"}
        ])

        assert sol != luna
