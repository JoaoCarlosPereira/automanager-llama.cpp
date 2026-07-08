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


def default_instances():
    return [
        make_instance(8085, MAIN_PATH, gpu_name="NVIDIA RTX 3090", gpu_index=0),
        make_instance(8086, AUX0_PATH, gpu_name="Tesla P100", gpu_index=1),
        make_instance(8087, AUX1_PATH, gpu_name="Tesla P100", gpu_index=2),
    ]


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[llama_manager.require_auth] = lambda: True
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
    )
    monkeypatch.setattr(llama_manager, "config_manager", cfg)
    monkeypatch.setattr(llama_manager, "proxy_router", router)
    monkeypatch.setattr(
        llama_manager.process_manager, "get_status", lambda: holder
    )
    return SimpleNamespace(cfg=cfg, router=router, holder=holder)


def _mock_response(payload: dict, port: int = 0):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.headers = httpx.Headers({"Content-Type": "application/json"})
    content = json.dumps(payload).encode()
    resp.content = content

    async def aiter_bytes():
        yield content

    resp.aiter_bytes = aiter_bytes
    return resp


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
    def test_secondary_by_real_name_bypasses_proxy(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "aux0.gguf"})
        response = client.post(
            "/v1/chat/completions", json=chat_body(model="aux0.gguf")
        )
        assert response.status_code == 200
        url = mock_post.call_args.args[0]
        assert "8086" in url
        # Sem sticky e sem reescrita: model interno permanece na resposta
        assert response.json()["model"] == "aux0.gguf"
        assert "x-automanager-backend" not in response.headers
        assert smart_env.router._sessions == {}

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_sticky_across_requests_through_handler(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        ports = []
        for _ in range(3):
            client.post("/v1/chat/completions", json=chat_body(tag="fixado"))
            ports.append(mock_post.call_args.args[0])
        assert len(set(ports)) == 1

    @patch("llama_manager.client.get", new_callable=AsyncMock)
    def test_v1_models_returns_only_primary(self, mock_get, smart_env):
        resp = MagicMock(spec=httpx.Response)
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "object": "list", "data": [{"id": "main.gguf", "object": "model"}]
        }
        mock_get.return_value = resp
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == 1
        assert data[0]["id"] == "main.gguf"

    def test_v1_models_empty_when_primary_offline(self, smart_env):
        smart_env.holder["instances"] = [make_instance(8086, AUX0_PATH)]
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_error_when_primary_offline_on_chat(self, smart_env):
        smart_env.holder["instances"] = [make_instance(8086, AUX0_PATH)]
        response = client.post("/v1/chat/completions", json=chat_body())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "primary_offline"

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
        mock_post.side_effect = [httpx.ConnectError("refused"), ok]
        response = client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert response.status_code == 200
        urls = [c.args[0] for c in mock_post.call_args_list]
        assert len(urls) == 2
        assert urls[0] != urls[1]

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_request_error_twice_returns_502_openai_format(
        self, mock_post, smart_env
    ):
        mock_post.side_effect = httpx.ConnectError("refused")
        response = client.post("/v1/chat/completions", json=chat_body(tag="a1"))
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "backend_unreachable"
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

    def test_backends_snapshot_shape(self, smart_env):
        backends = client.get("/proxy/backends").json()
        assert len(backends) == 3
        primary = next(b for b in backends if b["port"] == 8085)
        assert primary["role"] == "primary"
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
    def test_session_reassign_moves_backend(self, mock_post, smart_env):
        mock_post.return_value = _mock_response({"id": "x", "model": "m"})
        client.post("/v1/chat/completions", json=chat_body(tag="mover"))
        session = client.get("/proxy/sessions").json()[0]
        response = client.post(
            f"/proxy/sessions/{session['affinity_key']}/reassign"
        )
        assert response.status_code == 200
        assert response.json()["backend_port"] != session["backend_port"]
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
        assert payload["selected_backend"] in (8086, 8087)
        assert payload["internal_model"] in ("aux0.gguf", "aux1.gguf")
        assert payload["sticky_hit"] is False
        assert "reason" in payload
        # dry_run: nenhuma sessão criada, nenhum contador alterado
        assert client.get("/proxy/sessions").json() == []
        assert all(
            smart_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087)
        )

    def test_resolve_when_disabled_reports_flag(self, smart_env):
        smart_env.cfg.update_smart_proxy_settings({"enabled": False})
        response = client.post("/proxy/resolve", json=chat_body())
        assert response.json() == {"proxy_enabled": False}

    def test_primary_switch_affects_new_sessions_only(self, smart_env):
        first = client.post("/proxy/resolve", json=chat_body()).json()
        assert first["selected_backend"] == 8085
        client.post("/proxy/config", json={"primary_model_path": AUX1_PATH})
        second = client.post("/proxy/resolve", json=chat_body()).json()
        assert second["selected_backend"] == 8087
        assert second["external_model"] == "aux1.gguf"


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

        async def aiter_bytes():
            yield b'{"id": "legacy", "model": "main.gguf"}'

        mock_resp.aiter_bytes = aiter_bytes
        mock_post.return_value = mock_resp
        response = client.post("/v1/chat/completions", json=chat_body())
        assert response.status_code == 200
        assert response.json()["id"] == "legacy"
        assert "x-automanager-backend" not in response.headers
        assert smart_env.router._sessions == {}
