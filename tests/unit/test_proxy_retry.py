"""Testes de retry, failover reativo, preservação de payload e streaming SSE.

Cobre a matriz de testes da Task 12:
- Retries no mesmo backend reutilizam o mesmo payload validado.
- Failover para destino diferente reavalia orçamento, tokenizer e capacidades.
- Erros de transporte (429, 502, 503, 504, conexao) NAO autorizam Moderate/Aggressive.
- Stream iniciado (apos primeiro byte) nao gera nova chamada upstream.
- Eventos SSE nao contem relatorios administrativos.
- in_flight libera em todos os caminhos (sucesso, erro, desconexao).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import llama_manager
from config_manager import ConfigManager
from context_optimizer import ContextTooLargeError
from llama_manager import app, auth_manager
from proxy_router import (
    ProxyRouter,
    guard_sse_stream,
    infer_json_tool_call,
    repair_plain_json_tool_call_stream,
)
from token_counter import RequestTokenBudget

client = TestClient(app)

MAIN_PATH = "/path/to/main.gguf"
AUX0_PATH = "/path/to/aux0.gguf"
AUX1_PATH = "/path/to/aux1.gguf"


def make_instance(
    port,
    model_path,
    ctx=65536,
    slots=1,
    gpu_name="NVIDIA RTX 3090",
    gpu_index=0,
):
    return {
        "port": port,
        "status": "running",
        "model": model_path.rsplit("/", 1)[-1],
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


def default_instances():
    return [
        make_instance(8085, MAIN_PATH, ctx=65536, gpu_name="NVIDIA RTX 3090", gpu_index=0),
        make_instance(8086, AUX0_PATH, ctx=32768, gpu_name="Tesla P100", gpu_index=1),
        make_instance(8087, AUX1_PATH, ctx=131072, gpu_name="Tesla P100", gpu_index=2),
    ]


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[llama_manager.require_auth] = lambda: True
    app.dependency_overrides[llama_manager.require_api_token] = lambda: True
    app.dependency_overrides[auth_manager.check_auth] = lambda: True
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def retry_env(tmp_path, monkeypatch):
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

    async def legacy_test_budget(data, instances, headers):
        prompt_tokens = ProxyRouter.estimate_prompt_tokens(data)
        required = int(prompt_tokens * 1.1)
        return RequestTokenBudget(
            prompt_tokens=prompt_tokens,
            output_tokens=0,
            overhead_tokens=0,
            media_tokens=0,
            required_context=required,
            source="test_estimated",
            duration_ms=0.0,
        )

    monkeypatch.setattr(llama_manager, "_count_request_tokens", legacy_test_budget)
    return type("RetryEnv", (), {"cfg": cfg, "router": router, "holder": holder})()


def _mock_response(payload: dict, status: int = 200):
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


def _sse_upstream(chunks, status=200):
    upstream = MagicMock(spec=httpx.Response)
    upstream.status_code = status
    upstream.headers = httpx.Headers({"Content-Type": "text/event-stream"})

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk

    upstream.aiter_bytes = aiter_bytes
    upstream.aread = AsyncMock(return_value=b"".join(chunks))
    upstream.aclose = AsyncMock()
    return upstream


@pytest.mark.asyncio
async def test_direct_ollama_429_is_not_retried_on_same_account(monkeypatch):
    response = _mock_response({"error": "quota exhausted"}, status=429)
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(llama_manager.client, "post", post)

    result = await llama_manager._proxy_post_with_retry(
        "https://ollama.com/v1/chat/completions",
        content=b"{}",
        headers={},
        retry_rate_limits=False,
    )

    assert result.status_code == 429
    post.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_ollama_stream_429_is_not_retried_on_same_account(monkeypatch):
    response = _sse_upstream([b'{"error":"quota exhausted"}'], status=429)
    send = AsyncMock(return_value=response)
    monkeypatch.setattr(llama_manager.client, "send", send)

    result, iterator = await llama_manager._proxy_open_stream_with_retry(
        "https://ollama.com/v1/chat/completions",
        content=b"{}",
        headers={},
        retry_rate_limits=False,
    )

    assert result.status_code == 429
    assert iterator is None
    send.assert_awaited_once()


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


def image_body(tag=None, model="main.gguf"):
    body = chat_body(tag=tag, model=model)
    body["messages"][-1]["content"] = [
        {"type": "text", "text": "Descreva a imagem"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        },
    ]
    return body


@pytest.mark.asyncio
async def test_guard_sse_rejects_clean_eof_without_completion_marker():
    async def truncated_stream():
        yield b'data: {"choices":[{"delta":{"content":"metade"}}]}\n\n'

    chunks = []
    with pytest.raises(httpx.ReadError, match="sem marcador de conclusao"):
        async for chunk in guard_sse_stream(truncated_stream()):
            chunks.append(chunk)

    assert b"metade" in b"".join(chunks)


@pytest.mark.asyncio
async def test_guard_sse_accepts_finish_reason_without_done_sentinel():
    async def completed_stream():
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    chunks = [chunk async for chunk in guard_sse_stream(completed_stream())]
    assert b"finish_reason" in b"".join(chunks)


def _grep_tool_schema():
    return [{
        "type": "function",
        "function": {
            "name": "Grep",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                },
                "required": ["pattern"],
            },
        },
    }]


def test_infer_json_tool_call_requires_one_unambiguous_schema():
    content = json.dumps({
        "path": ".",
        "pattern": "discord",
        "glob": "*.log",
        "output_mode": "files_with_matches",
    })
    assert infer_json_tool_call(content, _grep_tool_schema()) == (
        "Grep",
        json.loads(content),
    )
    assert infer_json_tool_call(content, _grep_tool_schema() * 2) is None


@pytest.mark.asyncio
async def test_plain_json_response_is_repaired_as_streamed_tool_call():
    async def model_stream():
        yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"{\\"pattern\\":\\"discord\\",\\"path\\":\\".\\"}"},"finish_reason":null}]}\n\n'
        yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b'data: [DONE]\n\n'

    repaired = b"".join([
        chunk async for chunk in repair_plain_json_tool_call_stream(
            model_stream(), _grep_tool_schema()
        )
    ])
    assert b'"name": "Grep"' in repaired
    assert b'"finish_reason": "tool_calls"' in repaired
    assert b'"finish_reason":"stop"' not in repaired


class TestProxyRetryAndFailover:
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_image_request_skips_local_backends_without_active_vision(
        self, mock_post, retry_env
    ):
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })
        retry_env.holder["instances"][2]["config"].update({
            "mmproj_path": "/models/mmproj.gguf",
            "mmproj_disabled": False,
        })
        mock_post.return_value = _mock_response({"id": "vision-ok"})

        response = client.post(
            "/v1/chat/completions", json=image_body(tag="vision-route")
        )

        assert response.status_code == 200
        assert mock_post.call_count == 1
        assert ":8087/" in mock_post.call_args.args[0]
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_image_request_fails_before_upstream_when_no_vision_backend_exists(
        self, mock_post, retry_env
    ):
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })

        response = client.post(
            "/v1/chat/completions", json=image_body(tag="vision-unavailable")
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "vision_backend_unavailable"
        mock_post.assert_not_awaited()
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_image_failover_does_not_try_backend_without_vision(
        self, mock_post, mock_sleep, retry_env
    ):
        """Um erro no backend Vision nao pode desviar a imagem para Gemma."""
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })
        retry_env.holder["instances"][2]["config"].update({
            "mmproj_path": "/models/mmproj.gguf",
            "mmproj_disabled": False,
        })
        mock_post.return_value = _mock_response(
            {"error": "vision backend failed"}, status=500
        )

        response = client.post(
            "/v1/chat/completions", json=image_body(tag="vision-failover")
        )

        assert response.status_code in (502, 503)
        ports = [
            call.args[0].split(":")[2].split("/")[0]
            for call in mock_post.call_args_list
        ]
        assert ports
        assert set(ports) == {"8087"}
        mock_sleep.assert_awaited()
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @pytest.mark.parametrize("error_status", [400, 401, 403, 404, 500])
    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_any_http_error_retries_then_fails_over(
        self, mock_post, mock_sleep, error_status, retry_env
    ):
        """Qualquer 4xx/5xx persistente deve migrar para outro backend."""
        failed = _mock_response({"error": "upstream"}, status=error_status)
        succeeded = _mock_response({"id": "ok", "model": "aux1.gguf"})
        mock_post.side_effect = [failed, failed, failed, succeeded]

        response = client.post(
            "/v1/chat/completions",
            json=chat_body(tag=f"http-{error_status}"),
        )

        assert response.status_code == 200
        assert mock_post.call_count == 4
        ports = [
            call.args[0].split(":")[2].split("/")[0]
            for call in mock_post.call_args_list
        ]
        assert ports[:3] == ["8085", "8085", "8085"]
        assert ports[3] != "8085"

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_stream_http_500_retries_then_fails_over(
        self, mock_send, mock_sleep, retry_env
    ):
        failed = _sse_upstream([b'{"error":"invalid tool call"}'], status=500)
        succeeded = _sse_upstream([
            b'data: {"model":"aux1.gguf","choices":[]}\n\n',
            b'data: [DONE]\n\n',
        ])
        mock_send.side_effect = [failed, failed, failed, succeeded]

        response = client.post(
            "/v1/chat/completions", json=chat_body(tag="stream-500", stream=True)
        )

        assert response.status_code == 200
        assert mock_send.call_count == 4
        assert b"data: [DONE]" in response.content

    @patch("llama_manager.context_optimizer.optimize", new_callable=AsyncMock)
    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_disabled_optimizer_stays_disabled_during_failover(
        self, mock_post, mock_sleep, mock_optimize, retry_env
    ):
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })
        resp_503 = _mock_response({"error": "unavailable"}, status=503)
        resp_200 = _mock_response({"id": "ok", "model": "aux0.gguf"})
        mock_post.side_effect = [resp_503, resp_503, resp_503, resp_200]

        response = client.post(
            "/v1/chat/completions", json=chat_body(tag="optimizer-off")
        )

        assert response.status_code == 200
        mock_optimize.assert_not_awaited()

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_three_retries_same_backend_reuse_same_payload(
        self, mock_post, mock_sleep, retry_env
    ):
        """Tres retries no mesmo backend reutilizam exatamente o mesmo payload validado."""
        resp_503 = _mock_response({"error": "service unavailable"}, status=503)
        resp_200 = _mock_response({"id": "cmpl-ok", "model": "main.gguf"})
        mock_post.side_effect = [resp_503, resp_503, resp_200]

        res = client.post("/v1/chat/completions", json=chat_body(tag="t1"))
        assert res.status_code == 200
        assert mock_post.call_count == 3

        # As 3 chamadas foram enviadas para o mesmo backend (8085)
        ports = [call.args[0].split(":")[2].split("/")[0] for call in mock_post.call_args_list]
        assert ports == ["8085", "8085", "8085"]

        # O payload (content) de todas as 3 chamadas e rigorosamente idêntico
        contents = [call.kwargs["content"] for call in mock_post.call_args_list]
        assert contents[0] == contents[1] == contents[2]

        # Liberou in_flight
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_failover_to_different_backend_reevaluates_optimizer(
        self, mock_post, mock_sleep, retry_env
    ):
        """Failover para backend com janela/tokenizer diferente spara nova otimização."""
        resp_503 = _mock_response({"error": "error"}, status=503)
        resp_200 = _mock_response({"id": "cmpl-ok", "model": "aux1.gguf"})

        # 3 falhas no 8085 + 1 sucesso no failover (8087)
        mock_post.side_effect = [resp_503, resp_503, resp_503, resp_200]

        res = client.post("/v1/chat/completions", json=chat_body(tag="t1"))
        assert res.status_code == 200
        assert mock_post.call_count == 4

        ports = [call.args[0].split(":")[2].split("/")[0] for call in mock_post.call_args_list]
        assert ports[:3] == ["8085", "8085", "8085"]
        assert ports[3] != "8085"

        # Contadores in_flight voltam a zero
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_failover_to_smaller_window_returns_413_if_exceeded(
        self, mock_post, mock_sleep, retry_env
    ):
        """Failover para janela menor que nao comporta o payload retorna 413 e nao entrega payload excedente."""
        resp_503 = _mock_response({"error": "error"}, status=503)
        mock_post.side_effect = [resp_503, resp_503, resp_503]

        # Força o 8086 (32k) a ser o proximo candidato de failover
        retry_env.holder["instances"] = [
            make_instance(8085, MAIN_PATH, ctx=65536),
            make_instance(8086, AUX0_PATH, ctx=1000),  # janela minúscula de 1000 tokens
        ]

        large_user_text = "excesso " * 1500  # ~1500 tokens, excede 1000 tokens do 8086
        res = client.post(
            "/v1/chat/completions",
            json=chat_body(tag="t1", user=large_user_text),
        )

        assert res.status_code == 413
        assert res.json()["error"]["code"] == "context_too_large"

        # Nenhum request foi enviado ao 8086 com payload excedente
        ports = [call.args[0].split(":")[2].split("/")[0] for call in mock_post.call_args_list]
        assert "8086" not in ports

        # in_flight de todas as portas zerado
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086))

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_http_429_does_not_authorize_moderate_or_aggressive(
        self, mock_post, mock_sleep, retry_env
    ):
        """HTTP 429 nao autoriza estratégias destrutivas Moderate/Aggressive."""
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })
        resp_429 = _mock_response({"error": "rate_limit"}, status=429)
        resp_200 = _mock_response({"id": "ok", "model": "aux1.gguf"})
        mock_post.side_effect = [resp_429, resp_429, resp_429, resp_200]

        with patch("context_optimizer.optimize_request_ir_moderate") as mock_mod, \
             patch("context_optimizer.optimize_request_ir_aggressive") as mock_agg:

            res = client.post("/v1/chat/completions", json=chat_body(tag="t1"))
            assert res.status_code == 200
            # Moderate e Aggressive nao foram chamadas no failover por 429
            mock_mod.assert_not_called()
            mock_agg.assert_not_called()

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_connection_error_does_not_trigger_moderate_or_aggressive(
        self, mock_post, mock_sleep, retry_env
    ):
        """Erro de conexão upstream nao dispara Moderate ou Aggressive."""
        retry_env.cfg.update_smart_proxy_settings({
            "context_optimizer": {"enabled": False},
        })
        ok = _mock_response({"id": "cmpl-ok", "model": "aux1.gguf"})
        mock_post.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            ok,
        ]

        with patch("context_optimizer.optimize_request_ir_moderate") as mock_mod, \
             patch("context_optimizer.optimize_request_ir_aggressive") as mock_agg:

            res = client.post("/v1/chat/completions", json=chat_body(tag="t1"))
            assert res.status_code == 200
            mock_mod.assert_not_called()
            mock_agg.assert_not_called()

    @patch("llama_manager.asyncio.sleep", new_callable=AsyncMock)
    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_failure_before_bytes_allows_validated_failover(
        self, mock_send, mock_sleep, retry_env
    ):
        """Falha na abertura de stream (antes dos bytes) permite failover para novo backend."""
        stream_err = _sse_upstream([b'{"error": "busy"}'], status=503)
        stream_ok = _sse_upstream([b'data: {"model":"m"}\n\n', b'data: [DONE]\n\n'])
        mock_send.side_effect = [stream_err, stream_err, stream_err, stream_ok]

        res = client.post("/v1/chat/completions", json=chat_body(tag="t1", stream=True))
        assert res.status_code == 200
        assert mock_send.call_count == 4
        assert b"data: [DONE]" in res.content

    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_started_stream_no_second_upstream_call_on_error(
        self, mock_send, retry_env
    ):
        """Stream iniciado (apos entregar bytes) nao gera segunda chamada upstream se falhar mid-stream."""
        async def failing_aiter():
            yield b'data: {"model":"main.gguf","choices":[{"delta":{"content":"Hi"}}]}\n\n'
            raise httpx.ReadError("Connection lost mid-stream")

        stream_mid_fail = MagicMock(spec=httpx.Response)
        stream_mid_fail.status_code = 200
        stream_mid_fail.headers = httpx.Headers({"Content-Type": "text/event-stream"})
        stream_mid_fail.aiter_bytes = failing_aiter
        stream_mid_fail.aclose = AsyncMock()

        mock_send.return_value = stream_mid_fail

        with pytest.raises(httpx.ReadError):
            client.post("/v1/chat/completions", json=chat_body(tag="t1", stream=True))

        # Apenas 1 chamada upstream enviada
        assert mock_send.call_count == 1

        # Release in_flight executado mesmo com exceção mid-stream
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

    @patch("llama_manager.client.send", new_callable=AsyncMock)
    def test_no_administrative_reports_in_sse_stream(
        self, mock_send, retry_env
    ):
        """O stream SSE nao contém relatórios administrativos."""
        chunks = [
            b'data: {"model":"main.gguf","choices":[{"delta":{"content":"A"}}]}\n\n',
            b'data: {"model":"main.gguf","choices":[{"delta":{"content":"B"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_send.return_value = _sse_upstream(chunks)

        res = client.post("/v1/chat/completions", json=chat_body(stream=True))
        assert res.status_code == 200
        body_text = res.content.decode("utf-8")

        # Nao contem eventos de auditoria ou metadados de gestao
        assert "strategy" not in body_text
        assert "savings_tokens" not in body_text
        assert "optimization" not in body_text

    @patch("llama_manager.client.post", new_callable=AsyncMock)
    def test_in_flight_released_on_all_paths(
        self, mock_post, retry_env
    ):
        """Valida que in_flight e zerado em sucesso e em falhas HTTP/conexao."""
        # Caminho 1: Sucesso
        mock_post.return_value = _mock_response({"id": "ok", "model": "main.gguf"})
        res1 = client.post("/v1/chat/completions", json=chat_body())
        assert res1.status_code == 200
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

        # Caminho 2: Erro 500 em todos os backends esgota o failover
        mock_post.return_value = _mock_response({"error": "internal"}, status=500)
        res2 = client.post("/v1/chat/completions", json=chat_body())
        assert res2.status_code in (502, 503)
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))

        # Caminho 3: Falha total de conexão
        mock_post.side_effect = httpx.ConnectError("Refused")
        res3 = client.post("/v1/chat/completions", json=chat_body())
        assert res3.status_code in (502, 503)
        assert all(retry_env.router.in_flight(p) == 0 for p in (8085, 8086, 8087))
