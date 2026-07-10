import asyncio
import hashlib
import json
import socket
import os
import signal
import threading
import time
import re
import glob
import logging
import html
from urllib.parse import unquote
import uvicorn
import httpx
from typing import List, Optional, Tuple, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config_manager import (
    ConfigManager,
    TokenManager,
    AuthManager,
    SESSION_IDLE_SECONDS,
    normalize_model_path,
    CURSOR_COMPATIBLE_ALIAS_NAMES,
)
from proxy_router import (
    ProxyError,
    ProxyRouter,
    rewrite_json_model,
    rewrite_sse_stream,
)
from log_manager import LogManager, logger
from llama_server_bin import get_llama_server_bin, list_llama_server_bins
from process_manager import ProcessManager, OOMWatchdog, SERVER_PORT
from model_manager import ModelScanner, DownloadManager, _is_projector_filename
from cliproxy_auth import CLIProxyAuthManager
from platform_manager import (
    CLIProxySidecarError,
    CLIProxySidecarManager,
    PlatformIntegrationError,
    PlatformIntegrationManager,
    clear_platform_listing_registry,
    filter_models_for_provider,
    lookup_platform_bare_id,
    platform_client_facing_model,
    platform_listing_registry_populated,
    platform_model_listing_entry,
    platform_model_listing_id,
    platform_provider_for_listing,
    register_platform_model_listings,
    resolve_platform_listing_model,
    should_skip_platform_model_listing,
)
from version_manager import check_for_updates
from schemas import (
    BATCH_SIZE_PRESETS,
    CACHE_TYPE_PRESETS,
    DEFAULT_CACHE_TYPE,
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_FLASH_ATTN_ENABLED,
    GPUWeight,
    StartRequest,
    DeleteRequest,
    DownloadRequest,
    DownloadCancelRequest,
    RenameRequest,
    SetDefaultRequest,
    SetMmprojRequest,
    SetThinkingRequest,
    SetLlamaBinRequest,
    ProxyConfigRequest,
    SetModelProxyRequest,
    CLIProxyAuthStartRequest,
    CLIProxyAuthCallbackRequest,
    ModelAliasRequest,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_BATCH_SIZE,
    TURBOQUANT_PRESETS,
    TURBOQUANT_DEFAULT_CACHE_K,
    TURBOQUANT_DEFAULT_CACHE_V,
    TURBOQUANT_CACHE_K_PRESETS,
    TURBOQUANT_CACHE_V_PRESETS,
)
from gpu_manager import GPUManager, reasoning_cli_args, mtp_cli_args, compute_server_ctx_size
from paths import CONFIG_PATH, INSTALL_ROOT, update_models_dir, reload_module_paths

# Version tracking
_DASHBOARD_JS_V = "4.2.17"  # Alias/Copiar na aba plataforma

MANAGER_PORT = 8000
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 5

# httpx auto-decompresses bodies; forwarding these headers breaks browsers.
_PROXY_OMIT_HEADERS = frozenset({
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
})


def _filter_proxy_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _PROXY_OMIT_HEADERS
    }


_PROXY_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_PROXY_MAX_ATTEMPTS = 3


def _proxy_retry_delay(attempt: int) -> float:
    return min(0.25 * (2 ** attempt), 2.0)


def _is_retryable_upstream_status(status_code: int) -> bool:
    return status_code in _PROXY_RETRYABLE_STATUSES


async def _proxy_post_with_retry(
    url: str,
    *,
    content: bytes,
    headers: Dict[str, str],
    backend_label: str = "",
) -> httpx.Response:
    """Reenvia POST ao backend com backoff em falhas transitórias."""
    last_exc: Optional[Exception] = None
    for attempt in range(_PROXY_MAX_ATTEMPTS):
        try:
            resp = await client.post(url, content=content, headers=headers, timeout=None)
            if (
                _is_retryable_upstream_status(resp.status_code)
                and attempt < _PROXY_MAX_ATTEMPTS - 1
            ):
                logger.warning(
                    "[proxy] %s HTTP %s — retry %d/%d",
                    backend_label or url,
                    resp.status_code,
                    attempt + 1,
                    _PROXY_MAX_ATTEMPTS,
                )
                await asyncio.sleep(_proxy_retry_delay(attempt))
                continue
            return resp
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt >= _PROXY_MAX_ATTEMPTS - 1:
                raise
            logger.warning(
                "[proxy] %s indisponivel (%s) — retry %d/%d",
                backend_label or url,
                exc,
                attempt + 1,
                _PROXY_MAX_ATTEMPTS,
            )
            await asyncio.sleep(_proxy_retry_delay(attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("proxy post retry exhausted")


async def _proxy_open_stream_with_retry(
    url: str,
    *,
    content: bytes,
    headers: Dict[str, str],
    backend_label: str = "",
) -> httpx.Response:
    """Abre stream SSE no backend; retenta antes de entregar bytes ao cliente."""
    last_exc: Optional[Exception] = None
    for attempt in range(_PROXY_MAX_ATTEMPTS):
        try:
            upstream = await client.send(
                client.build_request(
                    "POST", url, content=content, headers=headers, timeout=None
                ),
                stream=True,
            )
            if (
                _is_retryable_upstream_status(upstream.status_code)
                and attempt < _PROXY_MAX_ATTEMPTS - 1
            ):
                logger.warning(
                    "[proxy] %s stream HTTP %s — retry %d/%d",
                    backend_label or url,
                    upstream.status_code,
                    attempt + 1,
                    _PROXY_MAX_ATTEMPTS,
                )
                await upstream.aread()
                await upstream.aclose()
                await asyncio.sleep(_proxy_retry_delay(attempt))
                continue
            return upstream
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt >= _PROXY_MAX_ATTEMPTS - 1:
                raise
            logger.warning(
                "[proxy] %s stream indisponivel (%s) — retry %d/%d",
                backend_label or url,
                exc,
                attempt + 1,
                _PROXY_MAX_ATTEMPTS,
            )
            await asyncio.sleep(_proxy_retry_delay(attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("proxy stream retry exhausted")


def _sse_stream_complete(body: bytes) -> bool:
    """True quando o corpo SSE parece completo (OpenAI-compat)."""
    if not body:
        return False
    return b"data: [DONE]" in body or body.rstrip().endswith(b"[DONE]")


async def _collect_upstream_stream_with_retry(
    url: str,
    *,
    content: bytes,
    headers: Dict[str, str],
    backend_label: str = "",
) -> Tuple[int, List[bytes], Dict[str, str]]:
    """Bufferiza o stream inteiro no AutoManager; retenta se falhar antes de [DONE]."""
    last_exc: Optional[Exception] = None
    for attempt in range(_PROXY_MAX_ATTEMPTS):
        upstream: Optional[httpx.Response] = None
        try:
            upstream = await client.send(
                client.build_request(
                    "POST", url, content=content, headers=headers, timeout=None
                ),
                stream=True,
            )
            upstream_headers = dict(upstream.headers)
            if (
                _is_retryable_upstream_status(upstream.status_code)
                and attempt < _PROXY_MAX_ATTEMPTS - 1
            ):
                logger.warning(
                    "[proxy] %s buffered stream HTTP %s — retry %d/%d",
                    backend_label or url,
                    upstream.status_code,
                    attempt + 1,
                    _PROXY_MAX_ATTEMPTS,
                )
                await upstream.aread()
                await upstream.aclose()
                await asyncio.sleep(_proxy_retry_delay(attempt))
                continue

            chunks: List[bytes] = []
            async for piece in upstream.aiter_bytes():
                chunks.append(piece)
            body = b"".join(chunks)
            status = upstream.status_code
            await upstream.aclose()
            upstream = None

            if (
                status == 200
                and not _sse_stream_complete(body)
                and attempt < _PROXY_MAX_ATTEMPTS - 1
            ):
                logger.warning(
                    "[proxy] %s buffered stream incompleto (%d bytes) — retry %d/%d",
                    backend_label or url,
                    len(body),
                    attempt + 1,
                    _PROXY_MAX_ATTEMPTS,
                )
                await asyncio.sleep(_proxy_retry_delay(attempt))
                continue
            return status, chunks, upstream_headers
        except httpx.RequestError as exc:
            last_exc = exc
            if upstream is not None:
                await upstream.aclose()
            if attempt >= _PROXY_MAX_ATTEMPTS - 1:
                raise
            logger.warning(
                "[proxy] %s buffered stream indisponivel (%s) — retry %d/%d",
                backend_label or url,
                exc,
                attempt + 1,
                _PROXY_MAX_ATTEMPTS,
            )
            await asyncio.sleep(_proxy_retry_delay(attempt))
        finally:
            if upstream is not None:
                await upstream.aclose()
    if last_exc:
        raise last_exc
    raise RuntimeError("proxy buffered stream retry exhausted")


def _inject_ui_base_tag(html: str, port: int) -> str:
    """Rewrite llama-server index HTML for reverse-proxy under /ui/{port}/."""
    base_tag = f'<base href="/ui/{port}/">'
    html, count = re.subn(
        r"(<head[^>]*>)",
        rf"\1{base_tag}",
        html,
        count=1,
        flags=re.IGNORECASE,
    )
    if count == 0:
        html = f"<head>{base_tag}</head>{html}"
    return html.replace(
        "base: new URL('.', location).pathname.slice(0, -1)",
        f'base: "/ui/{port}"',
    )


def _cfg_tip(text: str) -> str:
    """Balão explicativo exibido ao passar o mouse sobre um campo de configuração."""
    return (
        f'<div class="cfg-tip pointer-events-none absolute left-0 top-full mt-1.5 w-64 '
        f'max-w-[min(16rem,calc(100vw-2rem))] p-3 rounded-xl text-ui-body leading-relaxed '
        f'text-slate-200 bg-slate-900 border border-slate-600/80 shadow-2xl">'
        f'{html.escape(text)}</div>'
    )


def _escape_js_attr(value: str) -> str:
    """Escape a value for safe embedding inside a single-quoted JavaScript string within an HTML attribute.

    This prevents XSS via model paths containing single quotes, angle brackets, etc.
    """
    return (value
            .replace('\\', '\\\\')
            .replace("'", "\\'")
            .replace('<', '\\u003c')
            .replace('>', '\\u003e')
            .replace('&', '\\u0026')
            .replace('\n', '\\n')
            .replace('\r', '\\r'))


_CFG_FIELD = 'cfg-field group/tip relative space-y-2'

app = FastAPI(title="Automanager Llama.cpp")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rate limiter for login endpoint
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
app.state.limiter = limiter
# Sem o handler registrado, exceder o limite vira 500 em vez de 429.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Shared HTTP client for proxying
client = httpx.AsyncClient()

config_manager = ConfigManager()
log_manager = LogManager()
gpu_manager = GPUManager()
process_manager = ProcessManager(
    config_manager, None, gpu_manager, log_manager
)
token_manager = TokenManager(config_manager)
process_manager.token_mgr = token_manager
auth_manager = AuthManager(config_manager, token_manager)

model_scanner = ModelScanner(config_manager, process_manager)
platform_manager = PlatformIntegrationManager(config_manager)
cliproxy_sidecar = CLIProxySidecarManager(platform_manager, log_manager=log_manager)
cliproxy_auth_manager = CLIProxyAuthManager(platform_manager)
download_mgr = DownloadManager()
proxy_router = ProxyRouter(
    get_status=lambda: _hybrid_status(),
    config_manager=config_manager,
    sessions_path=os.path.join(
        os.path.dirname(CONFIG_PATH) or ".", "proxy_sessions.json"
    ),
)
oom_watchdog = OOMWatchdog(
    process_manager, config_manager, gpu_manager, log_manager
)

shutdown_event = threading.Event()


def require_auth(request: Request) -> bool:
    """FastAPI dependency wrapper so Request injection works reliably."""
    return auth_manager.check_auth(request)


def require_api_token(request: Request) -> bool:
    """OpenAI-compatible routes: Bearer API token only (same as llama-server --api-key)."""
    return auth_manager.check_api_token(request)


def _openai_auth_error() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Invalid API Key",
                "type": "authentication_error",
                "code": 401,
            }
        },
    )

# Context and Batch presets for the UI
CONTEXT_PRESET_VALUES = [4096, 8192, 16384, 32768, 65536, 131072, "custom"]
CONTEXT_K_MULTIPLIER = 1024


def _invalidate_models_cache():
    """Helper to force model list refresh on next scan."""
    model_scanner._last_scan_time = 0


def _local_instance_view(instance: Dict[str, Any]) -> Dict[str, Any]:
    view = dict(instance)
    view.setdefault("backend_type", "local")
    port = view.get("port")
    if port is not None:
        view.setdefault("backend_id", f"local:{port}")
    return view


def _hybrid_status() -> Dict[str, Any]:
    status = dict(process_manager.get_status() or {})
    local_instances = [
        _local_instance_view(inst) for inst in status.get("instances", []) or []
    ]
    platform_states = platform_manager.runtime_states()
    platform_instances = platform_manager.active_instances()
    status["local_instances"] = local_instances
    status["platforms"] = platform_states
    status["sidecar"] = cliproxy_sidecar.status()
    status["instances"] = local_instances + platform_instances
    return status


def _model_catalog_response() -> Dict[str, Any]:
    result = dict(model_scanner.scan() or {})
    auth_status = cliproxy_auth_manager.list_status()
    platforms = []
    for item in platform_manager.catalog():
        entry = dict(item)
        provider = entry.get("provider")
        entry["cliproxy_auth"] = auth_status.get(provider or "", {})
        platforms.append(entry)
    result["platforms"] = platforms
    return result


@app.post("/api/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, req: Dict[str, str]):
    username = req.get("username")
    password = req.get("password")
    result = auth_manager.authenticate(username, password)
    if not result:
        raise HTTPException(status_code=401, detail="Credenciais invalidas")
    session_token = result["token"]
    force_change = result.get("force_password_change", False)
    response = JSONResponse({"status": "ok", "force_password_change": force_change})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_IDLE_SECONDS,
        path="/",
    )
    return response


@app.post("/api/auth/logout")
async def logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token:
        auth_manager.logout(session_token)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(key="session_token", path="/")
    return response


@app.post("/api/auth/change-password")
async def change_password(req: Dict[str, str], authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    current = req.get("current") or req.get("username")
    new_pw = req.get("new") or req.get("password")
    if auth_manager.change_password(current, new_pw):
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Senha atual incorreta")


@app.get("/status")
async def get_status(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return _hybrid_status()


@app.get("/llama-bins")
async def get_llama_bins(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return list_llama_server_bins()


@app.get("/metrics")
async def get_metrics(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return gpu_manager.get_metrics()


@app.get("/models")
async def list_models(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return _model_catalog_response()


@app.post("/models/dir")
async def set_models_dir(req: Dict[str, str], authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    new_dir = req.get("models_dir")
    if not new_dir:
        raise HTTPException(status_code=400, detail="Diretorio invalido ou inacessivel")
    try:
        paths = update_models_dir(new_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Diretorio invalido ou inacessivel")
    reload_module_paths()
    model_scanner.models_dir = paths.models_dir
    download_mgr.models_dir = paths.models_dir
    _invalidate_models_cache()
    return _model_catalog_response()


@app.post("/start")
async def start_model(req: StartRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)

    try:
        total_layers = req.total_layers
        if not total_layers:
            total_layers = gpu_manager.detect_model_layers(req.path)
    except Exception:
        total_layers = 0

    base_settings = {
        "context_size": req.context_size,
        "parallel_slots": req.parallel_slots,
        "batch_size": req.batch_size,
        "ubatch_size": req.ubatch_size,
        "cache_type_k": req.cache_type_k,
        "cache_type_v": req.cache_type_v,
        "numa_enabled": req.numa_enabled,
        "flash_attn_enabled": req.flash_attn_enabled,
        "threads": req.threads,
        "threads_batch": req.threads_batch,
        "mmproj_path": req.mmproj_path,
        "mmproj_disabled": req.mmproj_disabled,
        "split_mode": req.split_mode,
        "auto_balance": req.auto_balance,
        "thinking_enabled": req.thinking_enabled,
        "mtp_enabled": req.mtp_enabled,
        "mtp_draft_tokens": req.mtp_draft_tokens,
        "total_layers": total_layers if total_layers else 0,
        "pinned_fields": req.pinned_fields or {},
    }
    if req.llama_server_bin:
        base_settings["llama_server_bin"] = req.llama_server_bin
    if req.turboquant_preset:
        base_settings["turboquant_preset"] = req.turboquant_preset

    if req.auto_balance:
        # Smart calibration respeita pins do usuário (GPU e campos); auto-balance clássico não.
        if not req.smart_calibration:
            for weight in req.gpu_weights:
                weight.pinned = False
        return process_manager.start_auto_balance(req)

    config_manager.update_model_settings(
        req.path,
        {
            **base_settings,
            "gpu_weights": [w.model_dump() for w in req.gpu_weights],
            "auto_balance_profile": (
                req.auto_balance_profile
                if req.auto_balance_profile is not None
                else False
            ),
        },
    )
    result = process_manager.start(
        model_path=req.path,
        gpu_weights=req.gpu_weights,
        context_size=req.context_size,
        mmproj_path=req.mmproj_path,
        mmproj_disabled=req.mmproj_disabled,
        split_mode=req.split_mode,
        parallel_slots=req.parallel_slots,
        batch_size=req.batch_size,
        ubatch_size=req.ubatch_size,
        cache_type_k=req.cache_type_k,
        cache_type_v=req.cache_type_v,
        numa_enabled=req.numa_enabled,
        flash_attn_enabled=req.flash_attn_enabled,
        threads=req.threads,
        threads_batch=req.threads_batch,
        thinking_enabled=req.thinking_enabled,
        mtp_enabled=req.mtp_enabled,
        mtp_draft_tokens=req.mtp_draft_tokens,
        total_layers=total_layers,
        cpu_enabled=req.cpu_enabled,
        port=req.port,
        llama_server_bin=req.llama_server_bin,
    )
    _invalidate_models_cache()
    return result


@app.post("/stop")
async def stop_model(port: Optional[int] = None, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    # stop() bloqueia até a porta liberar (até ~10s); fora do event loop para
    # não congelar /metrics, /logs e demais requisições concorrentes.
    await asyncio.to_thread(process_manager.stop, port)
    return {"message": "Parado"}


async def _fetch_sidecar_models(port: int) -> List[Dict[str, Any]]:
    """Lista modelos expostos pelo sidecar CLIProxyAPI, se estiver online."""
    try:
        resp = await client.get(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json().get("data") or []
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        logger.debug("Falha ao listar modelos do sidecar na porta %s: %s", port, exc)
        return []


def _platform_detail_payload(backend_id: str) -> Dict[str, Any]:
    item = platform_manager.get(backend_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Integracao de plataforma nao encontrada")
    runtime = platform_manager.runtime_state(backend_id) or {}
    provider = item.get("provider") or ""
    auth_status = cliproxy_auth_manager.list_status().get(provider, {})
    platform_configs = config_manager.get_platform_configs()
    p_cfg = platform_configs.get(backend_id, {})
    sidecar = cliproxy_sidecar.status()
    return {
        **item,
        **runtime,
        "cliproxy_auth": auth_status,
        "platform_config": p_cfg,
        "sidecar": sidecar,
        "smart_proxy": config_manager.get_smart_proxy_settings(),
    }


@app.get("/platforms/{backend_id}")
async def get_platform_detail(
    backend_id: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    payload = _platform_detail_payload(backend_id)
    sidecar_port = payload.get("sidecar", {}).get("port")
    if payload.get("status") == "running" and sidecar_port:
        all_models = await _fetch_sidecar_models(sidecar_port)
        filtered = filter_models_for_provider(
            all_models, payload.get("provider") or ""
        )
        provider = payload.get("provider") or ""
        payload["available_models"] = filtered
        payload["cursor_model_ids"] = [
            platform_model_listing_entry(m, provider=provider)["id"]
            for m in filtered
        ]
    else:
        payload["available_models"] = []
    return payload


@app.post("/platforms/{backend_id}/start")
async def start_platform(
    backend_id: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        state = await asyncio.to_thread(
            platform_manager.start_backend, backend_id, cliproxy_sidecar
        )
    except PlatformIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    _invalidate_models_cache()
    return {
        "message": "Integracao iniciada",
        "platform": state,
        "sidecar": cliproxy_sidecar.status(),
    }


@app.post("/platforms/{backend_id}/stop")
async def stop_platform(
    backend_id: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        state = await asyncio.to_thread(
            platform_manager.stop_backend, backend_id, cliproxy_sidecar
        )
    except PlatformIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    _invalidate_models_cache()
    return {
        "message": "Integracao parada",
        "platform": state,
        "sidecar": cliproxy_sidecar.status(),
    }


@app.get("/cliproxy/auth")
async def get_cliproxy_auth(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {
        "providers": cliproxy_auth_manager.list_status(),
        "sidecar": cliproxy_sidecar.status(),
    }


@app.post("/cliproxy/auth/{provider}/start")
async def start_cliproxy_auth(
    provider: str,
    req: CLIProxyAuthStartRequest,
    authenticated: bool = Depends(require_auth),
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        session = await asyncio.to_thread(
            cliproxy_auth_manager.start_login, provider, req.method
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"session": session}


@app.get("/cliproxy/auth/sessions/{session_id}")
async def get_cliproxy_auth_session(
    session_id: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    session = cliproxy_auth_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessao de autenticacao nao encontrada")
    return {"session": session}


@app.post("/cliproxy/auth/sessions/{session_id}/callback")
async def submit_cliproxy_auth_callback(
    session_id: str,
    req: CLIProxyAuthCallbackRequest,
    authenticated: bool = Depends(require_auth),
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        session = await asyncio.to_thread(
            cliproxy_auth_manager.submit_callback,
            session_id,
            req.callback_url,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Sessao de autenticacao nao encontrada")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"session": session}


@app.delete("/cliproxy/auth/sessions/{session_id}")
async def cancel_cliproxy_auth_session(
    session_id: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    session = await asyncio.to_thread(
        cliproxy_auth_manager.cancel_session, session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Sessao de autenticacao nao encontrada")
    return {"session": session}


@app.post("/cliproxy/restart")
async def restart_cliproxy_sidecar(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    active = platform_manager.active_instances()

    def _restart() -> dict:
        cliproxy_sidecar.stop()
        if active:
            try:
                cliproxy_sidecar.ensure_running()
            except CLIProxySidecarError as exc:
                raise RuntimeError(str(exc)) from exc
        return cliproxy_sidecar.status()

    try:
        status = await asyncio.to_thread(_restart)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    _invalidate_models_cache()
    return {"sidecar": status, "active_platforms": len(active)}


@app.post("/auto-balance/cancel")
async def cancel_auto_balance(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    process_manager.cancel_auto_balance()
    return {"message": "Cancelado"}


@app.post("/delete")
async def delete_model(req: DeleteRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    model_scanner.delete_model(req.path)
    _invalidate_models_cache()
    return {"message": "Excluido"}


@app.post("/rename")
async def rename_model(req: RenameRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    new_path = model_scanner.rename_model(req.path, req.new_name)
    _invalidate_models_cache()
    return {"message": "Renomeado", "new_path": new_path}


@app.post("/downloads")
async def start_download(req: DownloadRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    download_id = download_mgr.start_download(req.url, model_path=req.model_path)
    return {"download_id": download_id}


@app.get("/downloads")
async def list_downloads(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {"downloads": download_mgr.get_progress()}


@app.post("/downloads/cancel")
async def cancel_download(
    req: DownloadCancelRequest, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    if not download_mgr.cancel_download(req.download_id):
        raise HTTPException(status_code=404, detail="Download nao encontrado")
    return {"message": "Download cancelado"}


@app.post("/downloads/clear")
async def clear_downloads(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    download_mgr.clear_completed()
    return {"message": "Limpo"}


@app.get("/logs")
async def stream_logs(request: Request, port: Optional[int] = None):
    """SSE stream of llama-server logs."""
    authenticated = auth_manager.check_auth(request)
    if not authenticated:
        raise HTTPException(status_code=401, detail="Nao autenticado")
    return log_manager.stream_logs(stop_event=shutdown_event, request=request, port=port)


@app.api_route("/ui/{port}/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def ui_proxy(request: Request, port: int, path: str = ""):
    """Proxy llama-server web UI through the manager (fixes relative asset paths)."""
    if not require_auth(request):
        raise HTTPException(status_code=401)
    # Só faz proxy para portas de instâncias conhecidas — evita usar o manager
    # como proxy cego para qualquer serviço em loopback do host.
    known_ports = {
        inst.get("port") for inst in process_manager.get_status().get("instances", [])
    }
    if port not in known_ports:
        raise HTTPException(status_code=404, detail="Instancia inexistente")

    is_index = not path or path == "index.html"
    if is_index:
        path = ""

    target_url = f"http://127.0.0.1:{port}/{path}"
    query = str(request.query_params)
    if query:
        target_url += f"?{query}"

    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        if is_index:
            resp = await client.get(target_url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                html = _inject_ui_base_tag(resp.text, port)
                return HTMLResponse(content=html, status_code=200)

        resp = await client.request(
            request.method,
            target_url,
            content=await request.body(),
            headers=headers,
            timeout=30.0,
        )
        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers=_filter_proxy_headers(dict(resp.headers)),
        )
    except httpx.RequestError as exc:
        logger.error("UI proxy error to port %s: %s", port, exc)
        raise HTTPException(status_code=502, detail="Interface do modelo inacessivel")

@app.post("/set_default")
async def set_default_model(req: SetDefaultRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config_manager.set_default_model(req.path, req.add)
    return {"message": "Configuracao salva"}


@app.get("/config")
async def get_config(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config = config_manager.get_config()
    config.pop("admin_password_hash", None)
    return config


@app.get("/model-aliases")
async def list_model_aliases(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {
        "aliases": config_manager.get_model_aliases(),
        "cursor_compatible_names": list(CURSOR_COMPATIBLE_ALIAS_NAMES),
    }


@app.post("/model-aliases")
async def set_model_alias(
    req: ModelAliasRequest, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        aliases = config_manager.set_model_alias(req.alias, req.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "Alias salvo", "aliases": aliases}


@app.post("/models/mmproj")
async def set_mmproj(req: SetMmprojRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    settings = {"mmproj_path": req.mmproj_path}
    if req.mmproj_path == "__no_vision__":
        settings["mmproj_disabled"] = True
    elif req.mmproj_path is None or (req.mmproj_path and req.mmproj_path != "__no_vision__"):
        settings["mmproj_disabled"] = False
    config_manager.update_model_settings(req.model_path, settings)
    return {"status": "ok", "mmproj_path": req.mmproj_path}


@app.post("/models/thinking")
async def set_thinking(req: SetThinkingRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config_manager.update_model_settings(req.model_path, {"thinking_enabled": req.thinking_enabled})
    return {"message": "Configuracao salva"}


@app.post("/models/llama-bin")
async def set_llama_bin(req: SetLlamaBinRequest, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    settings: Dict[str, Any] = {}
    if req.llama_server_bin:
        settings["llama_server_bin"] = req.llama_server_bin
    if req.cache_type_k:
        settings["cache_type_k"] = req.cache_type_k
    if req.cache_type_v:
        settings["cache_type_v"] = req.cache_type_v
    if req.turboquant_preset:
        settings["turboquant_preset"] = req.turboquant_preset
    if not settings:
        raise HTTPException(status_code=400, detail="Nenhuma configuracao informada")
    config_manager.update_model_settings(req.model_path, settings)
    return {"message": "Configuracao salva", **settings}


@app.post("/system/shutdown")
async def system_shutdown(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    logger.info("Shutdown solicitado via API")
    shutdown_event.set()
    # Kill the process shortly after returning response
    def _kill():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill).start()
    return {"message": "Sistema encerrando..."}


@app.post("/system/update")
async def system_update(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    from version_manager import update_and_restart
    logger.info("Update solicitado via API")
    success, msg = update_and_restart(INSTALL_ROOT)
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"message": msg}


@app.get("/api/key")
async def get_api_key(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {"key": token_manager.get_or_create()}


@app.post("/api/key/renew")
async def renew_api_key(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {"key": token_manager.renew()}


@app.get("/api/system/version-check")
async def system_version_check(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    # check_for_updates faz `git fetch` (até ~30s); em thread para não bloquear
    # o event loop enquanto aguarda a rede.
    return await asyncio.to_thread(check_for_updates, INSTALL_ROOT)


async def _aggregate_models_response(
    instances: List[Dict[str, Any]], headers: Dict[str, str]
) -> JSONResponse:
    """Agrega o /v1/models de todas as instancias llama-server em execucao."""
    clear_platform_listing_registry()

    async def fetch_models(inst: Dict[str, Any]) -> List[Dict[str, Any]]:
        port = inst.get("port")
        if not port:
            return []
        url = f"http://127.0.0.1:{port}/v1/models"
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            models = resp.json().get("data") or []
            if inst.get("backend_type") == "platform":
                local_ids = _local_model_ids(instances)
                provider = str(inst.get("provider") or "")
                models = [
                    platform_model_listing_entry(m, provider=provider)
                    for m in models
                    if not should_skip_platform_model_listing(
                        str(m.get("id") or ""), local_ids
                    )
                ]
            return models
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            logger.warning(
                "Falha ao listar modelos da instancia na porta %s: %s",
                port, exc,
            )
            return []

    results = await asyncio.gather(*(fetch_models(inst) for inst in instances))
    merged: List[Dict[str, Any]] = []
    seen_ids = set()
    for models in results:
        for model in models:
            model_id = model.get("id")
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            merged.append(model)
    return JSONResponse({"object": "list", "data": merged})


def _inject_model_aliases(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adiciona entradas de alias (ex.: gpt-4o) com o mesmo shape dos modelos locais."""
    aliases = config_manager.get_model_aliases()
    if not aliases:
        return models
    existing_ids = {str(m.get("id") or "") for m in models}
    augmented = list(models)
    for alias, target in aliases.items():
        if alias in existing_ids:
            continue
        entry = platform_model_listing_entry(
            {"id": target, "owned_by": "platform"}, provider="platform"
        )
        entry["id"] = alias
        augmented.append(entry)
        existing_ids.add(alias)
    return augmented


async def _v1_models_payload(
    instances: List[Dict[str, Any]], headers: Dict[str, str]
) -> Dict[str, Any]:
    response = await _aggregate_models_response(instances, headers)
    payload = json.loads(response.body)
    payload["data"] = _inject_model_aliases(payload.get("data") or [])
    return payload


def _find_model_in_v1_list(
    models: List[Dict[str, Any]], model_id: str
) -> Optional[Dict[str, Any]]:
    """Busca modelo na listagem agregada (aceita id com ou sem .gguf)."""
    decoded = unquote(model_id or "").strip()
    if not decoded:
        return None
    candidates = [decoded]
    mapped = lookup_platform_bare_id(decoded)
    if mapped:
        candidates.extend([mapped, platform_model_listing_id(mapped)])
    listing = platform_model_listing_id(decoded)
    if listing not in candidates:
        candidates.append(listing)
    bare = resolve_platform_listing_model(decoded)
    if bare and bare not in candidates:
        candidates.append(bare)
        listing_bare = platform_model_listing_id(bare)
        if listing_bare not in candidates:
            candidates.append(listing_bare)
    by_id = {str(m.get("id") or ""): m for m in models}
    for candidate in candidates:
        if candidate in by_id:
            return by_id[candidate]
    return None


async def _ensure_platform_listing_registry(
    instances: List[Dict[str, Any]], headers: Optional[Dict[str, str]] = None
) -> None:
    """Garante mapa listing->sidecar antes de rotear chat (sem depender de GET /v1/models)."""
    if platform_listing_registry_populated():
        return
    hdrs = headers or {}
    for inst in instances:
        if inst.get("backend_type") != "platform":
            continue
        port = inst.get("port")
        if not port:
            continue
        provider = str(inst.get("provider") or "")
        try:
            resp = await client.get(
                f"http://127.0.0.1:{port}/v1/models", headers=hdrs, timeout=5.0
            )
            resp.raise_for_status()
            local_ids = _local_model_ids(instances)
            for m in resp.json().get("data") or []:
                root = str(m.get("id") or "")
                if not root or should_skip_platform_model_listing(root, local_ids):
                    continue
                register_platform_model_listings(root, provider)
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            logger.debug(
                "Falha ao montar registry de modelos plataforma na porta %s: %s",
                port, exc,
            )


def _find_target_instance(
    instances: List[Dict[str, Any]], requested_model: Optional[str]
) -> Dict[str, Any]:
    if requested_model:
        target = next(
            (
                inst
                for inst in instances
                if _instance_matches_model(inst, requested_model, instances)
            ),
            None,
        )
        if not target:
            listing_provider = platform_provider_for_listing(requested_model)
            if listing_provider:
                target = next(
                    (
                        inst for inst in instances
                        if inst.get("backend_type") == "platform"
                        and str(inst.get("provider") or "") == listing_provider
                    ),
                    None,
                )
        if not target:
            target = next(
                (
                    inst for inst in instances
                    if inst.get("backend_type") == "platform"
                ),
                None,
            )
        if not target:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{requested_model}' nao esta carregado.",
            )
        return target
    return next(
        (i for i in instances if i.get("port") == SERVER_PORT), instances[0]
    )


def _platform_response_model_name(
    client_requested_model: Optional[str],
    target_instance: Dict[str, Any],
    instances: List[Dict[str, Any]],
) -> Optional[str]:
    if target_instance.get("backend_type") != "platform" or not client_requested_model:
        return None
    return platform_client_facing_model(
        str(client_requested_model),
        _local_model_ids(instances),
        config_manager.get_model_aliases(),
        provider=str(target_instance.get("provider") or ""),
    )


def _local_model_ids(instances: List[Dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for inst in instances:
        if inst.get("backend_type") == "platform":
            continue
        if inst.get("model"):
            ids.add(inst["model"])
        path = inst.get("model_path")
        if path:
            ids.add(path)
            ids.add(os.path.basename(path))
    return ids


def _prepare_request_model(
    data: Dict[str, Any], instances: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Optional[str]]:
    """Resolve apenas alias no corpo; sufixo .gguf de plataforma é resolvido no encaminhamento."""
    requested = data.get("model")
    if not requested:
        return data, None
    original = str(requested)
    resolved = config_manager.resolve_model_alias(original)
    if resolved == original:
        return data, None
    return {**data, "model": resolved}, original


def _forward_model_for_backend(
    model_name: str,
    target_instance: Dict[str, Any],
    instances: List[Dict[str, Any]],
) -> str:
    if target_instance.get("backend_type") != "platform":
        return model_name
    return resolve_platform_listing_model(model_name, _local_model_ids(instances))


def _instance_matches_model(
    inst: Dict[str, Any],
    requested_model: str,
    instances: List[Dict[str, Any]],
) -> bool:
    if inst.get("model") == requested_model or inst.get("model_path") == requested_model:
        return True
    if inst.get("backend_type") != "platform":
        return False
    local_ids = _local_model_ids(instances)
    provider = str(inst.get("provider") or "")
    bare = resolve_platform_listing_model(requested_model, local_ids)
    listing = platform_model_listing_id(bare, provider)
    if requested_model in {bare, listing}:
        return True
    if lookup_platform_bare_id(requested_model):
        return True
    listing_provider = platform_provider_for_listing(requested_model)
    return listing_provider == provider if listing_provider else False


def _find_primary_instance(
    instances: List[Dict[str, Any]], proxy_settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Instância online do modelo principal (menor porta em caso de réplicas)."""
    primary_backend_id = proxy_settings.get("primary_backend_id")
    if primary_backend_id:
        matches = [
            inst for inst in instances
            if inst.get("backend_id") == primary_backend_id
        ]
        if not matches:
            return None
        return min(matches, key=lambda i: i["port"])
    primary_model_path = proxy_settings.get("primary_model_path")
    if not primary_model_path:
        return None
    norm = normalize_model_path(primary_model_path)
    matches = [
        inst for inst in instances
        if normalize_model_path(inst.get("model_path") or "") == norm
    ]
    if not matches:
        return None
    return min(matches, key=lambda i: i["port"])


def _is_primary_model_request(
    requested_model: str,
    proxy_settings: Dict[str, Any],
    primary_instance: Optional[Dict[str, Any]],
    instances: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """True quando o modelo pedido é o principal exposto (nome ou path)."""
    primary_backend_id = proxy_settings.get("primary_backend_id")
    if primary_backend_id:
        candidates = {primary_backend_id}
        if primary_instance:
            candidates.add(primary_instance.get("model") or "")
            candidates.add(primary_instance.get("backend_id") or "")
            candidates.add(primary_instance.get("provider") or "")
        if requested_model in candidates:
            return True
        local_matches = [
            inst for inst in (instances or [])
            if inst.get("backend_type", "local") != "platform"
            and (
                inst.get("model") == requested_model
                or inst.get("model_path") == requested_model
            )
        ]
        if local_matches:
            return False
        if primary_instance and primary_instance.get("backend_type") == "platform":
            primary_provider = str(primary_instance.get("provider") or "")
            listing_provider = platform_provider_for_listing(requested_model)
            if listing_provider == primary_provider:
                return True
            if listing_provider or lookup_platform_bare_id(requested_model):
                return False
            return False
        return not local_matches
    primary_model_path = proxy_settings.get("primary_model_path")
    if not primary_model_path:
        return False
    candidates = {primary_model_path, os.path.basename(primary_model_path)}
    if primary_instance:
        candidates.add(primary_instance.get("model") or "")
        candidates.add(primary_instance.get("model_path") or "")
    return requested_model in candidates


async def _primary_only_models_response(
    instances: List[Dict[str, Any]],
    proxy_settings: Dict[str, Any],
    headers: Dict[str, str],
) -> JSONResponse:
    """Com o modo proxy ativo, /v1/models expõe somente o principal (ADR-003)."""
    primary = _find_primary_instance(instances, proxy_settings)
    if primary is None:
        logger.warning(
            "[proxy] modo ativo sem modelo principal online; /v1/models vazio"
        )
        return JSONResponse({"object": "list", "data": []})
    entries: List[Dict[str, Any]] = []
    try:
        resp = await client.get(
            f"http://127.0.0.1:{primary['port']}/v1/models", headers=headers
        )
        resp.raise_for_status()
        entries = resp.json().get("data") or []
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        logger.warning(
            "Falha ao listar modelos da instancia principal na porta %s: %s",
            primary.get("port"), exc,
        )
    if not entries:
        entries = [{
            "id": primary.get("model"),
            "object": "model",
            "owned_by": "automanager",
        }]
    return JSONResponse({"object": "list", "data": entries[:1]})


async def _smart_proxy_forward(request: Request, path: str, data: Dict[str, Any]):
    """Encaminha uma requisição ao modelo principal via ProxyRouter (PRD F4-F8)."""
    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")
    is_stream = bool(data.get("stream"))

    try:
        decision = await proxy_router.resolve(
            headers=request.headers,
            body=data,
            client_ip=client_ip,
            user_agent=user_agent,
        )
    except ProxyError as exc:
        return JSONResponse(exc.payload(), status_code=exc.status_code)

    attempts = 0
    while True:
        logger.info(
            "[proxy] route external_model=%s internal_model=%s backend=%s gpu=%s "
            "affinity_key=%s sticky_hit=%s reason=%s stream=%s "
            "prompt_tokens_estimated=%s",
            decision.external_model, decision.internal_model,
            decision.backend_port, decision.gpu, decision.affinity_key,
            decision.sticky_hit, decision.reason, is_stream,
            decision.prompt_tokens_estimated,
        )
        instances = _hybrid_status().get("instances", [])
        decision_instance = next(
            (
                inst for inst in instances
                if inst.get("backend_id") == decision.backend_id
            ),
            None,
        )
        if decision_instance is None:
            decision_instance = next(
                (
                    inst for inst in instances
                    if inst.get("port") == decision.backend_port
                ),
                {"backend_type": decision.backend_type},
            )
        forward_model = _forward_model_for_backend(
            decision.internal_model,
            decision_instance,
            instances,
        )
        forward_body = json.dumps(
            {**data, "model": forward_model}, ensure_ascii=False
        ).encode("utf-8")
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        target_url = f"http://127.0.0.1:{decision.backend_port}/v1/{path}"
        backend_label = (
            f"platform:{decision.provider or decision.backend_port}"
            if decision.backend_type == "platform"
            else f"local:{decision.backend_port}"
        )
        telemetry = {
            "x-automanager-backend": str(decision.backend_port),
            "x-automanager-backend-model": forward_model,
            "x-automanager-backend-id": decision.backend_id,
            "x-automanager-backend-type": decision.backend_type,
        }

        try:
            if is_stream:
                if decision.backend_type == "platform":
                    status, chunks, upstream_headers = (
                        await _collect_upstream_stream_with_retry(
                            target_url,
                            content=forward_body,
                            headers=headers,
                            backend_label=backend_label,
                        )
                    )
                    await proxy_router.release(
                        decision.backend_id,
                        affinity_key=decision.affinity_key,
                    )

                    async def buffered_stream_generator(parts=chunks):
                        for part in parts:
                            yield part

                    response_headers = _filter_proxy_headers(upstream_headers)
                    response_headers.update(telemetry)
                    response_headers.pop("content-length", None)
                    return StreamingResponse(
                        buffered_stream_generator(),
                        status_code=status,
                        media_type=upstream_headers.get("content-type")
                        or "text/event-stream",
                        headers=response_headers,
                    )

                upstream = await _proxy_open_stream_with_retry(
                    target_url,
                    content=forward_body,
                    headers=headers,
                    backend_label=backend_label,
                )

                async def stream_generator(response=upstream, dec=decision):
                    usage_holder: Dict[str, Any] = {}
                    try:
                        if dec.rewrite:
                            # Reescrita por linha (ADR-006)
                            async for chunk in rewrite_sse_stream(
                                response.aiter_bytes(), dec.external_model,
                                usage_holder,
                            ):
                                yield chunk
                        else:
                            # Backend principal: repasse bruto, sem parse
                            async for chunk in response.aiter_bytes():
                                yield chunk
                    finally:
                        await response.aclose()
                        await proxy_router.release(
                            dec.backend_id,
                            affinity_key=dec.affinity_key,
                            usage=usage_holder.get("usage"),
                        )

                response_headers = _filter_proxy_headers(dict(upstream.headers))
                response_headers.update(telemetry)
                return StreamingResponse(
                    stream_generator(),
                    status_code=upstream.status_code,
                    media_type="text/event-stream",
                    headers=response_headers,
                )

            usage: Optional[Dict[str, Any]] = None
            try:
                resp = await _proxy_post_with_retry(
                    target_url,
                    content=forward_body,
                    headers=headers,
                    backend_label=backend_label,
                )
                content = resp.content
                if decision.rewrite:
                    content, usage = rewrite_json_model(
                        content, decision.external_model
                    )
                else:
                    _, usage = rewrite_json_model(content, decision.external_model)
                response_headers = _filter_proxy_headers(dict(resp.headers))
                response_headers.update(telemetry)
                return Response(
                    content=content,
                    status_code=resp.status_code,
                    headers=response_headers,
                    media_type=resp.headers.get("content-type"),
                )
            finally:
                await proxy_router.release(
                    decision.backend_id,
                    affinity_key=decision.affinity_key,
                    usage=usage,
                )
        except httpx.RequestError as exc:
            logger.warning(
                "[proxy] backend %s unavailable: %s", decision.backend_port, exc
            )
            if is_stream:
                # Slot ainda reservado (o gerador não chegou a rodar)
                await proxy_router.release(
                    decision.backend_id, affinity_key=decision.affinity_key
                )
            if attempts >= 1:
                return JSONResponse(
                    ProxyError(
                        502, "Erro ao conectar na instancia do modelo",
                        code="backend_unreachable",
                    ).payload(),
                    status_code=502,
                )
            attempts += 1
            try:
                new_decision = await proxy_router.reassign(decision.affinity_key)
            except ProxyError as pe:
                return JSONResponse(pe.payload(), status_code=pe.status_code)
            if new_decision is None:
                return JSONResponse(
                    ProxyError(
                        502, "Erro ao conectar na instancia do modelo",
                        code="backend_unreachable",
                    ).payload(),
                    status_code=502,
                )
            await proxy_router.acquire(new_decision.backend_id)
            new_decision.prompt_tokens_estimated = decision.prompt_tokens_estimated
            decision = new_decision


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(
    request: Request,
    path: str,
    authenticated: bool = Depends(require_api_token),
):
    """Proxy OpenAI-compatible requests to the correct llama-server instance,
    routing by the 'model' field in the request body (single-port multi-model).

    Com o Modo Proxy Inteligente ativo, requisições ao modelo principal são
    roteadas pelo ProxyRouter (sticky + least-busy); o restante segue o fluxo
    legado inalterado (ADR-003/ADR-004)."""
    if not authenticated:
        return _openai_auth_error()
    body = await request.body()
    data: Dict[str, Any] = {}
    requested_model = None

    if body:
        try:
            data = json.loads(body)
            requested_model = data.get("model")
        except json.JSONDecodeError:
            data = {}

    instances = _hybrid_status().get("instances", [])
    if not instances:
        raise HTTPException(status_code=503, detail="Nenhum modelo carregado")

    proxy_settings = config_manager.get_smart_proxy_settings()
    proxy_enabled = bool(proxy_settings.get("enabled"))

    # /v1/models: agregar todas as instancias (locais + plataforma) para clientes
    # OpenAI-compatíveis (Cursor, etc.) validarem nomes de modelo na listagem.
    # O Modo Proxy Inteligente continua atuando apenas em POST /v1/chat/completions.
    path_norm = path.strip("/")
    if request.method == "GET" and path_norm.startswith("models"):
        list_headers = _filter_proxy_headers(dict(request.headers))
        list_headers.pop("host", None)
        payload = await _v1_models_payload(instances, list_headers)
        if path_norm == "models":
            return JSONResponse(payload)
        if path_norm.startswith("models/"):
            model_id = path_norm.split("/", 1)[1]
            match = _find_model_in_v1_list(payload.get("data") or [], model_id)
            if match is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Modelo '{unquote(model_id)}' nao encontrado.",
                )
            return JSONResponse(match)

    route_headers = _filter_proxy_headers(dict(request.headers))
    route_headers.pop("host", None)
    await _ensure_platform_listing_registry(instances, route_headers)

    client_requested_model = requested_model
    external_model_name: Optional[str] = None
    if requested_model:
        data, external_model_name = _prepare_request_model(data, instances)
        requested_model = data.get("model")
        if body and data:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    if proxy_enabled and request.method == "POST" and requested_model:
        primary_instance = _find_primary_instance(instances, proxy_settings)
        if _is_primary_model_request(
            requested_model, proxy_settings, primary_instance, instances,
        ):
            return await _smart_proxy_forward(request, path, data)

    target_instance = _find_target_instance(instances, requested_model)
    forward_model = _forward_model_for_backend(
        str(requested_model or ""), target_instance, instances
    )
    if requested_model and forward_model != requested_model:
        data = {**data, "model": forward_model}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers_need_rebuild = True
    else:
        headers_need_rebuild = external_model_name is not None

    client_facing_model = _platform_response_model_name(
        client_requested_model, target_instance, instances
    )

    target_url = f"http://127.0.0.1:{target_instance['port']}/v1/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    if headers_need_rebuild:
        headers.pop("content-length", None)

    backend_label = (
        f"platform:{target_instance.get('provider') or target_instance.get('port')}"
        if target_instance.get("backend_type") == "platform"
        else f"local:{target_instance.get('port')}"
    )

    try:
        if request.method == "POST":
            is_stream = bool(data.get("stream"))
            if is_stream:
                if target_instance.get("backend_type") == "platform":
                    status, chunks, upstream_headers = (
                        await _collect_upstream_stream_with_retry(
                            target_url,
                            content=body,
                            headers=headers,
                            backend_label=backend_label,
                        )
                    )

                    async def buffered_stream_generator(parts=chunks):
                        for part in parts:
                            yield part

                    response_headers = _filter_proxy_headers(upstream_headers)
                    response_headers.pop("content-length", None)
                    return StreamingResponse(
                        buffered_stream_generator(),
                        status_code=status,
                        media_type=upstream_headers.get("content-type")
                        or "text/event-stream",
                        headers=response_headers,
                    )

                upstream = await _proxy_open_stream_with_retry(
                    target_url,
                    content=body,
                    headers=headers,
                    backend_label=backend_label,
                )

                async def stream_generator(response=upstream):
                    try:
                        byte_iter = response.aiter_bytes()
                        if client_facing_model:
                            async for chunk in rewrite_sse_stream(
                                byte_iter, client_facing_model
                            ):
                                yield chunk
                        else:
                            async for chunk in byte_iter:
                                yield chunk
                    finally:
                        await response.aclose()

                response_headers = _filter_proxy_headers(dict(upstream.headers))
                response_headers.pop("content-length", None)
                return StreamingResponse(
                    stream_generator(),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type")
                    or "text/event-stream",
                    headers=response_headers,
                )

            resp = await _proxy_post_with_retry(
                target_url,
                content=body,
                headers=headers,
                backend_label=backend_label,
            )
            content = resp.content
            if client_facing_model and target_instance.get("backend_type") != "platform":
                content, _ = rewrite_json_model(content, client_facing_model)
            response_headers = _filter_proxy_headers(dict(resp.headers))
            response_headers.pop("content-length", None)
            return Response(
                content=content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )
        elif request.method == "GET":
            resp = await client.get(
                target_url, params=request.query_params, headers=headers
            )
        else:
            resp = await client.request(
                request.method, target_url, content=body, headers=headers
            )

        content = await resp.aread()
        if client_facing_model and target_instance.get("backend_type") != "platform":
            content, _ = rewrite_json_model(content, client_facing_model)
        response_headers = _filter_proxy_headers(dict(resp.headers))
        response_headers.pop("content-length", None)
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )
    except httpx.RequestError as exc:
        logger.error(f"Proxy error to port {target_instance['port']}: {exc}")
        raise HTTPException(
            status_code=502, detail="Erro ao conectar na instancia do modelo"
        )


# ---------------------------------------------------------------------------
# Endpoints administrativos do Modo Proxy Inteligente (PRD F10)
# ---------------------------------------------------------------------------

def _known_model_path(path: str) -> bool:
    """Valida primary_model_path: arquivo existente, instância online ou config."""
    if not path:
        return False
    norm = normalize_model_path(path)
    if os.path.exists(norm):
        return True
    instances = process_manager.get_status().get("instances", [])
    if any(
        normalize_model_path(inst.get("model_path") or "") == norm
        for inst in instances
    ):
        return True
    model_configs = config_manager.get_config().get("model_configs", {})
    return any(normalize_model_path(key) == norm for key in model_configs)


def _known_backend_id(backend_id: str) -> bool:
    if not backend_id:
        return False
    if platform_manager.get(backend_id) is not None:
        return True
    return any(
        inst.get("backend_id") == backend_id
        for inst in _hybrid_status().get("instances", [])
    )


@app.post("/proxy/config")
async def set_proxy_config(
    req: ProxyConfigRequest, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    partial = req.model_dump(exclude_unset=True)
    primary = partial.get("primary_model_path")
    primary_backend_id = partial.get("primary_backend_id")
    if primary and not _known_model_path(primary):
        raise HTTPException(
            status_code=400,
            detail=f"Modelo principal desconhecido: {primary}",
        )
    if primary_backend_id and not _known_backend_id(primary_backend_id):
        raise HTTPException(
            status_code=400,
            detail=f"Backend principal desconhecido: {primary_backend_id}",
        )
    merged = config_manager.update_smart_proxy_settings(partial)
    logger.info(
        "[proxy] config updated enabled=%s primary=%s primary_backend=%s",
        merged["enabled"], merged["primary_model_path"],
        merged.get("primary_backend_id"),
    )
    return {"message": "Configuracao salva", "smart_proxy": merged}


@app.post("/models/proxy")
async def set_model_proxy(
    req: SetModelProxyRequest, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    if not req.model_path and not req.backend_id:
        raise HTTPException(
            status_code=400, detail="Informe model_path ou backend_id"
        )
    settings: Dict[str, Any] = {}
    if req.proxy_eligible is not None:
        settings["proxy_eligible"] = req.proxy_eligible
    if req.max_parallel_requests is not None:
        settings["max_parallel_requests"] = req.max_parallel_requests
    if req.auto_start is not None:
        settings["auto_start"] = req.auto_start
    if req.default_model is not None:
        if not req.backend_id:
            raise HTTPException(
                status_code=400,
                detail="default_model so e valido para plataformas (backend_id)",
            )
        settings["default_model"] = req.default_model
    if not settings:
        raise HTTPException(status_code=400, detail="Nenhuma configuracao informada")
    if req.backend_id:
        config_manager.update_platform_settings(req.backend_id, settings)
    else:
        config_manager.update_model_settings(req.model_path, settings)
    return {"message": "Configuracao salva"}


@app.get("/proxy/status")
async def proxy_status(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    settings = config_manager.get_smart_proxy_settings()
    backends = proxy_router.backends_snapshot()
    primary = next((b for b in backends if b["role"] == "primary"), None)
    sessions = await proxy_router.sessions()
    return {
        "enabled": settings["enabled"],
        "primary_model_path": settings["primary_model_path"],
        "primary_backend_id": settings.get("primary_backend_id"),
        "exposed_model": primary["model"] if primary else None,
        "primary": primary,
        "backends": backends,
        "sessions_count": len(sessions),
        "ttl_minutes": settings["ttl_minutes"],
        "max_wait_seconds": settings["max_wait_seconds"],
    }


@app.get("/proxy/backends")
async def proxy_backends(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return proxy_router.backends_snapshot()


@app.post("/proxy/backends/{port}/enable")
async def proxy_backend_enable(
    port: int, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    proxy_router.set_backend_enabled(port, True)
    return {"message": "Backend habilitado", "state": "online"}


@app.post("/proxy/backends/{port}/disable")
async def proxy_backend_disable(
    port: int, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    proxy_router.set_backend_enabled(port, False)
    return {"message": "Backend desabilitado", "state": "disabled"}


def _session_view(session, backends_by_port: Dict[int, Dict[str, Any]]) -> dict:
    backend = backends_by_port.get(session.backend_port, {})
    return {
        "affinity_key": session.affinity_key,
        "backend_port": session.backend_port,
        "external_model": session.external_model,
        "internal_model": session.internal_model,
        "backend_id": getattr(session, "backend_id", None) or backend.get("backend_id"),
        "backend_type": getattr(session, "backend_type", None) or backend.get("backend_type"),
        "provider": getattr(session, "provider", None) or backend.get("provider"),
        "detected_tag": session.detected_tag,
        "created_at": session.created_at,
        "last_used_at": session.last_used_at,
        "request_count": session.request_count,
        "tokens_processed": session.tokens_processed,
        "gpu": backend.get("gpu"),
        "backend_state": backend.get("state", "offline"),
    }


@app.get("/proxy/sessions")
async def proxy_sessions(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    backends_by_port = {b["port"]: b for b in proxy_router.backends_snapshot()}
    sessions = await proxy_router.sessions()
    return [_session_view(s, backends_by_port) for s in sessions]


@app.delete("/proxy/sessions")
async def proxy_sessions_clear(authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    removed = await proxy_router.clear_sessions()
    logger.info("[proxy] admin cleared %d sticky session(s)", removed)
    return {"removed": removed}


@app.delete("/proxy/sessions/{affinity_key:path}")
async def proxy_session_delete(
    affinity_key: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    removed = await proxy_router.clear_sessions(affinity_key)
    if not removed:
        raise HTTPException(status_code=404, detail="Sessao nao encontrada")
    return {"removed": removed}


@app.post("/proxy/sessions/{affinity_key:path}/reassign")
async def proxy_session_reassign(
    affinity_key: str, authenticated: bool = Depends(require_auth)
):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        decision = await proxy_router.reassign(affinity_key)
    except ProxyError as exc:
        return JSONResponse(exc.payload(), status_code=exc.status_code)
    if decision is None:
        raise HTTPException(status_code=404, detail="Sessao nao encontrada")
    return decision


@app.post("/proxy/resolve")
async def proxy_resolve(request: Request, authenticated: bool = Depends(require_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    try:
        data = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Corpo JSON invalido")
    settings = config_manager.get_smart_proxy_settings()
    if not settings.get("enabled"):
        return {"proxy_enabled": False}
    try:
        decision = await proxy_router.resolve(
            headers=request.headers,
            body=data,
            client_ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            dry_run=True,
        )
    except ProxyError as exc:
        return JSONResponse(exc.payload(), status_code=exc.status_code)
    return {
        "proxy_enabled": True,
        "external_model": decision.external_model,
        "detected_tag": decision.detected_tag,
        "affinity_key": decision.affinity_key,
        "selected_backend": decision.backend_port,
        "backend_id": decision.backend_id,
        "backend_type": decision.backend_type,
        "provider": decision.provider,
        "internal_model": decision.internal_model,
        "gpu": decision.gpu,
        "reason": decision.reason,
        "sticky_hit": decision.sticky_hit,
    }


def _resolved_mmproj_path(model: dict, model_cfg: dict) -> Optional[str]:
    candidates = model.get("mmproj_candidates") or []
    if not candidates:
        return None
    saved = model_cfg.get("mmproj_path")
    # Preserve the explicit "Sem visão" sentinel
    if saved == "__no_vision__":
        return None
    if saved and saved in candidates:
        return saved
    return candidates[0]


def _build_model_vision_controls(model: dict, model_js: str, model_cfg: dict) -> str:
    candidates = model.get("mmproj_candidates") or []
    import_btn = (
        f'<button type="button" onclick="event.stopPropagation(); '
        f"openVisionImportModal('{model_js}')\" "
        'class="vision-import-btn w-8 h-8 flex items-center justify-center rounded '
        'bg-slate-800/50 text-slate-500 hover:text-violet-400 hover:bg-violet-500/20 '
        'transition-all" title="Importar projetor de visão" '
        'aria-label="Importar projetor de visão">'
        '<i class="fas fa-eye text-ui-label"></i></button>'
    )
    if not candidates:
        return import_btn
    selected = _resolved_mmproj_path(model, model_cfg)
    options = ''
    # "Sem visão" option at the top
    no_vision_selected = " selected" if selected is None else ""
    options += (
        f'<option value="__no_vision__"{no_vision_selected}>Sem visão</option>'
    )
    for candidate in candidates:
        name = html.escape(os.path.basename(candidate))
        value = html.escape(candidate, quote=True)
        selected_attr = " selected" if candidate == selected else ""
        options += (
            f'<option value="{value}" class="bg-slate-900"{selected_attr}>{name}</option>'
        )
    safe_js = _escape_js_attr(model_js)
    return (
        f"{import_btn}"
        f'<select data-mmproj-for="{html.escape(model_js, quote=True)}" '
        'class="model-mmproj-select bg-slate-900 border border-slate-700 text-slate-300 '
        'rounded-lg px-2 py-1 text-ui-label font-bold focus:ring-2 focus:ring-violet-500/50 '
        'outline-none transition-all cursor-pointer min-w-[7rem] max-w-[11rem]" '
        f"onmousedown=\"event.stopPropagation()\" onpointerdown=\"event.stopPropagation()\" "
        f"onclick=\"event.stopPropagation()\" "
        f"onchange=\"onMmprojChange('{safe_js}', this)\" "
        'title="Projetor de visão para este modelo" '
        'aria-label="Projetor de visão para este modelo">'
        f"{options}</select>"
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    is_authenticated = auth_manager.check_auth_cookie(request)
    
    # Pre-render rows for the template (they will be cloned per tab)
    gpus = gpu_manager.detect_gpus()
    gpu_rows = ""
    for g in gpus:
        idx = g["index"]
        name = g["name"]
        vram = g["vram"]
        gpu_rows += f"""
            <tr class="gpu-row group/row" data-index="{idx}">
                <td class="px-6 py-4 text-center">
                    <input type="checkbox" checked class="gpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                </td>
                <td class="px-4 py-4 text-center">
                    <input type="radio" name="main-gpu-[TABID]" class="gpu-main-radio w-5 h-5 bg-slate-900 border-slate-700 text-blue-600 cursor-pointer" { 'checked' if idx == 0 else '' }>
                </td>
                <td class="px-4 py-4">
                    <div class="flex flex-col">
                        <span class="text-ui-body-sm font-black text-white uppercase tracking-tight">{name}</span>
                        <span class="text-ui-label text-slate-400 font-mono uppercase tracking-tighter">ID: {idx} · {vram} MB VRAM</span>
                    </div>
                </td>
                <td class="px-4 py-4">
                    <div class="flex items-center gap-5">
                        <div class="relative flex items-center group/input shrink-0 cfg-field group/tip">
                            {_cfg_tip("Peso percentual desta GPU no tensor-split. Pin (📌) trava o valor durante o auto-balance manual.")}
                            <input type="number" class="gpu-weight w-20 bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1.5 text-xs font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all pr-6"
                                   value="{100 if idx == 0 else 0}" min="0" max="100">
                            <span class="absolute right-2 text-ui-label font-black text-slate-600">%</span>
                            <label class="ml-3 flex items-center gap-2 cursor-pointer" title="Travar valor no auto-balance">
                                <input type="checkbox" class="gpu-pin hidden">
                                <i class="fas fa-thumbtack text-ui-body-sm text-slate-700 hover:text-blue-500 transition-colors pin-icon"></i>
                            </label>
                        </div>
                        <div class="flex-1 min-w-[100px]">
                            <div class="flex justify-between items-end mb-1">
                                <span class="text-ui-label font-black text-slate-500 uppercase tracking-widest">VRAM</span>
                                <span class="gpu-vram-text text-ui-label font-mono text-blue-400">0 / {vram} MB</span>
                            </div>
                            <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div class="gpu-vram-bar h-full bg-cyan-500 transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>"""

    cpu_rows = f"""
        <tr class="cpu-row group/row">
            <td class="px-6 py-4 text-center">
                <input type="checkbox" class="cpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-emerald-600 cursor-pointer">
            </td>
            <td class="px-4 py-4 text-center opacity-20 pointer-events-none">
                <input type="radio" disabled class="w-4 h-4">
            </td>
            <td class="px-4 py-4">
                <div class="flex flex-col">
                    <span class="text-ui-body-sm font-black text-white uppercase tracking-tight">System RAM / CPU Offload</span>
                    <span class="text-ui-label text-slate-400 font-mono uppercase tracking-tighter">Latencia superior a VRAM</span>
                </div>
            </td>
            <td class="px-4 py-4">
                <div class="relative flex items-center group/input cfg-field group/tip">
                    {_cfg_tip("Offload para RAM/CPU quando a VRAM não comporta o modelo. Mais lento que GPU; o auto-balance usa como último recurso se habilitado.")}
                    <input type="number" class="cpu-weight w-20 bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1.5 text-xs font-bold focus:ring-2 focus:ring-emerald-500/50 outline-none transition-all pr-6" 
                           value="0" min="0" max="100">
                    <span class="absolute right-2 text-ui-label font-black text-slate-600">%</span>
                    <label class="ml-3 flex items-center gap-2 cursor-pointer" title="Travar valor no auto-balance">
                        <input type="checkbox" class="cpu-pin hidden">
                        <i class="fas fa-thumbtack text-ui-body-sm text-slate-700 hover:text-emerald-500 transition-colors pin-icon"></i>
                    </label>
                </div>
            </td>
        </tr>"""

    status = process_manager.get_status()
    scan_result = model_scanner.scan()
    models = scan_result.get("models", [])
    default_model = config_manager.get_config().get("default_model")
    default_models = config_manager.get_config().get("default_models", [])
    model_configs = config_manager.get_config().get("model_configs", {})

    smart_proxy_cfg = config_manager.get_smart_proxy_settings()
    if not isinstance(smart_proxy_cfg, dict):
        smart_proxy_cfg = {}
    proxy_primary_path = smart_proxy_cfg.get("primary_model_path")
    proxy_enabled = bool(smart_proxy_cfg.get("enabled") is True)

    model_items = ""
    for m in models:
        m_path = m["path"]
        m_name = m["name"]
        m_dir = m["dir"]
        m_js = m_path.replace("\\", "/")
        m_cfg = model_configs.get(m_js, {})
        stable_id = hashlib.md5(m_js.encode("utf-8")).hexdigest()[:12]
        m_js_js = _escape_js_attr(m_js)
        
        is_default = "checked" if (m_path in default_models or m_path == default_model) else ""
        has_config = "text-blue-400" if m_cfg and not m_cfg.get("hardware_incapable") else "text-slate-100"
        
        is_proxy_primary = (
            "checked"
            if proxy_primary_path
            and normalize_model_path(m_path) == proxy_primary_path
            else ""
        )
        is_proxy_eligible = (
            "checked" if m_cfg.get("proxy_eligible", True) else ""
        )
        proxy_max_parallel = m_cfg.get("max_parallel_requests") or 1

        hardware_incapable = bool(m_cfg.get("hardware_incapable"))
        incapable_badge = '<span class="shrink-0 text-ui-label font-black uppercase tracking-wider text-red-400 bg-red-500/15 px-2 py-0.5 rounded-lg border border-red-500/30 ml-2">Incapaz</span>' if hardware_incapable else ''
        incapable_row_class = 'border-red-500/40 bg-red-950/20' if hardware_incapable else ''
        incapable_attr = 'true' if hardware_incapable else 'false'

        vision_controls = _build_model_vision_controls(m, m_js_js, m_cfg)

        model_items += f"""
        <div id="lib-{stable_id}" class="model-item-container group flex flex-col gap-3 p-3 mb-2 bg-slate-800/40 rounded-xl hover:bg-slate-700/60 transition-all border border-slate-700/50 hover:border-blue-500/50 shadow-sm {incapable_row_class}" data-path="{html.escape(m_js, quote=True)}" data-hardware-incapable="{incapable_attr}">
            <div class="w-full cursor-pointer" title="Clique para abrir · Ctrl+clique para nova aba" onclick="selectModelFromEvent(event, '{m_js_js}', '{stable_id}')" onauxclick="selectModelFromEvent(event, '{m_js_js}', '{stable_id}')">
                <div class="flex items-start justify-between">
                    <div class="flex items-center gap-2 overflow-hidden">
                        <i class="fas fa-cube text-blue-400 text-ui-body-sm shrink-0"></i>
                        <p class="model-name text-ui-body font-bold {has_config} truncate">{html.escape(m_name)}</p>
                    </div>
                    {incapable_badge}
                </div>
                <p class="text-ui-label text-slate-400 truncate font-mono mt-1 uppercase opacity-80">{html.escape(m_dir)}</p>
            </div>
            <div class="flex items-center justify-between gap-2 mt-1">
                <div class="flex items-center gap-1 flex-wrap">
                    <button onclick="event.stopPropagation(); renameModel('{m_js_js}')" title="Renomear modelo" aria-label="Renomear modelo" class="rename-btn w-8 h-8 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-blue-400 transition-all"><i class="fas fa-edit text-ui-label"></i></button>
                    <button onclick="event.stopPropagation(); deleteModel('{m_js_js}')" title="Excluir modelo" aria-label="Excluir modelo" class="w-8 h-8 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-red-400 transition-all"><i class="fas fa-trash-alt text-ui-label"></i></button>
                    {vision_controls}
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-ui-label font-black text-slate-600 uppercase">Padrão</span>
                    <input type="checkbox" class="w-3.5 h-3.5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" {is_default} onclick="event.stopPropagation(); setDefaultModel(this, '{m_js_js}')">
                </div>
            </div>
            <div class="proxy-model-controls flex flex-wrap items-center gap-x-3 gap-y-1.5 pt-2 border-t border-slate-800/40 min-w-0">
                <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Modelo principal exposto pela API no Modo Proxy Inteligente (apenas um por vez)">
                    <span class="text-ui-label font-black text-violet-400/80 uppercase">Principal</span>
                    <input type="checkbox" class="proxy-primary-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600 cursor-pointer" data-path="{html.escape(m_js, quote=True)}" {is_proxy_primary} onclick="event.stopPropagation(); setProxyPrimary(this, '{m_js_js}')">
                </label>
                <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Usar como backend secundário no proxy inteligente">
                    <span class="text-ui-label font-black text-slate-600 uppercase">Proxy</span>
                    <input type="checkbox" class="proxy-eligible-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600 cursor-pointer" {is_proxy_eligible} onclick="event.stopPropagation(); setProxyEligible(this, '{m_js_js}')">
                </label>
                <label class="flex items-center gap-1 shrink-0 ml-auto" title="Máximo de requisições paralelas roteadas para este backend">
                    <span class="text-ui-label font-black text-slate-600 uppercase">Paralelo</span>
                    <input type="number" min="1" max="16" value="{proxy_max_parallel}" class="proxy-max-parallel w-9 px-0.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-ui-label text-slate-300 text-center outline-none" onclick="event.stopPropagation()" onchange="setProxyMaxParallel(this, '{m_js_js}')">
                </label>
            </div>
        </div>"""

    # Presets for selects
    ctx_opts = ""
    for val in CONTEXT_PRESET_VALUES:
        label = f"{val}K" if isinstance(val, int) else "Personalizado"
        val_attr = val if isinstance(val, int) else "custom"
        selected = 'selected' if val == 65536 else ''
        ctx_opts += f'<option value="{val_attr}" class="bg-slate-900" {selected}>{label}</option>'

    batch_opts = ""
    for val in BATCH_SIZE_PRESETS:
        selected = "selected" if val == DEFAULT_BATCH_SIZE else ""
        batch_opts += f'<option value="{val}" class="bg-slate-900" {selected}>{val}</option>'

    cache_type_k_opts = ""
    for val in CACHE_TYPE_PRESETS:
        selected = "selected" if val == DEFAULT_CACHE_TYPE else ""
        cache_type_k_opts += f'<option value="{val}" class="bg-slate-900" {selected}>{val}</option>'

    cache_type_v_opts = cache_type_k_opts

    ubatch_opts = ""
    for val in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        selected = "selected" if val == 512 else ""
        ubatch_opts += f'<option value="{val}" class="bg-slate-900" {selected}>{val}</option>'

    # Só expõe o token da API no HTML quando há sessão válida; GET / é público
    # (serve a tela de login) e não deve vazar o segredo para anônimos.
    api_token = token_manager.get_or_create() if is_authenticated else ""
    local_ip = get_local_ip()

    return HTMLResponse(_build_html(
        gpu_rows=gpu_rows,
        cpu_rows=cpu_rows,
        model_items=model_items,
        ctx_opts=ctx_opts,
        batch_opts=batch_opts,
        ubatch_opts=ubatch_opts,
        cache_type_k_opts=cache_type_k_opts,
        cache_type_v_opts=cache_type_v_opts,
        default_model=default_model,
        local_ip=local_ip,
        api_token=api_token,
        is_authenticated=is_authenticated,
        context_preset_values=CONTEXT_PRESET_VALUES,
        default_context_size=DEFAULT_CONTEXT_SIZE,
        context_k_multiplier=CONTEXT_K_MULTIPLIER,
        default_parallel_slots=DEFAULT_PARALLEL_SLOTS,
        default_batch_size=DEFAULT_BATCH_SIZE,
        default_cache_type=DEFAULT_CACHE_TYPE,
        default_mtp_draft_tokens=DEFAULT_MTP_DRAFT_TOKENS,
        proxy_enabled=proxy_enabled,
    ))


def _build_html(
    gpu_rows: str,
    cpu_rows: str,
    model_items: str,
    ctx_opts: str,
    batch_opts: str,
    ubatch_opts: str,
    cache_type_k_opts: str,
    cache_type_v_opts: str,
    default_model: Optional[str],
    local_ip: str,
    api_token: str,
    is_authenticated: bool,
    context_preset_values: list,
    default_context_size: int,
    context_k_multiplier: int,
    default_parallel_slots: int,
    default_batch_size: int,
    default_cache_type: str,
    default_mtp_draft_tokens: int,
    proxy_enabled: bool = False,
) -> str:
    """Build the full HTML template."""

    proxy_enabled_attr = "checked" if proxy_enabled else ""
    login_overlay_style = "none" if is_authenticated else "flex"
    shell_style = "flex" if is_authenticated else "none"
    login_overlay = f"""
        <div id="login-overlay" class="fixed inset-0 z-50 flex items-center justify-center pointer-events-auto bg-slate-950/95 backdrop-blur-sm" style="display: {login_overlay_style};">
            <div class="glass p-8 md:p-10 rounded-3xl border border-slate-700/50 w-full max-w-md mx-4 shadow-2xl">
                <div class="flex flex-col items-center mb-8">
                    <div class="bg-blue-600 p-4 rounded-2xl shadow-xl shadow-blue-500/20 mb-4">
                        <i class="fas fa-brain text-white text-2xl"></i>
                    </div>
                    <h2 class="text-xl font-bold text-white">Automanager Llama.cpp</h2>
                    <p class="text-xs text-slate-500 mt-1 uppercase tracking-widest font-black">Sistema de Controle Neural</p>
                </div>
                <form id="login-form" onsubmit="handleLogin(event)">
                    <div class="mb-4">
                        <label class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest pl-1">Usuario</label>
                        <input type="text" id="login-username" value="admin" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" required>
                    </div>
                    <div class="mb-6">
                        <label class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest pl-1">Senha</label>
                        <input type="password" id="login-password" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" required>
                    </div>
                    <button type="submit" class="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white text-sm font-black rounded-xl transition-all uppercase tracking-widest shadow-xl active:scale-95">
                        AUTENTICAR
                    </button>
                    <p id="login-error" class="text-red-500 text-ui-body-sm mt-4 text-center font-bold uppercase hidden"></p>
                </form>
            </div>
        </div>"""

    vision_import_modal = """
        <div id="vision-import-modal" class="fixed inset-0 z-50 hidden items-center justify-center p-4" role="dialog" aria-modal="true">
            <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" onclick="closeVisionImportModal()"></div>
            <div class="relative glass w-full max-w-lg rounded-3xl border border-violet-500/30 shadow-2xl overflow-hidden">
                <div class="p-6 md:p-8 border-b border-slate-800/60 bg-slate-900/40">
                    <h2 class="text-lg font-bold text-white">Importar Projetor de Visão</h2>
                    <p class="text-xs text-slate-500 mt-1">Vincule um arquivo mmproj ao modelo selecionado</p>
                </div>
                <form id="vision-import-form" class="p-6 md:p-8 space-y-4" onsubmit="submitVisionImport(event)">
                    <input type="hidden" id="vision-import-model-path" value="">
                    <div>
                        <label class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest pl-1">URL mmproj</label>
                        <input type="url" id="vision-import-url" required placeholder="https://..." class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 outline-none focus:ring-2 focus:ring-violet-500/50">
                    </div>
                    <button type="submit" class="w-full py-3 bg-violet-600 hover:bg-violet-500 text-white text-xs font-black rounded-xl transition-all uppercase">BAIXAR E VINCULAR</button>
                </form>
            </div>
        </div>"""

    version_update_modal = """
        <div id="version-update-modal" class="fixed inset-0 z-50 hidden items-center justify-center p-4" role="dialog" aria-modal="true">
            <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"></div>
            <div class="relative glass w-full max-w-2xl max-h-[80vh] flex flex-col rounded-3xl border border-blue-500/30 shadow-2xl overflow-hidden">
                <div class="p-6 md:p-8 border-b border-slate-800/60 bg-slate-900/40">
                    <h2 class="text-xl font-bold text-white">Atualização Disponível</h2>
                    <p class="text-xs text-slate-400 mt-2">Novas melhorias e correções prontas para instalação</p>
                </div>
                <div id="version-commits-list" class="custom-scroll flex-1 overflow-y-auto p-6 md:p-8 space-y-4 font-mono text-xs"></div>
                <div class="p-6 md:p-8 border-t border-slate-800/60 bg-slate-900/40">
                    <button onclick="dismissVersionModal()" class="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white text-xs font-black rounded-xl uppercase">ENTENDI</button>
                </div>
            </div>
        </div>"""

    cliproxy_auth_modal = """
        <div id="cliproxy-auth-modal" class="fixed inset-0 z-50 hidden items-center justify-center p-3 sm:p-4 overflow-y-auto" role="dialog" aria-modal="true">
            <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" onclick="closeCliproxyAuthModal()"></div>
            <div class="relative glass w-full max-w-xl max-h-[min(90vh,720px)] flex flex-col rounded-2xl sm:rounded-3xl border border-amber-500/30 shadow-2xl overflow-hidden my-auto">
                <div class="shrink-0 p-4 sm:p-6 border-b border-slate-800/60 bg-slate-900/40">
                    <h2 id="cliproxy-auth-title" class="text-base sm:text-lg font-bold text-white">Autenticar plataforma</h2>
                    <p id="cliproxy-auth-subtitle" class="text-xs text-slate-500 mt-1">Conecte a conta do provedor ao CLIProxyAPI</p>
                </div>
                <div class="flex-1 min-h-0 overflow-y-auto custom-scroll p-4 sm:p-6 space-y-3">
                    <div id="cliproxy-auth-status" class="text-sm text-slate-300">Preparando autenticacao...</div>
                    <div id="cliproxy-auth-device" class="hidden rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 sm:p-4">
                        <p class="text-ui-label font-black uppercase tracking-widest text-amber-300">Codigo do dispositivo</p>
                        <p id="cliproxy-auth-device-code" class="mt-2 text-xl sm:text-2xl font-mono font-bold text-white tracking-widest break-all"></p>
                        <a id="cliproxy-auth-device-url" href="#" target="_blank" rel="noopener noreferrer" class="mt-3 inline-flex text-xs sm:text-sm text-amber-300 hover:text-amber-200 underline break-all"></a>
                    </div>
                    <div id="cliproxy-auth-oauth" class="hidden rounded-xl border border-blue-500/30 bg-blue-500/5 p-3 sm:p-4 space-y-2 sm:space-y-3">
                        <p class="text-ui-label font-black uppercase tracking-widest text-blue-300">1. Abra o link de login</p>
                        <a id="cliproxy-auth-oauth-url" href="#" target="_blank" rel="noopener noreferrer" class="inline-flex text-xs sm:text-sm text-blue-300 hover:text-blue-200 underline break-all line-clamp-4"></a>
                        <pre id="cliproxy-auth-instructions" class="max-h-28 overflow-y-auto custom-scroll text-[11px] sm:text-xs text-slate-400 whitespace-pre-wrap font-mono"></pre>
                        <div id="cliproxy-auth-callback-box" class="hidden space-y-2 pt-2 border-t border-blue-500/20">
                            <p class="text-ui-label font-black uppercase tracking-widest text-blue-300">2. Cole a URL de callback</p>
                            <p class="text-[11px] sm:text-xs text-slate-400">Depois do login, copie a URL completa que comeca com <code class="text-slate-300">http://localhost:1455/auth/callback</code>.</p>
                            <input type="url" id="cliproxy-auth-callback-input" placeholder="http://localhost:1455/auth/callback?code=..." class="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs text-slate-200 outline-none focus:ring-2 focus:ring-blue-500/40">
                            <button type="button" id="cliproxy-auth-callback-btn" onclick="submitCliproxyAuthCallback()" class="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-black rounded-xl uppercase">Enviar callback</button>
                        </div>
                    </div>
                    <pre id="cliproxy-auth-log" class="hidden max-h-24 overflow-y-auto custom-scroll text-[11px] text-slate-500 whitespace-pre-wrap font-mono"></pre>
                </div>
                <div class="shrink-0 p-4 sm:p-6 border-t border-slate-800/60 bg-slate-900/40 flex gap-3">
                    <button type="button" onclick="closeCliproxyAuthModal()" class="flex-1 py-2.5 sm:py-3 bg-slate-800 hover:bg-slate-700 text-white text-xs font-black rounded-xl uppercase">Fechar</button>
                    <button type="button" id="cliproxy-auth-cancel-btn" onclick="cancelCliproxyAuth()" class="hidden flex-1 py-2.5 sm:py-3 bg-red-600/20 hover:bg-red-600/30 text-red-300 text-xs font-black rounded-xl uppercase">Cancelar</button>
                </div>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automanager Llama.cpp</title>
    <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        :root {{
            --bg-deep: #020617;
            --card-bg: rgba(15, 23, 42, 0.6);
            --text-caption: 0.6875rem;
            --text-label: 0.75rem;
            --text-body-sm: 0.8125rem;
            --text-body: 0.875rem;
            --text-heading: 0.9375rem;
        }}
        .text-ui-caption {{ font-size: var(--text-caption); }}
        .text-ui-label {{ font-size: var(--text-label); }}
        .text-ui-body-sm {{ font-size: var(--text-body-sm); }}
        .text-ui-body {{ font-size: var(--text-body); }}
        .text-ui-heading {{ font-size: var(--text-heading); }}
        body {{ font-family: 'Space Grotesk', sans-serif; background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #020617 100%); background-attachment: fixed; }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .glass {{ background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }}
        .custom-scroll::-webkit-scrollbar {{ width: 6px; }}
        .custom-scroll::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 10px; }}
        @keyframes pulse-glow {{ 0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }} 70% {{ box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }} }}
        .glow-online {{ animation: pulse-glow 2s infinite; }}
        
        #sidebar {{ transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1); }}
        #sidebar.collapsed {{ transform: translateX(-100%); }}
        .main-content {{ transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1); margin-left: 320px; }}
        .main-content.full {{ margin-left: 0; }}
        
        .tab-btn {{ transition: all 0.3s ease; border-bottom: 3px solid transparent; }}
        .tab-btn.active {{ border-bottom-color: #3b82f6; color: #fff; background: rgba(59, 130, 246, 0.1); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: flex; flex-direction: column; width: 100%; }}

        #tabs-container {{
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
            min-height: 0;
        }}

        .tab-layout-row {{
            display: flex;
            flex-direction: column;
            width: 100%;
            align-items: stretch;
        }}

        @media (min-width: 1280px) {{
            .tab-layout-row {{
                flex-direction: row;
            }}
        }}

        .tab-config-panel {{
            position: relative;
            z-index: 1;
            overflow: visible;
        }}

        .tab-log-panel {{
            position: relative;
            z-index: 5;
            display: flex;
            flex-direction: column;
            min-height: 28rem;
            max-height: var(--tab-config-height, none);
            overflow: hidden;
        }}

        .tab-log-box {{
            flex: 1 1 0;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }}

        .custom-scroll {{
            scrollbar-width: thin;
            scrollbar-color: #334155 transparent;
        }}
        
        @media (max-width: 1024px) {{
            #sidebar {{ transform: translateX(-100%); z-index: 50; width: 300px; }}
            #sidebar.open {{ transform: translateX(0); }}
            .main-content {{ margin-left: 0 !important; }}
        }}

        .hide-scrollbar::-webkit-scrollbar {{ display: none; }}
        .hide-scrollbar {{ -ms-overflow-style: none; scrollbar-width: none; }}
        
        .tab-close-btn {{ opacity: 0; transition: opacity 0.2s; }}
        .tab-btn:hover .tab-close-btn {{ opacity: 1; }}
        
        .model-item-container.active-selection {{ border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }}

        /* --- Polish: cursor, foco por teclado, transições --- */
        button:not(:disabled), [onclick], label[onclick], a[href] {{ cursor: pointer; }}
        button:disabled {{ cursor: not-allowed; }}
        :focus-visible {{ outline: 2px solid rgba(59, 130, 246, 0.6); outline-offset: 2px; border-radius: 8px; }}
        .model-mmproj-select, input, select, textarea {{ transition: box-shadow 0.15s ease, border-color 0.15s ease; }}

        @keyframes panel-in {{ from {{ opacity: 0; transform: translateY(-6px); }} to {{ opacity: 1; transform: none; }} }}
        .tab-auto-balance-progress:not(.hidden),
        .tab-auto-balance-alert:not(.hidden),
        .tab-proposed-config:not(.hidden),
        .tab-mtp-warning:not(.hidden) {{ animation: panel-in 0.25s ease; }}
        .tab-content.active {{ animation: panel-in 0.2s ease; }}

        /* --- Toasts --- */
        #toast-container {{ position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 100; display: flex; flex-direction: column; gap: 0.5rem; max-width: min(92vw, 26rem); }}
        .toast {{
            display: flex; align-items: flex-start; gap: 0.65rem;
            padding: 0.85rem 1rem; border-radius: 0.9rem;
            background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
            color: #e2e8f0; font-size: var(--text-body-sm);
            transform: translateY(10px); opacity: 0;
            transition: transform 0.22s ease, opacity 0.22s ease;
        }}
        .toast.show {{ transform: translateY(0); opacity: 1; }}
        .toast.toast-error {{ border-color: rgba(244, 63, 94, 0.5); }}
        .toast.toast-success {{ border-color: rgba(16, 185, 129, 0.5); }}
        .toast.toast-info {{ border-color: rgba(59, 130, 246, 0.5); }}
        .toast .toast-icon {{ margin-top: 0.1rem; }}
        .toast-error .toast-icon {{ color: #fb7185; }}
        .toast-success .toast-icon {{ color: #34d399; }}
        .toast-info .toast-icon {{ color: #60a5fa; }}

        .cfg-field {{
            position: relative;
            overflow: visible;
        }}
        .cfg-field .cfg-tip {{
            font-weight: 400;
            letter-spacing: normal;
            text-transform: none;
            z-index: 99999;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.15s ease, visibility 0.15s ease;
        }}
        .cfg-field:hover .cfg-tip,
        .cfg-field:focus-within .cfg-tip {{
            opacity: 1;
            visibility: visible;
        }}
        .cfg-field:hover,
        .cfg-field:focus-within {{
            z-index: 9999;
        }}
        .glass:has(.cfg-field:hover),
        .glass:has(.cfg-field:focus-within) {{
            position: relative;
            z-index: 9998;
        }}
        .tab-config-panel:has(.cfg-field:hover),
        .tab-config-panel:has(.cfg-field:focus-within) {{
            z-index: 200;
        }}
        .tab-turboquant-panel .cfg-tip {{
            top: auto;
            bottom: 100%;
            margin-top: 0;
            margin-bottom: 0.375rem;
        }}
        thead .cfg-tip {{
            top: auto;
            bottom: 100%;
            margin-top: 0;
            margin-bottom: 0.375rem;
        }}
    </style>
</head>
<body class="min-h-screen text-slate-200 selection:bg-blue-500/30 flex overflow-x-hidden">
    <script>window.modelConfigs = {{}}; window.activeTabs = [];</script>
    {login_overlay}
    {vision_import_modal}
    {version_update_modal}
    {cliproxy_auth_modal}
    <div id="toast-container" aria-live="polite" aria-atomic="false"></div>

    <!-- SIDEBAR (MENU RETRATIL) -->
    <aside id="sidebar" class="fixed top-0 left-0 h-full w-80 glass border-r border-slate-800 z-40 overflow-y-auto custom-scroll flex flex-col shadow-2xl collapsed" style="display: {shell_style};">
        <div class="p-6 border-b border-slate-800 flex items-center justify-between shrink-0 bg-slate-950/20">
            <h2 class="font-bold text-lg text-white flex items-center gap-3">
                <i class="fas fa-layer-group text-blue-500"></i> Biblioteca
            </h2>
            <button onclick="toggleSidebar(false)" class="text-slate-500 hover:text-white transition-colors">
                <i class="fas fa-chevron-left"></i>
            </button>
        </div>
        
        <div class="flex-1 p-6 space-y-8">
            <!-- Biblioteca de Modelos -->
            <section>
                <div class="flex items-center justify-between mb-4">
                    <p class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest">Modelos Disponíveis</p>
                    <span id="model-count" class="text-ui-label bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700 font-mono">0</span>
                </div>
                <div id="model-list-container" class="space-y-2">
                    {model_items}
                </div>
            </section>

            <!-- Download -->
            <section class="pt-6 border-t border-slate-800/50">
                <div class="flex items-center justify-between mb-3">
                    <p class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest">Download GGUF</p>
                    <button type="button" onclick="clearCompletedDownloads()" title="Remover downloads concluídos, cancelados e com falha da lista" aria-label="Limpar lista de downloads" class="text-ui-label font-black uppercase tracking-widest text-slate-500 hover:text-slate-300 flex items-center gap-1.5 transition-colors">
                        <i class="fas fa-broom text-ui-label"></i> Limpar
                    </button>
                </div>
                <div class="space-y-3">
                    <div class="relative">
                        <input type="text" id="download-url" placeholder="URL HuggingFace..." class="w-full pl-4 pr-10 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-300 focus:ring-1 focus:ring-blue-500/50 outline-none">
                        <button onclick="downloadModel()" title="Iniciar download" aria-label="Iniciar download" class="absolute right-2 top-1/2 -translate-y-1/2 text-blue-500 hover:text-blue-400"><i class="fas fa-arrow-down"></i></button>
                    </div>
                </div>
                <div id="download-list" class="mt-4 space-y-2"></div>
            </section>

            <!-- Admin Config -->
            <section class="pt-6 border-t border-slate-800/50 space-y-4 pb-10">
                 <p class="text-ui-body-sm font-black text-slate-500 uppercase tracking-widest mb-3">Configurações Globais</p>
                 <div class="space-y-3">
                    <div class="space-y-1">
                        <label class="text-ui-label font-black text-slate-600 uppercase ml-1">Diretório de Modelos</label>
                        <div class="flex gap-2">
                            <input type="text" id="models-dir-input" class="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-300 font-mono outline-none">
                            <button onclick="saveModelsDir()" class="px-2 py-2 bg-slate-800 hover:bg-blue-600 rounded-lg text-ui-label font-bold transition-all"><i class="fas fa-save"></i></button>
                        </div>
                    </div>
                    <div class="flex justify-between items-center px-1">
                        <span class="text-ui-label text-slate-600 font-mono uppercase tracking-widest">Armazenamento</span>
                        <span class="text-ui-label text-slate-500 font-bold" id="repo-storage">-- GB</span>
                    </div>
                 </div>

                 <div class="space-y-2 pt-4 border-t border-slate-800/30">
                    <label class="text-ui-label font-black text-slate-600 uppercase ml-1">Acesso API (OpenAI)</label>
                    <div class="bg-slate-900 p-2 rounded-lg border border-slate-800 flex items-center justify-between">
                        <code id="api-token" data-full-token="{html.escape(api_token)}" class="text-ui-label text-amber-500/80 font-mono truncate mr-2">{html.escape(api_token[:11] + '…' + api_token[-8:]) if api_token else ''}</code>
                        <button type="button" onclick="copyApiToken()" title="Copiar token" class="text-slate-600 hover:text-white shrink-0"><i class="far fa-copy text-ui-body-sm"></i></button>
                    </div>
                 </div>

                 <div class="space-y-2 pt-4 border-t border-slate-800/30">
                    <label class="text-ui-label font-black text-slate-600 uppercase ml-1">Proxy Inteligente</label>
                    <div class="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
                        <label class="flex items-center justify-between cursor-pointer gap-2">
                            <span class="text-ui-body-sm text-slate-300 font-bold">Ativar Modo Proxy Inteligente</span>
                            <input type="checkbox" id="proxy-enabled-toggle" class="w-4 h-4 bg-slate-950 border-slate-700 rounded text-violet-600 cursor-pointer shrink-0" {proxy_enabled_attr} onchange="proxyToggleEnabled(this)">
                        </label>
                        <p id="proxy-primary-hint" class="text-ui-label text-amber-400/90 leading-relaxed hidden">Nenhum modelo principal definido. Marque "Principal" em um modelo da biblioteca para expor a API.</p>
                        <p class="text-ui-label text-slate-500 leading-relaxed">Com o modo ativo, a API /v1 expõe somente o modelo principal e distribui conversas/subagentes entre as instâncias online automaticamente.</p>
                    </div>
                 </div>

                 <div class="space-y-2 pt-4 border-t border-slate-800/30">
                    <label class="text-ui-label font-black text-slate-600 uppercase ml-1">Alterar Senha Admin</label>
                    <input type="password" id="current-password" placeholder="Senha atual" class="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm outline-none">
                    <input type="password" id="new-password" placeholder="Nova senha" class="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm outline-none">
                    <button onclick="changePassword()" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-ui-label font-bold rounded-lg transition-all uppercase tracking-widest border border-slate-700">ATUALIZAR SENHA</button>
                    <p id="password-change-status" class="text-ui-label font-bold text-center"></p>
                 </div>
            </section>
        </div>
    </aside>

    <!-- CONTEUDO PRINCIPAL -->
    <main id="main-content" class="main-content full flex-1 min-h-screen flex flex-col relative" style="display: {shell_style};">
        <!-- HEADER -->
        <header class="glass border-b border-slate-800 px-6 py-4 flex items-center justify-between h-16 shrink-0 sticky top-0 z-30 shadow-md">
            <div class="flex items-center gap-4">
                <button id="sidebar-toggle" onclick="toggleSidebar()" class="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all active:scale-90">
                    <i class="fas fa-bars"></i>
                </button>
                <div>
                    <h1 class="text-base font-bold text-white tracking-tight flex items-center gap-2">
                        Automanager <span class="text-blue-500 font-light">Llama.cpp</span>
                    </h1>
                </div>
            </div>

            <div class="flex items-center gap-6">
                <div id="status-badge" class="px-4 py-1.5 rounded-full text-ui-label font-black tracking-[0.2em] flex items-center gap-2 glass border-slate-700/50 text-slate-500 uppercase">
                    <div class="w-1.5 h-1.5 rounded-full bg-slate-600 status-dot"></div><span class="status-text">OFFLINE</span>
                </div>
                <div class="flex items-center gap-4 border-l border-slate-800 pl-6">
                    <button onclick="handleUpdate()" class="text-amber-500/50 hover:text-amber-500 transition-colors" title="Atualizar"><i class="fas fa-sync-alt"></i></button>
                    <button onclick="handleShutdown()" class="text-red-500/50 hover:text-red-500 transition-colors" title="Desligar"><i class="fas fa-power-off"></i></button>
                    <button onclick="handleLogout()" class="text-slate-500 hover:text-white transition-colors" title="Sair"><i class="fas fa-sign-out-alt"></i></button>
                </div>
            </div>
        </header>

        <div id="dashboard" class="flex-1 flex flex-col" style="display: {'flex' if is_authenticated else 'none'};">
            <!-- METRICAS (FIXAS) -->
            <div id="metrics-panel" class="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 md:p-8 bg-slate-950/20 shrink-0 border-b border-slate-800/30">
                <div class="glass p-4 rounded-2xl border-l-2 border-blue-600">
                    <div class="flex justify-between items-center mb-1">
                        <p class="text-ui-label font-black text-slate-500 uppercase tracking-widest">CPU HOST</p>
                        <i class="fas fa-microchip text-slate-700 text-ui-label"></i>
                    </div>
                    <div class="flex items-end justify-between gap-4">
                        <h3 id="cpu-val" class="text-2xl font-bold text-white tracking-tighter">0%</h3>
                        <div class="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden mb-1.5">
                            <div id="cpu-bar" class="h-full bg-blue-500 transition-all duration-700" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                <div class="glass p-4 rounded-2xl border-l-2 border-emerald-600">
                    <div class="flex justify-between items-center mb-1">
                        <p class="text-ui-label font-black text-slate-500 uppercase tracking-widest">RAM HOST</p>
                        <i class="fas fa-memory text-slate-700 text-ui-label"></i>
                    </div>
                    <div class="flex items-end justify-between gap-4">
                        <h3 id="ram-val" class="text-2xl font-bold text-white tracking-tighter">0%</h3>
                        <div class="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden mb-1.5">
                            <div id="ram-bar" class="h-full bg-emerald-500 transition-all duration-700" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                <div id="mini-gpu-metrics" class="col-span-2 flex gap-3 overflow-x-auto custom-scroll hide-scrollbar">
                    <!-- Dinâmico -->
                </div>
            </div>

            <!-- PROXY INTELIGENTE (PRD F9) -->
            <section id="proxy-panel" class="px-6 md:px-8 py-4 bg-slate-950/20 border-b border-slate-800/30 shrink-0">
                <div class="glass p-5 rounded-2xl border-l-2 border-violet-600 space-y-4">
                    <div class="flex items-center justify-between flex-wrap gap-3">
                        <div class="flex items-center gap-3">
                            <i class="fas fa-route text-violet-400"></i>
                            <p class="text-ui-body-sm font-black text-slate-400 uppercase tracking-widest">Proxy Inteligente</p>
                            <span id="proxy-mode-badge" class="px-3 py-1 rounded-full text-ui-label font-black tracking-widest uppercase glass border-slate-700/50 text-slate-500">INATIVO</span>
                        </div>
                        <div class="text-ui-label text-slate-500 uppercase tracking-widest">
                            Modelo exposto: <span id="proxy-exposed-model" class="text-slate-200 font-bold normal-case tracking-normal">—</span>
                        </div>
                    </div>
                    <div id="proxy-panel-body" class="space-y-4 hidden">
                        <div id="proxy-backends-list" class="grid grid-cols-1 md:grid-cols-3 gap-3"></div>
                        <div>
                            <p class="text-ui-label font-black text-slate-500 uppercase tracking-widest mb-2">
                                Sessões ativas (sticky) <span id="proxy-sessions-count" class="font-mono text-slate-400"></span>
                            </p>
                            <div id="proxy-sessions-list" class="space-y-1.5 max-h-56 overflow-y-auto custom-scroll pr-1"></div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- TABS AREA -->
            <div class="flex-1 flex flex-col bg-slate-900/10">
                <!-- BARRA DE ABAS -->
                <nav id="tab-bar" class="bg-slate-950/40 border-b border-slate-800 px-4 flex items-center gap-1 overflow-x-auto hide-scrollbar h-12 shrink-0">
                    <button type="button" onclick="toggleSidebar(true)" title="Abrir biblioteca para adicionar aba" class="tab-new-btn shrink-0 w-8 h-8 flex items-center justify-center rounded-lg border border-slate-800 text-slate-500 hover:text-blue-400 hover:border-blue-500/40 hover:bg-blue-500/10 transition-all">
                        <i class="fas fa-plus text-ui-body-sm"></i>
                    </button>
                    <!-- Tabs injetadas via JS -->
                </nav>

                <!-- CONTAINER DE CONTEUDO -->
                <div id="tabs-container" class="flex-1 flex flex-col">
                    <!-- Tela Vazia -->
                    <div id="no-tab-content" class="flex flex-col items-center justify-center p-6 md:p-10 text-center bg-slate-950/30 min-h-[60vh]">
                         <div class="w-16 h-16 rounded-[1.5rem] bg-slate-900 flex items-center justify-center mb-4 border border-slate-800 shadow-inner">
                             <i class="fas fa-cubes text-2xl text-slate-700"></i>
                         </div>
                         <h3 class="text-lg font-bold text-slate-300 tracking-tight">Arquitetura Multi-Modelo</h3>
                         <p class="text-ui-body-sm text-slate-500 mt-2 max-w-md leading-relaxed uppercase tracking-[0.2em]">
                             Selecione modelos na biblioteca lateral. Mantenha várias abas abertas para comparar configurações.
                         </p>

                         <div id="no-tab-shortcuts" class="w-full max-w-5xl mt-8 space-y-4 hidden">
                             <div class="flex items-center justify-between px-1">
                                 <p class="text-ui-label font-black text-slate-500 uppercase tracking-[0.25em]">Modelos Frequentes</p>
                                 <span id="no-tab-shortcuts-count" class="text-ui-label font-mono text-slate-600"></span>
                             </div>
                             <div id="no-tab-shortcuts-grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-left"></div>
                         </div>

                         <p id="no-tab-shortcuts-empty" class="text-ui-body-sm text-slate-600 mt-6 max-w-sm hidden">
                             Nenhum modelo configurado ainda. Abra a biblioteca para escolher um modelo e salvar suas preferências.
                         </p>

                         <button onclick="toggleSidebar(true)" class="mt-8 px-10 py-3.5 bg-blue-600 hover:bg-blue-500 text-white text-ui-body-sm font-black rounded-2xl uppercase tracking-[0.25em] transition-all shadow-2xl shadow-blue-600/20 active:scale-95">
                             ABRIR BIBLIOTECA
                         </button>
                    </div>
                    <!-- Conteúdo das tabs será injetado aqui -->
                </div>
            </div>
        </div>
    </main>

    <!-- TEMPLATE PARA ABA DE MODELO -->
    <template id="model-tab-template">
        <div class="tab-content w-full flex-col">
            <div class="tab-layout-row">
                
                <!-- PAINEL DE CONFIG (ESQUERDA) -->
                <div class="tab-config-panel flex-1 p-6 md:p-8 space-y-6 bg-slate-900/10">
                    <!-- Header da Tab -->
                    <div class="flex items-center justify-between gap-6 flex-wrap pb-6 border-b border-slate-800/60">
                        <div class="flex items-center gap-5">
                            <div class="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center shadow-inner">
                                <i class="fas fa-cube text-blue-500 text-xl"></i>
                            </div>
                            <div>
                                <h2 class="model-tab-name text-2xl font-bold text-white tracking-tight leading-none">NOME</h2>
                                <p class="model-tab-path text-ui-body-sm text-slate-500 font-mono mt-2 uppercase tracking-tighter opacity-50 truncate max-w-md"></p>
                            </div>
                        </div>
                        <div class="flex items-center gap-4">
                             <div class="tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500 shadow-sm transition-all">OFFLINE</div>
                             <div class="tab-actions flex items-center gap-3">
                                 <!-- Buttons Start/Stop/Chat -->
                             </div>
                        </div>
                    </div>

                    <div class="tab-auto-balance-progress hidden p-6 rounded-2xl border-2 border-amber-500/40 bg-amber-500/10 shadow-lg shadow-amber-500/10">
                        <div class="flex gap-4 items-start text-amber-300">
                            <i class="fas fa-sync animate-spin mt-1 shrink-0 text-lg"></i>
                            <div class="flex-1 min-w-0">
                                <p class="text-ui-body font-black uppercase tracking-widest mb-1">Auto-Balance em andamento</p>
                                <p class="tab-auto-balance-progress-msg text-sm leading-relaxed break-words text-amber-100/90"></p>
                                <p class="tab-auto-balance-progress-attempt text-ui-body-sm text-amber-200/60 mt-2 font-mono"></p>
                            </div>
                            <button type="button" class="tab-auto-balance-cancel-btn px-4 py-2 rounded-xl border border-red-500/40 text-ui-label font-black uppercase tracking-widest text-red-300 hover:bg-red-500/10 transition-all shrink-0">
                                Cancelar
                            </button>
                        </div>
                    </div>

                    <!-- Configurações em Grid -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 relative overflow-visible">
                        <!-- Configuração do Motor -->
                        <div class="glass rounded-[2rem] p-6 space-y-6 shadow-sm overflow-visible relative">
                             <div class="flex items-center justify-between">
                                <p class="text-ui-body-sm font-black text-blue-500 uppercase tracking-[0.25em]">Parâmetros do Motor</p>
                                <i class="fas fa-sliders-h text-slate-800 text-xs"></i>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Tokens de contexto por slot (histórico + resposta). Valores maiores consomem mais VRAM/RAM no cache KV. O servidor usa contexto × slots como tamanho total do buffer.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span><i class="fas fa-expand-arrows-alt text-ui-label mr-1"></i> Contexto / Slot</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-context hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <select class="tab-context-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-blue-500/50 outline-none transition-all">
                                            {ctx_opts}
                                        </select>
                                        <div class="relative tab-custom-ctx-wrap hidden">
                                            <input type="number" class="tab-context-size-custom w-24 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center">
                                            <span class="absolute right-2 top-1/2 -translate-y-1/2 text-ui-label font-black text-slate-600">K</span>
                                        </div>
                                    </div>
                                 </div>
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Conversas independentes atendidas em paralelo. Cada slot duplica o cache KV — mais slots = mais VRAM/RAM e maior o ctx-size total enviado ao llama-server.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span><i class="fas fa-clone text-ui-label mr-1"></i> Slots (Simultâneo)</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-slots hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <input type="number" class="tab-parallel-slots w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center" value="{DEFAULT_PARALLEL_SLOTS}" min="1" max="64">
                                 </div>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-4 border-t border-slate-800/30">
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Tamanho do lote no prefill (processamento do prompt). Batch maior acelera prompts longos, mas aumenta picos de VRAM temporária durante a entrada.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Batch Prefill</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-batch hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <select class="tab-batch-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-violet-500/50 outline-none transition-all">
                                        {batch_opts}
                                    </select>
                                 </div>
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Micro-lote físico (ubatch) usado internamente pelo llama.cpp. Afeta throughput e memória de trabalho; normalmente deve ser menor ou igual ao batch de prefill.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>U-Batch Físico</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-ubatch hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <select class="tab-ubatch-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-cyan-500/50 outline-none transition-all">
                                        {ubatch_opts}
                                    </select>
                                 </div>
                             </div>
                             <div class="pt-4 border-t border-slate-800/30">
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Binário llama.cpp usado para carregar este modelo. Quando há mais de uma instalação detectada no sistema, escolha qual versão usar.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center gap-2">
                                        <span><i class="fas fa-code-branch text-ui-label mr-1"></i> Versão llama.cpp</span>
                                    </label>
                                    <select class="tab-llama-bin bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-blue-500/50 outline-none transition-all">
                                        <option value="" class="bg-slate-900">Detectando...</option>
                                    </select>
                                 </div>
                             </div>
                             <div class="tab-turboquant-panel hidden pt-4 border-t border-amber-500/20 space-y-4">
                                 <div class="flex items-center gap-2">
                                     <i class="fas fa-bolt text-amber-500 text-ui-body-sm"></i>
                                     <p class="text-ui-label font-black text-amber-500 uppercase tracking-[0.25em]">TurboQuant+ KV Cache</p>
                                 </div>
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Presets assimétricos recomendados pelo TurboQuant+: K em alta precisão, V comprimido com turbo2/3/4. Boundary V e sparse dequant ativam automaticamente no binário.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1">Preset TurboQuant</label>
                                    <select class="tab-turboquant-preset bg-slate-950 border border-amber-500/30 text-amber-200 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none transition-all">
                                    </select>
                                 </div>
                                 <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                     <div class="{_CFG_FIELD}">
                                        {_cfg_tip("Precisão do cache K (keys). Mantenha f16 ou q8_0 — comprimir K com turbo degrada qualidade.")}
                                        <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1">Cache K (Keys)</label>
                                        <select class="tab-turbo-cache-k bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none"></select>
                                     </div>
                                     <div class="{_CFG_FIELD}">
                                        {_cfg_tip("Precisão do cache V (values). turbo4 = mais leve; turbo2 = mais agressivo (~4.6× compressão em turbo3).")}
                                        <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1">Cache V (Values)</label>
                                        <select class="tab-turbo-cache-v bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none"></select>
                                     </div>
                                 </div>
                                 <p class="text-ui-label text-slate-500 leading-relaxed">Valores injetados como <span class="font-mono text-amber-400/80">--cache-type-k</span> e <span class="font-mono text-amber-400/80">--cache-type-v</span> ao iniciar com TurboQuant+.</p>
                             </div>
                        </div>

                        <!-- Otimização & Threads -->
                        <div class="glass rounded-[2rem] p-6 space-y-6 shadow-sm overflow-visible relative">
                             <div class="flex items-center justify-between">
                                <p class="text-ui-body-sm font-black text-emerald-500 uppercase tracking-[0.25em]">Otimização Sistêmica</p>
                                <i class="fas fa-microchip text-slate-800 text-xs"></i>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                 <div class="{_CFG_FIELD}">
                                    {_cfg_tip("Threads de CPU para geração (token a token) e para prefill. 0 = automático. Mais threads ajudam quando há camadas na CPU; excesso pode reduzir desempenho.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Threads (Gen / Batch)</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-threads hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <input type="number" class="tab-threads w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center" placeholder="Auto">
                                        <input type="number" class="tab-threads-batch w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2.5 text-sm font-bold focus:ring-1 focus:ring-violet-500/50 outline-none text-center" placeholder="Auto">
                                    </div>
                                 </div>
                                 <div class="{_CFG_FIELD} tab-standard-cache-wrap">
                                    {_cfg_tip("Precisão do cache KV (K = keys, V = values). Com TurboQuant+, V aceita turbo2/3/4 além de f16/q8_0/q4_0.")}
                                    <label class="text-ui-label font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Quantização de Cache</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-cache hidden">
                                            <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <select class="tab-cache-type-k bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-2 py-2.5 text-ui-body font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none">
                                            {cache_type_k_opts}
                                        </select>
                                        <select class="tab-cache-type-v bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-2 py-2.5 text-ui-body font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none">
                                            {cache_type_v_opts}
                                        </select>
                                    </div>
                                 </div>
                             </div>
                             <div class="flex flex-wrap gap-3 pt-4 border-t border-slate-800/30">
                                <div class="{_CFG_FIELD} flex items-center gap-2 bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-violet-500/30 transition-all">
                                    {_cfg_tip("Ativa blocos de raciocínio interno em modelos compatíveis (ex.: DeepSeek). Aumenta tokens gerados e latência, mas melhora respostas complexas.")}
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="checkbox" class="tab-thinking-toggle w-5 h-5 bg-slate-950 border-slate-700 rounded text-violet-600">
                                        <span class="text-ui-body-sm font-bold uppercase text-slate-500">Thinking</span>
                                    </label>
                                    <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                        <input type="checkbox" class="tab-pin-thinking hidden">
                                        <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                    </label>
                                </div>
                                <div class="{_CFG_FIELD} flex items-center gap-2 bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-amber-500/30 transition-all">
                                    {_cfg_tip("Multi-Token Prediction: modelo draft prevê vários tokens à frente para acelerar a geração. Requer suporte no modelo/binário; consome VRAM extra.")}
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="checkbox" class="tab-mtp-toggle w-5 h-5 bg-slate-950 border-slate-700 rounded text-amber-600">
                                        <span class="text-ui-body-sm font-bold uppercase text-slate-500">MTP</span>
                                    </label>
                                    <input type="number" class="tab-mtp-draft-tokens w-12 bg-slate-950 border border-slate-800 text-slate-300 rounded-lg px-2 py-1 text-ui-body-sm font-bold text-center focus:ring-1 focus:ring-amber-500/50 outline-none disabled:opacity-40 disabled:cursor-not-allowed" value="{default_mtp_draft_tokens}">
                                    <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                        <input type="checkbox" class="tab-pin-mtp hidden">
                                        <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                    </label>
                                </div>
                                <div class="{_CFG_FIELD} flex items-center gap-2 bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-emerald-500/30 transition-all">
                                    {_cfg_tip("Flash Attention acelera a inferência e reduz uso de VRAM no cache KV. Desative se o modelo/binário apresentar instabilidade ou incompatibilidade.")}
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="checkbox" class="tab-flash-attn-toggle w-5 h-5 bg-slate-950 border-slate-700 rounded text-emerald-600" checked>
                                        <span class="text-ui-body-sm font-bold uppercase text-slate-500">Flash Attn</span>
                                    </label>
                                    <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                        <input type="checkbox" class="tab-pin-flash-attn hidden">
                                        <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                    </label>
                                </div>
                                <div class="{_CFG_FIELD} flex items-center gap-2 bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-cyan-500/30 transition-all">
                                    {_cfg_tip("NUMA: com 2+ GPUs usa distribute (threads em todos os nós); com 1 GPU usa isolate (nó da GPU principal). Útil em servidores multi-socket com offload grande para RAM.")}
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="checkbox" class="tab-numa-toggle w-5 h-5 bg-slate-950 border-slate-700 rounded text-cyan-600">
                                        <span class="text-ui-body-sm font-bold uppercase text-slate-500">NUMA</span>
                                    </label>
                                    <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                        <input type="checkbox" class="tab-pin-numa hidden">
                                        <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                    </label>
                                </div>
                                <div class="{_CFG_FIELD} flex items-center gap-2 bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-blue-500/30 transition-all">
                                    {_cfg_tip("Como dividir o modelo entre GPUs: Layer split reparte camadas; Row split reparte linhas (exige suporte). Afeta balanceamento multi-GPU e uso de tensor-split.")}
                                    <select class="tab-split-mode bg-slate-950 border border-slate-800 text-slate-400 rounded-xl px-4 py-2 text-ui-body-sm font-bold outline-none focus:ring-1 focus:ring-blue-500/50">
                                        <option value="layer">LAYER SPLIT</option>
                                        <option value="row">ROW SPLIT</option>
                                    </select>
                                    <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                        <input type="checkbox" class="tab-pin-split-mode hidden">
                                        <i class="fas fa-thumbtack text-ui-label text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                    </label>
                                </div>
                             </div>
                        </div>
                    </div>

                    <!-- Proposed Configuration Area (Hidden until results) -->
                    <div class="tab-proposed-config hidden glass rounded-[2rem] border border-blue-500/40 bg-blue-500/5 p-6 space-y-4">
                        <div class="flex items-center justify-between">
                            <h3 class="text-ui-body-sm font-black text-blue-400 uppercase tracking-widest flex items-center gap-2">
                                <i class="fas fa-magic"></i> Configuração Otimizada Sugerida
                            </h3>
                            <div class="flex items-center gap-3">
                                <button class="tab-apply-config-btn px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white text-ui-body-sm font-black rounded-xl uppercase tracking-widest transition-all active:scale-95 shadow-lg shadow-blue-600/20">
                                    EFETIVAR E SALVAR
                                </button>
                                <button class="tab-discard-config-btn text-ui-body-sm font-bold text-slate-500 hover:text-slate-300 uppercase tracking-widest transition-colors">
                                    DESCARTAR
                                </button>
                            </div>
                        </div>
                        <div class="tab-proposed-details grid grid-cols-2 md:grid-cols-4 gap-4 text-ui-label font-mono text-slate-400">
                            <!-- Details injected via JS -->
                        </div>
                    </div>

                    <!-- Alocação de GPU -->
                    <div class="glass rounded-[2rem] overflow-visible border border-slate-800/50 shadow-lg">
                        <table class="w-full text-left">
                            <thead class="text-ui-label font-black text-slate-500 uppercase tracking-[0.25em] bg-slate-950/50">
                                <tr>
                                    <th class="px-8 py-5 text-center w-16"><span class="cfg-field group/tip relative inline-block">{_cfg_tip("Marca se a GPU/CPU participa do carregamento do modelo.")}<span>USO</span></span></th>
                                    <th class="px-4 py-5 text-center w-24"><span class="cfg-field group/tip relative inline-block">{_cfg_tip("GPU principal (--main-gpu): recebe tensores centrais e costuma carregar mais peso no split.")}<span>PRINCIPAL</span></span></th>
                                    <th class="px-4 py-5">DISPOSITIVO</th>
                                    <th class="px-8 py-5 text-right"><span class="cfg-field group/tip relative inline-block ml-auto">{_cfg_tip("Percentual do modelo neste dispositivo (tensor-split). A soma das GPUs/CPU ativas deve totalizar 100%.")}<span>DISTRIBUIÇÃO</span></span></th>
                                </tr>
                            </thead>
                            <tbody class="tab-gpu-table-body divide-y divide-slate-800/40 bg-slate-950/10">
                                {cpu_rows}
                                {gpu_rows}
                            </tbody>
                        </table>
                        <div class="p-6 bg-slate-950/30 border-t border-slate-800/50 flex flex-wrap items-center justify-between gap-6">
                             <div class="flex items-center gap-6">
                                <button type="button" onclick="window.runSmartCalibration(this)" class="tab-smart-calibrate-btn px-6 py-3 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 rounded-2xl text-ui-body-sm font-black uppercase tracking-[0.2em] transition-all active:scale-95 flex items-center gap-3">
                                    <i class="fas fa-brain"></i> CALIBRAR SMART (AUTO-BALANCE)
                                </button>
                                <span class="tab-total-percent text-ui-body font-bold tracking-[0.1em] text-slate-500 uppercase">CARGA TOTAL: 100%</span>
                             </div>
                             <button class="tab-reset-defaults-btn px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-ui-label font-black rounded-xl border border-slate-700 transition-all uppercase tracking-widest text-slate-400 hover:text-white">
                                 <i class="fas fa-undo mr-2 text-ui-label"></i> RESETAR PADRÕES
                             </button>
                        </div>
                    </div>

                    <!-- Alertas Localizados -->
                    <div class="tab-alerts space-y-4">
                        <div class="tab-mtp-warning hidden p-6 rounded-2xl border border-amber-500/20 bg-amber-500/5">
                            <div class="flex gap-4 items-start text-amber-500/80">
                                <i class="fas fa-bolt mt-1"></i>
                                <div class="flex-1">
                                    <p class="text-ui-body-sm font-black uppercase tracking-widest mb-1">MTP Indisponível</p>
                                    <p class="tab-mtp-warning-msg text-xs leading-relaxed"></p>
                                </div>
                            </div>
                        </div>
                        <div class="tab-auto-balance-alert hidden p-6 rounded-2xl border border-red-500/20 bg-red-500/5">
                            <div class="flex gap-4 items-start text-red-500/80">
                                <i class="fas fa-microchip mt-1"></i>
                                <div class="flex-1">
                                    <p class="text-ui-body-sm font-black uppercase tracking-widest mb-1">Capacidade do Hardware Excedida</p>
                                    <p class="tab-auto-balance-msg text-xs leading-relaxed"></p>
                                    <ul class="tab-auto-balance-details mt-3 text-ui-body-sm text-slate-500 space-y-1 font-mono"></ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- PAINEL DE LOGS (DIREITA) -->
                <div class="tab-log-panel xl:w-1/3 xl:max-w-[40%] xl:border-l border-t xl:border-t-0 border-slate-800/60 bg-slate-950/40 shadow-2xl relative">
                    <div class="p-6 border-b border-slate-800 bg-slate-900/40 flex items-center justify-between shrink-0">
                        <div class="flex items-center gap-3">
                            <div class="flex gap-1">
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                            </div>
                            <p class="text-ui-body-sm font-black uppercase tracking-[0.25em] text-slate-500 ml-2">Console de Instância</p>
                        </div>
                        <button class="tab-clear-logs-btn text-slate-600 hover:text-red-400 transition-colors" title="Limpar logs">
                            <i class="fas fa-trash-alt text-ui-body-sm"></i>
                        </button>
                    </div>
                    <div class="tab-log-box p-8 font-mono text-sm text-slate-400 leading-6 custom-scroll whitespace-pre-wrap break-words selection:bg-blue-500/20 bg-slate-950/20">
                        <!-- Logs in realtime -->
                    </div>
                    <div class="p-4 bg-slate-900/60 border-t border-slate-800/80 flex items-center justify-between shrink-0">
                         <div class="flex items-center gap-3">
                             <div class="w-2 h-2 rounded-full bg-emerald-500/50 animate-pulse"></div>
                             <span class="tab-log-status text-ui-label font-black text-slate-600 uppercase tracking-widest">Aguardando instância</span>
                         </div>
                         <span class="tab-log-size text-ui-label font-mono text-slate-700">0 KB</span>
                    </div>
                </div>
            </div>
        </div>
    </template>

    <!-- TEMPLATE PARA ABA DE PLATAFORMA CLOUD -->
    <template id="platform-tab-template">
        <div class="tab-content platform-tab-content w-full flex-col" data-tab-kind="platform">
            <div class="tab-layout-row">
                <div class="tab-config-panel flex-1 p-6 md:p-8 space-y-6 bg-slate-900/10">
                    <div class="flex items-center justify-between gap-6 flex-wrap pb-6 border-b border-slate-800/60">
                        <div class="flex items-center gap-5">
                            <div class="w-14 h-14 rounded-2xl bg-violet-600/10 border border-violet-500/20 flex items-center justify-center shadow-inner">
                                <i class="fas fa-cloud text-violet-400 text-xl"></i>
                            </div>
                            <div>
                                <h2 class="platform-tab-name text-2xl font-bold text-white tracking-tight leading-none">Plataforma</h2>
                                <p class="platform-tab-provider text-ui-body-sm text-slate-500 font-mono mt-2 uppercase tracking-tighter"></p>
                            </div>
                        </div>
                        <div class="flex items-center gap-4">
                            <div class="tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500 shadow-sm transition-all">OFFLINE</div>
                            <div class="tab-actions flex items-center gap-3 flex-wrap justify-end"></div>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="glass rounded-[2rem] p-6 space-y-4 shadow-sm">
                            <p class="text-ui-body-sm font-black text-violet-400 uppercase tracking-[0.25em]">Integração</p>
                            <dl class="space-y-3 text-sm">
                                <div class="flex justify-between gap-4"><dt class="text-slate-500 uppercase text-ui-label font-black">Backend</dt><dd class="platform-info-backend-id text-slate-300 font-mono text-right break-all">—</dd></div>
                                <div class="flex justify-between gap-4"><dt class="text-slate-500 uppercase text-ui-label font-black">Executável</dt><dd class="platform-info-executable text-slate-400 font-mono text-right text-xs break-all">—</dd></div>
                                <div class="flex justify-between gap-4"><dt class="text-slate-500 uppercase text-ui-label font-black">CLIProxyAPI</dt><dd class="platform-info-cliproxy text-slate-400 font-mono text-right text-xs break-all">—</dd></div>
                                <div class="flex justify-between gap-4"><dt class="text-slate-500 uppercase text-ui-label font-black">Sidecar</dt><dd class="platform-info-sidecar text-slate-300 font-mono">—</dd></div>
                                <div class="flex justify-between gap-4"><dt class="text-slate-500 uppercase text-ui-label font-black">Início</dt><dd class="platform-info-start-time text-slate-400">—</dd></div>
                            </dl>
                            <p class="platform-info-error hidden text-ui-label text-rose-400/90 leading-relaxed border-t border-slate-800/50 pt-3"></p>
                        </div>

                        <div class="glass rounded-[2rem] p-6 space-y-4 shadow-sm">
                            <div class="flex items-center justify-between gap-3">
                                <p class="text-ui-body-sm font-black text-emerald-400 uppercase tracking-[0.25em]">Autenticação</p>
                                <button type="button" class="platform-auth-btn px-3 py-1.5 rounded-lg border border-amber-500/30 text-amber-300 text-ui-label font-black uppercase tracking-widest hover:bg-amber-500/10 transition-all">
                                    <i class="fas fa-key mr-1"></i> Gerenciar
                                </button>
                            </div>
                            <p class="platform-auth-summary text-ui-body-sm text-slate-400">—</p>
                            <ul class="platform-auth-accounts space-y-1.5 text-ui-label font-mono text-slate-500 max-h-32 overflow-y-auto custom-scroll"></ul>
                            <p class="platform-auth-methods text-ui-label text-slate-600">—</p>
                        </div>
                    </div>

                    <div class="glass rounded-[2rem] p-6 space-y-4 shadow-sm">
                        <div class="flex items-center justify-between gap-3">
                            <p class="text-ui-body-sm font-black text-blue-400 uppercase tracking-[0.25em]">Modelos Disponíveis</p>
                            <button type="button" class="platform-refresh-models-btn px-3 py-1.5 rounded-lg border border-slate-700 text-slate-400 text-ui-label font-black uppercase tracking-widest hover:bg-slate-800 transition-all">
                                <i class="fas fa-sync-alt mr-1"></i> Atualizar
                            </button>
                        </div>
                        <div class="platform-models-list space-y-2 min-h-[3rem]">
                            <p class="text-ui-label text-slate-600 italic">Inicie a integração para listar modelos.</p>
                        </div>
                    </div>

                    <div class="glass rounded-[2rem] p-6 space-y-4 shadow-sm border border-cyan-500/20">
                        <p class="text-ui-body-sm font-black text-cyan-400 uppercase tracking-[0.25em]">Uso no Cursor</p>
                        <p class="text-sm text-slate-400 leading-relaxed">No Cursor BYOK, use alias do catálogo (<span class="font-mono">gpt-4o</span>, <span class="font-mono">gpt-4o-mini</span>, …) ou o ID opaco gerado na listagem — ex.: <span class="font-mono">antigravity-31prolow.gguf</span>. Nomes com <span class="font-mono">gemini</span>/<span class="font-mono">claude</span> são bloqueados pelo Cursor.</p>
                        <ul class="platform-cursor-aliases-list space-y-2 text-ui-label font-mono text-slate-500"></ul>
                        <div class="flex flex-wrap items-end gap-3 pt-2 border-t border-slate-800/50">
                            <label class="space-y-1">
                                <span class="text-ui-label font-black text-slate-600 uppercase">Nome no Cursor</span>
                                <select class="platform-cursor-alias-select bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2 text-sm font-bold min-w-[10rem]"></select>
                            </label>
                            <label class="space-y-1 flex-1 min-w-[12rem]">
                                <span class="text-ui-label font-black text-slate-600 uppercase">Modelo real</span>
                                <select class="platform-cursor-target-select bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2 text-sm font-mono w-full"></select>
                            </label>
                            <button type="button" class="platform-cursor-save-alias px-4 py-2 rounded-xl bg-cyan-600/20 border border-cyan-500/30 text-cyan-300 text-ui-label font-black uppercase tracking-widest hover:bg-cyan-600/30 transition-all">Salvar alias</button>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="glass rounded-[2rem] p-6 space-y-4 shadow-sm">
                            <p class="text-ui-body-sm font-black text-violet-400/80 uppercase tracking-[0.25em]">Modo Proxy Inteligente</p>
                            <div class="flex flex-wrap items-center gap-x-4 gap-y-2 text-ui-label">
                                <label class="flex items-center gap-2 cursor-pointer">
                                    <span class="font-black text-violet-400/80 uppercase">Principal</span>
                                    <input type="checkbox" class="platform-proxy-primary w-4 h-4 bg-slate-900 border-slate-700 rounded text-violet-600">
                                </label>
                                <label class="flex items-center gap-2 cursor-pointer">
                                    <span class="font-black text-slate-500 uppercase">Proxy</span>
                                    <input type="checkbox" class="platform-proxy-eligible w-4 h-4 bg-slate-900 border-slate-700 rounded text-violet-600">
                                </label>
                                <label class="flex items-center gap-2">
                                    <span class="font-black text-slate-500 uppercase">Paralelo</span>
                                    <input type="number" min="1" max="16" class="platform-proxy-parallel w-12 px-1 py-0.5 bg-slate-900 border border-slate-700 rounded text-center text-slate-300">
                                </label>
                                <label class="flex items-center gap-2 cursor-pointer ml-auto">
                                    <span class="font-black text-slate-500 uppercase">Auto-Start</span>
                                    <input type="checkbox" class="platform-autostart w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600">
                                </label>
                            </div>
                            <div class="pt-3 border-t border-slate-800/60 space-y-1.5">
                                <label class="block text-ui-label font-black text-slate-500 uppercase tracking-wider">Modelo padrão (proxy secundário)</label>
                                <select class="platform-proxy-default-model w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-300 text-ui-label">
                                    <option value="">— Nenhum (não encaminhar) —</option>
                                </select>
                                <p class="text-xs text-slate-600 leading-snug">Usado quando esta plataforma não é a principal: requisições encaminhadas ao proxy são atendidas por este modelo.</p>
                            </div>
                        </div>

                        <div class="glass rounded-[2rem] p-6 space-y-3 shadow-sm">
                            <p class="text-ui-body-sm font-black text-amber-500/80 uppercase tracking-[0.25em]">Limites & Uso</p>
                            <p class="platform-limits-info text-sm text-slate-400 leading-relaxed">—</p>
                            <dl class="space-y-2 text-ui-label">
                                <div class="flex justify-between"><dt class="text-slate-600 uppercase font-black">Req. paralelas (proxy)</dt><dd class="platform-limits-parallel text-slate-400 font-mono">1</dd></div>
                                <div class="flex justify-between"><dt class="text-slate-600 uppercase font-black">API AutoManager</dt><dd class="text-slate-500">Chave obrigatória</dd></div>
                            </dl>
                        </div>
                    </div>
                </div>

                <div class="tab-log-panel xl:w-1/3 xl:max-w-[40%] xl:border-l border-t xl:border-t-0 border-slate-800/60 bg-slate-950/40 shadow-2xl relative">
                    <div class="p-6 border-b border-slate-800 bg-slate-900/40 flex items-center justify-between shrink-0">
                        <div class="flex items-center gap-3">
                            <div class="flex gap-1">
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                            </div>
                            <p class="text-ui-body-sm font-black uppercase tracking-[0.25em] text-slate-500 ml-2">Console de Requisições</p>
                        </div>
                        <button class="tab-clear-logs-btn text-slate-600 hover:text-red-400 transition-colors" title="Limpar console">
                            <i class="fas fa-trash-alt text-ui-body-sm"></i>
                        </button>
                    </div>
                    <div class="tab-log-box p-8 font-mono text-sm text-slate-400 leading-6 custom-scroll whitespace-pre-wrap break-words selection:bg-violet-500/20 bg-slate-950/20"></div>
                    <div class="p-4 bg-slate-900/60 border-t border-slate-800/80 flex items-center justify-between shrink-0">
                        <div class="flex items-center gap-3">
                            <div class="w-2 h-2 rounded-full bg-violet-500/50 animate-pulse"></div>
                            <span class="tab-log-status text-ui-label font-black text-slate-600 uppercase tracking-widest">Aguardando instância</span>
                        </div>
                        <span class="tab-log-size text-ui-label font-mono text-slate-700">0 KB</span>
                    </div>
                </div>
            </div>
        </div>
    </template>

    <script>
        window.fixedIp = "{local_ip}";
        window.__constants = {{
            CONTEXT_PRESET_VALUES: {json.dumps(context_preset_values)},
            DEFAULT_CONTEXT_SIZE: {default_context_size},
            CONTEXT_K_MULTIPLIER: {context_k_multiplier},
            DEFAULT_PARALLEL_SLOTS: {default_parallel_slots},
            DEFAULT_BATCH_SIZE: {default_batch_size},
            DEFAULT_CACHE_TYPE: {json.dumps(default_cache_type)},
            DEFAULT_MTP_DRAFT_TOKENS: {default_mtp_draft_tokens},
            DEFAULT_FLASH_ATTN_ENABLED: {json.dumps(DEFAULT_FLASH_ATTN_ENABLED)},
            DEFAULT_MODEL: {json.dumps(default_model)},
            TURBOQUANT_PRESETS: {json.dumps(TURBOQUANT_PRESETS)},
            TURBOQUANT_DEFAULT_CACHE_K: {json.dumps(TURBOQUANT_DEFAULT_CACHE_K)},
            TURBOQUANT_DEFAULT_CACHE_V: {json.dumps(TURBOQUANT_DEFAULT_CACHE_V)},
            TURBOQUANT_CACHE_K_PRESETS: {json.dumps(TURBOQUANT_CACHE_K_PRESETS)},
            TURBOQUANT_CACHE_V_PRESETS: {json.dumps(TURBOQUANT_CACHE_V_PRESETS)},
        }};
    </script>
    <script type="module" src="/static/js/index.js?v={_DASHBOARD_JS_V}"></script>
</body>
</html>"""


      # ─────────────────────────────────────────────────────────
# Startup event — auto-start default model + OOM watchdog
# ─────────────────────────────────────────────────────────


def _chain_shutdown_signals() -> None:
    """Set shutdown_event on SIGTERM/SIGINT before uvicorn drains SSE streams."""
    if os.name != "posix":
        return

    def _wrap(sig: signal.Signals):
        previous = signal.getsignal(sig)

        def handler(signum, frame):
            shutdown_event.set()
            if callable(previous) and previous not in (
                signal.SIG_IGN,
                signal.SIG_DFL,
            ):
                previous(signum, frame)

        signal.signal(sig, handler)

    _wrap(signal.SIGTERM)
    _wrap(signal.SIGINT)


def _auto_start_default_model() -> None:
    """Load the default models in the background so HTTP starts immediately."""
    default_models = config_manager.get_default_models()
    if not default_models:
        return

    logger.info(f"Auto-start requested for: {', '.join(default_models)}")
    
    # Track assigned ports to avoid collisions during batch start
    assigned_ports = set()

    for model_path in default_models:
        if not os.path.exists(model_path):
            logger.warning(f"Auto-start: model file not found: {model_path}")
            continue

        try:
            saved_cfg = config_manager.get_model_settings(model_path)
            if saved_cfg.get("gpu_weights"):
                weights = [
                    GPUWeight(**w) if isinstance(w, dict) else w
                    for w in saved_cfg["gpu_weights"]
                ]
                weights = gpu_manager.normalize_gpu_weights(weights)
                context_size = saved_cfg.get("context_size", DEFAULT_CONTEXT_SIZE)
                parallel_slots = saved_cfg.get("parallel_slots", DEFAULT_PARALLEL_SLOTS)
                batch_size = saved_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
                mmproj_path = saved_cfg.get("mmproj_path")
                mmproj_disabled = saved_cfg.get("mmproj_disabled", False)
                split_mode = saved_cfg.get("split_mode", "layer")
                thinking_enabled = saved_cfg.get("thinking_enabled", True)
                mtp_enabled = saved_cfg.get("mtp_enabled", False)
                mtp_draft_tokens = saved_cfg.get(
                    "mtp_draft_tokens", DEFAULT_MTP_DRAFT_TOKENS
                )
            else:
                gpus = gpu_manager.detect_gpus()
                weights = []
                max_vram = max((g["vram"] for g in gpus), default=0)
                main_gpu_idx = next(
                    (g["index"] for g in gpus if g["vram"] == max_vram), -1
                )
                for g in gpus:
                    val = 100.0 if g["index"] == main_gpu_idx else 0.0
                    weights.append(
                        GPUWeight(
                            index=g["index"],
                            weight=val,
                            name=g["name"],
                        )
                    )
                context_size = DEFAULT_CONTEXT_SIZE
                parallel_slots = DEFAULT_PARALLEL_SLOTS
                batch_size = DEFAULT_BATCH_SIZE
                mmproj_path = None
                mmproj_disabled = False
                split_mode = "layer"
                thinking_enabled = True
                mtp_enabled = False
                mtp_draft_tokens = DEFAULT_MTP_DRAFT_TOKENS

            # Auto-allocate port for this model
            port = SERVER_PORT
            while port in assigned_ports or not process_manager._is_port_free(port):
                port += 1
            assigned_ports.add(port)

            # Wait for port to be truly free before starting
            if not process_manager._wait_port_released(port, timeout=5.0):
                logger.warning(f"Auto-start: port {port} may not be fully released")

            start_result = process_manager.start(
                model_path=model_path,
                gpu_weights=weights,
                context_size=context_size,
                mmproj_path=mmproj_path,
                mmproj_disabled=mmproj_disabled,
                split_mode=split_mode,
                parallel_slots=parallel_slots,
                batch_size=batch_size,
                ubatch_size=saved_cfg.get("ubatch_size", 512),
                cache_type_k=saved_cfg.get("cache_type_k", DEFAULT_CACHE_TYPE),
                cache_type_v=saved_cfg.get("cache_type_v", DEFAULT_CACHE_TYPE),
                numa_enabled=saved_cfg.get("numa_enabled", False),
                flash_attn_enabled=saved_cfg.get(
                    "flash_attn_enabled", DEFAULT_FLASH_ATTN_ENABLED
                ),
                threads=saved_cfg.get("threads", 0),
                threads_batch=saved_cfg.get("threads_batch", 0),
                thinking_enabled=thinking_enabled,
                mtp_enabled=mtp_enabled,
                mtp_draft_tokens=mtp_draft_tokens,
                total_layers=saved_cfg.get("total_layers", 0),
                port=port,
                llama_server_bin=saved_cfg.get("llama_server_bin"),
            )
            logger.info(f"Auto-start: {model_path} started on port {port} (result: {start_result})")
            # Small delay between starts to avoid resource contention peaks
            time.sleep(3)
        except Exception as e:
            logger.error(f"Auto-start error for {model_path}: {e}")


def _auto_start_platforms() -> None:
    """Start platform integrations that are marked for Auto-Start."""
    platform_configs = config_manager.get_platform_configs()
    backend_ids = [
        item["backend_id"]
        for item in platform_manager.catalog()
        if platform_configs.get(item["backend_id"], {}).get("auto_start")
    ]
    if not backend_ids:
        return

    logger.info("Platform auto-start requested for: %s", ", ".join(backend_ids))
    for backend_id in backend_ids:
        try:
            state = platform_manager.start_backend(backend_id, cliproxy_sidecar)
            logger.info(
                "Platform auto-start: %s status=%s port=%s",
                backend_id,
                state.get("status"),
                state.get("sidecar_port"),
            )
        except Exception as exc:
            logger.error("Platform auto-start error for %s: %s", backend_id, exc)


@app.on_event("startup")
async def startup_event():
    """Start OOM watchdog, download runner, and optionally auto-start default model."""
    _chain_shutdown_signals()
    get_llama_server_bin()
    oom_watchdog.start()
    threading.Thread(target=_run_downloads, args=(shutdown_event,), daemon=True).start()
    threading.Thread(
        target=_auto_start_default_model,
        daemon=True,
        name="auto-start",
    ).start()
    threading.Thread(
        target=_auto_start_platforms,
        daemon=True,
        name="platform-auto-start",
    ).start()


@app.on_event("shutdown")
async def shutdown_event_handler():
    """Signal all background tasks to stop and kill llama-server."""
    logger.info("Encerrando Automanager Llama.cpp...")
    shutdown_event.set()
    await proxy_router.flush()
    oom_watchdog.stop()
    process_manager.stop()


# ─────────────────────────────────────────────────────────
# Background download task runner
# ─────────────────────────────────────────────────────────


def _persist_vision_download_mmproj(download_id: str, dest_path: str) -> None:
    """Save mmproj_path to the parent model after a vision download completes."""
    with download_mgr._lock:
        entry = download_mgr._downloads.get(download_id) or {}
    if entry.get("status") != "completed":
        return
    model_path = entry.get("model_path")
    if not model_path:
        return
    if not _is_projector_filename(os.path.basename(dest_path).lower()):
        return
    mmproj_path = os.path.normpath(dest_path).replace("\\", "/")
    config_manager.update_model_settings(model_path, {"mmproj_path": mmproj_path})
    _invalidate_models_cache()


def _run_downloads(stop_event: threading.Event):
    """Periodically process background downloads."""
    while not stop_event.is_set():
        with download_mgr._lock:
            to_process = list(download_mgr._downloads_queue)
            download_mgr._downloads_queue.clear()
        for download_id, url, filename, path in to_process:
            if stop_event.is_set():
                break
            with download_mgr._lock:
                if download_mgr._downloads.get(download_id, {}).get("cancel_requested"):
                    continue
            download_mgr._do_download(download_id, url, filename, path)
            _persist_vision_download_mmproj(download_id, path)
        time.sleep(1)


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────


def get_local_ip() -> str:
    """Detect local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MANAGER_PORT,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    )
