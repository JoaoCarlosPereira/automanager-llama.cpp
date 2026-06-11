#!/usr/bin/env python3
"""
Automanager Llama.cpp - Control Plane para llama-server
FastAPI application that orchestrates llama-server instances with multi-GPU
tensor split management, OOM auto-recovery, and real-time hardware monitoring.
"""

import json
import os
import asyncio
import signal
import socket
import subprocess
import threading
import time
import httpx

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

from config_manager import (
    ConfigManager,
    TokenManager,
    AuthManager,
    lookup_model_config,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_BATCH_SIZE,
)
from gpu_manager import GPUManager
from log_manager import LogManager, logger
from process_manager import ProcessManager, OOMWatchdog, SERVER_PORT
from model_manager import ModelScanner, DownloadManager
from schemas import (
    BATCH_SIZE_PRESETS,
    DEFAULT_MTP_DRAFT_TOKENS,
    GPUWeight,
    StartRequest,
    DeleteRequest,
    DownloadRequest,
    SetMmprojRequest,
    SetThinkingRequest,
    SetDefaultRequest,
    RenameRequest,
    LoginRequest,
    SetModelsDirRequest,
    VersionCheckResponse,
    VersionCommit,
)
from paths import INSTALL_ROOT, ensure_directories, reload_module_paths, update_models_dir
from version_manager import check_for_updates
from llama_server_bin import get_llama_server_bin

MANAGER_PORT = 8000

CONTEXT_PRESETS = [
    (2048, "2K"),
    (4096, "4K"),
    (8192, "8K"),
    (16384, "16K"),
    (32768, "32K"),
    (65536, "64K"),
    (131072, "128K"),
    (262144, "256K"),
    (524288, "512K"),
    (1048576, "1M"),
]
CONTEXT_PRESET_VALUES = [v for v, _ in CONTEXT_PRESETS]
# Custom context input is in K (100 → 100_000 tokens)
CONTEXT_K_MULTIPLIER = 1000


def context_tokens_to_k(tokens: int) -> str:
    """Convert token count to K for the custom input (100 → '100')."""
    k = tokens / CONTEXT_K_MULTIPLIER
    if k == int(k):
        return str(int(k))
    rounded = round(k, 3)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


# Initialize logging and services
_install_paths = ensure_directories()
log_manager = LogManager(
    project_root=_install_paths.install_root,
    server_log_path=_install_paths.server_log,
    manager_log_path=_install_paths.manager_log,
)
config_manager = ConfigManager(_install_paths.config_file)
token_manager = TokenManager(config_manager)
auth_manager = AuthManager(config_manager, token_manager)
gpu_manager = GPUManager()
gpu_detector = gpu_manager
process_manager = ProcessManager(
    config_manager, token_manager, gpu_manager, log_manager
)
model_scanner = ModelScanner(
    config_manager, process_manager, models_dir=_install_paths.models_dir
)
download_mgr = DownloadManager(models_dir=_install_paths.models_dir)
oom_watchdog = OOMWatchdog(
    process_manager, config_manager, gpu_manager, log_manager
)

app = FastAPI(title="Automanager Llama.cpp")
shutdown_event = threading.Event()

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─────────────────────────────────────────────────────────
# Security Headers Middleware
# ─────────────────────────────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "connect-src 'self' ws://localhost:8000 http://localhost:8085; "
            "img-src 'self' data: blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)


def _dashboard_js_version() -> str:
    """Cache-bust dashboard bundles when any core JS file changes."""
    js_dir = os.path.join(_static_dir, "js")
    mtimes = []
    for name in ("gpu.js", "auth.js", "metrics.js", "models.js", "version.js", "index.js"):
        path = os.path.join(js_dir, name)
        if os.path.isfile(path):
            mtimes.append(os.path.getmtime(path))
    return str(int(max(mtimes))) if mtimes else "0"


_DASHBOARD_JS_V = _dashboard_js_version()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = os.path.join(_static_dir, "favicon.svg")
    if os.path.isfile(icon_path):
        return FileResponse(icon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404)

# Security scheme for API key
security = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────────────────────
# Auth middleware / dependencies
# ─────────────────────────────────────────────────────────


def get_current_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    Authenticate via session cookie or API key.
    Returns 'session' or 'api' token on success.
    """
    # Check session cookie first (for UI routes)
    session_token = request.cookies.get("session_token")
    if session_token and auth_manager.verify_session(session_token):
        return session_token

    # Check API key (for API routes)
    if credentials and auth_manager.verify_api_key(credentials):
        return credentials.credentials

    raise HTTPException(
        status_code=401,
        detail="Nao autorizado. Faça login ou envie um token valido.",
    )


def optional_auth(request: Request) -> Optional[str]:
    """Optional auth — returns token or None (for index page that handles auth on frontend)."""
    session_token = request.cookies.get("session_token")
    if session_token and auth_manager.verify_session(session_token):
        return session_token
    credentials = None  # We can't use Depends here
    return None


# ─────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────

# --- Authentication ---


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    if len(req.username) < 1 or len(req.username) > 128:
        raise HTTPException(status_code=400, detail="Nome de usuario invalido")
    if len(req.password) > 256:
        raise HTTPException(status_code=400, detail="Senha muito longa")
    result = auth_manager.authenticate(req.username, req.password)
    if not result:
        raise HTTPException(status_code=401, detail="Credenciais invalidas")
    # Backward compatible: handle both old (string) and new (dict) return formats
    if isinstance(result, dict):
        session_token = result.get("token", result.get("session_token"))
        force_change = result.get("force_password_change", False)
    else:
        session_token = result
        force_change = False
    response = JSONResponse({"status": "ok", "force_password_change": force_change})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=3600,
        path="/",
    )
    return response


@app.post("/api/auth/logout")
async def api_logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token:
        auth_manager.logout(session_token)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(key="session_token")
    return response


@app.post("/api/auth/change-password")
async def change_password(
    req: LoginRequest,
    _auth: str = Depends(get_current_auth),
):
    # Old password in req.username, new in req.password
    if auth_manager.change_password(req.username, req.password):
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Senha atual incorreta")


# --- llma-server Control ---

# Lightweight cache for /status (psutil.process_iter is blocking)
_status_cache: dict = {}
_status_cache_time: float = 0.0
_STATUS_CACHE_TTL: float = 1.0  # seconds


@app.get("/status")
async def get_status(_auth: str = Depends(get_current_auth)):
    global _status_cache, _status_cache_time
    loop = asyncio.get_event_loop()
    # Check sync cache first to avoid executor overhead
    now = time.monotonic()
    if _status_cache and (now - _status_cache_time) < _STATUS_CACHE_TTL:
        return _status_cache

    result = await loop.run_in_executor(None, process_manager.get_status)
    _status_cache = result
    _status_cache_time = now
    return result


@app.post("/start")
async def start_model(
    req: StartRequest, _auth: str = Depends(get_current_auth)
):
    process_manager.recovery_state = {
        "active": False,
        "failed": False,
        "message": "",
        "auto_balance": False,
    }

    # Sob Auto-Balance, a cascata controla 100% da distribuição: pins são
    # ignorados (ADR-001). Normaliza no ponto único de entrada, garantindo o
    # comportamento mesmo em chamadas diretas à API.
    if req.auto_balance:
        for w in req.gpu_weights:
            w.pinned = False

    # Auto-detect total_layers from model file
    total_layers = req.total_layers if req.total_layers and req.total_layers > 0 else 0
    try:
        if not total_layers:
            total_layers = gpu_manager.detect_model_layers(req.path)
    except Exception:
        total_layers = 0

    base_settings = {
        "context_size": req.context_size,
        "parallel_slots": req.parallel_slots,
        "batch_size": req.batch_size,
        "mmproj_path": req.mmproj_path,
        "split_mode": req.split_mode,
        "auto_balance": req.auto_balance,
        "thinking_enabled": req.thinking_enabled,
        "mtp_enabled": req.mtp_enabled,
        "mtp_draft_tokens": req.mtp_draft_tokens,
        "total_layers": total_layers if total_layers else 0,
    }

    if req.auto_balance:
        return process_manager.start_auto_balance(req)

    config_manager.update_model_settings(
        req.path,
        {
            **base_settings,
            "gpu_weights": [w.model_dump() for w in req.gpu_weights],
            "auto_balance_profile": False,
        },
    )
    result = process_manager.start(
        model_path=req.path,
        gpu_weights=req.gpu_weights,
        context_size=req.context_size,
        mmproj_path=req.mmproj_path,
        split_mode=req.split_mode,
        parallel_slots=req.parallel_slots,
        batch_size=req.batch_size,
        thinking_enabled=req.thinking_enabled,
        mtp_enabled=req.mtp_enabled,
        mtp_draft_tokens=req.mtp_draft_tokens,
        total_layers=total_layers,
        cpu_enabled=req.cpu_enabled,
        port=req.port,
    )
    _invalidate_models_cache()
    return result


@app.post("/stop")
async def stop_model(port: Optional[int] = None, _auth: str = Depends(get_current_auth)):
    result = process_manager.stop(port=port)
    _invalidate_status_cache()
    return result


@app.post("/auto-balance/cancel")
async def cancel_auto_balance(_auth: str = Depends(get_current_auth)):
    return process_manager.cancel_auto_balance()


# --- Hardware Metrics ---


@app.get("/metrics")
async def get_metrics(_auth: str = Depends(get_current_auth)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, gpu_manager.get_metrics)


# --- Model Management ---

# Cache for the expensive model scan (os.walk + disk stats)
_models_cache: dict = {}
_models_cache_time: float = 0.0
_MODELS_CACHE_TTL: float = 5.0  # seconds — models don't change while running

# Cache for the dashboard GET / endpoint (chained blocking calls)
_dashboard_cache: dict = {}
_dashboard_cache_time: float = 0.0
_DASHBOARD_CACHE_TTL: float = 30.0  # seconds — dashboard data is semi-static


def _invalidate_models_cache() -> None:
    """Clear the models and dashboard cache (call after model operations that change the filesystem)."""
    global _models_cache, _models_cache_time, _dashboard_cache, _dashboard_cache_time
    _models_cache = {}
    _models_cache_time = 0.0
    _dashboard_cache = {}
    _dashboard_cache_time = 0.0


def _invalidate_status_cache() -> None:
    """Clear the status cache (call after process changes)."""
    global _status_cache, _status_cache_time
    _status_cache = {}
    _status_cache_time = 0.0


async def _scan_models_async() -> dict:
    """Run the blocking model scan off the event loop thread with caching."""
    global _models_cache, _models_cache_time
    now = time.monotonic()
    if _models_cache and (now - _models_cache_time) < _MODELS_CACHE_TTL:
        return _models_cache

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model_scanner.scan)
    _models_cache = result
    _models_cache_time = now
    return result


@app.get("/models")
async def list_models(_auth: str = Depends(get_current_auth)):
    return await _scan_models_async()


@app.post("/models/dir")
async def set_models_dir(
    req: SetModelsDirRequest, _auth: str = Depends(get_current_auth)
):
    try:
        paths = update_models_dir(req.models_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("Failed to update models_dir: %s", exc)
        raise HTTPException(
            status_code=500, detail="Não foi possível criar ou salvar o diretório"
        ) from exc

    reload_module_paths()
    model_scanner.models_dir = paths.models_dir
    download_mgr.models_dir = paths.models_dir
    logger.info("models_dir atualizado para %s", paths.models_dir)
    _invalidate_models_cache()
    return await _scan_models_async()


@app.post("/rename")
async def rename_model(
    req: RenameRequest, _auth: str = Depends(get_current_auth)
):
    new_path = model_scanner.rename_model(req.path, req.new_name)
    _invalidate_models_cache()
    return {"status": "renamed", "new_path": new_path}


@app.post("/delete")
async def delete_model(
    req: DeleteRequest, _auth: str = Depends(get_current_auth)
):
    model_scanner.delete_model(req.path)
    _invalidate_models_cache()
    return {"status": "deleted"}


@app.post("/models/mmproj")
async def set_model_mmproj(
    req: SetMmprojRequest, _auth: str = Depends(get_current_auth)
):
    config_manager.update_model_settings(
        req.model_path, {"mmproj_path": req.mmproj_path}
    )
    return {"status": "ok", "mmproj_path": req.mmproj_path}


@app.post("/models/thinking")
async def set_model_thinking(
    req: SetThinkingRequest, _auth: str = Depends(get_current_auth)
):
    config_manager.update_model_settings(
        req.model_path, {"thinking_enabled": req.thinking_enabled}
    )
    return {"status": "ok", "thinking_enabled": req.thinking_enabled}


# --- Downloads ---


@app.post("/downloads")
async def start_download_endpoint(
    req: DownloadRequest,
    background_tasks: BackgroundTasks,
    _auth: str = Depends(get_current_auth),
):
    download_id = download_mgr.start_download(
        req.url, req.filename, model_path=req.model_path
    )
    return {"download_id": download_id}


@app.get("/downloads")
async def get_downloads(_auth: str = Depends(get_current_auth)):
    return download_mgr.get_progress()


@app.post("/downloads/clear")
async def clear_downloads(_auth: str = Depends(get_current_auth)):
    download_mgr.clear_completed()
    return {"status": "cleared"}


# --- API Key Management ---


@app.get("/api/key")
async def get_api_key(_auth: str = Depends(get_current_auth)):
    return {"key": token_manager.get_or_create()}


@app.post("/api/key/renew")
async def renew_api_key(_auth: str = Depends(get_current_auth)):
    return {"key": token_manager.renew()}


# --- Logging ---


@app.get("/logs")
async def stream_logs(
    request: Request,
    port: Optional[int] = None,
    _auth: str = Depends(get_current_auth),
):
    return log_manager.stream_logs(stop_event=shutdown_event, request=request, port=port)


# --- OpenAI Proxy (Multi-Model Support) ---

# Client HTTP compartilhado para o proxy
client = httpx.AsyncClient()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(request: Request, path: str, _auth: str = Depends(get_current_auth)):
    """
    Proxy OpenAI-compatible requests to the correct llama-server instance
    based on the 'model' field in the request body.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Proxy request from {client_ip}: {request.method} /v1/{path}")
    
    body = await request.body()
    requested_model = None

    # Tentar extrair o modelo do corpo (JSON)
    if body:
        try:
            data = json.loads(body)
            requested_model = data.get("model")
        except json.JSONDecodeError:
            pass

    status = process_manager.get_status()
    instances = status.get("instances", [])

    if not instances:
        raise HTTPException(status_code=503, detail="Nenhum modelo carregado")

    target_instance = None
    if requested_model:
        # Procurar por nome exato ou basename
        for inst in instances:
            if inst["model"] == requested_model or inst["model_path"] == requested_model:
                target_instance = inst
                break
        
        if not target_instance:
             # Fallback: se o usuário pediu um modelo mas ele não está rodando, 
             # opcionalmente poderíamos tentar carregar, mas por agora retornamos erro.
             raise HTTPException(status_code=404, detail=f"Modelo '{requested_model}' nao esta carregado.")
    else:
        # Se nenhum modelo foi especificado, usar o da porta 8085 (default)
        target_instance = next((i for i in instances if i["port"] == 8085), instances[0])

    target_url = f"http://127.0.0.1:{target_instance['port']}/v1/{path}"
    
    # Repassar headers, removendo o host original
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        # Encaminhar a requisição
        if request.method == "POST":
            # Streaming support
            if requested_model and data.get("stream"):
                async def stream_generator():
                    async with client.stream(
                        "POST", target_url, content=body, headers=headers, timeout=None
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            
            resp = await client.post(target_url, content=body, headers=headers, timeout=None)
        elif request.method == "GET":
            resp = await client.get(target_url, params=request.query_params, headers=headers)
        else:
            # Outros métodos simples
            resp = await client.request(request.method, target_url, content=body, headers=headers)

        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )
    except httpx.RequestError as exc:
        logger.error(f"Proxy error to port {target_instance['port']}: {exc}")
        raise HTTPException(status_code=502, detail="Erro ao conectar na instancia do modelo")


@app.api_route("/ui/{port}/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def ui_proxy(request: Request, port: int, path: str):
    """
    Proxy web UI requests to a specific llama-server instance.
    Injects a <base> tag into HTML to fix relative asset paths.
    """
    # Simple path mapping for the llama.cpp UI
    is_index = False
    if not path or path == "index.html":
        path = ""
        is_index = True
    
    target_url = f"http://127.0.0.1:{port}/{path}"
    query = str(request.query_params)
    if query:
        target_url += f"?{query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    # Important: some UIs check referer for CSRF
    # headers.pop("referer", None) 

    try:
        if is_index:
            # Fetch index and rewrite
            resp = await client.get(target_url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                html = resp.text
                base_tag = f'<base href="/ui/{port}/">'
                # Insert base tag after <head>
                if "<head>" in html:
                    html = html.replace("<head>", f"<head>{base_tag}")
                else:
                    html = f"<head>{base_tag}</head>{html}"
                
                # SvelteKit specific: fix __sveltekit__ base if found
                html = html.replace("base: new URL('.', location).pathname.slice(0, -1)", f'base: "/ui/{port}"')

                return HTMLResponse(content=html, status_code=200)
        
        # Proxy other assets/API calls directly
        resp = await client.request(
            request.method, 
            target_url, 
            content=await request.body(), 
            headers=headers,
            timeout=10.0
        )
        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )
    except httpx.RequestError as exc:
        logger.error(f"UI Proxy error to port {port}: {exc}")
        raise HTTPException(status_code=502, detail="Interface do modelo inacessivel")


# --- System Management ---


@app.post("/api/system/shutdown")
async def system_shutdown(
    _auth: str = Depends(get_current_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Shut down the entire system."""
    background_tasks.add_task(_execute_shutdown)
    return {"status": "shutdown_initiated", "message": "Desligamento do sistema iniciado"}


@app.post("/api/system/update")
async def system_update(
    _auth: str = Depends(get_current_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Pull latest code via git and restart the service."""
    background_tasks.add_task(_execute_update)
    return {"status": "update_initiated", "message": "Atualização e reinício do serviço iniciado"}


@app.get("/api/system/version-check", response_model=VersionCheckResponse)
async def system_version_check(_auth: str = Depends(get_current_auth)):
    """Compare local git HEAD with origin and return ahead commits."""
    result = check_for_updates(INSTALL_ROOT)
    return VersionCheckResponse(
        status=result.status,
        update_available=result.update_available,
        current_ref=result.current_ref,
        remote_ref=result.remote_ref,
        branch=result.branch,
        commits=[
            VersionCommit(
                sha=c.sha,
                message=c.message,
                author=c.author,
                date=c.date,
            )
            for c in result.commits
        ],
        error_message=result.error_message,
    )


def _execute_shutdown():
    """Execute poweroff command."""
    logger.info("Solicitado desligamento do sistema")
    try:
        subprocess.run(["poweroff"], check=False)
    except Exception as e:
        logger.error(f"Erro ao desligar: {e}")


def _execute_update():
    """Pull latest code and restart llama-manager service."""
    logger.info("Solicitada atualização do sistema")
    app_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Git pull
        result = subprocess.run(
            ["git", "-C", app_dir, "pull"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            logger.error(f"Git pull falhou: {result.stderr}")
        else:
            logger.info("Git pull concluído com sucesso")
            
        # Restart service
        # Use Popen to avoid waiting for a result that might be interrupted by the restart itself
        subprocess.Popen(["systemctl", "restart", "llama-manager.service"])
    except Exception as e:
        logger.error(f"Erro na atualizacao: {e}")


# --- Configuration ---


@app.get("/config")
async def get_config(_auth: str = Depends(get_current_auth)):
    config = config_manager.load()
    # Don't expose password hash
    config.pop("admin_password_hash", None)
    return config


@app.post("/set_default")
async def set_default(
    req: SetDefaultRequest, _auth: str = Depends(get_current_auth)
):
    config_manager.set_default_model(req.path, add=req.add)
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────
# Frontend (Steps 10–16)
# Embedded SPA with login overlay, dashboard, and all JS
# ─────────────────────────────────────────────────────────


def _resolve_thinking_enabled(
    status: dict, default_model: Optional[str], model_configs: dict
) -> bool:
    """UI/server default for thinking toggle from running or saved config."""
    if status.get("running") and status.get("config"):
        te = status["config"].get("thinking_enabled")
        if te is not None:
            return bool(te)
    if default_model:
        saved = lookup_model_config(model_configs, default_model)
        if "thinking_enabled" in saved:
            return bool(saved["thinking_enabled"])
    return True


def _resolved_mmproj_path(model: dict, model_cfg: dict) -> Optional[str]:
    candidates = model.get("mmproj_candidates") or []
    if not candidates:
        return None
    saved = model_cfg.get("mmproj_path")
    if saved and saved in candidates:
        return saved
    return candidates[0]


def _build_model_vision_controls(model: dict, model_js: str, model_cfg: dict) -> str:
    candidates = model.get("mmproj_candidates") or []
    import_btn = (
        f'<button type="button" onclick="event.stopPropagation(); '
        f"openVisionImportModal('{model_js}')\" "
        'class="vision-import-btn w-10 h-10 flex items-center justify-center rounded-xl '
        'hover:bg-violet-500/20 text-slate-600 hover:text-violet-400 transition-all" '
        'title="Importar projetor de visao" aria-label="Importar projetor de visao">'
        '<i class="fas fa-eye text-[10px] md:text-xs"></i></button>'
    )
    if not candidates:
        return import_btn
    selected = _resolved_mmproj_path(model, model_cfg)
    options = ""
    for candidate in candidates:
        name = os.path.basename(candidate)
        selected_attr = " selected" if candidate == selected else ""
        options += (
            f'<option value="{candidate}" class="bg-slate-900"{selected_attr}>'
            f"{name}</option>"
        )
    return (
        f"{import_btn}"
        f'<select data-mmproj-for="{model_js}" '
        'class="model-mmproj-select bg-slate-900 border border-slate-700 text-slate-300 '
        'rounded-xl px-3 py-2 text-[10px] font-bold focus:ring-2 focus:ring-violet-500/50 '
        'outline-none transition-all cursor-pointer max-w-[180px]" '
        f"onchange=\"onMmprojChange('{model_js}', this)\" "
        'onclick="event.stopPropagation()" '
        'title="Projetor de visao para este modelo" '
        'aria-label="Projetor de visao para este modelo">'
        f"{options}</select>"
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the dashboard SPA."""
    # Check auth — if not authenticated, show login overlay
    session_token = request.cookies.get("session_token")
    is_authenticated = (
        session_token and auth_manager.verify_session(session_token)
    )

    # Run blocking calls off the event loop thread
    loop = asyncio.get_event_loop()
    models_data, gpus, cpu_info, config, status = await loop.run_in_executor(
        None,
        lambda: (
            model_scanner.scan(),
            gpu_manager.detect_gpus(),
            gpu_manager.detect_cpu_info(),
            config_manager.load(),
            process_manager.get_status(),
        ),
    )
    models = models_data["models"]
    projectors = models_data["projectors"]
    model_configs = config.get("model_configs", {})
    default_model = config.get("default_model")
    default_models_list = config.get("default_models")
    if isinstance(default_models_list, list) and default_models_list:
        default_model = default_models_list[0]

    # Build GPU rows
    max_vram = max((g["vram"] for g in gpus), default=0)
    main_gpu_idx = next(
        (g["index"] for g in gpus if g["vram"] == max_vram), -1
    )
    gpu_rows = ""
    for g in gpus:
        idx = g["index"]
        is_main = False
        if (
            status.get("running")
            and status.get("config", {}).get("gpu_weights")
        ):
            w_list = status["config"]["gpu_weights"]
            if isinstance(w_list, list) and len(w_list) > 0:
                # Find weight for this GPU
                w_obj = next(
                    (w for w in w_list if isinstance(w, dict) and w.get("index") == idx),
                    None,
                )
                if w_obj:
                    is_checked = (
                        "checked"
                        if w_obj.get("active", True)
                        else ""
                    )
                    weight_val = int(w_obj.get("weight", 0))
                    is_main = w_obj.get("is_main", False)
                    pin_checked = "checked" if w_obj.get("pinned") else ""
                else:
                    is_checked = "checked"
                    weight_val = 100 if idx == main_gpu_idx else 0
                    is_main = (idx == main_gpu_idx)
                    pin_checked = ""
            else:
                is_checked = "checked"
                weight_val = 100 if idx == main_gpu_idx else 0
                is_main = (idx == main_gpu_idx)
                pin_checked = ""
        else:
            is_checked = "checked"
            weight_val = 100 if idx == main_gpu_idx else 0
            is_main = (idx == main_gpu_idx)
            pin_checked = ""

        main_checked = "checked" if is_main else ""

        gpu_rows += f"""
        <tr class="gpu-row group border-b border-slate-800/50" data-index="{idx}">
            <td class="px-3 md:px-6 py-4 md:py-6 text-center">
                <div class="flex flex-col items-center gap-2">
                    <span class="gpu-util-val text-xs font-black text-blue-400 font-mono">0%</span>
                    <div class="w-12 h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div class="gpu-util-bar h-full bg-blue-500 transition-all duration-1000" style="width: 0%"></div>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6 text-center">
                <input type="radio" name="main-gpu-radio" {main_checked} class="gpu-main-radio w-4 h-4 bg-slate-900 border-slate-700 rounded-full text-blue-600 cursor-pointer" title="Definir como GPU Principal">
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-4">
                    <input type="checkbox" {is_checked} class="gpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                    <div class="flex flex-col">
                        <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-0.5">ID {idx}</span>
                        <span class="text-sm font-bold text-slate-100 whitespace-nowrap">{g['name']}</span>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col md:flex-row gap-2 md:gap-6">
                    <div class="flex flex-col">
                        <span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Temp</span>
                        <span class="gpu-temp-val text-xs font-bold text-slate-300 font-mono">--°C</span>
                    </div>
                    <div class="flex flex-col">
                        <span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Power</span>
                        <span class="gpu-power-val text-xs font-bold text-slate-300 font-mono">--W</span>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col gap-2 min-w-[100px] md:min-w-[160px]">
                    <div class="flex justify-between items-end">
                        <span class="text-[8px] font-black text-slate-500 uppercase">VRAM</span>
                        <span class="gpu-vram-text text-[9px] font-mono text-blue-400">0 / {g['vram']} MB</span>
                    </div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div class="gpu-vram-bar h-full bg-cyan-500 transition-all duration-1000" style="width: 0%"></div>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-3">
                    <label class="flex items-center gap-1.5 cursor-pointer shrink-0" title="Fixar % — o auto balance nao altera esta GPU">
                        <input type="checkbox" {pin_checked} class="gpu-pin w-4 h-4 bg-slate-900 border-slate-700 rounded text-amber-500 cursor-pointer">
                        <span class="text-[8px] font-black text-slate-500 uppercase tracking-wider hidden md:inline">Fixar</span>
                    </label>
                    <div class="relative">
                        <input type="number" value="{weight_val}" min="0" max="100"
                               class="gpu-weight w-20 md:w-24 pl-2 md:pl-4 pr-7 md:pr-9 py-2 bg-slate-900/80 border border-slate-700 rounded-xl text-sm font-black text-blue-400 outline-none transition-all"
                               oninput="balanceWeights(this)">
                        <span class="absolute right-2 md:right-3 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-600">%</span>
                    </div>
                </div>
            </td>
        </tr>"""

    # Build CPU row (follows the same visual pattern as GPU rows, no radio button)
    cpu_name_display = cpu_info.name if cpu_info.name else "CPU Desconhecido"
    cpu_ram_total = cpu_info.ram_total_mb if cpu_info.ram_total_mb else 0
    cpu_ram_used = cpu_info.ram_used_mb if cpu_info.ram_used_mb else 0
    cpu_checked = ""
    cpu_weight_val = 0
    cpu_pin_checked = ""
    if (
        status.get("running")
        and status.get("config", {}).get("gpu_weights")
    ):
        w_list = status["config"]["gpu_weights"]
        if isinstance(w_list, list):
            cpu_obj = next(
                (
                    w
                    for w in w_list
                    if isinstance(w, dict)
                    and (
                        w.get("device") == "cpu"
                        or w.get("index") == -1
                    )
                ),
                None,
            )
            if cpu_obj:
                cpu_checked = "checked" if cpu_obj.get("active", False) else ""
                cpu_weight_val = int(cpu_obj.get("weight", 0))
                cpu_pin_checked = "checked" if cpu_obj.get("pinned") else ""
    cpu_rows = f"""
        <tr id="cpu-row" class="cpu-row group border-b border-slate-800/50" data-index="cpu">
            <td class="px-3 md:px-6 py-4 md:py-6 text-center">
                <div class="flex flex-col items-center gap-2">
                    <span class="cpu-util-val text-xs font-black text-blue-400 font-mono">0%</span>
                    <div class="w-12 h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div class="cpu-util-bar h-full bg-blue-500 transition-all duration-1000" style="width: 0%"></div>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6 text-center">
                <!-- CPU nao tem radio principal -->
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-4">
                    <input type="checkbox" {cpu_checked} class="cpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                    <div class="flex flex-col">
                        <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-0.5">CPU</span>
                        <span class="text-sm font-bold text-slate-100 whitespace-nowrap">{cpu_name_display}</span>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col md:flex-row gap-2 md:gap-6">
                    <div class="flex flex-col">
                        <span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Temp</span>
                        <span class="cpu-temp-val text-xs font-bold text-slate-300 font-mono">--°C</span>
                    </div>
                    <div class="flex flex-col">
                        <span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Power</span>
                        <span class="cpu-power-val text-xs font-bold text-slate-300 font-mono">--W</span>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col gap-2 min-w-[100px] md:min-w-[160px]">
                    <div class="flex justify-between items-end">
                        <span class="text-[8px] font-black text-slate-500 uppercase">RAM</span>
                        <span class="cpu-ram-text text-[9px] font-mono text-blue-400">0 / {cpu_ram_total} MB</span>
                    </div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div class="cpu-ram-bar h-full bg-amber-500 transition-all duration-1000" style="width: 0%"></div>
                    </div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-3">
                    <label class="flex items-center gap-1.5 cursor-pointer shrink-0" title="Fixar % — o auto balance nao altera esta CPU">
                        <input type="checkbox" {cpu_pin_checked} class="cpu-pin w-4 h-4 bg-slate-900 border-slate-700 rounded text-amber-500 cursor-pointer">
                        <span class="text-[8px] font-black text-slate-500 uppercase tracking-wider hidden md:inline">Fixar</span>
                    </label>
                    <div class="relative">
                        <input type="number" value="{cpu_weight_val}" min="0" max="100"
                               class="cpu-weight w-20 md:w-24 pl-2 md:pl-4 pr-7 md:pr-9 py-2 bg-slate-900/80 border border-slate-700 rounded-xl text-sm font-black text-blue-400 outline-none transition-all"
                               oninput="balanceWeights(this)">
                        <span class="absolute right-2 md:right-3 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-600">%</span>
                    </div>
                </div>
            </td>
        </tr>"""

    # Context options (presets + manual via "Personalizado")
    running_ctx = (
        status.get("config", {}).get("context_size")
        if status.get("running")
        else None
    )
    use_custom_ctx = (
        running_ctx is not None and running_ctx not in CONTEXT_PRESET_VALUES
    )
    ctx_opts = ""
    for val, label in CONTEXT_PRESETS:
        if use_custom_ctx:
            selected = ""
        elif running_ctx is not None:
            selected = "selected" if running_ctx == val else ""
        else:
            selected = "selected" if val == DEFAULT_CONTEXT_SIZE else ""
        ctx_opts += f'<option value="{val}" class="bg-slate-900" {selected}>{label}</option>'
    custom_selected = "selected" if use_custom_ctx else ""
    ctx_opts += (
        f'<option value="custom" class="bg-slate-900" {custom_selected}>'
        "Personalizado</option>"
    )
    custom_ctx_value = (
        context_tokens_to_k(running_ctx) if use_custom_ctx else ""
    )
    custom_ctx_class = "" if use_custom_ctx else "hidden"

    running_batch = (
        status.get("config", {}).get("batch_size")
        if status.get("running")
        else None
    )
    batch_opts = ""
    for val in BATCH_SIZE_PRESETS:
        if running_batch is not None:
            selected = "selected" if running_batch == val else ""
        else:
            selected = "selected" if val == DEFAULT_BATCH_SIZE else ""
        batch_opts += (
            f'<option value="{val}" class="bg-slate-900" {selected}>{val}</option>'
        )

    thinking_enabled_ui = _resolve_thinking_enabled(
        status, default_model, model_configs
    )
    thinking_checked = "checked" if thinking_enabled_ui else ""
    thinking_badge_class = (
        "text-violet-400" if thinking_enabled_ui else "text-slate-500"
    )
    thinking_badge_text = "ON" if thinking_enabled_ui else "OFF"

    # Model items
    model_items = ""
    for m in models:
        m_path = m["path"]
        m_js = m_path.replace("\\", "/")
        m_name = m["name"]
        m_dir = m["dir"]
        is_default = "checked" if m_path == default_model else ""
        stable_id = f"model-item-{abs(sum(ord(c) << (i % 8) for i, c in enumerate(m_path))) % 1000000}"
        initial_cfg_js = ""
        m_cfg = lookup_model_config(model_configs, m_path)
        if m_cfg:
            initial_cfg_js = (
                f"<script>window.modelConfigs['{m_js}'] = {json.dumps(m_cfg)};</script>"
            )
        has_config = "text-blue-400" if m_cfg else "text-slate-100"
        hardware_incapable = bool(m_cfg.get("hardware_incapable"))
        incapable_row_class = (
            "border-red-500/40 bg-red-950/20" if hardware_incapable else ""
        )
        incapable_badge = (
            '<span class="shrink-0 text-[8px] font-black uppercase tracking-wider '
            'text-red-400 bg-red-500/15 px-2 py-0.5 rounded-lg border border-red-500/30" '
            'title="Incompativel com o hardware atual (auto balance)">Incapaz</span>'
            if hardware_incapable
            else ""
        )
        incapable_attr = "true" if hardware_incapable else "false"
        vision_controls = _build_model_vision_controls(m, m_js, m_cfg)

        model_items += f"""
        {initial_cfg_js}
        <div id="{stable_id}" class="model-item-container group flex flex-col gap-4 p-4 md:p-5 mb-3 md:mb-4 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg {incapable_row_class}" data-path="{m_js}" data-hardware-incapable="{incapable_attr}">
            <div class="w-full cursor-pointer" onclick="selectModel('{m_js}', '{stable_id}')" title="Clique para selecionar e carregar configuracoes">
                <div class="flex items-start gap-2 mb-1 flex-wrap">
                    <i class="fas fa-cube text-blue-400 text-[10px] mt-1"></i>
                    <p class="model-name text-sm font-bold {has_config} break-words">{m_name}</p>
                    {incapable_badge}
                    {'<i class="fas fa-history text-[8px] text-blue-500/50 history-icon" title="Configuracao salva disponivel"></i>' if m_cfg and not hardware_incapable else ''}
                </div>
                <p class="text-[9px] text-slate-500 break-all uppercase tracking-tighter font-mono">{m_dir}</p>
            </div>
            <div class="flex flex-wrap items-center gap-3 md:gap-4 w-full">
                <div class="flex items-center gap-1">
                    <button onclick="renameModel('{m_js}')" class="rename-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all" title="Renomear Modelo">
                        <i class="fas fa-edit text-[10px]"></i>
                    </button>
                    <button onclick="deleteModel('{m_js}')" class="delete-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all" title="Excluir Modelo">
                        <i class="fas fa-trash-alt text-[10px]"></i>
                    </button>
                    {vision_controls}
                </div>
                <div class="flex flex-col items-center gap-1">
                    <span class="text-[8px] font-black text-slate-600 uppercase tracking-tighter">Padrao</span>
                    <input type="checkbox" class="model-default-checkbox w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer"
                           {is_default} onclick="setDefaultModel(this, '{m_js}')">
                </div>
                <div class="action-btn-container">
                    <button onclick="startModel('{m_js}', '{stable_id}')" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-xl active:scale-95 flex items-center gap-2 uppercase tracking-widest shadow-xl">
                        <i class="fas fa-play text-[8px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span>
                    </button>
                </div>
            </div>
        </div>"""

    # API key
    api_token = token_manager.get_or_create()

    local_ip = get_local_ip()

    html = _build_html(
        gpu_rows=gpu_rows,
        cpu_rows=cpu_rows,
        model_items=model_items,
        ctx_opts=ctx_opts,
        custom_ctx_value=custom_ctx_value,
        custom_ctx_class=custom_ctx_class,
        batch_opts=batch_opts,
        thinking_checked=thinking_checked,
        thinking_badge_class=thinking_badge_class,
        thinking_badge_text=thinking_badge_text,
        default_model=default_model,
        local_ip=local_ip,
        api_token=api_token,
        is_authenticated=is_authenticated,
    )
    return HTMLResponse(html)


def _build_html(
    gpu_rows: str,
    cpu_rows: str,
    model_items: str,
    ctx_opts: str,
    custom_ctx_value: str,
    custom_ctx_class: str,
    batch_opts: str,
    thinking_checked: str,
    thinking_badge_class: str,
    thinking_badge_text: str,
    default_model: Optional[str],
    local_ip: str,
    api_token: str,
    is_authenticated: bool,
) -> str:
    """Build the full HTML template."""

    login_overlay_style = "none" if is_authenticated else "flex"
    login_overlay = f"""
        <div id="login-overlay" class="fixed inset-0 z-50 flex items-center justify-center pointer-events-auto" style="display: {login_overlay_style};">
            <div class="glass p-8 md:p-10 rounded-3xl border border-slate-700/50 w-full max-w-md mx-4">
                <div class="flex flex-col items-center mb-8">
                    <div class="bg-blue-600 p-4 rounded-2xl shadow-xl shadow-blue-500/20 mb-4">
                        <i class="fas fa-brain text-white text-2xl"></i>
                    </div>
                    <h2 class="text-xl font-bold text-white">Automanager Llama.cpp</h2>
                    <p class="text-xs text-slate-500 mt-1">Insira suas credenciais</p>
                </div>
                <form id="login-form" onsubmit="handleLogin(event)">
                    <div class="mb-4">
                        <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Usuario</label>
                        <input type="text" id="login-username" value="admin" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none" required>
                    </div>
                    <div class="mb-6">
                        <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Senha</label>
                        <input type="password" id="login-password" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none" required>
                    </div>
                    <button type="submit" class="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white text-sm font-black rounded-xl transition-all uppercase tracking-widest">
                        ENTRAR
                    </button>
                    <p id="login-error" class="text-red-500 text-xs mt-3 text-center hidden"></p>
                </form>
            </div>
        </div>"""

    vision_import_modal = """
        <div id="vision-import-modal" class="fixed inset-0 z-50 hidden items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="vision-import-title">
            <div id="vision-import-backdrop" class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" aria-hidden="true" onclick="closeVisionImportModal()"></div>
            <div class="relative glass w-full max-w-lg rounded-3xl border border-violet-500/30 shadow-2xl shadow-violet-500/10 overflow-hidden">
                <div class="p-6 md:p-8 border-b border-slate-800/60">
                    <div class="flex items-start justify-between gap-4">
                        <div>
                            <p class="text-[10px] font-black uppercase tracking-[0.25em] text-violet-400 mb-2">Projetor de visao</p>
                            <h2 id="vision-import-title" class="text-lg font-bold text-white">Importar mmproj</h2>
                            <p class="text-sm text-slate-400 mt-2 leading-relaxed">
                                Informe o link de download do arquivo mmproj. O arquivo sera salvo no diretorio deste modelo.
                            </p>
                        </div>
                        <button type="button" onclick="closeVisionImportModal()" class="shrink-0 w-9 h-9 flex items-center justify-center rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition-all" title="Fechar" aria-label="Fechar">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                <form id="vision-import-form" class="p-6 md:p-8 space-y-4" onsubmit="submitVisionImport(event)">
                    <input type="hidden" id="vision-import-model-path" value="">
                    <div>
                        <label for="vision-import-url" class="text-[10px] font-black text-slate-500 uppercase tracking-widest">URL de download</label>
                        <input type="url" id="vision-import-url" required placeholder="https://..."
                               class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-violet-500/50 outline-none">
                    </div>
                    <button type="submit" class="w-full py-3 bg-violet-600 hover:bg-violet-500 text-white text-sm font-black rounded-xl transition-all uppercase tracking-widest">
                        BAIXAR PROJETOR
                    </button>
                </form>
            </div>
        </div>"""

    version_update_modal = """
        <div id="version-update-modal" class="fixed inset-0 z-50 hidden items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="version-update-title">
            <div id="version-update-backdrop" class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" aria-hidden="true"></div>
            <div class="relative glass w-full max-w-2xl max-h-[85vh] flex flex-col rounded-3xl border border-blue-500/30 shadow-2xl shadow-blue-500/10 overflow-hidden">
                <div class="p-6 md:p-8 border-b border-slate-800/60 shrink-0">
                    <div class="flex items-start justify-between gap-4">
                        <div>
                            <p class="text-[10px] font-black uppercase tracking-[0.25em] text-blue-400 mb-2">Atualizacao disponivel</p>
                            <h2 id="version-update-title" class="text-lg md:text-xl font-bold text-white">Nova atualizacao disponivel</h2>
                            <p class="text-sm text-slate-300 mt-2 leading-relaxed">
                                Ha uma nova atualizacao disponivel. Convem atualizar o servidor para aplicar correcoes e melhorias.
                            </p>
                            <p class="text-xs text-slate-400 mt-3 font-mono">
                                Atual: <span id="version-current-ref" class="text-slate-200">--</span>
                                <span class="text-slate-600 mx-2">→</span>
                                Disponivel: <span id="version-remote-ref" class="text-emerald-400">--</span>
                            </p>
                        </div>
                        <button type="button" id="version-dismiss-btn" onclick="dismissVersionModal()" class="shrink-0 w-9 h-9 flex items-center justify-center rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition-all" title="Fechar" aria-label="Fechar">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                <div id="version-commits-list" class="custom-scroll flex-1 overflow-y-auto p-6 md:p-8 space-y-4 min-h-0"></div>
                <div class="p-6 md:p-8 border-t border-slate-800/60 bg-slate-900/40 shrink-0">
                    <p class="text-[10px] text-slate-500 leading-relaxed mb-4">
                        Recomendamos atualizar em breve. No servidor, execute <code class="text-slate-400">git pull</code> e reinicie o servico, ou use o botao ATUALIZAR no cabecalho.
                    </p>
                    <button type="button" onclick="dismissVersionModal()" class="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white text-sm font-black rounded-xl transition-all uppercase tracking-widest">
                        ENTENDI
                    </button>
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
    <link rel="shortcut icon" href="/favicon.ico">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        :root {{ --bg-deep: #020617; --card-bg: rgba(15, 23, 42, 0.6); }}
        body {{ font-family: 'Space Grotesk', sans-serif; background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #020617 100%); background-attachment: fixed; }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .glass {{ background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }}
        .custom-scroll::-webkit-scrollbar {{ width: 6px; }}
        .custom-scroll::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 10px; }}
        @keyframes pulse-glow {{ 0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }} 70% {{ box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }} }}
        .glow-online {{ animation: pulse-glow 2s infinite; }}
        .terminal-line {{ animation: fadeIn 0.3s ease-out; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .model-item-container.active-selection {{ border-color: rgba(59, 130, 246, 0.8) !important; background-color: rgba(30, 41, 59, 0.8) !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }}
        .model-item-container.running-now {{ border-color: rgba(16, 185, 129, 0.5) !important; background-color: rgba(6, 78, 59, 0.2) !important; }}
        #dashboard {{ position: relative; z-index: 1; }}
        .btn-gradient {{ background: linear-gradient(135deg, #1e30f3 0%, #e21e80 100%); }}
        .text-gradient {{ background: linear-gradient(315deg, #1e30f3 0%, #e21e80 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .metric-dimmed {{ opacity: 0.45; filter: grayscale(0.4); }}
    </style>
</head>
<body class="min-h-screen text-slate-200 pb-16 selection:bg-blue-500/30">
    <script>
        window.modelConfigs = {{}};
    </script>
    {login_overlay}
    {vision_import_modal}
    {version_update_modal}

    <div id="dashboard" class="max-w-[1800px] mx-auto px-4 md:px-8 pt-6 md:pt-10" style="display: {'block' if is_authenticated else 'none'};">
        <header class="flex flex-col md:flex-row items-center justify-between mb-8 md:mb-10 glass p-4 md:p-5 rounded-3xl md:rounded-[2rem] gap-4">
            <div class="flex items-center gap-4 md:gap-6">
                <div class="bg-blue-600 p-3 rounded-2xl shadow-xl shadow-blue-500/20"><i class="fas fa-brain text-white text-xl md:text-2xl"></i></div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2 md:gap-3">Automanager <span class="text-blue-500 font-light">Llama.cpp</span></h1>
                    <p class="text-[10px] text-slate-500 font-mono tracking-wider uppercase">Interface Avançada de Computação Neural</p>
                </div>
            </div>
            <div class="flex items-center gap-4 md:gap-8 w-full md:auto justify-center md:justify-end">
                <div class="hidden sm:flex items-center gap-3 md:gap-5 px-4 md:px-6 py-2 bg-slate-900/50 rounded-xl border border-slate-800">
                    <div class="flex flex-col items-end">
                        <span class="text-[9px] md:text-[10px] text-slate-500 font-black uppercase tracking-tighter">IP do Motor</span>
                        <span id="display-ip" class="text-xs font-mono text-blue-400">{local_ip}</span>
                    </div>
                    <i class="fas fa-network-wired text-slate-600 text-sm md:text-base"></i>
                </div>
                <div id="status-badge" class="px-6 md:px-8 py-2 md:py-2.5 rounded-xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 glass border-slate-700/50 text-slate-500 uppercase transition-all duration-500">
                    <div class="w-2 h-2 rounded-full bg-slate-600"></div>OFFLINE
                </div>
                <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                    <button onclick="handleUpdate()" class="px-3 py-2 bg-amber-600/10 hover:bg-amber-600/20 text-amber-500 border border-amber-500/30 rounded-xl text-[10px] font-black hover:tracking-widest transition-all uppercase" title="Atualizar codigo e reiniciar servico">
                        <i class="fas fa-sync-alt text-[9px]"></i> <span class="hidden lg:inline">ATUALIZAR</span>
                    </button>
                    <button onclick="handleShutdown()" class="px-3 py-2 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/30 rounded-xl text-[10px] font-black hover:tracking-widest transition-all uppercase" title="Desligar o sistema">
                        <i class="fas fa-power-off text-[9px]"></i> <span class="hidden lg:inline">DESLIGAR</span>
                    </button>
                </div>
                <button onclick="handleLogout()" class="text-slate-500 hover:text-white transition-colors" title="Sair">
                    <i class="fas fa-sign-out-alt"></i>
                </button>
            </div>
        </header>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10 items-start">
            <div class="lg:col-span-7 space-y-6 md:space-y-10">
                <div id="metrics-panel" class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">
                    <div class="glass p-5 rounded-[1.5rem] border-l-4 border-blue-600">
                        <div class="flex justify-between items-start mb-4">
                            <p class="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">Processador (Host)</p>
                            <i class="fas fa-microchip text-slate-700 text-xs"></i>
                        </div>
                        <div class="flex items-end justify-between gap-4">
                            <h3 id="cpu-val" class="text-3xl font-bold text-white leading-none">0%</h3>
                            <div class="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                <div id="cpu-bar" class="h-full bg-blue-500 transition-all duration-700 shadow-[0_0_10px_rgba(37,99,235,0.5)]" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="glass p-5 rounded-[1.5rem] border-l-4 border-emerald-600">
                        <div class="flex justify-between items-start mb-4">
                            <p class="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">Memória RAM (Host)</p>
                            <i class="fas fa-memory text-slate-700 text-xs"></i>
                        </div>
                        <div class="flex items-end justify-between gap-4">
                            <h3 id="ram-val" class="text-3xl font-bold text-white leading-none">0%</h3>
                            <div class="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                <div id="ram-bar" class="h-full bg-emerald-500 transition-all duration-700" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="glass rounded-[2rem] overflow-hidden p-6 md:p-8">
                    <div class="flex flex-col md:flex-row md:items-center justify-between gap-6 md:gap-8 mb-8 border-b border-slate-800/50 pb-6">
                        <div>
                            <h3 class="font-bold text-lg text-white flex items-center gap-3"><i class="fas fa-microchip text-blue-500"></i>Recursos de GPU & Configuração</h3>
                            <p class="text-xs text-slate-500 mt-1 font-medium italic">Monitore e distribua a carga de processamento entre as GPUs</p>
                        </div>
                        <div class="flex flex-wrap items-center gap-4 md:gap-6 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800">
                            <div class="flex items-center gap-2">
                                <label class="text-[9px] font-black uppercase text-slate-400 pl-3 md:pl-4 tracking-widest whitespace-nowrap" title="Tokens por slot (--ctx-size / --parallel no llama-server)">Contexto:</label>
                                <select id="context-size" onchange="onContextSizePresetChange()" class="bg-blue-600/20 border border-blue-500/30 text-blue-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                    {ctx_opts}
                                </select>
                                <div class="relative {custom_ctx_class}" id="context-size-custom-wrap">
                                    <input type="number" id="context-size-custom" value="{custom_ctx_value}"
                                           class="w-28 min-w-[7rem] bg-slate-800 border border-slate-700 text-slate-300 rounded-xl pl-3 pr-8 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all text-center tabular-nums"
                                           min="1" step="1" placeholder="K"
                                           title="Contexto em K (ex.: 100 = 100K tokens por slot)"
                                           oninput="onContextSizeCustomInput()">
                                    <span class="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-500 pointer-events-none">K</span>
                                </div>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap" title="Requisições simultâneas (--parallel)"><i class="fas fa-clone text-blue-400 mr-2"></i>Slots:</label>
                                <input type="number" id="parallel-slots" value="{DEFAULT_PARALLEL_SLOTS}" min="1" max="64"
                                       class="w-16 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-3 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all text-center">
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap" title="Tamanho do batch de prefill (--batch-size)"><i class="fas fa-boxes-stacked text-violet-400 mr-2"></i>Batch:</label>
                                <select id="batch-size" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-3 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-violet-500/50 outline-none transition-all cursor-pointer min-w-[5.5rem]">
                                    {batch_opts}
                                </select>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-layer-group text-blue-400 mr-2"></i>Split:</label>
                                <select id="split-mode" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                    <option value="layer">Layer (Sqn)</option>
                                    <option value="row">Row (Par)</option>
                                </select>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-brain text-violet-400 mr-2"></i>Thinking:</label>
                                <label class="flex items-center gap-2 cursor-pointer select-none bg-slate-800/80 px-3 py-1.5 rounded-xl border border-slate-700 hover:border-violet-500/30 transition-all">
                                    <input type="checkbox" id="thinking-toggle" {thinking_checked} class="w-4 h-4 bg-slate-900 border-slate-700 rounded text-violet-600 cursor-pointer">
                                    <span id="thinking-badge" class="text-[9px] font-black uppercase tracking-wider {thinking_badge_class}">{thinking_badge_text}</span>
                                </label>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap" title="Multi-Token Prediction — acelera inferência em modelos compatíveis"><i class="fas fa-bolt text-amber-400 mr-2"></i>MTP:</label>
                                <label class="flex items-center gap-2 cursor-pointer select-none bg-slate-800/80 px-3 py-1.5 rounded-xl border border-slate-700 hover:border-amber-500/30 transition-all">
                                    <input type="checkbox" id="mtp-toggle" class="w-4 h-4 bg-slate-900 border-slate-700 rounded text-amber-600 cursor-pointer">
                                    <span id="mtp-badge" class="text-[9px] font-black uppercase tracking-wider text-slate-500">OFF</span>
                                </label>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap" title="Tokens de predição MTP (--spec-draft-n-max); quanto maior, mais tokens tenta prever por passo"><i class="fas fa-forward text-amber-400 mr-2"></i>MTP Tok:</label>
                                <input type="number" id="mtp-draft-tokens" value="{DEFAULT_MTP_DRAFT_TOKENS}" min="1"
                                       class="w-16 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-3 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-amber-500/50 outline-none transition-all text-center">
                            </div>
                        </div>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left">
                            <thead class="text-[9px] md:text-[10px] font-black text-slate-500 uppercase tracking-widest">
                                <tr>
                                    <th class="px-4 md:px-6 py-4 text-center">Uso</th>
                                    <th class="px-4 py-4 text-center">Principal</th>
                                    <th class="px-4 py-4">Dispositivo</th>
                                    <th class="px-4 py-4">Monitoramento</th>
                                    <th class="px-4 py-4">VRAM Status</th>
                                    <th class="px-4 py-4">Distribuição</th>
                                </tr>
                            </thead>
                            <tbody id="gpu-table-body" class="divide-y divide-slate-800/50">
                                {cpu_rows}
                                {gpu_rows}
                            </tbody>
                        </table>
                    </div>
                    <div class="flex flex-col sm:flex-row justify-between items-center pt-8 gap-4">
                        <div class="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                            <div class="flex items-center gap-3 text-[10px] md:text-xs text-slate-500">
                                <i class="fas fa-info-circle text-blue-500"></i>
                                Distribua 100% da carga total entre as GPUs selecionadas
                            </div>
                            <label class="flex items-center gap-3 cursor-pointer select-none bg-slate-900/60 px-4 py-2.5 rounded-xl border border-slate-800 hover:border-blue-500/30 transition-all">
                                <input type="checkbox" id="auto-balance-toggle" class="w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                                <span class="text-[10px] md:text-xs font-black uppercase tracking-widest text-slate-300">Auto Balance</span>
                                <span id="auto-balance-badge" class="hidden text-[9px] font-black uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-lg border border-emerald-500/20">Salvo</span>
                            </label>
                            <button type="button" id="auto-balance-cancel-btn" onclick="cancelAutoBalance()"
                                    class="hidden px-4 py-2.5 rounded-xl border border-red-500/40 bg-red-500/10 text-red-400 hover:bg-red-500/20 text-[10px] font-black uppercase tracking-widest transition-all"
                                    title="Interromper calibracao de GPUs">
                                <i class="fas fa-stop mr-2"></i>Cancelar balance
                            </button>
                        </div>
                        <span id="total-percent" class="text-xs md:text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl transition-all duration-300">CARGA TOTAL: 100%</span>
                    </div>
                    <div id="mtp-warning" class="hidden mt-4 p-4 md:p-5 rounded-2xl border border-amber-500/40 bg-amber-950/30">
                        <div class="flex gap-4 items-start">
                            <div class="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center shrink-0">
                                <i class="fas fa-bolt text-amber-400"></i>
                            </div>
                            <div class="flex-1 min-w-0">
                                <p class="text-[10px] font-black uppercase tracking-widest text-amber-400 mb-1">MTP não foi aplicado</p>
                                <p id="mtp-warning-msg" class="text-xs md:text-sm text-amber-100/80 leading-relaxed"></p>
                                <p class="mt-2 text-[10px] text-slate-400">Use um modelo GGUF com suporte a Multi-Token Prediction (heads MTP embutidos). Os modelos compatíveis têm <code class="text-amber-300/80">nextn_predict_layers &gt; 0</code>.</p>
                            </div>
                        </div>
                    </div>
                    <div id="auto-balance-capacity-alert" class="hidden mt-6 p-5 md:p-6 rounded-2xl border border-red-500/40 bg-red-950/40">
                        <div class="flex gap-4 items-start">
                            <div class="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center shrink-0">
                                <i class="fas fa-microchip text-red-400"></i>
                            </div>
                            <div class="flex-1 min-w-0">
                                <p class="text-[10px] font-black uppercase tracking-widest text-red-400 mb-2">Modelo além da capacidade do hardware</p>
                                <p id="auto-balance-capacity-msg" class="text-xs md:text-sm text-red-100/90 leading-relaxed"></p>
                                <ul id="auto-balance-capacity-details" class="mt-3 text-[10px] text-slate-400 space-y-1"></ul>
                                <p class="mt-3 text-[10px] font-black uppercase tracking-wider text-slate-500">Sugestões</p>
                                <ul id="auto-balance-capacity-suggestions" class="mt-1 text-[10px] text-slate-400 space-y-1 list-disc list-inside"></ul>
                            </div>
                            <button type="button" onclick="hideAutoBalanceCapacityAlert()" class="text-slate-500 hover:text-white shrink-0 p-1" title="Fechar">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                </div>
                <div id="active-cards-container" class="space-y-6">
                    <!-- Cards de instâncias serão injetados via JS -->
                </div>
                <div class="glass rounded-[2rem] overflow-hidden shadow-2xl border border-slate-800">
                    <div class="px-6 md:px-10 py-4 bg-slate-900/60 border-b border-slate-800 flex justify-between items-center">
                        <div class="flex items-center gap-3 md:gap-4">
                            <div class="flex gap-1.5 md:gap-2">
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                            </div>
                            <p id="log-panel-title" class="text-slate-400 text-[9px] md:text-[10px] font-black uppercase tracking-widest font-mono ml-2 md:ml-4">Saída de logs do sistema</p>
                        </div>
                        <button onclick="document.getElementById('log-box').innerHTML=''" class="text-[9px] md:text-[10px] text-slate-600 hover:text-blue-400 font-bold uppercase transition-colors tracking-widest">
                            <i class="fas fa-trash-alt mr-2"></i> Limpar
                        </button>
                    </div>
                    <div id="log-box" class="custom-scroll p-6 md:p-10 h-[300px] md:h-[400px] overflow-y-auto font-mono text-[10px] md:text-xs text-slate-400 leading-relaxed whitespace-pre-wrap bg-slate-950/40"></div>
                </div>
            </div>
            <div class="lg:col-span-5 space-y-6 md:space-y-10">
                <div class="glass rounded-[2rem] border border-slate-800">
                    <div class="p-8 border-b border-slate-800/50 flex items-center justify-between">
                        <div class="flex items-center gap-4 md:gap-5">
                            <div class="w-10 h-10 bg-slate-800 rounded-xl flex items-center justify-center border border-slate-700">
                                <i class="fas fa-database text-amber-500 text-sm md:text-base"></i>
                            </div>
                            <h3 class="font-bold text-base md:text-lg text-white tracking-tight">Model Repository</h3>
                        </div>
                        <div class="flex flex-col items-end gap-1">
                            <span class="text-[10px] bg-slate-800 text-slate-400 px-3 py-1 rounded-full font-mono border border-slate-700" id="model-count">0 UNIDADES</span>
                            <span class="text-[9px] bg-slate-800/80 text-slate-400 px-2 py-0.5 rounded-full font-mono border border-slate-700 whitespace-nowrap" id="repo-storage" title="Espaço usado pelo repositório / capacidade total do disco">-- / -- GB</span>
                        </div>
                    </div>
                    <div class="px-6 md:px-8 py-3 border-b border-slate-800/30 bg-slate-900/20 flex items-center gap-2">
                        <span class="text-[9px] font-black text-slate-500 uppercase tracking-widest shrink-0 hidden sm:inline">Diretório</span>
                        <input type="text" id="models-dir-input" placeholder="Caminho do repositório de modelos" class="flex-1 min-w-0 px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-[10px] text-slate-300 font-mono focus:ring-2 focus:ring-blue-500/50 outline-none transition-all placeholder:text-slate-600">
                        <button type="button" onclick="saveModelsDir()" title="Salvar caminho e atualizar lista" class="shrink-0 w-9 h-9 flex items-center justify-center rounded-xl bg-slate-800 hover:bg-blue-600 text-slate-400 hover:text-white border border-slate-700 transition-all active:scale-95">
                            <i class="fas fa-save text-xs"></i>
                        </button>
                    </div>
                    <div class="p-8 border-b border-slate-800/30 bg-blue-600/5">
                        <p class="text-[10px] font-black text-slate-500 uppercase mb-4 md:mb-6 tracking-widest">Ingerir GGUF via URL</p>
                        <div class="space-y-3 md:space-y-4">
                            <div class="relative group">
                                <i class="fas fa-link absolute left-4 top-1/2 -translate-y-1/2 text-slate-600 text-xs md:text-sm transition-colors group-focus-within:text-blue-500"></i>
                                <input type="text" id="download-url" placeholder="https://..." class="w-full pl-10 pr-4 py-3 bg-slate-900 border border-slate-700 rounded-2xl text-xs text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all placeholder:text-slate-600">
                            </div>
                            <button onclick="downloadModel()" class="w-full py-4 bg-slate-100 hover:bg-white text-slate-950 text-[10px] font-black rounded-2xl transition-all shadow-xl active:scale-[0.98] uppercase tracking-[0.2em] flex items-center justify-center gap-3 md:gap-4">
                                <i class="fas fa-cloud-download-alt text-sm"></i> EXECUTAR DOWNLOAD
                            </button>
                        </div>
                        <div id="download-status" class="mt-6 md:mt-8 space-y-3"></div>
                    </div>
                    <div id="model-list-container" class="p-6 md:p-8 space-y-2">
                        {model_items}
                    </div>
                </div>
                <div class="glass rounded-[2rem] border border-slate-800 p-8">
                        <div class="flex flex-col gap-4">
                            <div class="flex flex-col gap-2">
                                <div class="flex items-center justify-between">
                                    <p class="text-[9px] text-slate-500 font-black uppercase tracking-widest">Endpoint OpenAI API</p>
                                    <span class="text-[8px] bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded border border-emerald-500/20 uppercase font-black">Ativo</span>
                                </div>
                                <div class="flex items-center gap-3 md:gap-4 bg-slate-900 p-3 rounded-xl border border-slate-800 group">
                                    <code id="api-link" class="text-[10px] text-blue-400 font-mono flex-1 truncate"></code>
                                    <button onclick="navigator.clipboard.writeText(document.getElementById('api-link').innerText)" class="text-slate-600 hover:text-white transition-colors">
                                        <i class="far fa-copy"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="flex flex-col gap-2">
                                <p class="text-[9px] text-slate-500 font-black uppercase tracking-widest">Chave de API (Bearer Token)</p>
                                <div class="flex items-center gap-3 md:gap-4 bg-slate-900 p-3 rounded-xl border border-slate-800 group">
                                    <code id="api-token" class="text-[10px] text-amber-400 font-mono flex-1 truncate">{api_token}</code>
                                    <div class="flex gap-3">
                                        <button onclick="renewToken()" class="text-slate-600 hover:text-amber-500 transition-colors" title="Renovar Chave">
                                            <i class="fas fa-sync-alt"></i>
                                        </button>
                                        <button onclick="navigator.clipboard.writeText(document.getElementById('api-token').innerText)" class="text-slate-600 hover:text-white transition-colors" title="Copiar Chave">
                                            <i class="far fa-copy"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="flex flex-col gap-3 border-t border-slate-800 pt-4">
                                <p class="text-[9px] text-slate-500 font-black uppercase tracking-widest">Senha do Administrador</p>
                                <div class="grid grid-cols-1 gap-3">
                                    <input type="password" id="current-password" placeholder="Senha atual" autocomplete="current-password" class="w-full px-4 py-3 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none placeholder:text-slate-600">
                                    <input type="password" id="new-password" placeholder="Nova senha" autocomplete="new-password" class="w-full px-4 py-3 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none placeholder:text-slate-600">
                                    <button onclick="changePassword()" class="w-full py-3 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 text-[10px] font-black rounded-xl transition-all uppercase tracking-[0.2em]">
                                        ALTERAR SENHA
                                    </button>
                                    <p id="password-change-status" class="text-[10px] font-bold min-h-[1rem]"></p>
                                </div>
                            </div>
                        </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        window.fixedIp = "{local_ip}";
        window.__constants = {{
            CONTEXT_PRESET_VALUES: {json.dumps(CONTEXT_PRESET_VALUES)},
            DEFAULT_CONTEXT_SIZE: {DEFAULT_CONTEXT_SIZE},
            CONTEXT_K_MULTIPLIER: {CONTEXT_K_MULTIPLIER},
            DEFAULT_PARALLEL_SLOTS: {DEFAULT_PARALLEL_SLOTS},
            DEFAULT_BATCH_SIZE: {DEFAULT_BATCH_SIZE},
            DEFAULT_MODEL: {json.dumps(default_model)},
        }};
    </script>
    <script type="module" src="/static/js/index.js?v={_DASHBOARD_JS_V}"></script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
# Startup event — auto-start default model + OOM watchdog
# ─────────────────────────────────────────────────────────

GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 5


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
                split_mode=split_mode,
                parallel_slots=parallel_slots,
                batch_size=batch_size,
                thinking_enabled=thinking_enabled,
                mtp_enabled=mtp_enabled,
                mtp_draft_tokens=mtp_draft_tokens,
                total_layers=saved_cfg.get("total_layers", 0),
                port=port
            )
            logger.info(f"Auto-start: {model_path} started on port {port} (result: {start_result})")
            # Small delay between starts to avoid resource contention peaks
            time.sleep(3)
        except Exception as e:
            logger.error(f"Auto-start error for {model_path}: {e}")


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


@app.on_event("shutdown")
async def shutdown_event_handler():
    """Signal all background tasks to stop and kill llama-server."""
    logger.info("Encerrando Automanager Llama.cpp...")
    shutdown_event.set()
    oom_watchdog.stop()
    process_manager.stop()


# ─────────────────────────────────────────────────────────
# Background download task runner
# ─────────────────────────────────────────────────────────


def _run_downloads(stop_event: threading.Event):
    """Periodically process background downloads."""
    while not stop_event.is_set():
        with download_mgr._lock:
            to_process = list(download_mgr._downloads_queue)
            download_mgr._downloads_queue.clear()
        for download_id, url, filename, path in to_process:
            if stop_event.is_set():
                break
            download_mgr._do_download(download_id, url, filename, path)
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
