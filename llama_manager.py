import json
import socket
import os
import signal
import threading
import time
import re
import glob
import logging
import uvicorn
import httpx
from typing import List, Optional, Tuple, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_manager import ConfigManager
from log_manager import LogManager, logger
from llama_server_bin import get_llama_server_bin
from process_manager import ProcessManager, OOMWatchdog, SERVER_PORT
from model_manager import ModelScanner, DownloadManager
from schemas import (
    BATCH_SIZE_PRESETS,
    CACHE_TYPE_PRESETS,
    DEFAULT_CACHE_TYPE,
    DEFAULT_MTP_DRAFT_TOKENS,
    GPUWeight,
    StartRequest,
    DeleteRequest,
    DownloadRequest,
    RenameRequest,
    SetDefaultRequest,
    SetMmprojRequest,
    SetThinkingRequest,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_BATCH_SIZE,
)
from gpu_manager import GPUManager, reasoning_cli_args, mtp_cli_args, compute_server_ctx_size

# Version tracking
_DASHBOARD_JS_V = "4.0.0"  # Major UI Refactor

MANAGER_PORT = 8000
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 5

app = FastAPI(title="Automanager Llama.cpp")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared HTTP client for proxying
client = httpx.AsyncClient()

config_manager = ConfigManager()
log_manager = LogManager()
gpu_manager = GPUManager()
process_manager = ProcessManager(
    config_manager, None, gpu_manager, log_manager
)
# Fix circular dependency in initialization
from process_manager import TokenManager, AuthManager
token_manager = TokenManager(config_manager)
process_manager.token_mgr = token_manager
auth_manager = AuthManager(config_manager, token_manager)

model_scanner = ModelScanner(config_manager, process_manager)
download_mgr = DownloadManager()
oom_watchdog = OOMWatchdog(process_manager)

shutdown_event = threading.Event()

# Context and Batch presets for the UI
CONTEXT_PRESET_VALUES = [4096, 8192, 16384, 32768, 65536, 131072, "custom"]
CONTEXT_K_MULTIPLIER = 1024


def _invalidate_models_cache():
    """Helper to force model list refresh on next scan."""
    model_scanner._last_scan_time = 0


@app.post("/api/auth/login")
async def login(req: Dict[str, str]):
    username = req.get("username")
    password = req.get("password")
    if auth_manager.verify_admin(username, password):
        token = token_manager.get_or_create()
        return {"token": token}
    raise HTTPException(status_code=401, detail="Credenciais invalidas")


@app.post("/api/auth/change-password")
async def change_password(req: Dict[str, str], authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    current = req.get("current")
    new_pw = req.get("new")
    if auth_manager.change_admin_password(current, new_pw):
        return {"message": "Senha alterada"}
    raise HTTPException(status_code=400, detail="Senha atual incorreta")


@app.get("/status")
async def get_status():
    return process_manager.get_status()


@app.get("/metrics")
async def get_metrics():
    return gpu_manager.get_metrics()


@app.get("/models")
async def list_models():
    models = model_scanner.scan()
    storage = model_scanner.get_storage_info()
    return {"models": models, "storage": storage}


@app.post("/models/dir")
async def set_models_dir(req: Dict[str, str], authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    new_dir = req.get("models_dir")
    if config_manager.set_models_dir(new_dir):
        _invalidate_models_cache()
        return {"message": "Diretorio atualizado"}
    raise HTTPException(status_code=400, detail="Diretorio invalido ou inacessivel")


@app.post("/start")
async def start_model(req: StartRequest, authenticated: bool = Depends(auth_manager.check_auth)):
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
        "threads": req.threads,
        "threads_batch": req.threads_batch,
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
        ubatch_size=req.ubatch_size,
        cache_type_k=req.cache_type_k,
        cache_type_v=req.cache_type_v,
        numa_enabled=req.numa_enabled,
        threads=req.threads,
        threads_batch=req.threads_batch,
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
async def stop_model(port: Optional[int] = None, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    process_manager.stop(port)
    return {"message": "Parado"}


@app.post("/auto-balance/cancel")
async def cancel_auto_balance(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    process_manager.cancel_auto_balance()
    return {"message": "Cancelado"}


@app.post("/delete")
async def delete_model(req: DeleteRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    if model_scanner.delete_model(req.path):
        _invalidate_models_cache()
        return {"message": "Excluido"}
    raise HTTPException(status_code=400, detail="Erro ao excluir arquivo")


@app.post("/rename")
async def rename_model(req: RenameRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    if model_scanner.rename_model(req.path, req.new_name):
        _invalidate_models_cache()
        return {"message": "Renomeado"}
    raise HTTPException(status_code=400, detail="Erro ao renomear arquivo")


@app.post("/downloads")
async def start_download(req: DownloadRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    download_id = download_mgr.queue_download(req.url, req.model_path)
    return {"download_id": download_id}


@app.get("/downloads")
async def list_downloads():
    return {"downloads": download_mgr.get_status()}


@app.post("/downloads/clear")
async def clear_downloads(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    download_mgr.clear_completed()
    return {"message": "Limpo"}


@app.get("/logs")
async def get_logs(port: Optional[int] = None):
    return {"logs": log_manager.get_server_log(port)}


@app.post("/set_default")
async def set_default_model(req: SetDefaultRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config_manager.set_default_model(req.path, req.add)
    return {"message": "Configuracao salva"}


@app.get("/config")
async def get_config():
    return config_manager.get_config()


@app.post("/models/mmproj")
async def set_mmproj(req: SetMmprojRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config_manager.update_model_settings(req.model_path, {"mmproj_path": req.mmproj_path})
    return {"message": "Configuracao salva"}


@app.post("/models/thinking")
async def set_thinking(req: SetThinkingRequest, authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    config_manager.update_model_settings(req.model_path, {"thinking_enabled": req.thinking_enabled})
    return {"message": "Configuracao salva"}


@app.post("/system/shutdown")
async def system_shutdown(authenticated: bool = Depends(auth_manager.check_auth)):
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
async def system_update(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    from version_manager import VersionManager
    vm = VersionManager()
    logger.info("Update solicitado via API")
    success, msg = vm.update_and_restart()
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"message": msg}


@app.get("/api/key")
async def get_api_key(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {"api_key": token_manager.get_or_create()}


@app.post("/api/key/renew")
async def renew_api_key(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    return {"api_key": token_manager.renew()}


@app.get("/api/system/version-check")
async def system_version_check(authenticated: bool = Depends(auth_manager.check_auth)):
    if not authenticated:
        raise HTTPException(status_code=401)
    from version_manager import check_for_updates
    return check_for_updates(INSTALL_ROOT)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(request: Request, path: str):
    """Proxy OpenAI-compatible requests to the correct llama-server instance,
    routing by the 'model' field in the request body (single-port multi-model)."""
    body = await request.body()
    data: Dict[str, Any] = {}
    requested_model = None

    if body:
        try:
            data = json.loads(body)
            requested_model = data.get("model")
        except json.JSONDecodeError:
            data = {}

    instances = process_manager.get_status().get("instances", [])
    if not instances:
        raise HTTPException(status_code=503, detail="Nenhum modelo carregado")

    if requested_model:
        target_instance = next(
            (
                inst
                for inst in instances
                if inst.get("model") == requested_model
                or inst.get("model_path") == requested_model
            ),
            None,
        )
        if not target_instance:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{requested_model}' nao esta carregado.",
            )
    else:
        # Sem modelo especificado: usar a instancia da porta default (8085).
        target_instance = next(
            (i for i in instances if i.get("port") == SERVER_PORT), instances[0]
        )

    target_url = f"http://127.0.0.1:{target_instance['port']}/v1/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        if request.method == "POST":
            if data.get("stream"):
                async def stream_generator():
                    async with client.stream(
                        "POST", target_url, content=body, headers=headers, timeout=None
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            resp = await client.post(target_url, content=body, headers=headers, timeout=None)
        elif request.method == "GET":
            resp = await client.get(
                target_url, params=request.query_params, headers=headers
            )
        else:
            resp = await client.request(
                request.method, target_url, content=body, headers=headers
            )

        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except httpx.RequestError as exc:
        logger.error(f"Proxy error to port {target_instance['port']}: {exc}")
        raise HTTPException(
            status_code=502, detail="Erro ao conectar na instancia do modelo"
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
                    <input type="checkbox" checked class="gpu-checkbox w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                </td>
                <td class="px-4 py-4 text-center">
                    <input type="radio" name="main-gpu-[TABID]" class="gpu-main-radio w-4 h-4 bg-slate-900 border-slate-700 text-blue-600 cursor-pointer" { 'checked' if idx == 0 else '' }>
                </td>
                <td class="px-4 py-4">
                    <div class="flex flex-col">
                        <span class="text-[10px] font-black text-white uppercase tracking-tight">{name}</span>
                        <span class="text-[9px] text-slate-500 font-mono uppercase tracking-tighter">ID: {idx} · {vram} MB VRAM</span>
                    </div>
                </td>
                <td class="px-4 py-4">
                    <div class="relative flex items-center group/input">
                        <input type="number" class="gpu-weight w-20 bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1.5 text-xs font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all pr-6" 
                               value="{100 if idx == 0 else 0}" min="0" max="100">
                        <span class="absolute right-2 text-[9px] font-black text-slate-600">%</span>
                        <label class="ml-3 flex items-center gap-2 cursor-pointer" title="Travar valor">
                            <input type="checkbox" class="gpu-pin hidden">
                            <i class="fas fa-thumbtack text-[10px] text-slate-700 hover:text-blue-500 transition-colors pin-icon"></i>
                        </label>
                    </div>
                </td>
            </tr>"""

    cpu_rows = f"""
        <tr class="cpu-row group/row">
            <td class="px-6 py-4 text-center">
                <input type="checkbox" class="cpu-checkbox w-4 h-4 bg-slate-900 border-slate-700 rounded text-emerald-600 cursor-pointer">
            </td>
            <td class="px-4 py-4 text-center opacity-20 pointer-events-none">
                <input type="radio" disabled class="w-4 h-4">
            </td>
            <td class="px-4 py-4">
                <div class="flex flex-col">
                    <span class="text-[10px] font-black text-white uppercase tracking-tight">System RAM / CPU Offload</span>
                    <span class="text-[9px] text-slate-500 font-mono uppercase tracking-tighter">Latencia superior a VRAM</span>
                </div>
            </td>
            <td class="px-4 py-4">
                <div class="relative flex items-center group/input">
                    <input type="number" class="cpu-weight w-20 bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1.5 text-xs font-bold focus:ring-2 focus:ring-emerald-500/50 outline-none transition-all pr-6" 
                           value="0" min="0" max="100">
                    <span class="absolute right-2 text-[9px] font-black text-slate-600">%</span>
                    <label class="ml-3 flex items-center gap-2 cursor-pointer" title="Travar valor">
                        <input type="checkbox" class="cpu-pin hidden">
                        <i class="fas fa-thumbtack text-[10px] text-slate-700 hover:text-emerald-500 transition-colors pin-icon"></i>
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

    model_items = ""
    for m in models:
        m_path = m["path"]
        m_name = m["name"]
        m_dir = m["dir"]
        m_js = m_path.replace("\\", "/")
        m_cfg = model_configs.get(m_js, {})
        stable_id = m_js
        
        is_default = "checked" if (m_path in default_models or m_path == default_model) else ""
        has_config = "text-blue-400" if m_cfg and not m_cfg.get("hardware_incapable") else "text-slate-100"
        
        hardware_incapable = bool(m_cfg.get("hardware_incapable"))
        incapable_badge = '<span class="shrink-0 text-[8px] font-black uppercase tracking-wider text-red-400 bg-red-500/15 px-2 py-0.5 rounded-lg border border-red-500/30 ml-2">Incapaz</span>' if hardware_incapable else ''
        incapable_row_class = 'border-red-500/40 bg-red-950/20' if hardware_incapable else ''
        incapable_attr = 'true' if hardware_incapable else 'false'

        # Build vision controls
        vision_controls = ""
        candidates = m.get("mmproj_candidates", [])
        if candidates:
            vision_name = os.path.basename(candidates[0])
            vision_controls = f"""
                <div class="flex items-center gap-2 ml-2">
                    <i class="fas fa-eye text-violet-500 text-[10px]" title="Visao suportada"></i>
                    <span class="text-[8px] text-slate-500 font-mono truncate max-w-[80px]">{vision_name}</span>
                </div>"""

        model_items += f"""
        <div id="lib-{stable_id}" class="model-item-container group flex flex-col gap-3 p-3 mb-2 bg-slate-800/40 rounded-xl hover:bg-slate-700/60 transition-all border border-slate-700/50 hover:border-blue-500/50 shadow-sm {incapable_row_class}" data-path="{m_js}" data-hardware-incapable="{incapable_attr}">
            <div class="w-full cursor-pointer" onclick="selectModel('{m_js}', '{stable_id}')">
                <div class="flex items-start justify-between">
                    <div class="flex items-center gap-2 overflow-hidden">
                        <i class="fas fa-cube text-blue-400 text-[10px] shrink-0"></i>
                        <p class="model-name text-[11px] font-bold {has_config} truncate">{m_name}</p>
                    </div>
                    {incapable_badge}
                </div>
                <p class="text-[8px] text-slate-500 truncate font-mono mt-1 uppercase opacity-60">{m_dir}</p>
            </div>
            <div class="flex items-center justify-between gap-2 mt-1">
                <div class="flex items-center gap-1">
                    <button onclick="renameModel('{m_js}')" class="w-6 h-6 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-blue-400 transition-all"><i class="fas fa-edit text-[9px]"></i></button>
                    <button onclick="deleteModel('{m_js}')" class="w-6 h-6 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-red-400 transition-all"><i class="fas fa-trash-alt text-[9px]"></i></button>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-[8px] font-black text-slate-600 uppercase">Padrão</span>
                    <input type="checkbox" class="w-3.5 h-3.5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" {is_default} onclick="setDefaultModel(this, '{m_js}')">
                </div>
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

    api_token = token_manager.get_or_create()
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
) -> str:
    """Build the full HTML template."""

    login_overlay_style = "none" if is_authenticated else "flex"
    login_overlay = f"""
        <div id="login-overlay" class="fixed inset-0 z-50 flex items-center justify-center pointer-events-auto" style="display: {login_overlay_style};">
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
                        <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Usuario</label>
                        <input type="text" id="login-username" value="admin" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" required>
                    </div>
                    <div class="mb-6">
                        <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Senha</label>
                        <input type="password" id="login-password" class="w-full mt-2 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" required>
                    </div>
                    <button type="submit" class="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white text-sm font-black rounded-xl transition-all uppercase tracking-widest shadow-xl active:scale-95">
                        AUTENTICAR
                    </button>
                    <p id="login-error" class="text-red-500 text-[10px] mt-4 text-center font-bold uppercase hidden"></p>
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
                        <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">URL mmproj</label>
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
        :root {{ --bg-deep: #020617; --card-bg: rgba(15, 23, 42, 0.6); }}
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
        .tab-content.active {{ display: flex; }}
        
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
    </style>
</head>
<body class="min-h-screen text-slate-200 selection:bg-blue-500/30 overflow-hidden flex">
    <script>window.modelConfigs = {{}}; window.activeTabs = [];</script>
    {login_overlay}
    {vision_import_modal}
    {version_update_modal}

    <!-- SIDEBAR (MENU RETRATIL) -->
    <aside id="sidebar" class="fixed top-0 left-0 h-full w-80 glass border-r border-slate-800 z-40 overflow-y-auto custom-scroll flex flex-col shadow-2xl">
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
                    <p class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Modelos Disponíveis</p>
                    <span id="model-count" class="text-[9px] bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700 font-mono">0</span>
                </div>
                <div id="model-list-container" class="space-y-2">
                    {model_items}
                </div>
            </section>

            <!-- Download -->
            <section class="pt-6 border-t border-slate-800/50">
                <p class="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">Download GGUF</p>
                <div class="space-y-3">
                    <div class="relative">
                        <input type="text" id="download-url" placeholder="URL HuggingFace..." class="w-full pl-4 pr-10 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-[10px] text-slate-300 focus:ring-1 focus:ring-blue-500/50 outline-none">
                        <button onclick="downloadModel()" class="absolute right-2 top-1/2 -translate-y-1/2 text-blue-500 hover:text-blue-400"><i class="fas fa-arrow-down"></i></button>
                    </div>
                </div>
                <div id="download-list" class="mt-4 space-y-2"></div>
            </section>

            <!-- Admin Config -->
            <section class="pt-6 border-t border-slate-800/50 space-y-4 pb-10">
                 <p class="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">Configurações Globais</p>
                 <div class="space-y-3">
                    <div class="space-y-1">
                        <label class="text-[8px] font-black text-slate-600 uppercase ml-1">Diretório de Modelos</label>
                        <div class="flex gap-2">
                            <input type="text" id="models-dir-input" class="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-[9px] text-slate-400 font-mono outline-none">
                            <button onclick="saveModelsDir()" class="px-2 py-2 bg-slate-800 hover:bg-blue-600 rounded-lg text-[9px] font-bold transition-all"><i class="fas fa-save"></i></button>
                        </div>
                    </div>
                    <div class="flex justify-between items-center px-1">
                        <span class="text-[9px] text-slate-600 font-mono uppercase tracking-widest">Armazenamento</span>
                        <span class="text-[9px] text-slate-500 font-bold" id="repo-storage">-- GB</span>
                    </div>
                 </div>

                 <div class="space-y-2 pt-4 border-t border-slate-800/30">
                    <label class="text-[8px] font-black text-slate-600 uppercase ml-1">Acesso API (OpenAI)</label>
                    <div class="bg-slate-900 p-2 rounded-lg border border-slate-800 flex items-center justify-between">
                        <code id="api-token" class="text-[9px] text-amber-500/80 font-mono truncate mr-2">{api_token}</code>
                        <button onclick="navigator.clipboard.writeText(document.getElementById('api-token').innerText)" class="text-slate-600 hover:text-white"><i class="far fa-copy text-[10px]"></i></button>
                    </div>
                 </div>

                 <div class="space-y-2 pt-4 border-t border-slate-800/30">
                    <label class="text-[8px] font-black text-slate-600 uppercase ml-1">Alterar Senha Admin</label>
                    <input type="password" id="current-password" placeholder="Senha atual" class="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-[10px] outline-none">
                    <input type="password" id="new-password" placeholder="Nova senha" class="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-[10px] outline-none">
                    <button onclick="changePassword()" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-[9px] font-bold rounded-lg transition-all uppercase tracking-widest border border-slate-700">ATUALIZAR SENHA</button>
                    <p id="password-change-status" class="text-[8px] font-bold text-center"></p>
                 </div>
            </section>
        </div>
    </aside>

    <!-- CONTEUDO PRINCIPAL -->
    <main id="main-content" class="main-content flex-1 h-screen flex flex-col relative overflow-hidden">
        <!-- HEADER -->
        <header class="glass border-b border-slate-800 px-6 py-4 flex items-center justify-between h-16 shrink-0 z-30 shadow-md">
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
                <div id="status-badge" class="px-4 py-1.5 rounded-full text-[9px] font-black tracking-[0.2em] flex items-center gap-2 glass border-slate-700/50 text-slate-500 uppercase">
                    <div class="w-1.5 h-1.5 rounded-full bg-slate-600 status-dot"></div><span class="status-text">OFFLINE</span>
                </div>
                <div class="flex items-center gap-4 border-l border-slate-800 pl-6">
                    <button onclick="handleUpdate()" class="text-amber-500/50 hover:text-amber-500 transition-colors" title="Atualizar"><i class="fas fa-sync-alt"></i></button>
                    <button onclick="handleShutdown()" class="text-red-500/50 hover:text-red-500 transition-colors" title="Desligar"><i class="fas fa-power-off"></i></button>
                    <button onclick="handleLogout()" class="text-slate-500 hover:text-white transition-colors" title="Sair"><i class="fas fa-sign-out-alt"></i></button>
                </div>
            </div>
        </header>

        <div id="dashboard" class="flex-1 flex flex-col min-h-0" style="display: {'flex' if is_authenticated else 'none'};">
            <!-- METRICAS (FIXAS) -->
            <div id="metrics-panel" class="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 md:p-8 bg-slate-950/20 shrink-0 border-b border-slate-800/30">
                <div class="glass p-4 rounded-2xl border-l-2 border-blue-600">
                    <div class="flex justify-between items-center mb-1">
                        <p class="text-[8px] font-black text-slate-500 uppercase tracking-widest">CPU HOST</p>
                        <i class="fas fa-microchip text-slate-700 text-[8px]"></i>
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
                        <p class="text-[8px] font-black text-slate-500 uppercase tracking-widest">RAM HOST</p>
                        <i class="fas fa-memory text-slate-700 text-[8px]"></i>
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

            <!-- TABS AREA -->
            <div class="flex-1 flex flex-col min-h-0 bg-slate-900/10">
                <!-- BARRA DE ABAS -->
                <nav id="tab-bar" class="bg-slate-950/40 border-b border-slate-800 px-4 flex items-center gap-1 overflow-x-auto hide-scrollbar h-12 shrink-0">
                    <!-- Tabs injetadas via JS -->
                </nav>

                <!-- CONTAINER DE CONTEUDO -->
                <div id="tabs-container" class="flex-1 relative overflow-hidden">
                    <!-- Tela Vazia -->
                    <div id="no-tab-content" class="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-slate-950/30">
                         <div class="w-20 h-20 rounded-[2rem] bg-slate-900 flex items-center justify-center mb-6 border border-slate-800 shadow-inner">
                             <i class="fas fa-cubes text-3xl text-slate-700"></i>
                         </div>
                         <h3 class="text-xl font-bold text-slate-300 tracking-tight">Arquitetura Multi-Modelo</h3>
                         <p class="text-xs text-slate-500 mt-3 max-w-sm leading-relaxed uppercase tracking-[0.2em]">
                             Selecione modelos na biblioteca lateral para gerenciar configurações e instâncias independentes.
                         </p>
                         <button onclick="toggleSidebar(true)" class="mt-8 px-10 py-3.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-2xl uppercase tracking-[0.25em] transition-all shadow-2xl shadow-blue-600/20 active:scale-95">
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
        <div class="tab-content h-full flex-col overflow-hidden">
            <div class="flex-1 flex flex-col xl:flex-row min-h-0 overflow-y-auto xl:overflow-hidden custom-scroll">
                
                <!-- PAINEL DE CONFIG (ESQUERDA) -->
                <div class="flex-1 p-6 md:p-8 space-y-6 xl:overflow-y-auto custom-scroll bg-slate-900/10">
                    <!-- Header da Tab -->
                    <div class="flex items-center justify-between gap-6 flex-wrap pb-6 border-b border-slate-800/60">
                        <div class="flex items-center gap-5">
                            <div class="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center shadow-inner">
                                <i class="fas fa-cube text-blue-500 text-xl"></i>
                            </div>
                            <div>
                                <h2 class="model-tab-name text-2xl font-bold text-white tracking-tight leading-none">NOME</h2>
                                <p class="model-tab-path text-[10px] text-slate-500 font-mono mt-2 uppercase tracking-tighter opacity-50 truncate max-w-md"></p>
                            </div>
                        </div>
                        <div class="flex items-center gap-4">
                             <div class="tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500 shadow-sm transition-all">OFFLINE</div>
                             <div class="tab-actions flex items-center gap-3">
                                 <!-- Buttons Start/Stop/Chat -->
                             </div>
                        </div>
                    </div>

                    <!-- Configurações em Grid -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <!-- Configuração do Motor -->
                        <div class="glass rounded-[2rem] p-6 space-y-6 shadow-sm">
                             <div class="flex items-center justify-between">
                                <p class="text-[10px] font-black text-blue-500 uppercase tracking-[0.25em]">Parâmetros do Motor</p>
                                <i class="fas fa-sliders-h text-slate-800 text-xs"></i>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span><i class="fas fa-expand-arrows-alt text-[8px] mr-1"></i> Contexto / Slot</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-context hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <select class="tab-context-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-blue-500/50 outline-none transition-all">
                                            {ctx_opts}
                                        </select>
                                        <div class="relative tab-custom-ctx-wrap hidden">
                                            <input type="number" class="tab-context-size-custom w-24 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center">
                                            <span class="absolute right-2 top-1/2 -translate-y-1/2 text-[9px] font-black text-slate-600">K</span>
                                        </div>
                                    </div>
                                 </div>
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span><i class="fas fa-clone text-[8px] mr-1"></i> Slots (Simultâneo)</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-slots hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <input type="number" class="tab-parallel-slots w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center" value="{DEFAULT_PARALLEL_SLOTS}" min="1" max="64">
                                 </div>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-4 border-t border-slate-800/30">
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Batch Prefill</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-batch hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <select class="tab-batch-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-violet-500/50 outline-none transition-all">
                                        {batch_opts}
                                    </select>
                                 </div>
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>U-Batch Físico</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-ubatch hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <select class="tab-ubatch-size bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-4 py-2.5 text-sm font-bold w-full focus:ring-1 focus:ring-cyan-500/50 outline-none transition-all">
                                        {ubatch_opts}
                                    </select>
                                 </div>
                             </div>
                        </div>

                        <!-- Otimização & Threads -->
                        <div class="glass rounded-[2rem] p-6 space-y-6 shadow-sm">
                             <div class="flex items-center justify-between">
                                <p class="text-[10px] font-black text-emerald-500 uppercase tracking-[0.25em]">Otimização Sistêmica</p>
                                <i class="fas fa-microchip text-slate-800 text-xs"></i>
                             </div>
                             <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Threads (Gen / Batch)</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-threads hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <input type="number" class="tab-threads w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2.5 text-sm font-bold focus:ring-1 focus:ring-blue-500/50 outline-none text-center" placeholder="Auto" title="Threads para geracao">
                                        <input type="number" class="tab-threads-batch w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-3 py-2.5 text-sm font-bold focus:ring-1 focus:ring-violet-500/50 outline-none text-center" placeholder="Auto" title="Threads para prefill">
                                    </div>
                                 </div>
                                 <div class="space-y-2">
                                    <label class="text-[9px] font-black text-slate-500 uppercase tracking-widest pl-1 flex items-center justify-between">
                                        <span>Quantização de Cache</span>
                                        <label class="cursor-pointer group/pin" title="Fixar valor no Auto Balance">
                                            <input type="checkbox" class="tab-pin-cache hidden">
                                            <i class="fas fa-thumbtack text-[8px] text-slate-700 group-hover/pin:text-blue-500 transition-colors"></i>
                                        </label>
                                    </label>
                                    <div class="flex gap-2">
                                        <select class="tab-cache-type-k bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-2 py-2.5 text-[11px] font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none" title="Cache Key">
                                            {cache_type_k_opts}
                                        </select>
                                        <select class="tab-cache-type-v bg-slate-950 border border-slate-800 text-slate-300 rounded-xl px-2 py-2.5 text-[11px] font-bold w-full focus:ring-1 focus:ring-amber-500/50 outline-none" title="Cache Value">
                                            {cache_type_v_opts}
                                        </select>
                                    </div>
                                 </div>
                             </div>
                             <div class="flex flex-wrap gap-3 pt-4 border-t border-slate-800/30">
                                <label class="flex items-center gap-2 cursor-pointer bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-violet-500/30 transition-all">
                                    <input type="checkbox" class="tab-thinking-toggle w-4 h-4 bg-slate-950 border-slate-700 rounded text-violet-600">
                                    <span class="text-[10px] font-bold uppercase text-slate-500">Thinking</span>
                                </label>
                                <label class="flex items-center gap-2 cursor-pointer bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-amber-500/30 transition-all">
                                    <input type="checkbox" class="tab-mtp-toggle w-4 h-4 bg-slate-950 border-slate-700 rounded text-amber-600">
                                    <span class="text-[10px] font-bold uppercase text-slate-500">MTP</span>
                                </label>
                                <label class="flex items-center gap-2 cursor-pointer bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-cyan-500/30 transition-all">
                                    <input type="checkbox" class="tab-numa-toggle w-4 h-4 bg-slate-950 border-slate-700 rounded text-cyan-600">
                                    <span class="text-[10px] font-bold uppercase text-slate-500">NUMA</span>
                                </label>
                                <label class="flex items-center gap-2 cursor-pointer bg-slate-950/60 px-4 py-2 rounded-xl border border-slate-800 hover:border-emerald-500/30 transition-all">
                                    <input type="checkbox" class="tab-auto-balance-toggle w-4 h-4 bg-slate-950 border-slate-700 rounded text-emerald-600">
                                    <span class="text-[10px] font-bold uppercase text-slate-500">Auto Balance</span>
                                </label>
                                <select class="tab-split-mode bg-slate-950 border border-slate-800 text-slate-400 rounded-xl px-4 py-2 text-[10px] font-bold outline-none focus:ring-1 focus:ring-blue-500/50">
                                    <option value="layer">LAYER SPLIT</option>
                                    <option value="row">ROW SPLIT</option>
                                </select>
                             </div>
                        </div>
                    </div>

                    <!-- Proposed Configuration Area (Hidden until results) -->
                    <div class="tab-proposed-config hidden glass rounded-[2rem] border border-blue-500/40 bg-blue-500/5 p-6 space-y-4">
                        <div class="flex items-center justify-between">
                            <h3 class="text-[10px] font-black text-blue-400 uppercase tracking-widest flex items-center gap-2">
                                <i class="fas fa-magic"></i> Configuração Otimizada Sugerida
                            </h3>
                            <div class="flex items-center gap-3">
                                <button class="tab-apply-config-btn px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-xl uppercase tracking-widest transition-all active:scale-95 shadow-lg shadow-blue-600/20">
                                    EFETIVAR E SALVAR
                                </button>
                                <button class="tab-discard-config-btn text-[10px] font-bold text-slate-500 hover:text-slate-300 uppercase tracking-widest transition-colors">
                                    DESCARTAR
                                </button>
                            </div>
                        </div>
                        <div class="tab-proposed-details grid grid-cols-2 md:grid-cols-4 gap-4 text-[9px] font-mono text-slate-400">
                            <!-- Details injected via JS -->
                        </div>
                    </div>

                    <!-- Alocação de GPU -->
                    <div class="glass rounded-[2rem] overflow-hidden border border-slate-800/50 shadow-lg">
                        <table class="w-full text-left">
                            <thead class="text-[9px] font-black text-slate-500 uppercase tracking-[0.25em] bg-slate-950/50">
                                <tr>
                                    <th class="px-8 py-5 text-center w-16">USO</th>
                                    <th class="px-4 py-5 text-center w-24">PRINCIPAL</th>
                                    <th class="px-4 py-5">DISPOSITIVO</th>
                                    <th class="px-8 py-5 text-right">DISTRIBUIÇÃO</th>
                                </tr>
                            </thead>
                            <tbody class="tab-gpu-table-body divide-y divide-slate-800/40 bg-slate-950/10">
                                {cpu_rows}
                                {gpu_rows}
                            </tbody>
                        </table>
                        <div class="p-6 bg-slate-950/30 border-t border-slate-800/50 flex flex-wrap items-center justify-between gap-6">
                             <div class="flex items-center gap-6">
                                <button class="tab-smart-calibrate-btn px-6 py-3 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] transition-all active:scale-95 flex items-center gap-3">
                                    <i class="fas fa-brain"></i> CALIBRAR SMART (AUTO-BALANCE)
                                </button>
                                <span class="tab-total-percent text-[11px] font-bold tracking-[0.1em] text-slate-500 uppercase">CARGA TOTAL: 100%</span>
                             </div>
                             <button class="tab-reset-defaults-btn px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-[9px] font-black rounded-xl border border-slate-700 transition-all uppercase tracking-widest text-slate-400 hover:text-white">
                                 <i class="fas fa-undo mr-2 text-[8px]"></i> RESETAR PADRÕES
                             </button>
                        </div>
                    </div>

                    <!-- Alertas Localizados -->
                    <div class="tab-alerts space-y-4">
                        <div class="tab-mtp-warning hidden p-6 rounded-2xl border border-amber-500/20 bg-amber-500/5">
                            <div class="flex gap-4 items-start text-amber-500/80">
                                <i class="fas fa-bolt mt-1"></i>
                                <div class="flex-1">
                                    <p class="text-[10px] font-black uppercase tracking-widest mb-1">MTP Indisponível</p>
                                    <p class="tab-mtp-warning-msg text-xs leading-relaxed"></p>
                                </div>
                            </div>
                        </div>
                        <div class="tab-auto-balance-alert hidden p-6 rounded-2xl border border-red-500/20 bg-red-500/5">
                            <div class="flex gap-4 items-start text-red-500/80">
                                <i class="fas fa-microchip mt-1"></i>
                                <div class="flex-1">
                                    <p class="text-[10px] font-black uppercase tracking-widest mb-1">Capacidade do Hardware Excedida</p>
                                    <p class="tab-auto-balance-msg text-xs leading-relaxed"></p>
                                    <ul class="tab-auto-balance-details mt-3 text-[10px] text-slate-500 space-y-1 font-mono"></ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- PAINEL DE LOGS (DIREITA) -->
                <div class="xl:w-1/3 xl:border-l border-slate-800/60 bg-slate-950/40 flex flex-col h-[500px] xl:h-auto shadow-2xl relative">
                    <div class="p-6 border-b border-slate-800 bg-slate-900/40 flex items-center justify-between shrink-0">
                        <div class="flex items-center gap-3">
                            <div class="flex gap-1">
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                                <div class="w-1.5 h-1.5 rounded-full bg-slate-700"></div>
                            </div>
                            <p class="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500 ml-2">Console de Instância</p>
                        </div>
                        <button class="tab-clear-logs-btn text-slate-600 hover:text-red-400 transition-colors" title="Limpar logs">
                            <i class="fas fa-trash-alt text-[10px]"></i>
                        </button>
                    </div>
                    <div class="tab-log-box flex-1 p-8 font-mono text-[11px] text-slate-500 leading-relaxed overflow-y-auto custom-scroll whitespace-pre-wrap selection:bg-blue-500/20 bg-slate-950/20">
                        <!-- Logs in realtime -->
                    </div>
                    <div class="p-4 bg-slate-900/60 border-t border-slate-800/80 flex items-center justify-between shrink-0">
                         <div class="flex items-center gap-3">
                             <div class="w-2 h-2 rounded-full bg-emerald-500/50 animate-pulse"></div>
                             <span class="text-[9px] font-black text-slate-600 uppercase tracking-widest">Fluxo de Dados Ativo</span>
                         </div>
                         <span class="tab-log-size text-[8px] font-mono text-slate-700">0 KB</span>
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
                ubatch_size=saved_cfg.get("ubatch_size", 512),
                cache_type_k=saved_cfg.get("cache_type_k", DEFAULT_CACHE_TYPE),
                cache_type_v=saved_cfg.get("cache_type_v", DEFAULT_CACHE_TYPE),
                numa_enabled=saved_cfg.get("numa_enabled", False),
                threads=saved_cfg.get("threads", 0),
                threads_batch=saved_cfg.get("threads_batch", 0),
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
