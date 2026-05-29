#!/usr/bin/env python3
"""
Automanager Llama.cpp - Control Plane para llama-server
FastAPI application that orchestrates llama-server instances with multi-GPU
tensor split management, OOM auto-recovery, and real-time hardware monitoring.
"""

import json
import os
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from config_manager import (
    ConfigManager,
    TokenManager,
    AuthManager,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
)
from gpu_manager import GPUManager
from log_manager import LogManager, logger
from process_manager import ProcessManager, OOMWatchdog
from model_manager import ModelScanner, DownloadManager
from schemas import (
    GPUWeight,
    StartRequest,
    DeleteRequest,
    DownloadRequest,
    SetDefaultRequest,
    RenameRequest,
    LoginRequest,
)
from gpu_manager import GPUDetector
from model_manager import ModelScanner
from config_manager import ConfigManager, TokenManager

MANAGER_PORT = 8000

# Initialize logging and services
log_manager = LogManager()
config_manager = ConfigManager()
token_manager = TokenManager(config_manager)
auth_manager = AuthManager(config_manager, token_manager)
gpu_manager = GPUManager()
gpu_detector = gpu_manager
process_manager = ProcessManager(
    config_manager, token_manager, gpu_manager, log_manager
)
model_scanner = ModelScanner(config_manager, process_manager)
download_mgr = DownloadManager()
oom_watchdog = OOMWatchdog(
    process_manager, config_manager, gpu_manager, log_manager
)

app = FastAPI(title="Automanager Llama.cpp")

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

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
    session_token = auth_manager.authenticate(req.username, req.password)
    if not session_token:
        raise HTTPException(status_code=401, detail="Credenciais invalidas")
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=3600,
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


@app.get("/status")
async def get_status(_auth: str = Depends(get_current_auth)):
    return process_manager.get_status()


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

    base_settings = {
        "context_size": req.context_size,
        "parallel_slots": req.parallel_slots,
        "mmproj_path": req.mmproj_path,
        "split_mode": req.split_mode,
        "auto_balance": req.auto_balance,
    }

    if req.manual_gpu_override:
        config_manager.update_model_settings(
            req.path,
            {
                **base_settings,
                "gpu_weights": [w.model_dump() for w in req.gpu_weights],
                "auto_balance_profile": False,
            },
        )
        return process_manager.start(
            model_path=req.path,
            gpu_weights=req.gpu_weights,
            context_size=req.context_size,
            mmproj_path=req.mmproj_path,
            split_mode=req.split_mode,
            parallel_slots=req.parallel_slots,
        )

    if req.auto_balance:
        saved = config_manager.get_model_settings(req.path)
        if saved.get("auto_balance_profile") and saved.get("gpu_weights"):
            weights = [
                GPUWeight(**w) if isinstance(w, dict) else w
                for w in saved["gpu_weights"]
            ]
            config_manager.update_model_settings(
                req.path,
                {
                    **base_settings,
                    "gpu_weights": [w.model_dump() for w in weights],
                    "auto_balance_profile": True,
                },
            )
            return process_manager.start(
                model_path=req.path,
                gpu_weights=weights,
                context_size=req.context_size,
                mmproj_path=req.mmproj_path,
                split_mode=req.split_mode,
                parallel_slots=req.parallel_slots,
            )
        return process_manager.start_auto_balance(req)

    config_manager.update_model_settings(
        req.path,
        {
            **base_settings,
            "gpu_weights": [w.model_dump() for w in req.gpu_weights],
            "auto_balance_profile": False,
        },
    )
    return process_manager.start(
        model_path=req.path,
        gpu_weights=req.gpu_weights,
        context_size=req.context_size,
        mmproj_path=req.mmproj_path,
        split_mode=req.split_mode,
        parallel_slots=req.parallel_slots,
    )


@app.post("/stop")
async def stop_model(_auth: str = Depends(get_current_auth)):
    return process_manager.stop()


# --- Hardware Metrics ---


@app.get("/metrics")
async def get_metrics(_auth: str = Depends(get_current_auth)):
    return gpu_manager.get_metrics()


# --- Model Management ---


@app.get("/models")
async def list_models(_auth: str = Depends(get_current_auth)):
    return model_scanner.scan()


@app.post("/rename")
async def rename_model(
    req: RenameRequest, _auth: str = Depends(get_current_auth)
):
    new_path = model_scanner.rename_model(req.path, req.new_name)
    return {"status": "renamed", "new_path": new_path}


@app.post("/delete")
async def delete_model(
    req: DeleteRequest, _auth: str = Depends(get_current_auth)
):
    model_scanner.delete_model(req.path)
    return {"status": "deleted"}


# --- Downloads ---


@app.post("/downloads")
async def start_download_endpoint(
    req: DownloadRequest,
    background_tasks: BackgroundTasks,
    _auth: str = Depends(get_current_auth),
):
    download_id = download_mgr.start_download(req.url, req.filename)
    return {"download_id": download_id}


@app.get("/downloads")
async def get_downloads(_auth: str = Depends(get_current_auth)):
    return download_mgr.get_progress()


# --- API Key Management ---


@app.get("/api/key")
async def get_api_key(_auth: str = Depends(get_current_auth)):
    return {"key": token_manager.get_or_create()}


@app.post("/api/key/renew")
async def renew_api_key(_auth: str = Depends(get_current_auth)):
    return {"key": token_manager.renew()}


# --- Logging ---


@app.get("/logs")
async def stream_logs():
    return log_manager.stream_logs()


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
    config_manager.set_default_model(req.path)
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────
# Frontend (Steps 10–16)
# Embedded SPA with login overlay, dashboard, and all JS
# ─────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the dashboard SPA."""
    # Check auth — if not authenticated, show login overlay
    session_token = request.cookies.get("session_token")
    is_authenticated = (
        session_token and auth_manager.verify_session(session_token)
    )

    models_data = model_scanner.scan()
    models = models_data["models"]
    projectors = models_data["projectors"]
    gpus = gpu_manager.detect_gpus()
    config = config_manager.load()
    default_model = config.get("default_model")
    model_configs = config.get("model_configs", {})
    status = process_manager.get_status()

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

    # Vision options
    vision_options = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>'
    for p in projectors:
        vision_options += f'<option value="{p["path"]}" class="bg-slate-900">{p["name"]}</option>'

    # Context options
    ctx_opts = ""
    for val, label in [
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
    ]:
        selected = "selected" if (
            status.get("running")
            and status.get("config", {}).get("context_size") == val
        ) else ""
        if not selected and val == DEFAULT_CONTEXT_SIZE:
            selected = "selected"
        ctx_opts += f'<option value="{val}" class="bg-slate-900" {selected}>{label}</option>'

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
        if m_path in model_configs:
            initial_cfg_js = (
                f"<script>window.modelConfigs['{m_js}'] = {json.dumps(model_configs[m_path])};</script>"
            )
        has_config = "text-blue-400" if m_path in model_configs else "text-slate-100"

        model_items += f"""
        {initial_cfg_js}
        <div id="{stable_id}" class="model-item-container group flex items-center justify-between p-4 mb-3 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg" data-path="{m_js}">
            <div class="flex-1 min-w-0 mr-4 cursor-pointer" onclick="selectModel('{m_js}', '{stable_id}')" title="Clique para selecionar e carregar configuracoes">
                <div class="flex items-center gap-2 mb-1">
                    <i class="fas fa-cube text-blue-400 text-[10px]"></i>
                    <p class="model-name text-sm font-bold {has_config} break-all line-clamp-2">{m_name}</p>
                    {'<i class="fas fa-history text-[8px] text-blue-500/50 history-icon" title="Configuracao salva disponivel"></i>' if m_path in model_configs else ''}
                </div>
                <p class="text-[9px] text-slate-500 truncate uppercase tracking-tighter font-mono">{m_dir}</p>
            </div>
            <div class="flex items-center gap-3 md:gap-4">
                <div class="flex items-center gap-1">
                    <button onclick="renameModel('{m_js}')" class="rename-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all" title="Renomear Modelo">
                        <i class="fas fa-edit text-[10px]"></i>
                    </button>
                    <button onclick="deleteModel('{m_js}')" class="delete-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all" title="Excluir Modelo">
                        <i class="fas fa-trash-alt text-[10px]"></i>
                    </button>
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
        model_items=model_items,
        vision_options=vision_options,
        ctx_opts=ctx_opts,
        local_ip=local_ip,
        api_token=api_token,
        is_authenticated=is_authenticated,
    )
    return HTMLResponse(html)


def _build_html(
    gpu_rows: str,
    model_items: str,
    vision_options: str,
    ctx_opts: str,
    local_ip: str,
    api_token: str,
    is_authenticated: bool,
) -> str:
    """Build the full HTML template."""

    login_overlay = ""
    if not is_authenticated:
        login_overlay = """
        <div id="login-overlay" class="fixed inset-0 z-50 flex items-center justify-center" style="background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #020617 100%);">
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

    return f"""<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automanager Llama.cpp</title>
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
        #pacman-background {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0; opacity: 0.35; pointer-events: none; }}
        #dashboard {{ position: relative; z-index: 1; }}
        .btn-gradient {{ background: linear-gradient(135deg, #1e30f3 0%, #e21e80 100%); }}
        .text-gradient {{ background: linear-gradient(315deg, #1e30f3 0%, #e21e80 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .metric-dimmed {{ opacity: 0.45; filter: grayscale(0.4); }}
        @media (prefers-reduced-motion: reduce) {{ #pacman-background {{ display: none; }} }}
    </style>
</head>
<body class="min-h-screen text-slate-200 pb-16 selection:bg-blue-500/30">
    <canvas id="pacman-background" aria-hidden="true"></canvas>
    {login_overlay}
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
                <button onclick="handleLogout()" class="text-slate-500 hover:text-white transition-colors" title="Sair">
                    <i class="fas fa-sign-out-alt"></i>
                </button>
            </div>
        </header>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10">
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
                                <label class="text-[9px] font-black uppercase text-slate-400 pl-3 md:pl-4 tracking-widest whitespace-nowrap" title="Tokens por slot (--ctx-size / --parallel no llama-server)">Contexto/slot:</label>
                                <select id="context-size" class="bg-blue-600/20 border border-blue-500/30 text-blue-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                    {ctx_opts}
                                </select>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap" title="Requisições simultâneas (--parallel)"><i class="fas fa-clone text-blue-400 mr-2"></i>Slots:</label>
                                <input type="number" id="parallel-slots" value="{DEFAULT_PARALLEL_SLOTS}" min="1" max="64"
                                       class="w-16 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-3 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all text-center">
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-eye text-blue-400 mr-2"></i>Vision:</label>
                                <select id="mmproj-path" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer max-w-[200px]">
                                    {vision_options}
                                </select>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-layer-group text-blue-400 mr-2"></i>Split:</label>
                                <select id="split-mode" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                    <option value="layer">Layer (Sqn)</option>
                                    <option value="row">Row (Par)</option>
                                </select>
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
                        </div>
                        <span id="total-percent" class="text-xs md:text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl transition-all duration-300">CARGA TOTAL: 100%</span>
                    </div>
                </div>
                <div id="active-card" class="bg-gradient-to-r from-blue-900/40 to-slate-900/40 backdrop-blur-xl p-6 md:p-10 rounded-[2rem] md:rounded-[2.5rem] border border-blue-500/30 hidden transition-all duration-700">
                    <div class="flex flex-col lg:flex-row items-center justify-between gap-8 md:gap-10">
                        <div class="flex items-center gap-5 md:gap-8 w-full">
                            <div class="w-16 h-16 rounded-3xl bg-blue-600 flex items-center justify-center text-white shadow-2xl shadow-blue-500/40 shrink-0">
                                <i class="fas fa-robot text-2xl md:text-3xl"></i>
                            </div>
                            <div class="min-w-0">
                                <p class="text-blue-400 text-[10px] font-black uppercase tracking-[0.3em] mb-2 font-mono">Motor de Computação Primário</p>
                                <h2 id="active-model-name" class="text-xl md:text-2xl font-bold text-white truncate max-w-[200px] sm:max-w-md">--</h2>
                                <div class="flex gap-4 mt-3">
                                    <div class="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                                        Ativo há: <span id="uptime-val">Calculando...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="flex flex-col sm:flex-row gap-4 md:gap-6 w-full lg:w-auto">
                            <a id="chat-link" href="#" target="_blank" class="px-6 md:px-10 py-4 btn-gradient text-white rounded-2xl text-[10px] md:text-xs font-black transition-all shadow-xl shadow-blue-600/30 active:scale-95 flex items-center justify-center gap-3 md:gap-4 uppercase tracking-widest whitespace-nowrap pointer-events-none opacity-40" aria-disabled="true">
                                <i class="fas fa-comments text-sm"></i> ABRIR CHAT
                            </a>
                            <button onclick="stopModel()" class="px-6 md:px-10 py-4 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/30 rounded-2xl text-[10px] md:text-xs font-black transition-all active:scale-95 uppercase tracking-widest whitespace-nowrap">
                                ENCERRAR
                            </button>
                        </div>
                    </div>
                </div>
                <div class="glass rounded-[2rem] overflow-hidden shadow-2xl border border-slate-800">
                    <div class="px-6 md:px-10 py-4 bg-slate-900/60 border-b border-slate-800 flex justify-between items-center">
                        <div class="flex items-center gap-3 md:gap-4">
                            <div class="flex gap-1.5 md:gap-2">
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                                <div class="w-2 h-2 rounded-full bg-slate-700"></div>
                            </div>
                            <p class="text-slate-400 text-[9px] md:text-[10px] font-black uppercase tracking-widest font-mono ml-2 md:ml-4">Saída de logs do sistema</p>
                        </div>
                        <button onclick="document.getElementById('log-box').innerHTML=''" class="text-[9px] md:text-[10px] text-slate-600 hover:text-blue-400 font-bold uppercase transition-colors tracking-widest">
                            <i class="fas fa-trash-alt mr-2"></i> Limpar
                        </button>
                    </div>
                    <div id="log-box" class="custom-scroll p-6 md:p-10 h-[300px] md:h-[400px] overflow-y-auto font-mono text-[10px] md:text-xs text-slate-400 leading-relaxed whitespace-pre-wrap bg-slate-950/40"></div>
                </div>
            </div>
            <div class="lg:col-span-5 space-y-6 md:space-y-10">
                <div class="glass rounded-[2rem] border border-slate-800 flex flex-col h-auto md:h-[1100px] xl:h-[1200px]">
                    <div class="p-8 border-b border-slate-800/50 flex items-center justify-between">
                        <div class="flex items-center gap-4 md:gap-5">
                            <div class="w-10 h-10 bg-slate-800 rounded-xl flex items-center justify-center border border-slate-700">
                                <i class="fas fa-database text-amber-500 text-sm md:text-base"></i>
                            </div>
                            <h3 class="font-bold text-base md:text-lg text-white tracking-tight">Model Repository</h3>
                        </div>
                        <span class="text-[10px] bg-slate-800 text-slate-400 px-3 py-1 rounded-full font-mono border border-slate-700" id="model-count">0 UNIDADES</span>
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
                    <div id="model-list-container" class="p-6 flex-1 overflow-y-auto custom-scroll space-y-2">
                        {model_items}
                    </div>
                    <div class="p-8 bg-slate-950/40 border-t border-slate-800 rounded-b-[2rem] md:rounded-b-[2.5rem]">
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
    </div>
    <script>
        let logStream = null;
        let startTime = null;
        const fixedIp = "{local_ip}";
        window.modelConfigs = window.modelConfigs || {{}};
        let currentSelectedModel = null;
        let currentRunningModelPath = null;
        let manualGpuOverride = false;

        document.getElementById('chat-link').href = `http://${{fixedIp}}:8085/`;
        document.getElementById('api-link').innerText = `http://${{fixedIp}}:8085/v1`;

        // ─── Authentication ───
        async function handleLogin(event) {{
            event.preventDefault();
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;
            try {{
                const res = await fetch('/api/auth/login', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{username, password}}),
                }});
                if (res.ok) {{
                    document.getElementById('login-overlay').style.display = 'none';
                    document.getElementById('dashboard').style.display = 'block';
                    initDashboard();
                }} else {{
                    const err = await res.json();
                    const el = document.getElementById('login-error');
                    el.textContent = err.detail || 'Erro no login';
                    el.classList.remove('hidden');
                }}
            }} catch (e) {{
                const el = document.getElementById('login-error');
                el.textContent = 'Erro de rede';
                el.classList.remove('hidden');
            }}
        }}

        async function handleLogout() {{
            try {{ await fetch('/api/auth/logout', {{method: 'POST'}}); }} catch (e) {{}}
            location.reload();
        }}

        async function changePassword() {{
            const currentPassword = document.getElementById('current-password').value;
            const newPassword = document.getElementById('new-password').value;
            const statusEl = document.getElementById('password-change-status');

            statusEl.textContent = '';
            statusEl.className = 'text-[10px] font-bold min-h-[1rem]';

            if (!currentPassword || !newPassword) {{
                statusEl.textContent = 'Informe a senha atual e a nova senha.';
                statusEl.classList.add('text-amber-500');
                return;
            }}

            if (newPassword.length < 6) {{
                statusEl.textContent = 'A nova senha deve ter pelo menos 6 caracteres.';
                statusEl.classList.add('text-amber-500');
                return;
            }}

            try {{
                const res = await fetch('/api/auth/change-password', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{username: currentPassword, password: newPassword}}),
                }});
                if (res.ok) {{
                    document.getElementById('current-password').value = '';
                    document.getElementById('new-password').value = '';
                    statusEl.textContent = 'Senha alterada com sucesso.';
                    statusEl.classList.add('text-emerald-500');
                }} else {{
                    const err = await res.json();
                    statusEl.textContent = err.detail || 'Erro ao alterar senha.';
                    statusEl.classList.add('text-red-500');
                }}
            }} catch (e) {{
                statusEl.textContent = 'Erro de rede ao alterar senha.';
                statusEl.classList.add('text-red-500');
            }}
        }}

        function markManualGpuChange() {{
            manualGpuOverride = true;
            const badge = document.getElementById('auto-balance-badge');
            if (badge) badge.classList.add('hidden');
        }}

        function updateAutoBalanceProfileBadge(hasProfile) {{
            const badge = document.getElementById('auto-balance-badge');
            if (!badge) return;
            let show = !!hasProfile;
            if (hasProfile === undefined && currentSelectedModel) {{
                const cfg = window.modelConfigs[currentSelectedModel];
                show = !!(cfg && cfg.auto_balance_profile);
            }}
            badge.classList.toggle('hidden', !show);
        }}

        function bindGpuManualListeners() {{
            document.querySelectorAll('.gpu-weight').forEach(el => {{
                el.addEventListener('input', markManualGpuChange);
            }});
            document.querySelectorAll('.gpu-checkbox').forEach(el => {{
                el.addEventListener('change', () => {{
                    markManualGpuChange();
                    redistributeUnpinnedWeights(null);
                }});
            }});
            document.querySelectorAll('.gpu-main-radio').forEach(el => {{
                el.addEventListener('change', markManualGpuChange);
            }});
            document.querySelectorAll('.gpu-pin').forEach(el => {{
                el.addEventListener('change', () => onGpuPinToggle(el));
            }});
            document.querySelectorAll('.gpu-pin:checked').forEach(pin => {{
                pin.closest('.gpu-row')?.querySelector('.gpu-weight')
                    ?.classList.add('ring-2', 'ring-amber-500/40');
            }});
        }}

        function onGpuPinToggle(pinCheckbox) {{
            markManualGpuChange();
            const row = pinCheckbox.closest('.gpu-row');
            const weightInput = row?.querySelector('.gpu-weight');
            if (weightInput) {{
                if (pinCheckbox.checked) {{
                    weightInput.classList.add('ring-2', 'ring-amber-500/40');
                }} else {{
                    weightInput.classList.remove('ring-2', 'ring-amber-500/40');
                }}
            }}
            redistributeUnpinnedWeights(weightInput);
        }}

        function redistributeUnpinnedWeights(changedInput) {{
            const rows = Array.from(document.querySelectorAll('.gpu-row'))
                .filter(r => r.querySelector('.gpu-checkbox').checked);
            const pinnedRows = rows.filter(r => r.querySelector('.gpu-pin').checked);
            const unpinnedRows = rows.filter(r => !r.querySelector('.gpu-pin').checked);

            if (rows.length === 0) {{
                updateTotal();
                return;
            }}
            if (rows.length === 1) {{
                rows[0].querySelector('.gpu-weight').value = 100;
                updateTotal();
                return;
            }}

            const pinnedSum = pinnedRows.reduce(
                (s, r) => s + (parseInt(r.querySelector('.gpu-weight').value, 10) || 0), 0
            );

            if (unpinnedRows.length === 0) {{
                updateTotal();
                return;
            }}

            if (unpinnedRows.length === 1) {{
                unpinnedRows[0].querySelector('.gpu-weight').value = Math.max(0, 100 - pinnedSum);
                updateTotal();
                return;
            }}

            const changedRow = changedInput?.closest('.gpu-row');
            const changedIsPinned = changedRow && changedRow.querySelector('.gpu-pin').checked;

            if (changedIsPinned || !changedInput) {{
                let remaining = Math.max(0, 100 - pinnedSum);
                for (let i = 0; i < unpinnedRows.length; i++) {{
                    const input = unpinnedRows[i].querySelector('.gpu-weight');
                    if (i === unpinnedRows.length - 1) {{
                        input.value = remaining;
                    }} else {{
                        const share = Math.min(
                            remaining,
                            Math.max(0, Math.round(remaining / (unpinnedRows.length - i)))
                        );
                        input.value = share;
                        remaining -= share;
                    }}
                }}
                updateTotal();
                return;
            }}

            let val = parseInt(changedInput.value, 10) || 0;
            const maxForChanged = Math.max(0, 100 - pinnedSum);
            if (val > maxForChanged) {{ val = maxForChanged; changedInput.value = val; }}
            if (val < 0) {{ val = 0; changedInput.value = 0; }}

            const otherUnpinned = unpinnedRows.filter(
                r => r.querySelector('.gpu-weight') !== changedInput
            );
            let remaining = maxForChanged - val;
            for (let i = 0; i < otherUnpinned.length; i++) {{
                const input = otherUnpinned[i].querySelector('.gpu-weight');
                if (i === otherUnpinned.length - 1) {{
                    input.value = Math.max(0, remaining);
                }} else {{
                    const share = Math.min(
                        remaining,
                        Math.max(0, Math.round(remaining / (otherUnpinned.length - i)))
                    );
                    input.value = share;
                    remaining -= share;
                }}
            }}
            updateTotal();
        }}

        // ─── Dashboard functions ───
        function initDashboard() {{
            bindGpuManualListeners();
            updateStatus();
            updateMetrics();
            updateDownloads();
            updateModels();
            updateTotal();
        }}

        function getModelButtonsHtml(path, elementId, isRunning) {{
            if (isRunning) {{
                return `<div class="flex items-center gap-3">
                    <a href="http://${{fixedIp}}:8085/" target="_blank" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[9px] font-black rounded-xl flex items-center gap-2 uppercase tracking-widest shadow-lg shadow-blue-600/20 transition-all whitespace-nowrap">
                        <i class="fas fa-comments text-[8px]"></i> ABRIR INTERFACE
                    </a>
                    <button onclick="stopModel()" class="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 text-[9px] font-black rounded-xl transition-all uppercase tracking-widest whitespace-nowrap">
                        ENCERRAR
                    </button>
                    <div class="flex items-center gap-2 text-[9px] font-mono text-emerald-400 bg-emerald-500/5 px-3 py-2 rounded-xl border border-emerald-500/10">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span id="uptime-val">--</span>
                    </div>
                </div>`;
            }}
            return `<button onclick="startModel('${{path}}', '${{elementId}}')" class="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-2xl active:scale-95 flex items-center gap-3 uppercase tracking-widest shadow-xl shadow-blue-600/20 transition-all">
                <i class="fas fa-play text-[9px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span>
            </button>`;
        }}

        function resetToDefaults() {{
            document.getElementById('context-size').value = "{DEFAULT_CONTEXT_SIZE}"; // 65536
            document.getElementById('parallel-slots').value = "{DEFAULT_PARALLEL_SLOTS}";
            document.getElementById('mmproj-path').value = "";
            document.getElementById('split-mode').value = "layer";
            const toggle = document.getElementById('auto-balance-toggle');
            if (toggle) toggle.checked = false;
            document.querySelectorAll('.gpu-row').forEach((row, idx) => {{
                row.querySelector('.gpu-checkbox').checked = true;
                row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
                row.querySelector('.gpu-main-radio').checked = (idx === 0);
                const pin = row.querySelector('.gpu-pin');
                if (pin) pin.checked = false;
                row.querySelector('.gpu-weight')?.classList.remove('ring-2', 'ring-amber-500/40');
            }});
            markManualGpuChange();
            updateTotal();
        }}

        function selectModel(path, elementId) {{
            currentSelectedModel = path;
            document.querySelectorAll('.model-item-container').forEach(el => {{
                el.classList.remove('active-selection');
            }});
            const selectedEl = document.getElementById(elementId);
            if (selectedEl) selectedEl.classList.add('active-selection');
            if (window.modelConfigs[path]) {{
                applyModelConfig(path);
            }} else {{
                resetToDefaults();
            }}
        }}

        function applyModelConfig(path) {{
            const cfg = window.modelConfigs[path];
            if (!cfg) return;
            if (cfg.context_size) document.getElementById('context-size').value = cfg.context_size;
            if (cfg.parallel_slots) document.getElementById('parallel-slots').value = cfg.parallel_slots;
            if (cfg.split_mode) document.getElementById('split-mode').value = cfg.split_mode;
            if (cfg.mmproj_path !== undefined) {{
                const select = document.getElementById('mmproj-path');
                let found = false;
                for (let i = 0; i < select.options.length; i++) {{
                    if (select.options[i].value === cfg.mmproj_path) {{
                        select.value = cfg.mmproj_path;
                        found = true;
                        break;
                    }}
                }}
                if (!found && cfg.mmproj_path) {{
                    const opt = document.createElement('option');
                    opt.value = cfg.mmproj_path;
                    opt.text = cfg.mmproj_path.split('/').pop() + " (Salvo)";
                    select.add(opt);
                    select.value = cfg.mmproj_path;
                }} else if (!cfg.mmproj_path) {{
                    select.value = "";
                }}
            }}
            const abToggle = document.getElementById('auto-balance-toggle');
            if (abToggle) abToggle.checked = !!cfg.auto_balance;
            updateAutoBalanceProfileBadge(cfg.auto_balance_profile);
            manualGpuOverride = false;
            if (cfg.gpu_weights) {{
                cfg.gpu_weights.forEach(w => {{
                    const row = document.querySelector(`.gpu-row[data-index="${{w.index}}"]`);
                    if (row) {{
                        const cb = row.querySelector('.gpu-checkbox');
                        const input = row.querySelector('.gpu-weight');
                        const radio = row.querySelector('.gpu-main-radio');
                        const pin = row.querySelector('.gpu-pin');
                        cb.checked = w.active !== undefined ? w.active : (w.weight > 0);
                        input.value = Math.round(w.weight);
                        if (w.is_main) radio.checked = true;
                        if (pin) {{
                            pin.checked = !!w.pinned;
                            if (w.pinned) input.classList.add('ring-2', 'ring-amber-500/40');
                            else input.classList.remove('ring-2', 'ring-amber-500/40');
                        }}
                    }}
                }});
                updateTotal();
            }}
            const nameEl = document.querySelector(`[data-path="${{path}}"] .model-name`);
            if (nameEl) {{
                nameEl.classList.add('text-emerald-400');
                setTimeout(() => {{ nameEl.classList.remove('text-emerald-400'); }}, 1000);
            }}
        }}

        function balanceWeights(changedInput) {{
            redistributeUnpinnedWeights(changedInput);
        }}

        function updateTotal() {{
            let sum = 0;
            document.querySelectorAll('.gpu-weight').forEach(i => {{
                const isChecked = i.closest('.gpu-row').querySelector('.gpu-checkbox').checked;
                if (isChecked) sum += parseInt(i.value || 0);
                else i.value = 0;
            }});
            const badge = document.getElementById('total-percent');
            badge.innerText = `CARGA TOTAL: ${{sum}}%`;
            badge.className = sum === 100
                ? 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20'
                : 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/20';
        }}

        async function updateMetrics() {{
            try {{
                const res = await fetch('/metrics');
                const data = await res.json();
                const metricsPanel = document.getElementById('metrics-panel');
                if (metricsPanel) {{
                    if (!currentRunningModelPath) metricsPanel.classList.add('metric-dimmed');
                    else metricsPanel.classList.remove('metric-dimmed');
                }}
                document.getElementById('cpu-val').innerText = data.cpu + '%';
                document.getElementById('cpu-bar').style.width = data.cpu + '%';
                document.getElementById('ram-val').innerText = data.ram + '%';
                document.getElementById('ram-bar').style.width = data.ram + '%';
                data.gpus.forEach(g => {{
                    const row = document.querySelector(`.gpu-row[data-index="${{g.index}}"]`);
                    if (row) {{
                        row.querySelector('.gpu-util-val').innerText = g.util + '%';
                        row.querySelector('.gpu-util-bar').style.width = g.util + '%';
                        row.querySelector('.gpu-temp-val').innerText = (g.temp || '--') + '°C';
                        row.querySelector('.gpu-power-val').innerText = (g.power || '--') + 'W';
                        row.querySelector('.gpu-vram-text').innerText = `${{g.mem_used}} / ${{g.mem_total}} MB`;
                        row.querySelector('.gpu-vram-bar').style.width = g.vram_pct + '%';
                    }}
                }});
            }} catch (e) {{}}
        }}

        async function startLogs() {{
            if (logStream) logStream.abort();
            logStream = new AbortController();
            const box = document.getElementById('log-box');
            box.innerHTML = '';
            try {{
                const response = await fetch('/logs', {{ signal: logStream.signal }});
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                while (true) {{
                    const {{ value, done }} = await reader.read();
                    if (done) break;
                    const formatted = decoder.decode(value)
                        .replace(/error/gi, '<span class="text-red-500 font-black px-1 rounded bg-red-500/10">ERRO</span>')
                        .replace(/warn/gi, '<span class="text-amber-500 font-black px-1 rounded bg-amber-500/10">AVISO</span>')
                        .replace(/info/gi, '<span class="text-blue-400 font-bold uppercase tracking-tighter">info</span>');
                    const line = document.createElement('div');
                    line.className = 'terminal-line mb-1 md:mb-2 border-l border-slate-800 md:border-l-2 pl-3 md:pl-4';
                    line.innerHTML = formatted;
                    box.appendChild(line);
                    box.scrollTop = box.scrollHeight;
                    if (box.childNodes.length > 500) box.removeChild(box.firstChild);
                }}
            }} catch (e) {{}}
        }}

        function updateUptime(serverStartTime) {{
            let diff;
            if (serverStartTime) {{
                diff = Math.floor(Date.now() / 1000 - serverStartTime);
            }} else if (startTime) {{
                diff = Math.floor((new Date() - startTime) / 1000);
            }} else {{ return; }}
            document.getElementById('uptime-val').innerText = `${{Math.floor(diff/3600)}}h ${{Math.floor((diff%3600)/60)}}m ${{diff%60}}s`;
        }}

        async function renewToken() {{
            if (!confirm("Deseja realmente gerar uma nova chave de API?")) return;
            try {{
                const res = await fetch('/api/key/renew', {{ method: 'POST' }});
                const data = await res.json();
                document.getElementById('api-token').innerText = data.key;
                alert("Nova chave gerada!");
            }} catch (e) {{
                alert("Erro ao renovar token.");
            }}
        }}

        async function updateStatus() {{
            try {{
                const res = await fetch('/status');
                const data = await res.json();
                const badge = document.getElementById('status-badge');
                const card = document.getElementById('active-card');

                if (data.running && data.config && data.config.path) {{
                    window.modelConfigs[data.config.path] = window.modelConfigs[data.config.path] || {{}};
                    Object.assign(window.modelConfigs[data.config.path], data.config);
                    if (currentSelectedModel === data.config.path) {{
                        updateAutoBalanceProfileBadge(data.config.auto_balance_profile);
                    }}
                }}

                if (data.config && data.config.gpu_weights && (!data.recovery || !data.recovery.active)) {{
                    data.config.gpu_weights.forEach(w => {{
                        const row = document.querySelector(`.gpu-row[data-index="${{w.index}}"]`);
                        if (row) {{
                            const input = row.querySelector('.gpu-weight');
                            const cb = row.querySelector('.gpu-checkbox');
                            const isPinned = row.querySelector('.gpu-pin')?.checked;
                            if (!isPinned && document.activeElement !== input) {{
                                const newWeight = Math.round(w.weight);
                                if (parseInt(input.value, 10) !== newWeight) input.value = newWeight;
                            }}
                            if (w.active !== undefined) cb.checked = w.active;
                            if (w.pinned !== undefined && row.querySelector('.gpu-pin')) {{
                                row.querySelector('.gpu-pin').checked = !!w.pinned;
                                if (w.pinned) input.classList.add('ring-2', 'ring-amber-500/40');
                            }}
                        }}
                    }});
                    updateTotal();
                    if (data.running && !currentSelectedModel) {{
                        if (data.config.context_size) document.getElementById('context-size').value = data.config.context_size;
                        if (data.config.parallel_slots) document.getElementById('parallel-slots').value = data.config.parallel_slots;
                        if (data.config.mmproj_path !== undefined) document.getElementById('mmproj-path').value = data.config.mmproj_path || "";
                    }}
                }}

                if (data.recovery && data.recovery.failed) {{
                    badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-red-500/50 text-red-500 uppercase';
                    badge.innerHTML = `<i class="fas fa-exclamation-triangle mr-1"></i> FALHA: ${{data.recovery.message.toUpperCase()}}`;
                    return;
                }}

                if (data.recovery && data.recovery.active) {{
                    badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
                    if (data.recovery.auto_balance) {{
                        const msg = data.recovery.message || 'calibrando GPUs...';
                        badge.innerHTML = `<i class="fas fa-sync animate-spin mr-1"></i> AUTO BALANCE: ${{msg.toUpperCase()}}`;
                    }} else {{
                        badge.innerHTML = '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
                    }}
                    return;
                }}

                if (data.running) {{
                    badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
                    badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE';
                    card.classList.remove('hidden');
                    document.getElementById('active-model-name').innerText = data.model;
                    if (!logStream) startLogs();
                    updateUptime(data.start_time);
                    currentRunningModelPath = data.model_path;
                    if (!currentSelectedModel && currentRunningModelPath) {{
                        currentSelectedModel = currentRunningModelPath.replace(/\\\\\\\\/g, '/');
                    }}
                    const chatLink = document.getElementById('chat-link');
                    if (chatLink) {{
                        chatLink.classList.remove('pointer-events-none', 'opacity-40');
                        chatLink.setAttribute('aria-disabled', 'false');
                    }}
                }} else {{
                    startTime = null;
                    badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase';
                    badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE';
                    card.classList.add('hidden');
                    if (logStream) {{ logStream.abort(); logStream = null; }}
                    currentRunningModelPath = null;
                    const chatLinkOff = document.getElementById('chat-link');
                    if (chatLinkOff) {{
                        chatLinkOff.classList.add('pointer-events-none', 'opacity-40');
                        chatLinkOff.setAttribute('aria-disabled', 'true');
                    }}
                }}

                document.querySelectorAll('.model-item-container').forEach(el => {{
                    const m_js = el.dataset.path;
                    const actionBtnContainer = el.querySelector('.action-btn-container');
                    const renameBtn = el.querySelector('.rename-btn');
                    const deleteBtn = el.querySelector('.delete-btn');
                    const normalizedM = m_js.replace(/\\\\\\\\/g, '/');
                    const normalizedR = currentRunningModelPath ? currentRunningModelPath.replace(/\\\\\\\\/g, '/') : null;
                    const isRunning = normalizedR && normalizedM === normalizedR;

                    if (isRunning) {{
                        el.classList.add('running-now');
                        if (renameBtn) renameBtn.classList.add('hidden');
                        if (deleteBtn) deleteBtn.classList.add('hidden');
                    }} else {{
                        el.classList.remove('running-now');
                        if (renameBtn) renameBtn.classList.remove('hidden');
                        if (deleteBtn) deleteBtn.classList.remove('hidden');
                    }}
                    if (currentSelectedModel === m_js) el.classList.add('active-selection');
                    else el.classList.remove('active-selection');

                    const newButtonsHtml = getModelButtonsHtml(m_js, el.id, isRunning);
                    if (actionBtnContainer.innerHTML.trim() !== newButtonsHtml.trim()) {{
                        actionBtnContainer.innerHTML = newButtonsHtml;
                    }}
                }});
            }} catch (e) {{ console.error("updateStatus error:", e); }}
        }}

        async function setDefaultModel(checkbox, path) {{
            if (checkbox.checked) document.querySelectorAll('.model-default-checkbox').forEach(cb => {{
                if (cb !== checkbox) cb.checked = false;
            }});
            try {{
                await fetch('/set_default', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{path: checkbox.checked ? path : null}}),
                }});
            }} catch (e) {{
                alert("Erro ao salvar configuracao.");
            }}
        }}

        async function downloadModel() {{
            const url = document.getElementById('download-url').value.trim();
            if (!url) return;
            try {{
                const res = await fetch('/downloads', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{url}}),
                }});
                if (res.ok) document.getElementById('download-url').value = '';
                updateDownloads();
            }} catch (e) {{}}
        }}

        async function updateDownloads() {{
            try {{
                const res = await fetch('/downloads');
                const data = await res.json();
                const container = document.getElementById('download-status');
                const entries = Object.entries(data);
                if (entries.length === 0) {{ container.innerHTML = ''; return; }}
                container.innerHTML = entries.map(([id, d]) => `
                    <div class="p-4 md:p-5 bg-slate-900 border border-slate-800 rounded-2xl">
                        <div class="flex justify-between items-center mb-3 md:mb-4">
                            <p class="text-xs md:text-sm font-bold truncate flex-1 mr-3 md:mr-4 text-slate-300 font-mono" title="${{d.filename}}">${{d.filename}}</p>
                            <span class="text-[8px] md:text-[10px] font-black uppercase px-2 md:px-3 py-0.5 md:py-1 rounded ${{d.status === 'completed' ? 'bg-emerald-500/10 text-emerald-500' : d.status === 'failed' ? 'bg-red-500/10 text-red-500' : 'bg-blue-500/10 text-blue-500'}}">
                                ${{d.status === 'completed' ? 'concluído' : d.status === 'failed' ? 'falhou' : 'baixando'}}
                            </span>
                        </div>
                        <div class="w-full h-1.5 md:h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500 shadow-[0_0_10px_rgba(37,99,235,0.5)] transition-all duration-500" style="width: ${{d.progress}}%"></div>
                        </div>
                    </div>
                `).join('');
                if (entries.some(([_, d]) => d.status === 'completed')) updateModels();
            }} catch (e) {{}}
        }}

        async function updateModels() {{
            try {{
                const [res, cfgRes] = await Promise.all([fetch('/models'), fetch('/config')]);
                const data = await res.json();
                const cfg = await cfgRes.json();
                document.getElementById('model-count').innerText = `${{data.models.length}} UNIDADES`;
                const oldContainer = document.getElementById('model-list-container');
                const newHtml = data.models.map(m => {{
                    const m_js = m.path.replace(/\\\\\\\\/g, '/');
                    if (m.last_config) window.modelConfigs[m.path] = m.last_config;
                    const hasConfigClass = m.last_config ? 'text-blue-400' : 'text-slate-100';
                    const historyIcon = m.last_config ? '<i class="fas fa-history text-[8px] text-blue-500/50" title="Configuração salva disponível"></i>' : '';
                    const isRunning = currentRunningModelPath && m_js === currentRunningModelPath.replace(/\\\\\\\\/g, '/');
                    const isActive = currentSelectedModel === m_js ? 'active-selection' : '';
                    const runningClass = isRunning ? 'running-now' : '';
                    const hashId = m.id;
                    const buttonsHtml = getModelButtonsHtml(m_js, hashId, isRunning);
                    return `<div id="${{hashId}}" class="model-item-container group flex items-center justify-between p-4 md:p-5 mb-3 md:mb-4 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg ${{isActive}} ${{runningClass}}" data-path="${{m_js}}">
                        <div class="flex-1 min-w-0 mr-4 md:mr-6 cursor-pointer" onclick="selectModel('${{m_js}}', '${{hashId}}')">
                            <div class="flex items-center gap-2 md:gap-3 mb-1 md:mb-2">
                                <i class="fas fa-cube text-blue-400 text-[10px] md:text-xs"></i>
                                <p class="model-name text-sm md:text-base font-bold ${{hasConfigClass}} break-all line-clamp-2" title="${{m.name}}">${{m.name}}</p>
                                ${{historyIcon}}
                            </div>
                            <p class="text-[9px] md:text-xs text-slate-500 truncate uppercase tracking-tighter font-mono">${{m.dir}}</p>
                        </div>
                        <div class="flex items-center gap-3 md:gap-6">
                            <div class="flex items-center gap-1">
                                <button onclick="renameModel('${{m_js}}')" class="rename-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all ${{isRunning ? 'hidden' : ''}}" title="Renomear Modelo">
                                    <i class="fas fa-edit text-[10px] md:text-xs"></i>
                                </button>
                                <button onclick="deleteModel('${{m_js}}')" class="delete-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all ${{isRunning ? 'hidden' : ''}}" title="Excluir Modelo">
                                    <i class="fas fa-trash-alt text-[10px] md:text-xs"></i>
                                </button>
                            </div>
                            <div class="flex flex-col items-center gap-1 md:gap-1.5">
                                <span class="text-[8px] md:text-[10px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span>
                                <input type="checkbox" class="model-default-checkbox w-4 h-4 md:w-5 md:h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" ${{m.path === cfg.default_model ? 'checked' : ''}} onclick="setDefaultModel(this, '${{m_js}}')">
                            </div>
                            <div class="action-btn-container">${{buttonsHtml}}</div>
                        </div>
                    </div>`;
                }}).join('');
                if (oldContainer.innerHTML !== newHtml) oldContainer.innerHTML = newHtml;

                const projSelect = document.getElementById('mmproj-path');
                const currentVal = projSelect.value;
                let projHtml = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>';
                data.projectors.forEach(p => {{
                    projHtml += `<option value="${{p.path}}" class="bg-slate-900">${{p.name}}</option>`;
                }});
                if (projSelect.innerHTML.trim() !== projHtml.trim()) {{
                    projSelect.innerHTML = projHtml;
                    projSelect.value = currentVal;
                    if (projSelect.value !== currentVal) projSelect.value = "";
                }}
            }} catch (e) {{}}
        }}

        async function renameModel(path) {{
            const currentName = path.split('/').pop().replace('.gguf', '');
            const newName = prompt("Digite o novo nome para o modelo:", currentName);
            if (!newName || newName === currentName) return;
            try {{
                const res = await fetch('/rename', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{path, new_name: newName}}),
                }});
                if (res.ok) updateModels();
                else {{ const err = await res.json(); alert("Erro ao renomear: " + (err.detail || "Erro desconhecido")); }}
            }} catch (e) {{
                alert("Erro de rede ao renomear modelo.");
            }}
        }}

        async function deleteModel(path) {{
            if (!confirm("TEM CERTEZA QUE DESEJA EXCLUIR ESTE MODELO DO DISCO?\\nEsta ação é irreversível.")) return;
            try {{
                const res = await fetch('/delete', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{path}}),
                }});
                if (res.ok) updateModels();
                else {{ const err = await res.json(); alert("Erro ao excluir: " + (err.detail || "Erro desconhecido")); }}
            }} catch (e) {{
                alert("Erro de rede ao excluir modelo.");
            }}
        }}

        async function startModel(path, elementId) {{
            if (currentSelectedModel !== path) {{
                selectModel(path, elementId);
                await new Promise(r => setTimeout(r, 100));
            }}
            const weights = [];
            document.getElementById('log-box').innerHTML = '';
            document.querySelectorAll('.gpu-row').forEach(r => {{
                const isChecked = r.querySelector('.gpu-checkbox').checked;
                const isMain = r.querySelector('.gpu-main-radio').checked;
                weights.push({{
                    index: parseInt(r.dataset.index, 10),
                    weight: parseInt(r.querySelector('.gpu-weight').value || 0, 10),
                    name: "GPU",
                    active: isChecked,
                    is_main: isMain,
                    pinned: r.querySelector('.gpu-pin')?.checked || false,
                }});
            }});
            if (!weights.some(w => w.active)) return alert("SELECIONE PELO MENOS UMA GPU");
            if (!weights.some(w => w.is_main)) return alert("DEFINA A GPU PRINCIPAL (coluna Principal)");
            const mmprojPath = document.getElementById('mmproj-path').value;
            const splitMode = document.getElementById('split-mode').value;
            const parallelSlots = Math.max(1, Math.min(64, parseInt(document.getElementById('parallel-slots').value) || {DEFAULT_PARALLEL_SLOTS}));
            const autoBalance = document.getElementById('auto-balance-toggle').checked;
            document.getElementById('parallel-slots').value = parallelSlots;
            document.getElementById('status-badge').innerHTML = autoBalance && !manualGpuOverride
                ? '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> AUTO BALANCE...'
                : '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> INICIALIZANDO...';
            try {{
                const res = await fetch('/start', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        path,
                        mmproj_path: mmprojPath || null,
                        gpu_weights: weights,
                        context_size: parseInt(document.getElementById('context-size').value),
                        parallel_slots: parallelSlots,
                        split_mode: splitMode,
                        auto_balance: autoBalance,
                        manual_gpu_override: manualGpuOverride,
                    }}),
                }});
                if (!res.ok) {{
                    const err = await res.json();
                    alert("Erro ao iniciar: " + (err.detail || "Erro desconhecido"));
                    return;
                }}
                const startData = await res.json();
                if (startData.probing) {{
                    manualGpuOverride = false;
                }} else if (!manualGpuOverride && autoBalance) {{
                    manualGpuOverride = false;
                    if (window.modelConfigs[path]) {{
                        window.modelConfigs[path].auto_balance = true;
                        window.modelConfigs[path].auto_balance_profile = true;
                    }}
                    updateAutoBalanceProfileBadge(true);
                }} else if (manualGpuOverride) {{
                    manualGpuOverride = false;
                    if (window.modelConfigs[path]) {{
                        window.modelConfigs[path].auto_balance_profile = false;
                    }}
                    updateAutoBalanceProfileBadge(false);
                }}
            }} catch (e) {{
                alert("Erro ao iniciar modelo.");
            }}
            setTimeout(updateStatus, 2000);
        }}

        async function stopModel() {{
            if (confirm("ENCERRAR PROCESSO?")) {{
                await fetch('/stop', {{method: 'POST'}});
                setTimeout(updateStatus, 1000);
            }}
        }}

        setInterval(updateMetrics, 2000);
        setInterval(updateStatus, 3000);
        setInterval(updateDownloads, 3000);
        setInterval(updateModels, 5000);
    </script>
    <script src="/static/js/pacman_bg.js"></script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
# Startup event — auto-start default model + OOM watchdog
# ─────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Start OOM watchdog, download runner, and optionally auto-start default model."""
    oom_watchdog.start()
    threading.Thread(target=_run_downloads, daemon=True).start()

    # Auto-start default model
    default_model = config_manager.get_default_model()
    if default_model and os.path.exists(default_model):
        if not process_manager.get_status().get("running"):
            logger.info(f"Auto-start: {default_model}")
            try:
                saved_cfg = config_manager.get_model_settings(default_model)
                if saved_cfg.get("gpu_weights"):
                    weights = [
                        GPUWeight(**w) if isinstance(w, dict) else w
                        for w in saved_cfg["gpu_weights"]
                    ]
                    for w in weights:
                        if not hasattr(w, "active"):
                            w.active = True
                    context_size = saved_cfg.get("context_size", DEFAULT_CONTEXT_SIZE)
                    parallel_slots = saved_cfg.get("parallel_slots", DEFAULT_PARALLEL_SLOTS)
                    mmproj_path = saved_cfg.get("mmproj_path")
                    split_mode = saved_cfg.get("split_mode", "layer")
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
                    mmproj_path = None
                    split_mode = "layer"

                process_manager.start(
                    model_path=default_model,
                    gpu_weights=weights,
                    context_size=context_size,
                    mmproj_path=mmproj_path,
                    split_mode=split_mode,
                    parallel_slots=parallel_slots,
                )
            except Exception as e:
                logger.error(f"Auto-start error: {e}")


# ─────────────────────────────────────────────────────────
# Background download task runner
# ─────────────────────────────────────────────────────────


def _run_downloads():
    """Periodically process background downloads."""
    while True:
        with download_mgr._lock:
            to_process = list(download_mgr._downloads_queue)
            download_mgr._downloads_queue.clear()
        for download_id, url, filename, path in to_process:
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
    uvicorn.run(app, host="0.0.0.0", port=MANAGER_PORT)
