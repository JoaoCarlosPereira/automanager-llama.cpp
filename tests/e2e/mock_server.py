"""Servidor FastAPI mock para testes E2E Playwright."""

from __future__ import annotations

import copy
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

from llama_manager import app, auth_manager, get_current_auth, gpu_manager

MOCK_PORT = 8001

_FAKE_GPUS = [{"index": 0, "name": "Mock GPU 0", "vram": 24564}]

_FAKE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "m1",
        "name": "llama-3.1-8b.gguf",
        "path": "/models/llama-3.1-8b.gguf",
        "dir": "/models",
    },
    {
        "id": "m2",
        "name": "mistral-7b.gguf",
        "path": "/models/mistral-7b.gguf",
        "dir": "/models",
    },
    {
        "id": "m3",
        "name": "qwen2.5-14b.gguf",
        "path": "/models/qwen2.5-14b.gguf",
        "dir": "/models",
    },
]

_mock_state: Dict[str, Any] = {
    "running": False,
    "model": None,
    "model_path": None,
    "start_time": None,
    "config": None,
}

_recovery_state: Dict[str, Any] = {
    "active": False,
    "failed": False,
    "message": "",
    "auto_balance": False,
}

_downloads_state: Dict[str, Dict[str, Any]] = {}
_download_counter = 0
_mocks_installed = False
_original_routes: Optional[list] = None
_original_detect_gpus: Any = None

_DEFAULT_START_PATH = "/models/llama-3.1-8b.gguf"


def _reset_mock_data() -> None:
    global _download_counter
    _mock_state.update(
        {
            "running": False,
            "model": None,
            "model_path": None,
            "start_time": None,
            "config": None,
        }
    )
    _recovery_state.update(
        {
            "active": False,
            "failed": False,
            "message": "",
            "auto_balance": False,
        }
    )
    _downloads_state.clear()
    _download_counter = 0
    _FAKE_MODELS[:] = [
        {
            "id": "m1",
            "name": "llama-3.1-8b.gguf",
            "path": "/models/llama-3.1-8b.gguf",
            "dir": "/models",
        },
        {
            "id": "m2",
            "name": "mistral-7b.gguf",
            "path": "/models/mistral-7b.gguf",
            "dir": "/models",
        },
        {
            "id": "m3",
            "name": "qwen2.5-14b.gguf",
            "path": "/models/qwen2.5-14b.gguf",
            "dir": "/models",
        },
    ]


def _status_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "running": _mock_state["running"],
        "model": _mock_state["model"],
        "config": _mock_state["config"] or {},
        "recovery": copy.deepcopy(_recovery_state),
    }
    if _mock_state["running"]:
        path = _mock_state["model_path"] or _DEFAULT_START_PATH
        name = _mock_state["model"] or path.split("/")[-1]
        payload.update(
            {
                "model": name,
                "model_path": path,
                "start_time": _mock_state["start_time"] or time.time(),
                "config": _mock_state["config"]
                or {
                    "path": path,
                    "context_size": 65536,
                    "parallel_slots": 1,
                    "batch_size": 512,
                    "split_mode": "layer",
                    "gpu_weights": [
                        {
                            "index": 0,
                            "weight": 100,
                            "active": True,
                            "is_main": True,
                            "pinned": False,
                        }
                    ],
                    "mmproj_path": None,
                },
            }
        )
    return payload


def _apply_start(path: str, body: Optional[Dict[str, Any]] = None) -> None:
    name = path.split("/")[-1]
    _mock_state.update(
        {
            "running": True,
            "model": name,
            "model_path": path,
            "start_time": time.time(),
            "config": (body or {}).get("config")
            or {
                "path": path,
                "context_size": (body or {}).get("context_size", 65536),
                "parallel_slots": (body or {}).get("parallel_slots", 1),
                "batch_size": (body or {}).get("batch_size", 512),
                "split_mode": (body or {}).get("split_mode", "layer"),
                "gpu_weights": (body or {}).get("gpu_weights")
                or [
                    {
                        "index": 0,
                        "weight": 100,
                        "active": True,
                        "is_main": True,
                        "pinned": False,
                    }
                ],
                "mmproj_path": (body or {}).get("mmproj_path"),
            },
        }
    )


mock_api = FastAPI()


@mock_api.post("/api/auth/login")
async def mock_login(request: Request):
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    if username == "invalid" or password == "wrong":
        raise HTTPException(status_code=401, detail="Credenciais invalidas")
    session_token = secrets.token_urlsafe(32)
    with auth_manager._lock:
        auth_manager._sessions[session_token] = datetime.utcnow()
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=3600,
    )
    return response


@mock_api.post("/api/auth/logout")
async def mock_logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token:
        auth_manager.logout(session_token)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(key="session_token")
    return response


@mock_api.post("/api/auth/change-password")
async def mock_change_password():
    return {"status": "ok"}


@mock_api.get("/status")
async def mock_status():
    return _status_payload()


@mock_api.post("/stop")
async def mock_stop():
    _mock_state.update(
        {
            "running": False,
            "model": None,
            "model_path": None,
            "start_time": None,
            "config": None,
        }
    )
    return {"status": "ok"}


@mock_api.post("/start")
async def mock_start(request: Request):
    body = await request.json()
    path = body.get("path", _DEFAULT_START_PATH)
    _apply_start(path, body)
    return {"probing": False, "status": "ok"}


@mock_api.get("/metrics")
async def mock_metrics():
    return {
        "cpu": 42.5,
        "ram": 67.2,
        "gpus": [
            {
                "index": 0,
                "util": 85,
                "temp": 72,
                "power": 240,
                "mem_used": 12000,
                "mem_total": 24564,
                "vram_pct": 48,
            }
        ],
    }


@mock_api.get("/models")
async def mock_models():
    return {"models": copy.deepcopy(_FAKE_MODELS), "projectors": []}


@mock_api.get("/config")
async def mock_config():
    return {"default_model": None, "model_configs": {}}


@mock_api.get("/logs")
async def mock_logs():
    async def event_stream():
        for line in ["[INFO] llama server started", "[INFO] model loaded"]:
            yield f"data: {line}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@mock_api.get("/downloads")
async def mock_get_downloads():
    return copy.deepcopy(_downloads_state)


@mock_api.post("/downloads")
async def mock_post_downloads(request: Request):
    global _download_counter
    body = await request.json()
    _download_counter += 1
    download_id = f"dl-{_download_counter}"
    filename = body.get("filename") or "e2e-model.gguf"
    _downloads_state[download_id] = {
        "filename": filename,
        "status": "downloading",
        "progress": 45,
    }
    return {"download_id": download_id}


@mock_api.post("/rename")
async def mock_rename(request: Request):
    body = await request.json()
    path = body.get("path")
    new_name = body.get("new_name", "").strip()
    if not path or not new_name:
        raise HTTPException(status_code=400, detail="path e new_name obrigatorios")
    if not new_name.endswith(".gguf"):
        new_name = f"{new_name}.gguf"
    new_path = None
    for model in _FAKE_MODELS:
        if model["path"] == path:
            model["name"] = new_name
            model["path"] = f"{model['dir']}/{new_name}"
            new_path = model["path"]
            break
    if not new_path:
        raise HTTPException(status_code=404, detail="Modelo nao encontrado")
    return {"ok": True, "status": "renamed", "new_path": new_path}


@mock_api.post("/delete")
async def mock_delete(request: Request):
    body = await request.json()
    path = body.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="path obrigatorio")
    before = len(_FAKE_MODELS)
    _FAKE_MODELS[:] = [m for m in _FAKE_MODELS if m["path"] != path]
    if len(_FAKE_MODELS) == before:
        raise HTTPException(status_code=404, detail="Modelo nao encontrado")
    if _mock_state.get("model_path") == path:
        _mock_state.update(
            {
                "running": False,
                "model": None,
                "model_path": None,
                "start_time": None,
                "config": None,
            }
        )
    return {"ok": True, "status": "deleted"}


@mock_api.post("/set_default")
async def mock_set_default():
    return {"ok": True}


@mock_api.get("/api/key")
async def mock_get_api_key():
    return {"key": "e2e-test-api-key"}


@mock_api.post("/api/key/renew")
async def mock_renew_api_key():
    return {"key": "e2e-renewed-api-key"}


@mock_api.post("/auto-balance/cancel")
async def mock_cancel_auto_balance():
    _recovery_state.update(
        {"active": False, "failed": False, "message": "", "auto_balance": False}
    )
    return {"status": "ok"}


@mock_api.post("/__e2e/reset")
async def mock_e2e_reset():
    _reset_mock_data()
    return {"ok": True}


def _fake_detect_gpus():
    return copy.deepcopy(_FAKE_GPUS)


def _fake_auth(
    request: Request = None,
    credentials=None,
) -> str:
    return "e2e-mock-session"


def _replace_routes() -> None:
    mock_paths = {
        (getattr(route, "path", None), frozenset(getattr(route, "methods", set()) or set()))
        for route in mock_api.routes
        if isinstance(route, APIRoute)
    }
    kept_routes = []
    for route in app.router.routes:
        if not isinstance(route, APIRoute):
            kept_routes.append(route)
            continue
        key = (route.path, frozenset(route.methods or set()))
        if key in mock_paths:
            continue
        kept_routes.append(route)
    for route in mock_api.routes:
        if isinstance(route, APIRoute):
            kept_routes.append(route)
    app.router.routes = kept_routes


def _install_mocks() -> None:
    global _mocks_installed, _original_routes, _original_detect_gpus
    if _mocks_installed:
        _reset_mock_data()
        return
    _original_routes = list(app.router.routes)
    _original_detect_gpus = gpu_manager.detect_gpus
    gpu_manager.detect_gpus = _fake_detect_gpus  # type: ignore[method-assign]
    app.dependency_overrides[get_current_auth] = _fake_auth
    _replace_routes()
    _reset_mock_data()
    _mocks_installed = True


def _remove_mocks() -> None:
    """Restaura rotas e dependências reais após testes que usam apply_mocks."""
    global _mocks_installed, _original_routes, _original_detect_gpus
    if not _mocks_installed:
        return
    if _original_routes is not None:
        app.router.routes = list(_original_routes)
    if _original_detect_gpus is not None:
        gpu_manager.detect_gpus = _original_detect_gpus  # type: ignore[method-assign]
    app.dependency_overrides.pop(get_current_auth, None)
    _mocks_installed = False


apply_mocks = _install_mocks
remove_mocks = _remove_mocks
reset_mock_state = _reset_mock_data


def main() -> None:
    _install_mocks()
    uvicorn.run(app, host="127.0.0.1", port=MOCK_PORT, log_level="warning")


if __name__ == "__main__":
    main()
