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

from llama_manager import app, auth_manager, gpu_manager, require_auth

MOCK_PORT = 8001

_FAKE_GPUS = [{"index": 0, "name": "Mock GPU 0", "vram": 24564}]

_FAKE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "m1",
        "name": "llama-3.1-8b.gguf",
        "path": "/models/llama/llama-3.1-8b.gguf",
        "dir": "llama",
    },
    {
        "id": "m2",
        "name": "mistral-7b.gguf",
        "path": "/models/text/mistral-7b.gguf",
        "dir": "text",
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
_model_configs: Dict[str, Dict[str, Any]] = {}
_FAKE_PROJECTORS: List[Dict[str, Any]] = []
_mocks_installed = False
_original_routes: Optional[list] = None
_original_detect_gpus: Any = None

_DEFAULT_START_PATH = "/models/llama/llama-3.1-8b.gguf"


def _model_directory(model_path: str) -> str:
    if "/" in model_path:
        return model_path.rsplit("/", 1)[0]
    return "/models"


def _sync_mmproj_candidates() -> None:
    for model in _FAKE_MODELS:
        model_dir = _model_directory(model["path"])
        candidates = sorted(
            projector["path"]
            for projector in _FAKE_PROJECTORS
            if _model_directory(projector["path"]) == model_dir
        )
        model["mmproj_candidates"] = candidates
        model["auto_mmproj"] = candidates[0] if candidates else None
        saved = _model_configs.get(model["path"], {})
        if saved:
            model["last_config"] = copy.deepcopy(saved)


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
    _model_configs.clear()
    _FAKE_PROJECTORS.clear()
    _FAKE_MODELS[:] = [
        {
            "id": "m1",
            "name": "llama-3.1-8b.gguf",
            "path": "/models/llama/llama-3.1-8b.gguf",
            "dir": "llama",
        },
        {
            "id": "m2",
            "name": "mistral-7b.gguf",
            "path": "/models/text/mistral-7b.gguf",
            "dir": "text",
        },
        {
            "id": "m3",
            "name": "qwen2.5-14b.gguf",
            "path": "/models/qwen2.5-14b.gguf",
            "dir": "/models",
        },
    ]
    _FAKE_PROJECTORS.append(
        {
            "path": "/models/llama/llama-3.1-8b-mmproj.gguf",
            "name": "llama-3.1-8b-mmproj.gguf",
            "dir": "llama",
        }
    )
    _model_configs["/models/llama/llama-3.1-8b.gguf"] = {
        "mmproj_path": "/models/llama/llama-3.1-8b-mmproj.gguf",
    }
    _sync_mmproj_candidates()


def _status_payload() -> Dict[str, Any]:
    instances = []
    if _mock_state["running"]:
        path = _mock_state["model_path"] or _DEFAULT_START_PATH
        name = _mock_state["model"] or path.split("/")[-1]
        instances.append({
            "port": 8085,
            "status": "running",
            "model": name,
            "model_path": path,
            "start_time": _mock_state["start_time"] or time.time(),
            "config": _mock_state["config"] or {
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
                        "name": "Mock GPU 0",
                        "device": "gpu"
                    }
                ],
            }
        })
    
    return {
        "instances": instances,
        "recovery": copy.deepcopy(_recovery_state),
    }


def _apply_start(path: str, body: Optional[Dict[str, Any]] = None) -> None:
    name = path.split("/")[-1]
    _mock_state.update(
        {
            "running": True,
            "model": name,
            "model_path": path,
            "start_time": time.time(),
            "config": (body or {}),
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
    response = JSONResponse({"status": "ok"})
    response.set_cookie(key="session_token", value="e2e-mock-session-token", httponly=True)
    return response


@mock_api.get("/status")
async def mock_status():
    instances = []
    if _mock_state["running"]:
        path = _mock_state["model_path"] or _DEFAULT_START_PATH
        name = _mock_state["model"] or path.split("/")[-1]
        instances.append({
            "port": 8085,
            "status": "running",
            "model": name,
            "model_path": path,
            "start_time": _mock_state["start_time"] or time.time(),
            "config": _mock_state["config"] or {
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
                        "name": "Mock GPU 0",
                        "device": "gpu"
                    }
                ],
            }
        })
    return {
        "running": len(instances) > 0,
        "model": instances[0]["model"] if instances else None,
        "config": instances[0].get("config") if instances else None,
        "instances": instances,
    }


@mock_api.post("/start")
async def mock_start(request: Request):
    body = await request.json()
    path = body.get("path", _DEFAULT_START_PATH)
    
    if body.get("auto_balance") and body.get("smart_calibration"):
        _recovery_state.update({
            "active": True,
            "model": path,
            "message": "Otimizando...",
            "auto_balance": True,
            "smart_calibration": True
        })
        # Simulate background finish after 1s
        def finish():
            time.sleep(1)
            _recovery_state.update({
                "active": False,
                "smart_proposal": {"threads": 8, "batch_size": 4096}
            })
        import threading
        threading.Thread(target=finish).start()
        return {"probing": True, "status": "ok"}

    _apply_start(path, body)
    return {"probing": False, "status": "ok", "port": 8085}


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
    return {"message": "Parado"}


@mock_api.get("/metrics")
async def mock_metrics():
    return {
        "cpu": 42.5,
        "ram": 67.2,
        "cpu_temp": "58",
        "cpu_power": "95",
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
    _sync_mmproj_candidates()
    return {
        "models": copy.deepcopy(_FAKE_MODELS),
        "projectors": copy.deepcopy(_FAKE_PROJECTORS),
        "storage": {
            "path": "/models",
            "used_gb": 42.0,
            "total_gb": 500.0,
        },
    }


@mock_api.get("/config")
async def mock_config():
    return {
        "default_model": None,
        "model_configs": copy.deepcopy(_model_configs),
    }


@mock_api.post("/models/mmproj")
async def mock_set_mmproj(request: Request):
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path obrigatorio")
    mmproj_path = body.get("mmproj_path")
    entry = _model_configs.setdefault(model_path, {})
    entry["mmproj_path"] = mmproj_path
    if body.get("user_initiated"):
        entry["mmproj_disabled"] = mmproj_path == "__no_vision__"
    _sync_mmproj_candidates()
    return {
        "status": "ok",
        "mmproj_path": mmproj_path,
        "mmproj_disabled": entry.get("mmproj_disabled", False),
    }


@mock_api.post("/models/proxy")
async def mock_set_model_proxy(request: Request):
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path or body.get("vision_enabled") is None:
        raise HTTPException(status_code=400, detail="model_path e vision_enabled obrigatorios")
    entry = _model_configs.setdefault(model_path, {})
    entry["vision_enabled"] = bool(body["vision_enabled"])
    _sync_mmproj_candidates()
    return {"message": "Configuracao salva"}


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
    url = body.get("url", "")
    model_path = body.get("model_path")
    if model_path:
        filename = body.get("filename") or url.split("/")[-1].split("?")[0] or "e2e-vision.mmproj"
        if not any(
            marker in filename.lower()
            for marker in ("mmproj", "clip", "vision", "projector")
        ):
            filename = f"{filename}.mmproj" if not filename.endswith(".gguf") else filename.replace(
                ".gguf", ".mmproj"
            )
        projector_path = f"{_model_directory(model_path)}/{filename}"
        if not any(p["path"] == projector_path for p in _FAKE_PROJECTORS):
            _FAKE_PROJECTORS.append(
                {
                    "path": projector_path,
                    "name": filename,
                    "dir": _model_directory(model_path),
                }
            )
        _sync_mmproj_candidates()
    else:
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
    app.dependency_overrides[require_auth] = _fake_auth
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
    app.dependency_overrides.pop(require_auth, None)
    _mocks_installed = False


apply_mocks = _install_mocks
remove_mocks = _remove_mocks
reset_mock_state = _reset_mock_data


def main() -> None:
    _install_mocks()
    uvicorn.run(app, host="127.0.0.1", port=MOCK_PORT, log_level="warning")


if __name__ == "__main__":
    main()
