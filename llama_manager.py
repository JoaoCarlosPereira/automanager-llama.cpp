#!/usr/bin/env python3
"""
Automanager Llama.cpp - Control Plane para llama-server
FastAPI application that orchestrates llama-server instances with multi-GPU
tensor split management, OOM auto-recovery, and real-time hardware monitoring.
"""

import os
import re
import sys
import json
import uuid
import time
import glob
import secrets
import logging
import socket
import signal
import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────

MODELS_DIR = "/media/docker/models"
SERVER_LOG_PATH = "/root/llama_server.log"
MANAGER_LOG_PATH = "/root/manager.log"
CONFIG_PATH = "/root/automanager_config.json"
MODEL_SETTINGS_PATH = "/root/model_settings.json"
LLAMA_SERVER_BIN = "llama-server"
SERVER_PORT = 8085
MANAGER_PORT = 8000
DEFAULT_CONTEXT_SIZE = 65536

# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────

try:
    os.makedirs(os.path.dirname(MANAGER_LOG_PATH), exist_ok=True)
    _manager_log_path = MANAGER_LOG_PATH
except OSError:
    _manager_log_path = os.path.join(os.getcwd(), "manager.log")

logging.basicConfig(
    filename=_manager_log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("automanager")

# ─────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────


class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True


class StartRequest(BaseModel):
    path: str
    mmproj_path: Optional[str] = None
    gpu_weights: List[GPUWeight]
    context_size: int = DEFAULT_CONTEXT_SIZE


class DeleteRequest(BaseModel):
    path: str


class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None


class SetDefaultRequest(BaseModel):
    path: Optional[str] = None


class RenameRequest(BaseModel):
    path: str
    new_name: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ─────────────────────────────────────────────────────────
# ConfigManager (Step 1)
# Atomic JSON file persistence — no external database
# ─────────────────────────────────────────────────────────


class ConfigManager:
    """Thread-safe JSON config manager with atomic writes."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self._lock = threading.Lock()

    def load(self) -> dict:
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
            return {}

    def save(self, data: dict) -> None:
        with self._lock:
            tmp_path = self.config_path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.config_path)
            except Exception as e:
                logger.error(f"Config save error: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def get_model_settings(self, model_path: str) -> dict:
        config = self.load()
        return config.get("model_configs", {}).get(model_path, {})

    def update_model_settings(self, model_path: str, settings: dict) -> None:
        config = self.load()
        if "model_configs" not in config:
            config["model_configs"] = {}
        config["model_configs"][model_path] = {
            "context_size": settings.get("context_size", DEFAULT_CONTEXT_SIZE),
            "mmproj_path": settings.get("mmproj_path"),
            "gpu_weights": settings.get("gpu_weights"),
            "last_started": datetime.utcnow().isoformat(),
        }
        self.save(config)

    def set_default_model(self, path: Optional[str]) -> None:
        config = self.load()
        config["default_model"] = path
        self.save(config)

    def get_default_model(self) -> Optional[str]:
        return self.load().get("default_model")


# ─────────────────────────────────────────────────────────
# TokenManager (Step 2)
# Generates and validates OpenAI-style API keys
# ─────────────────────────────────────────────────────────


class TokenManager:
    """Manages global API token in sk-... format."""

    PREFIX = "sk-"
    TOKEN_LENGTH = 32

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def generate(self) -> str:
        return f"{self.PREFIX}{secrets.token_hex(24)}"

    def validate(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        return key.startswith(self.PREFIX) and len(key) >= len(self.PREFIX) + 32

    def get_or_create(self) -> str:
        config = self.config.load()
        if "api_token" not in config or not self.validate(config["api_token"]):
            config["api_token"] = self.generate()
            self.config.save(config)
        return config["api_token"]

    def renew(self) -> str:
        config = self.config.load()
        config["api_token"] = self.generate()
        self.config.save(config)
        return config["api_token"]


# ─────────────────────────────────────────────────────────
# AuthManager (Step 3)
# Form-based login sessions + API key middleware
# ─────────────────────────────────────────────────────────


class AuthManager:
    """Handles UI login (form-based sessions) and API key auth."""

    def __init__(self, config_manager: ConfigManager, token_manager: TokenManager):
        self.config = config_manager
        self.token_mgr = token_manager
        self._sessions: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._init_admin_password()

    def _init_admin_password(self) -> None:
        """Initialize admin password hash if not present."""
        config = self.config.load()
        if "admin_password_hash" not in config:
            # Default password: "admin" — force user to change on first login
            config["admin_password_hash"] = self._hash_password("admin")
            self.config.save(config)

    @staticmethod
    def _hash_password(password: str) -> str:
        """Simple SHA-256 hash — in production, use bcrypt via passlib."""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Returns session token on success, None on failure."""
        config = self.config.load()
        expected_hash = config.get("admin_password_hash", "")
        actual_hash = hashlib.sha256(password.encode()).hexdigest()
        if actual_hash != expected_hash:
            logger.warning(f"Failed login attempt for user: {username}")
            return None
        session_token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[session_token] = datetime.utcnow()
        return session_token

    def verify_session(self, session_token: str) -> bool:
        with self._lock:
            if session_token in self._sessions:
                # Extend session
                self._sessions[session_token] = datetime.utcnow()
                return True
            return False

    def logout(self, session_token: str) -> None:
        with self._lock:
            self._sessions.pop(session_token, None)

    def verify_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
        return self.token_mgr.validate(credentials.credentials)

    def change_password(self, old_password: str, new_password: str) -> bool:
        config = self.config.load()
        current_hash = hashlib.sha256(old_password.encode()).hexdigest()
        if config.get("admin_password_hash") != current_hash:
            return False
        config["admin_password_hash"] = self._hash_password(new_password)
        self.config.save(config)
        return True


# ─────────────────────────────────────────────────────────
# GPUDetector (Step 4)
# Parses nvidia-smi for GPU metrics
# ─────────────────────────────────────────────────────────


class GPUDetector:
    """Detects GPUs and parses metrics from nvidia-smi."""

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs using llama-server --help first, fallback to nvidia-smi."""
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            output = subprocess.check_output(
                f"{LLAMA_SERVER_BIN} --help 2>&1",
                shell=True, env=env, timeout=10,
            ).decode()
            pattern = r"Device (\d+): (.*?), compute capability.*?, VRAM: (\d+) MiB"
            matches = re.findall(pattern, output)
            gpus = []
            for match in matches:
                idx, name, vram = match
                gpus.append({
                    "index": int(idx),
                    "name": name.strip(),
                    "vram": int(vram),
                })
            if gpus:
                return gpus
        except Exception:
            pass

        # Fallback to nvidia-smi
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=10,
            ).decode()
            gpus = []
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram": int(parts[2]),
                    })
            return gpus
        except Exception as e:
            logger.error(f"GPU detection error: {e}")
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get real-time hardware metrics."""
        try:
            output = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=10,
            ).decode()
            gpus = []
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    mem_used = float(parts[2])
                    mem_total = float(parts[3])
                    gpus.append({
                        "index": int(parts[0]),
                        "util": parts[1],
                        "mem_used": parts[2],
                        "mem_total": parts[3],
                        "vram_pct": round(
                            (mem_used / mem_total) * 100, 1
                        ) if mem_total > 0 else 0,
                        "temp": parts[4],
                        "power": parts[5].split(".")[0] if "." in parts[5] else parts[5],
                    })
            return {
                "cpu": psutil.cpu_percent(interval=0.1),
                "ram": psutil.virtual_memory().percent,
                "gpus": gpus,
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {"cpu": 0, "ram": 0, "gpus": []}


# ─────────────────────────────────────────────────────────
# ProcessManager (Step 5)
# Manages llama-server lifecycle via subprocess
# ─────────────────────────────────────────────────────────


class ProcessManager:
    """Manages llama-server process lifecycle."""

    def __init__(
        self, config_manager: ConfigManager, token_manager: TokenManager
    ):
        self.config = config_manager
        self.token_mgr = token_manager
        self._current_process: Optional[subprocess.Popen] = None
        self._last_request: Optional[StartRequest] = None
        self._lock = threading.Lock()
        self._recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
        }

    @property
    def recovery_state(self) -> dict:
        with self._lock:
            return dict(self._recovery_state)

    @recovery_state.setter
    def recovery_state(self, state: dict) -> None:
        with self._lock:
            self._recovery_state = state

    def get_status(self) -> dict:
        """Check if llama-server is running."""
        status = {"running": False, "recovery": self.recovery_state}
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                name = proc.info["name"] or ""
                cmdline = proc.info["cmdline"] or []
                if "llama-server" in name or (
                    cmdline and "llama-server" in cmdline[0]
                ):
                    model_name = None
                    model_path = None
                    for i in range(len(cmdline) - 1):
                        if cmdline[i] in ["-m", "--model"]:
                            model_name = os.path.basename(cmdline[i + 1])
                            model_path = cmdline[i + 1]
                            break
                    if model_name:
                        status.update(
                            {
                                "running": True,
                                "pid": proc.info["pid"],
                                "model": model_name,
                                "model_path": model_path,
                                "start_time": proc.info["create_time"],
                            }
                        )
                        # Attach last known config
                        with self._lock:
                            if self._last_request:
                                status["config"] = {
                                    "path": self._last_request.path,
                                    "context_size": self._last_request.context_size,
                                    "gpu_weights": [
                                        w.model_dump()
                                        if hasattr(w, "model_dump")
                                        else w
                                        for w in self._last_request.gpu_weights
                                    ],
                                    "mmproj_path": self._last_request.mmproj_path,
                                }
                        return status
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return status

    def stop(self) -> dict:
        """Stop running llama-server."""
        subprocess.run(["pkill", "-9", LLAMA_SERVER_BIN], check=False)
        with self._lock:
            if self._current_process:
                try:
                    os.killpg(
                        os.getpgid(self._current_process.pid), signal.SIGKILL
                    )
                except (ProcessLookupError, OSError):
                    pass
                self._current_process = None
            self._last_request = None
        self.recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
        }
        # Clear server log
        try:
            open(SERVER_LOG_PATH, "w").close()
        except OSError:
            pass
        logger.info("llama-server stopped")
        return {"message": "Stopped"}

    def start(
        self,
        model_path: str,
        gpu_weights: List[GPUWeight],
        context_size: int,
        mmproj_path: Optional[str] = None,
    ) -> dict:
        """Start llama-server with given configuration."""
        # Stop any existing process first
        self.stop()

        with self._lock:
            # Build tensor split from weights
            active_weights = [w for w in gpu_weights if w.active]
            if not active_weights:
                raise HTTPException(
                    status_code=400, detail="SELECIONE PELO MENOS UMA GPU"
                )

            weights_map = {w.index: w.weight for w in active_weights}
            total = sum(weights_map.values()) or 1

            # Build split array for all GPUs
            all_gpus = GPUDetector().detect_gpus()
            max_idx = max((g["index"] for g in all_gpus), default=0)
            split = [
                f"{weights_map.get(i, 0) / total:.4f}"
                for i in range(max_idx + 1)
            ]

            # Main GPU = highest weight among active
            main_gpu = str(max(weights_map, key=weights_map.get))

            # Build command
            api_token = self.token_mgr.get_or_create()
            cmd = [
                LLAMA_SERVER_BIN,
                "-m", model_path,
                "-ngl", "99",
                "--flash-attn", "on",
                "--host", "0.0.0.0",
                "--port", str(SERVER_PORT),
                "--tools", "all",
                "--parallel", "1",
                "--ctx-size", str(context_size),
                "--mlock",
                "--main-gpu", main_gpu,
                "--tensor-split", ",".join(split),
                "--api-key", api_token,
            ]

            if mmproj_path and os.path.exists(mmproj_path):
                cmd.extend(["--mmproj", mmproj_path])
            else:
                cmd.append("--mmproj-auto")

            # Setup environment
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get(
                "LD_LIBRARY_PATH", ""
            )

            # Clear log file
            try:
                with open(SERVER_LOG_PATH, "w") as f:
                    f.write("")
            except OSError as e:
                logger.error(f"Failed to clear log: {e}")

            logger.info(f"START: {' '.join(cmd)}")

            try:
                self._current_process = subprocess.Popen(
                    cmd,
                    stdout=open(SERVER_LOG_PATH, "a"),
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    env=env,
                )
                self._last_request = StartRequest(
                    path=model_path,
                    mmproj_path=mmproj_path,
                    gpu_weights=gpu_weights,
                    context_size=context_size,
                )
                return {
                    "message": "Started",
                    "pid": self._current_process.pid,
                }
            except Exception as e:
                logger.error(f"Start error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Erro ao iniciar: {e}"
                )


# ─────────────────────────────────────────────────────────
# OOMWatchdog (Step 6)
# Background thread that detects OOM and auto-recovers
# ─────────────────────────────────────────────────────────


class OOMWatchdog(threading.Thread):
    """
    Background watchdog that monitors llama_server.log for OOM errors.
    Uses conservative recovery: reduces overloaded GPU by 10%.
    Falls back to 50/50 split after 3 consecutive OOMs.
    """

    OOM_PATTERNS = re.compile(
        r"(?i)(out of memory|cuda error|malloc failed|c10\.Error)"
    )
    REDUCTION_PCT = 10.0
    MAX_CONSECUTIVE_OOM = 3
    SILENCE_TIMEOUT = 30  # seconds

    def __init__(
        self,
        process_manager: ProcessManager,
        config_manager: ConfigManager,
        gpu_detector: GPUDetector,
    ):
        super().__init__(daemon=True)
        self.process_manager = process_manager
        self.config = config_manager
        self.gpu_detector = gpu_detector
        self._consecutive_oom = 0
        self._last_oom_time = 0.0
        self._stopping = False
        self._lock = threading.Lock()

    def run(self) -> None:
        logger.info("OOMWatchdog started")
        while not self._stopping:
            try:
                if os.path.exists(SERVER_LOG_PATH):
                    self._check_log()
                time.sleep(5)
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(5)

    def _check_log(self) -> None:
        try:
            with open(SERVER_LOG_PATH, "r") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if self.OOM_PATTERNS.search(line):
                        self._handle_oom()
                        break
        except OSError:
            pass

    def _handle_oom(self) -> None:
        now = time.time()
        with self._lock:
            # Reset if too much time since last OOM
            if now - self._last_oom_time > self.SILENCE_TIMEOUT:
                self._consecutive_oom = 0
            self._consecutive_oom += 1
            self._last_oom_time = now
            consecutive = self._consecutive_oom

        logger.warning(f"OOM detected! Consecutive: {consecutive}")

        pm = self.process_manager
        with pm._lock:
            req = pm._last_request
            if not req:
                return

        if consecutive >= self.MAX_CONSECUTIVE_OOM:
            # Fallback to 50/50 split
            logger.warning(
                "Max OOM reached. Applying 50/50 fallback split."
            )
            pm.recovery_state = {
                "active": True,
                "failed": False,
                "message": "OOM repetido. Divisao 50/50.",
            }
            # Equal weights for all active GPUs
            new_weights = [
                GPUWeight(
                    index=w.index,
                    weight=50.0 if w.active else 0.0,
                    name=w.name,
                    active=w.active,
                )
                for w in req.gpu_weights
            ]
        else:
            # Conservative reduction: 10% from highest weighted GPU
            pm.recovery_state = {
                "active": True,
                "failed": False,
                "message": "OOM detectado. Reduzindo carga...",
            }
            weights = list(req.gpu_weights)
            active = [w for w in weights if w.active]
            if len(active) <= 1:
                pm.recovery_state = {
                    "active": False,
                    "failed": True,
                    "message": "Single GPU ou sem pesos.",
                }
                return
            main_gpu = max(active, key=lambda w: w.weight)
            other_gpus = [w for w in active if w != main_gpu]
            main_gpu.weight -= self.REDUCTION_PCT
            share = self.REDUCTION_PCT / len(other_gpus) if other_gpus else 0
            for og in other_gpus:
                og.weight += share
            new_weights = weights

        # Update model config and restart
        self.config.update_model_settings(
            req.path,
            {
                "context_size": req.context_size,
                "mmproj_path": req.mmproj_path,
                "gpu_weights": [
                    w.model_dump() for w in new_weights
                ],
            },
        )

        logger.info(
            f"Recovery: {[f'{w.index}:{w.weight:.1f}' for w in new_weights]}"
        )
        try:
            pm.start(
                model_path=req.path,
                gpu_weights=new_weights,
                context_size=req.context_size,
                mmproj_path=req.mmproj_path,
            )
        except Exception as e:
            logger.error(f"Recovery start failed: {e}")

        with pm._lock:
            pm._last_request = req

        # Reset recovery state after delay
        time.sleep(3)
        pm.recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
        }

    def stop(self) -> None:
        self._stopping = True


# ─────────────────────────────────────────────────────────
# ModelScanner (Step 7)
# Discovers .gguf and .mmproj files in models directory
# ─────────────────────────────────────────────────────────


class ModelScanner:
    """Scans models directory for .gguf and .mmproj files."""

    def scan(self) -> dict:
        """Scan for models and classify by type."""
        models = []
        projectors = []
        try:
            for root, _dirs, files in os.walk(MODELS_DIR):
                for f in files:
                    full_path = os.path.join(root, f)
                    name_lower = f.lower()
                    item = {
                        "path": full_path,
                        "name": f,
                        "dir": os.path.relpath(root, MODELS_DIR) or "/",
                    }
                    if any(
                        x in name_lower
                        for x in ["mmproj", "clip", "vision", "projector"]
                    ):
                        projectors.append(item)
                    else:
                        models.append(item)
        except OSError as e:
            logger.error(f"Scan error: {e}")

        # Attach saved configs
        config = ConfigManager().load()
        model_configs = config.get("model_configs", {})
        for m in models:
            m["last_config"] = model_configs.get(m["path"])
        for p in projectors:
            p["last_config"] = model_configs.get(p["path"])

        # Auto-detect mmproj candidates for each model
        for m in models:
            base_name = os.path.splitext(m["name"])[0]
            candidates = []
            for proj in projectors:
                proj_base = os.path.splitext(proj["name"])[0]
                if proj_base == base_name or base_name in proj_base:
                    candidates.append(proj["path"])
            m["mmproj_candidates"] = candidates
            m["auto_mmproj"] = candidates[0] if candidates else None

        return {"models": models, "projectors": projectors}

    def rename_model(self, old_path: str, new_name: str) -> str:
        """Rename a model file."""
        if not old_path.startswith(MODELS_DIR):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(old_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        # Check if currently running
        pm = process_manager
        status = pm.get_status()
        if status["running"]:
            normalized_old = old_path.replace("\\", "/")
            normalized_run = (
                status.get("model_path", "").replace("\\", "/")
            )
            if normalized_old == normalized_run:
                raise HTTPException(
                    status_code=400,
                    detail="Impossivel renomear modelo em execucao",
                )

        # Build new path
        dir_name = os.path.dirname(old_path)
        if not new_name.endswith(".gguf"):
            new_name += ".gguf"
        new_path = os.path.join(dir_name, new_name)

        if os.path.exists(new_path):
            raise HTTPException(
                status_code=400, detail="Ja existe um arquivo com este nome"
            )

        os.rename(old_path, new_path)

        # Update config references
        config = ConfigManager()
        data = config.load()
        updated = False

        if data.get("default_model") == old_path:
            data["default_model"] = new_path
            updated = True

        if "model_configs" in data and old_path in data["model_configs"]:
            data["model_configs"][new_path] = data["model_configs"].pop(
                old_path
            )
            updated = True

        if updated:
            config.save(data)

        logger.info(f"Renamed: {old_path} -> {new_path}")
        return new_path

    def delete_model(self, file_path: str) -> None:
        """Delete a model file."""
        if not file_path.startswith(MODELS_DIR):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        # Check if running — stop if needed
        pm = process_manager
        status = pm.get_status()
        if status["running"]:
            normalized = file_path.replace("\\", "/")
            normalized_run = status.get("model_path", "").replace("\\", "/")
            if normalized == normalized_run:
                pm.stop()

        os.remove(file_path)
        logger.info(f"Deleted: {file_path}")


# ─────────────────────────────────────────────────────────
# DownloadManager (Step 8)
# Background HTTP downloads with progress tracking
# ─────────────────────────────────────────────────────────


class DownloadManager:
    """Manages model downloads with progress tracking."""

    def __init__(self):
        self._downloads: Dict[str, dict] = {}
        self._downloads_queue: List[tuple] = []
        self._lock = threading.Lock()

    def start_download(self, url: str, filename: Optional[str] = None) -> str:
        """Start a background download. Returns download_id."""
        download_id = str(uuid.uuid4())
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".gguf"):
                filename += ".gguf"

        model_name_folder = filename.replace(".gguf", "")
        model_specific_dir = os.path.join(MODELS_DIR, model_name_folder)
        os.makedirs(model_specific_dir, exist_ok=True)
        path = os.path.join(model_specific_dir, filename)

        # Handle conflicts
        if os.path.exists(path):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"
            path = os.path.join(model_specific_dir, filename)

        with self._lock:
            self._downloads[download_id] = {
                "filename": filename,
                "path": path,
                "url": url,
                "status": "downloading",
                "progress": 0,
            }

        with self._lock:
            self._downloads_queue.append(
                (download_id, url, filename, path)
            )
        return download_id

    def get_progress(self) -> dict:
        with self._lock:
            return dict(self._downloads)

    def _do_download(
        self, download_id: str, url: str, filename: str, path: str
    ) -> None:
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(
                response.headers.get("content-length", 0)
            )
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            with self._lock:
                                if download_id in self._downloads:
                                    self._downloads[download_id][
                                        "progress"
                                    ] = round(
                                        (downloaded / total_size) * 100, 2
                                    )
            with self._lock:
                if download_id in self._downloads:
                    self._downloads[download_id]["status"] = "completed"
                    self._downloads[download_id]["progress"] = 100
            logger.info(f"Download completed: {filename}")
        except Exception as e:
            logger.error(f"Download error {download_id}: {e}")
            with self._lock:
                if download_id in self._downloads:
                    self._downloads[download_id]["status"] = "failed"
                    self._downloads[download_id]["error"] = str(e)


# ─────────────────────────────────────────────────────────
# SSEStreamer (Step 9)
# Streams llama_server.log to browser via Server-Sent Events
# ─────────────────────────────────────────────────────────


class SSEStreamer:
    """Streams server log via SSE with 500-line cap."""

    @staticmethod
    def stream() -> StreamingResponse:
        def generate():
            if not os.path.exists(SERVER_LOG_PATH):
                yield "data: Arquivo de log nao encontrado.\n\n"
                return
            with open(SERVER_LOG_PATH, "r") as f:
                lines = f.readlines()
                for line in lines[-500:]:
                    yield f"data: {line}"
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    yield f"data: {line}"

        return StreamingResponse(
            generate(), media_type="text/event-stream"
        )


# ─────────────────────────────────────────────────────────
# Application initialization
# ─────────────────────────────────────────────────────────

app = FastAPI(title="Automanager Llama.cpp")

# Service singletons
config_manager = ConfigManager()
token_manager = TokenManager(config_manager)
auth_manager = AuthManager(config_manager, token_manager)
gpu_detector = GPUDetector()
process_manager = ProcessManager(config_manager, token_manager)
model_scanner = ModelScanner()
download_mgr = DownloadManager()
oom_watchdog = OOMWatchdog(
    process_manager, config_manager, gpu_detector
)

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
    }
    # Save model config
    config_manager.update_model_settings(
        req.path,
        {
            "context_size": req.context_size,
            "mmproj_path": req.mmproj_path,
            "gpu_weights": [
                w.model_dump() for w in req.gpu_weights
            ],
        },
    )
    return process_manager.start(
        model_path=req.path,
        gpu_weights=req.gpu_weights,
        context_size=req.context_size,
        mmproj_path=req.mmproj_path,
    )


@app.post("/stop")
async def stop_model(_auth: str = Depends(get_current_auth)):
    return process_manager.stop()


# --- Hardware Metrics ---


@app.get("/metrics")
async def get_metrics(_auth: str = Depends(get_current_auth)):
    return gpu_detector.get_metrics()


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
    return SSEStreamer.stream()


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
    gpus = gpu_detector.detect_gpus()
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
                else:
                    is_checked = "checked"
                    weight_val = 100 if idx == main_gpu_idx else 0
            else:
                is_checked = "checked"
                weight_val = 100 if idx == main_gpu_idx else 0
        else:
            is_checked = "checked"
            weight_val = 100 if idx == main_gpu_idx else 0

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
                <div class="relative">
                    <input type="number" value="{weight_val}" min="0" max="100"
                           class="gpu-weight w-24 pl-2 md:pl-4 pr-7 md:pr-9 py-2 bg-slate-900/80 border border-slate-700 rounded-xl text-sm font-black text-blue-400 outline-none transition-all"
                           oninput="balanceWeights(this)">
                    <span class="absolute right-2 md:right-3 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-600">%</span>
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
    </style>
</head>
<body class="min-h-screen text-slate-200 pb-16 selection:bg-blue-500/30">
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
                <div class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">
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
                                <label class="text-[9px] font-black uppercase text-slate-400 pl-3 md:pl-4 tracking-widest whitespace-nowrap">Contexto:</label>
                                <select id="context-size" class="bg-blue-600/20 border border-blue-500/30 text-blue-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                    {ctx_opts}
                                </select>
                            </div>
                            <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-eye text-blue-400 mr-2"></i>Vision:</label>
                                <select id="mmproj-path" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer max-w-[200px]">
                                    {vision_options}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left">
                            <thead class="text-[9px] md:text-[10px] font-black text-slate-500 uppercase tracking-widest">
                                <tr>
                                    <th class="px-4 md:px-6 py-4 text-center">Uso</th>
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
                        <div class="flex items-center gap-3 text-[10px] md:text-xs text-slate-500">
                            <i class="fas fa-info-circle text-blue-500"></i>
                            Distribua 100% da carga total entre as GPUs selecionadas
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
                            <a id="chat-link" href="#" target="_blank" class="px-6 md:px-10 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl text-[10px] md:text-xs font-black transition-all shadow-xl shadow-blue-600/30 active:scale-95 flex items-center justify-center gap-3 md:gap-4 uppercase tracking-widest whitespace-nowrap">
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
                <div class="glass rounded-[2rem] border border-slate-800 flex flex-col h-auto md:h-[900px]">
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

        // ─── Dashboard functions ───
        function initDashboard() {{
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
            document.getElementById('mmproj-path').value = "";
            document.querySelectorAll('.gpu-row').forEach((row, idx) => {{
                row.querySelector('.gpu-checkbox').checked = true;
                row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
            }});
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
            if (cfg.gpu_weights) {{
                cfg.gpu_weights.forEach(w => {{
                    const row = document.querySelector(`.gpu-row[data-index="${{w.index}}"]`);
                    if (row) {{
                        const cb = row.querySelector('.gpu-checkbox');
                        const input = row.querySelector('.gpu-weight');
                        cb.checked = w.active !== undefined ? w.active : (w.weight > 0);
                        input.value = Math.round(w.weight);
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
            const weights = Array.from(document.querySelectorAll('.gpu-weight'));
            const checkedWeights = weights.filter(w => w.closest('.gpu-row').querySelector('.gpu-checkbox').checked);
            if (checkedWeights.length <= 1) {{
                if (checkedWeights.length === 1) checkedWeights[0].value = 100;
                updateTotal();
                return;
            }}
            let val = parseInt(changedInput.value) || 0;
            if (val > 100) {{ val = 100; changedInput.value = 100; }}
            if (val < 0) {{ val = 0; changedInput.value = 0; }}
            const otherInputs = checkedWeights.filter(w => w !== changedInput);
            let remaining = 100 - val;
            for (let i = 0; i < otherInputs.length; i++) {{
                if (i === otherInputs.length - 1) {{ otherInputs[i].value = Math.max(0, remaining); }}
                else {{
                    let share = Math.min(remaining, Math.round(remaining / otherInputs.length));
                    otherInputs[i].value = share;
                    remaining -= share;
                }}
            }}
            updateTotal();
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

                if (data.config && data.config.gpu_weights && (!data.recovery || !data.recovery.active)) {{
                    data.config.gpu_weights.forEach(w => {{
                        const row = document.querySelector(`.gpu-row[data-index="${{w.index}}"]`);
                        if (row) {{
                            const input = row.querySelector('.gpu-weight');
                            const cb = row.querySelector('.gpu-checkbox');
                            if (document.activeElement !== input) {{
                                const newWeight = Math.round(w.weight);
                                if (parseInt(input.value) !== newWeight) input.value = newWeight;
                            }}
                            if (w.active !== undefined) cb.checked = w.active;
                        }}
                    }});
                    updateTotal();
                    if (data.running && !currentSelectedModel) {{
                        if (data.config.context_size) document.getElementById('context-size').value = data.config.context_size;
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
                    badge.innerHTML = '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
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
                }} else {{
                    startTime = null;
                    badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase';
                    badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE';
                    card.classList.add('hidden');
                    if (logStream) {{ logStream.abort(); logStream = null; }}
                    currentRunningModelPath = null;
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
                weights.push({{
                    index: parseInt(r.dataset.index),
                    weight: parseInt(r.querySelector('.gpu-weight').value || 0),
                    name: "GPU",
                    active: isChecked,
                }});
            }});
            if (!weights.some(w => w.active)) return alert("SELECIONE PELO MENOS UMA GPU");
            const mmprojPath = document.getElementById('mmproj-path').value;
            document.getElementById('status-badge').innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> INICIALIZANDO...';
            try {{
                await fetch('/start', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        path,
                        mmproj_path: mmprojPath || null,
                        gpu_weights: weights,
                        context_size: parseInt(document.getElementById('context-size').value),
                    }}),
                }});
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
                    mmproj_path = saved_cfg.get("mmproj_path")
                else:
                    gpus = gpu_detector.detect_gpus()
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
                    mmproj_path = None

                process_manager.start(
                    model_path=default_model,
                    gpu_weights=weights,
                    context_size=context_size,
                    mmproj_path=mmproj_path,
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
