#!/usr/bin/env python3
"""Rebuild slim llama_manager.py from original."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "llama_manager.py").read_text(encoding="utf-8").splitlines()

# Keep from app = FastAPI (line 997) through end
start_idx = next(i for i, line in enumerate(src) if line.strip().startswith("app = FastAPI"))
tail = src[start_idx:]

header = '''#!/usr/bin/env python3
"""
Automanager Llama.cpp - Control Plane para llama-server
FastAPI application that orchestrates llama-server instances with multi-GPU
tensor split management, OOM auto-recovery, and real-time hardware monitoring.
"""

import os
import socket
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from config_manager import ConfigManager, TokenManager, AuthManager, DEFAULT_CONTEXT_SIZE
from gpu_manager import GPUManager
from log_manager import LogManager
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

MANAGER_PORT = 8000

# Initialize logging and services
log_manager = LogManager()
config_manager = ConfigManager()
token_manager = TokenManager(config_manager)
auth_manager = AuthManager(config_manager, token_manager)
gpu_manager = GPUManager()
process_manager = ProcessManager(
    config_manager, token_manager, gpu_manager, log_manager
)
model_scanner = ModelScanner(config_manager, process_manager)
download_mgr = DownloadManager()
oom_watchdog = OOMWatchdog(
    process_manager, config_manager, gpu_manager, log_manager
)
gpu_detector = gpu_manager  # alias for routes/tests compatibility

'''

# Fix tail references
text = header + "\n".join(tail) + "\n"
text = text.replace("gpu_detector.detect_gpus()", "gpu_manager.detect_gpus()")
text = text.replace("gpu_detector.get_metrics()", "gpu_manager.get_metrics()")
text = text.replace("GPUDetector()", "GPUManager()")
text = text.replace("SSEStreamer.stream()", "log_manager.stream_logs()")
text = text.replace(
    "oom_watchdog = OOMWatchdog(\n    process_manager, config_manager, gpu_detector\n)",
    "",
)
text = text.replace("config_manager = ConfigManager()\n", "", 1)
text = text.replace("token_manager = TokenManager(config_manager)\n", "", 1)
text = text.replace("auth_manager = AuthManager(config_manager, token_manager)\n", "", 1)
text = text.replace("gpu_detector = GPUDetector()\n", "", 1)
text = text.replace(
    "process_manager = ProcessManager(config_manager, token_manager)\n", "", 1
)
text = text.replace("model_scanner = ModelScanner()\n", "", 1)
text = text.replace("download_mgr = DownloadManager()\n", "", 1)

out = ROOT / "llama_manager.py"
out.write_text(text, encoding="utf-8")
print(f"Wrote {out} ({len(text.splitlines())} lines)")
