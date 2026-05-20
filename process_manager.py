"""llama-server process lifecycle and OOM watchdog."""

import os
import re
import time
import signal
import logging
import subprocess
import threading
from typing import List, Optional

import psutil
from fastapi import HTTPException

from config_manager import ConfigManager
from gpu_manager import GPUManager, LLAMA_SERVER_BIN
from log_manager import LogManager
from schemas import GPUWeight, StartRequest
from config_manager import TokenManager

SERVER_PORT = 8085
logger = logging.getLogger("automanager")


class ProcessManager:
    """Manages llama-server process lifecycle."""

    def __init__(
        self,
        config_manager: ConfigManager,
        token_manager: TokenManager,
        gpu_manager: GPUManager,
        log_manager: LogManager,
    ):
        self.config = config_manager
        self.token_mgr = token_manager
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
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
                        with self._lock:
                            if self._last_request:
                                status["config"] = {
                                    "path": self._last_request.path,
                                    "context_size": self._last_request.context_size,
                                    "split_mode": self._last_request.split_mode,
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
        self.log_manager.clear_server_log()
        logger.info("llama-server stopped")
        return {"message": "Stopped"}

    def start(
        self,
        model_path: str,
        gpu_weights: List[GPUWeight],
        context_size: int,
        mmproj_path: Optional[str] = None,
        split_mode: str = "layer",
    ) -> dict:
        self.stop()

        ok, err = self.gpu_manager.validate_gpu_weights(gpu_weights)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

        visible = self.gpu_manager.get_visible_devices(gpu_weights)
        if not visible:
            raise HTTPException(
                status_code=400, detail="SELECIONE PELO MENOS UMA GPU"
            )

        split = self.gpu_manager.compute_tensor_split(gpu_weights)
        active_weights = [w for w in gpu_weights if w.active and w.weight > 0]
        main_gpu_obj = next((w for w in active_weights if w.is_main), None)
        if main_gpu_obj:
            main_gpu = "0"
            for i, w in enumerate(active_weights):
                if w.index == main_gpu_obj.index:
                    main_gpu = str(i)
                    break
        else:
            main_gpu = "0"

        api_token = self.token_mgr.get_or_create()
        cmd = [
            LLAMA_SERVER_BIN,
            "-m",
            model_path,
            "-ngl",
            "99",
            "--flash-attn",
            "on",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--tools",
            "all",
            "--parallel",
            "1",
            "--ctx-size",
            str(context_size),
            "--mlock",
            "--main-gpu",
            main_gpu,
            "--split-mode",
            split_mode,
            "--tensor-split",
            ",".join(split),
            "--api-key",
            api_token,
        ]

        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])
        else:
            cmd.append("--mmproj-auto")

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get(
            "LD_LIBRARY_PATH", ""
        )

        self.log_manager.clear_server_log()
        logger.info(f"START: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={visible})")

        try:
            log_file = self.log_manager.open_server_log_append()
            with self._lock:
                self._current_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    env=env,
                )
                self._last_request = StartRequest(
                    path=model_path,
                    mmproj_path=mmproj_path,
                    gpu_weights=gpu_weights,
                    context_size=context_size,
                    split_mode=split_mode,
                )
                return {
                    "message": "Started",
                    "pid": self._current_process.pid,
                }
        except Exception as e:
            logger.error(f"Start error: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao iniciar: {e}")


class OOMWatchdog(threading.Thread):
    """Monitors server log for OOM and auto-recovers."""

    OOM_PATTERNS = re.compile(
        r"(?i)(out of memory|cuda error|malloc failed|c10\.Error)"
    )
    REDUCTION_PCT = 10.0
    MAX_CONSECUTIVE_OOM = 3
    SILENCE_TIMEOUT = 30

    def __init__(
        self,
        process_manager: ProcessManager,
        config_manager: ConfigManager,
        gpu_manager: GPUManager,
        log_manager: LogManager,
    ):
        super().__init__(daemon=True)
        self.process_manager = process_manager
        self.config = config_manager
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
        self._consecutive_oom = 0
        self._last_oom_time = 0.0
        self._stopping = False
        self._lock = threading.Lock()

    def run(self) -> None:
        logger.info("OOMWatchdog started")
        path = self.log_manager.get_server_log_path()
        while not self._stopping:
            try:
                if os.path.exists(path):
                    self._check_log(path)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(5)

    def _check_log(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
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
            pm.recovery_state = {
                "active": True,
                "failed": False,
                "message": "OOM repetido. Divisao 50/50.",
            }
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

        self.config.update_model_settings(
            req.path,
            {
                "context_size": req.context_size,
                "mmproj_path": req.mmproj_path,
                "gpu_weights": [w.model_dump() for w in new_weights],
            },
        )

        try:
            pm.start(
                model_path=req.path,
                gpu_weights=new_weights,
                context_size=req.context_size,
                mmproj_path=req.mmproj_path,
                split_mode=req.split_mode,
            )
        except Exception as e:
            logger.error(f"Recovery start failed: {e}")

        time.sleep(3)
        pm.recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
        }

    def stop(self) -> None:
        self._stopping = True
