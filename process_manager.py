"""llama-server process lifecycle and OOM watchdog."""

import json
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

from config_manager import ConfigManager, TokenManager
from gpu_manager import GPUManager, LLAMA_SERVER_BIN, DEFAULT_TOTAL_LAYERS
from log_manager import LogManager
from schemas import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_PARALLEL_SLOTS,
    GPUWeight,
    MTP_DRAFT_TOKENS_MAX,
    MTP_DRAFT_TOKENS_MIN,
    StartRequest,
)

SERVER_PORT = 8085
logger = logging.getLogger("automanager")


def compute_server_ctx_size(context_size: int, parallel_slots: int) -> int:
    """
    llama-server divides --ctx-size by --parallel (per-slot context).
    Pass the product so each slot receives the requested context_size.
    """
    slots = max(1, parallel_slots)
    ctx = max(1, context_size)
    return ctx * slots


def reasoning_cli_args(thinking_enabled: bool) -> List[str]:
    """
    Build llama-server flags for reasoning/thinking mode.

    Qwen3.x and Gemma4 read enable_thinking from the Jinja chat template;
    --reasoning off / --reasoning-budget 0 alone are not enough on those models.
    """
    kwargs = json.dumps(
        {"enable_thinking": thinking_enabled},
        separators=(",", ":"),
    )
    if thinking_enabled:
        return [
            "--jinja",
            "--reasoning",
            "on",
            "--chat-template-kwargs",
            kwargs,
        ]
    return [
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-kwargs",
        kwargs,
    ]


def mtp_cli_args(
    mtp_enabled: bool,
    mtp_draft_tokens: int,
    model_path: str,
    gpu_manager: GPUManager,
) -> List[str]:
    """Build llama-server flags for Multi-Token Prediction when applicable."""
    if not mtp_enabled:
        return []
    if not gpu_manager.detect_model_mtp(model_path):
        logger.info(
            "MTP requested but model has no MTP head, skipping flags"
        )
        return []
    n = max(
        MTP_DRAFT_TOKENS_MIN,
        min(MTP_DRAFT_TOKENS_MAX, mtp_draft_tokens or DEFAULT_MTP_DRAFT_TOKENS),
    )
    return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]


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
            "auto_balance": False,
        }
        self._auto_balance_active = False
        self._auto_balance_cancel = False

    @property
    def auto_balance_active(self) -> bool:
        with self._lock:
            return self._auto_balance_active

    @property
    def auto_balance_cancel_requested(self) -> bool:
        with self._lock:
            return self._auto_balance_cancel

    @property
    def recovery_state(self) -> dict:
        with self._lock:
            return dict(self._recovery_state)

    @recovery_state.setter
    def recovery_state(self, state: dict) -> None:
        with self._lock:
            self._recovery_state = state

    def get_status(self) -> dict:
        recovery = self.recovery_state
        status = {"running": False, "recovery": recovery}
        if recovery.get("auto_balance") and recovery.get("gpu_weights"):
            status["config"] = {"gpu_weights": recovery["gpu_weights"]}
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
                                    "parallel_slots": self._last_request.parallel_slots,
                                    "batch_size": self._last_request.batch_size,
                                    "split_mode": self._last_request.split_mode,
                                    "gpu_weights": [
                                        w.model_dump()
                                        if hasattr(w, "model_dump")
                                        else w
                                        for w in self._last_request.gpu_weights
                                    ],
                                    "mmproj_path": self._last_request.mmproj_path,
                                    "thinking_enabled": (
                                        self._last_request.thinking_enabled
                                    ),
                                    "mtp_enabled": self._last_request.mtp_enabled,
                                    "mtp_draft_tokens": (
                                        self._last_request.mtp_draft_tokens
                                    ),
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
        if not self.auto_balance_active:
            self.recovery_state = {
                "active": False,
                "failed": False,
                "message": "",
                "auto_balance": False,
            }
        self.log_manager.clear_server_log()
        logger.info("llama-server stopped")
        return {"message": "Stopped"}

    def cancel_auto_balance(self) -> dict:
        """Request cancellation of the running auto-balance probe loop."""
        with self._lock:
            if not self._auto_balance_active:
                raise HTTPException(
                    status_code=409,
                    detail="Nenhum auto-balance em andamento.",
                )
            self._auto_balance_cancel = True
        self.stop()
        logger.info("Auto-balance cancel requested by user")
        return {"message": "Cancelando auto-balance..."}

    def start_auto_balance(self, request: StartRequest) -> dict:
        """Start progressive GPU discovery in a background thread."""
        with self._lock:
            if self._auto_balance_active:
                raise HTTPException(
                    status_code=409,
                    detail="Auto-balance ja esta em andamento.",
                )
            self._auto_balance_cancel = False

        thread = threading.Thread(
            target=self._run_auto_balance,
            args=(request,),
            daemon=True,
            name="auto-balance",
        )
        thread.start()
        self.recovery_state = {
            "active": True,
            "failed": False,
            "message": "Iniciando auto-balance...",
            "auto_balance": True,
            "attempt": 0,
        }
        return {"message": "Auto-balance em andamento", "probing": True}

    def _run_auto_balance(self, request: StartRequest) -> None:
        from auto_balance import AutoBalanceCancelled, AutoBalanceProber

        with self._lock:
            self._auto_balance_active = True

        try:
            prober = AutoBalanceProber(
                self, self.config, self.gpu_manager, self.log_manager
            )
            success, gpu_weights, message, failure = prober.discover(request)
            if success:
                saved_weights = [w.model_dump() for w in gpu_weights]
                self.config.update_model_settings(
                    request.path,
                    {
                        "context_size": request.context_size,
                        "parallel_slots": request.parallel_slots,
                        "batch_size": request.batch_size,
                        "mmproj_path": request.mmproj_path,
                        "split_mode": request.split_mode,
                        "thinking_enabled": request.thinking_enabled,
                        "mtp_enabled": request.mtp_enabled,
                        "mtp_draft_tokens": request.mtp_draft_tokens,
                        "gpu_weights": saved_weights,
                        "auto_balance": False,
                        "auto_balance_profile": True,
                        "hardware_incapable": False,
                        "hardware_incapable_message": None,
                    },
                )
                self.recovery_state = {
                    "active": False,
                    "failed": False,
                    "message": message,
                    "auto_balance": False,
                    "auto_balance_completed": True,
                    "gpu_weights": saved_weights,
                }
            else:
                hardware_exceeded = bool(
                    failure
                    and failure.get("code") == "hardware_capacity_exceeded"
                )
                existing = self.config.get_model_settings(request.path)
                failure_settings = {
                    "context_size": request.context_size,
                    "parallel_slots": request.parallel_slots,
                    "batch_size": request.batch_size,
                    "mmproj_path": request.mmproj_path
                    or existing.get("mmproj_path"),
                    "split_mode": request.split_mode,
                    "thinking_enabled": request.thinking_enabled,
                    "mtp_enabled": request.mtp_enabled,
                    "mtp_draft_tokens": request.mtp_draft_tokens,
                    "gpu_weights": existing.get("gpu_weights")
                    or [w.model_dump() for w in request.gpu_weights],
                    "auto_balance": False,
                    "auto_balance_profile": False,
                }
                if hardware_exceeded:
                    failure_settings["hardware_incapable"] = True
                    failure_settings["hardware_incapable_message"] = message
                self.config.update_model_settings(
                    request.path, failure_settings
                )
                self.recovery_state = {
                    "active": False,
                    "failed": True,
                    "message": message,
                    "auto_balance": False,
                    "hardware_capacity_exceeded": hardware_exceeded,
                    "failure_details": failure,
                }
        except AutoBalanceCancelled:
            self.stop()
            self.recovery_state = {
                "active": False,
                "failed": False,
                "cancelled": True,
                "message": "Auto-balance cancelado.",
                "auto_balance": False,
            }
            logger.info("Auto-balance cancelled")
        except Exception as exc:
            logger.exception(f"Auto-balance error: {exc}")
            self.stop()
            self.recovery_state = {
                "active": False,
                "failed": True,
                "message": f"Erro no auto-balance: {exc}",
                "auto_balance": False,
                "hardware_capacity_exceeded": False,
                "failure_details": None,
            }
        finally:
            with self._lock:
                self._auto_balance_active = False
                self._auto_balance_cancel = False

    def start(
        self,
        model_path: str,
        gpu_weights: List[GPUWeight],
        context_size: int,
        mmproj_path: Optional[str] = None,
        split_mode: str = "layer",
        parallel_slots: int = DEFAULT_PARALLEL_SLOTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thinking_enabled: bool = True,
        mtp_enabled: bool = False,
        mtp_draft_tokens: int = DEFAULT_MTP_DRAFT_TOKENS,
        total_layers: int = 0,
    ) -> dict:
        self.stop()

        has_active_cpu = any(
            w.active and w.device == "cpu" for w in gpu_weights
        )
        if has_active_cpu:
            ok, err = self.gpu_manager.validate_weights(gpu_weights)
        else:
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
        server_ctx_size = compute_server_ctx_size(context_size, parallel_slots)
        # Resolve total_layers: use provided value, auto-detect from model, or fall back to default
        if not total_layers or total_layers <= 0:
            total_layers = self.gpu_manager.detect_model_layers(model_path)
        total_layers = max(1, total_layers)
        n_gpu_layers = self.gpu_manager.compute_n_gpu_layers(gpu_weights, total_layers)
        cmd = [
            LLAMA_SERVER_BIN,
            "-m",
            model_path,
            "-ngl",
            str(n_gpu_layers),
            "--flash-attn",
            "on",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--tools",
            "all",
            "--parallel",
            str(parallel_slots),
            "--ctx-size",
            str(server_ctx_size),
            "--batch-size",
            str(batch_size),
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

        cmd.extend(reasoning_cli_args(thinking_enabled))
        mtp_args = mtp_cli_args(
            mtp_enabled, mtp_draft_tokens, model_path, self.gpu_manager
        )
        cmd.extend(mtp_args)
        mtp_applied = bool(mtp_args)

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get(
            "LD_LIBRARY_PATH", ""
        )

        self.log_manager.clear_server_log()
        logger.info(
            f"START: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={visible}, "
            f"ctx_per_slot={context_size}, server_ctx={server_ctx_size}, "
            f"batch_size={batch_size}, thinking_enabled={thinking_enabled}, "
            f"mtp_enabled={mtp_enabled}, mtp_draft_tokens={mtp_draft_tokens}, "
            f"mtp_applied={mtp_applied})"
        )

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
                    parallel_slots=parallel_slots,
                    batch_size=batch_size,
                    split_mode=split_mode,
                    thinking_enabled=thinking_enabled,
                    mtp_enabled=mtp_enabled,
                    mtp_draft_tokens=mtp_draft_tokens,
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
        if self.process_manager.auto_balance_active:
            logger.info("OOM ignored during auto-balance probing")
            return

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
                "auto_balance": False,
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
                "auto_balance": False,
            }
            weights = list(req.gpu_weights)
            active = [w for w in weights if w.active]
            if len(active) <= 1:
                pm.recovery_state = {
                    "active": False,
                    "failed": True,
                    "message": "Single GPU ou sem pesos.",
                    "auto_balance": False,
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
                "parallel_slots": req.parallel_slots,
                "batch_size": req.batch_size,
                "mmproj_path": req.mmproj_path,
                "thinking_enabled": req.thinking_enabled,
                "mtp_enabled": req.mtp_enabled,
                "mtp_draft_tokens": req.mtp_draft_tokens,
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
                parallel_slots=req.parallel_slots,
                batch_size=req.batch_size,
                thinking_enabled=req.thinking_enabled,
                mtp_enabled=req.mtp_enabled,
                mtp_draft_tokens=req.mtp_draft_tokens,
                total_layers=getattr(req, "total_layers", 0),
            )
        except Exception as e:
            logger.error(f"Recovery start failed: {e}")

        time.sleep(3)
        pm.recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
            "auto_balance": False,
        }

    def stop(self) -> None:
        self._stopping = True
