"""llama-server process lifecycle and OOM watchdog."""

import json
import os
import re
import socket
import time
import signal
import logging
import subprocess
import threading
from typing import List, Optional

import psutil
from fastapi import HTTPException

from config_manager import ConfigManager, TokenManager
from gpu_manager import GPUManager, DEFAULT_TOTAL_LAYERS
from llama_server_bin import resolve_llama_server_bin
from log_manager import LogManager
from schemas import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_PARALLEL_SLOTS,
    GPUWeight,
    StartRequest,
)

SERVER_PORT = 8085
# How long to wait for the OS to release the server port after killing the
# process, so a follow-up start() (e.g. the next auto-balance probe) does not
# fail to bind it. SIGKILL is async: the socket lingers briefly after the kill.
PORT_RELEASE_TIMEOUT_SEC = 10.0
PORT_RELEASE_POLL_SEC = 0.25
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
    gpu_manager: "GPUManager",
):
    """Build llama-server flags for Multi-Token Prediction."""
    if not mtp_enabled:
        return ([], False, "MTP desativado na configuracao")

    # Forcamos a ativacao se solicitado pelo usuario (Opcao B)
    n = max(1, min(64, mtp_draft_tokens or 1))
    return (
        ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)],
        True,
        "",
    )
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

    @staticmethod
    def _is_port_free(port: int) -> bool:
        """True when nothing is listening on *port* (connect is refused)."""
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return False  # something answered -> still bound
        except (ConnectionRefusedError, socket.timeout, OSError):
            return True

    def _wait_port_released(
        self,
        port: int = SERVER_PORT,
        timeout: float = PORT_RELEASE_TIMEOUT_SEC,
    ) -> bool:
        """Block until *port* is free, or *timeout* elapses.

        Prevents the next start()/probe from failing with
        "couldn't bind HTTP server socket" while the just-killed server is still
        releasing the socket.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_port_free(port):
                return True
            time.sleep(PORT_RELEASE_POLL_SEC)
        free = self._is_port_free(port)
        if not free:
            logger.warning(
                "Port %d still bound after %.1fs wait — start() may fail to bind",
                port,
                timeout,
            )
        return free

    def stop(self) -> dict:
        if os.name == "posix":
            subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
        with self._lock:
            proc = self._current_process
            if proc:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                self._current_process = None
            self._last_request = None
        # Reap the killed process and wait for the OS to release the port so a
        # follow-up start() can bind it (fixes transient probe bind crashes).
        if proc is not None:
            try:
                proc.wait(timeout=PORT_RELEASE_TIMEOUT_SEC)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                pass
        self._wait_port_released()
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
                # Leave the model loaded with the final auto-balanced weights
                # (reload so the running server matches the chosen split, not
                # whatever the last probe happened to leave running).
                final_cpu_enabled = any(
                    w.device == "cpu" and w.active and w.weight > 0
                    for w in gpu_weights
                )
                model_loaded = False
                try:
                    self.start(
                        model_path=request.path,
                        gpu_weights=gpu_weights,
                        context_size=request.context_size,
                        mmproj_path=request.mmproj_path,
                        split_mode=request.split_mode,
                        parallel_slots=request.parallel_slots,
                        batch_size=request.batch_size,
                        thinking_enabled=request.thinking_enabled,
                        mtp_enabled=request.mtp_enabled,
                        mtp_draft_tokens=request.mtp_draft_tokens,
                        total_layers=request.total_layers,
                        cpu_enabled=final_cpu_enabled,
                    )
                    model_loaded = True
                    logger.info(
                        "Auto-balance: modelo recarregado com o split final"
                    )
                except Exception as exc:
                    logger.error(
                        "Auto-balance: falha ao recarregar o modelo final: %s",
                        exc,
                    )
                # active=False signals the frontend that auto-balance finished
                # (it polls for !recovery.active to apply the final weights). The
                # loaded model is reflected by the live process (data.running),
                # not by recovery.active — keeping it True would stick the UI on
                # "REALOCANDO...".
                self.recovery_state = {
                    "active": False,
                    "failed": False,
                    "message": message,
                    "auto_balance": False,
                    "auto_balance_completed": True,
                    "model_loaded": model_loaded,
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
        cpu_enabled: Optional[bool] = None,
    ) -> dict:
        self.stop()

        gpu_weights = self.gpu_manager.normalize_gpu_weights(gpu_weights)

        # Validate GPU weights only (cpu_enabled is controlled by checkbox valve)
        ok, err = self.gpu_manager.validate_gpu_weights(gpu_weights)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

        visible = self.gpu_manager.get_visible_devices(gpu_weights)
        if not visible:
            raise HTTPException(
                status_code=400, detail="SELECIONE PELO MENOS UMA GPU"
            )

        if not total_layers or total_layers <= 0:
            total_layers = self.gpu_manager.detect_model_layers(model_path)
        total_layers = max(1, total_layers)

        plan = self.gpu_manager.compute_offload_plan(
            gpu_weights, total_layers, cpu_enabled=cpu_enabled
        )
        split = plan.tensor_split
        main_gpu = self.gpu_manager.resolve_main_gpu_index(gpu_weights)
        n_gpu_layers = plan.n_gpu_layers
        llama_bin = resolve_llama_server_bin()
        if not llama_bin:
            raise HTTPException(
                status_code=500,
                detail=(
                    "llama-server nao encontrado. Instale o binario, adicione ao PATH, "
                    "defina LLAMA_SERVER_BIN ou configure llama_server_bin em paths.json."
                ),
            )

        api_token = self.token_mgr.get_or_create()
        server_ctx_size = compute_server_ctx_size(context_size, parallel_slots)
        cmd = [
            llama_bin,
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
        mtp_args, mtp_applied, mtp_reason = mtp_cli_args(
            mtp_enabled, mtp_draft_tokens, model_path, self.gpu_manager
        )
        cmd.extend(mtp_args)

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
            f"mtp_applied={mtp_applied}, gpu_pct={plan.gpu_pct}, "
            f"cpu_pct={plan.cpu_pct}, n_gpu_layers={plan.n_gpu_layers}, "
            f"n_cpu_layers={plan.n_cpu_layers})"
        )

        try:
            log_file = self.log_manager.open_server_log_append()
            popen_kwargs = {
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "posix":
                popen_kwargs["preexec_fn"] = os.setsid
            with self._lock:
                self._current_process = subprocess.Popen(cmd, **popen_kwargs)
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
                    "mtp_applied": mtp_applied,
                    "mtp_reason": mtp_reason,
                }
        except FileNotFoundError:
            logger.error("llama-server nao encontrado ao iniciar processo")
            raise HTTPException(
                status_code=500,
                detail=(
                    "llama-server nao encontrado. Instale o binario, adicione ao PATH, "
                    "defina LLAMA_SERVER_BIN ou configure llama_server_bin em paths.json."
                ),
            )
        except Exception as e:
            logger.error(f"Start error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


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
