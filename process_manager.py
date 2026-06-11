"""llama-server process lifecycle and OOM watchdog."""

import os
import re
import socket
import time
import signal
import logging
import subprocess
import threading
from typing import List, Optional, Dict

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
    """Build llama-server flags for reasoning/thinking mode."""
    if thinking_enabled:
        return ["--reasoning", "on"]
    return ["--reasoning", "off", "--reasoning-budget", "0"]


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
        self._processes: Dict[int, subprocess.Popen] = {}
        self._requests: Dict[int, StartRequest] = {}
        self._lock = threading.Lock()
        self._recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
            "auto_balance": False,
        }
        self._auto_balance_active = False
        self._auto_balance_cancel = False
        self._auto_balance_port = SERVER_PORT

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
        instances = []
        found_pids = set()

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                name = proc.info["name"] or ""
                cmdline = proc.info["cmdline"] or []
                if "llama-server" in name or (
                    cmdline and "llama-server" in cmdline[0]
                ):
                    model_name = None
                    model_path = None
                    port = None
                    for i in range(len(cmdline) - 1):
                        if cmdline[i] in ["-m", "--model"]:
                            model_name = os.path.basename(cmdline[i + 1])
                            model_path = cmdline[i + 1]
                        if cmdline[i] in ["-p", "--port"]:
                            try:
                                port = int(cmdline[i + 1])
                            except (ValueError, TypeError):
                                pass

                    if model_name:
                        if port is None:
                            port = SERVER_PORT

                        found_pids.add(proc.info["pid"])
                        inst = {
                            "running": True,
                            "pid": proc.info["pid"],
                            "model": model_name,
                            "model_path": model_path,
                            "start_time": proc.info["create_time"],
                            "port": port,
                        }
                        with self._lock:
                            req = self._requests.get(port)
                            if req:
                                inst["config"] = {
                                    "path": req.path,
                                    "context_size": req.context_size,
                                    "parallel_slots": req.parallel_slots,
                                    "batch_size": req.batch_size,
                                    "split_mode": req.split_mode,
                                    "gpu_weights": [
                                        w.model_dump()
                                        if hasattr(w, "model_dump")
                                        else w
                                        for w in req.gpu_weights
                                    ],
                                    "mmproj_path": req.mmproj_path,
                                    "thinking_enabled": req.thinking_enabled,
                                    "mtp_enabled": req.mtp_enabled,
                                    "mtp_draft_tokens": req.mtp_draft_tokens,
                                }
                        instances.append(inst)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        with self._lock:
            dead_ports = [
                p
                for p, proc in self._processes.items()
                if proc.pid not in found_pids
            ]
            for p in dead_ports:
                self._processes.pop(p, None)

        main_inst = next(
            (i for i in instances if i["port"] == SERVER_PORT), None
        ) or (instances[0] if instances else None)

        res = {
            "running": len(instances) > 0,
            "instances": instances,
            "recovery": recovery,
        }
        if main_inst:
            res.update(main_inst)

        return res

    @staticmethod
    def _is_port_free(port: int) -> bool:
        """True when nothing is listening on *port* (connect is refused)."""
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return False
        except (ConnectionRefusedError, socket.timeout, OSError):
            return True

    def _wait_port_released(
        self,
        port: int = SERVER_PORT,
        timeout: float = PORT_RELEASE_TIMEOUT_SEC,
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_port_free(port):
                return True
            time.sleep(PORT_RELEASE_POLL_SEC)
        return self._is_port_free(port)

    def stop(self, port: Optional[int] = None) -> dict:
        if port is None:
            port = SERVER_PORT

        with self._lock:
            proc = self._processes.pop(port, None)

        if proc:
            try:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.wait(timeout=PORT_RELEASE_TIMEOUT_SEC)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                pass

        self._wait_port_released(port)

        if port == SERVER_PORT and not self.auto_balance_active:
            self.recovery_state = {
                "active": False,
                "failed": False,
                "message": "",
                "auto_balance": False,
            }

        logger.info(f"llama-server on port {port} stopped")
        return {"message": f"Instancia na porta {port} encerrada"}

    def cancel_auto_balance(self) -> dict:
        """Request cancellation of the running auto-balance probe loop."""
        with self._lock:
            if not self._auto_balance_active:
                raise HTTPException(
                    status_code=409,
                    detail="Nenhum auto-balance em andamento.",
                )
            self._auto_balance_cancel = True
        self.stop(self._auto_balance_port)
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
            port = request.port or SERVER_PORT
            if request.port is None:
                while not self._is_port_free(port):
                    port += 1
            self._auto_balance_port = port

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
            # Monkeypatch request to use our allocated port
            request.port = self._auto_balance_port
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
                        port=self._auto_balance_port,
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
            self.stop(self._auto_balance_port)
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
            self.stop(self._auto_balance_port)
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
        port: Optional[int] = None,
    ) -> dict:
        if port is None:
            port = SERVER_PORT
            while not self._is_port_free(port):
                port += 1

        self.stop(port)

        gpu_weights = self.gpu_manager.normalize_gpu_weights(gpu_weights)

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
            str(port),
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

        self.log_manager.clear_server_log(port)
        logger.info(
            f"START (port {port}): {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={visible}, "
            f"ctx_per_slot={context_size}, server_ctx={server_ctx_size}, "
            f"batch_size={batch_size}, thinking_enabled={thinking_enabled}, "
            f"mtp_enabled={mtp_enabled}, mtp_draft_tokens={mtp_draft_tokens}, "
            f"mtp_applied={mtp_applied}, gpu_pct={plan.gpu_pct}, "
            f"cpu_pct={plan.cpu_pct}, n_gpu_layers={plan.n_gpu_layers}, "
            f"n_cpu_layers={plan.n_cpu_layers})"
        )

        try:
            log_file = self.log_manager.open_server_log_append(port)
            popen_kwargs = {
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "posix":
                popen_kwargs["preexec_fn"] = os.setsid
            with self._lock:
                proc = subprocess.Popen(cmd, **popen_kwargs)
                self._processes[port] = proc
                self._requests[port] = StartRequest(
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
                    port=port,
                )
                return {
                    "message": "Started",
                    "pid": proc.pid,
                    "port": port,
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
    """Monitors server logs for OOM and auto-recovers."""

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
        while not self._stopping:
            try:
                ports = []
                with self.process_manager._lock:
                    ports = list(self.process_manager._processes.keys())

                for port in ports:
                    path = self.log_manager.get_server_log_path(port)
                    if os.path.exists(path):
                        self._check_log(path, port)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(5)

    def _check_log(self, path: str, port: int) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if self.OOM_PATTERNS.search(line):
                        self._handle_oom(port)
                        break
        except OSError:
            pass

    def _handle_oom(self, port: int) -> None:
        if self.process_manager.auto_balance_active:
            logger.info(f"OOM on port {port} ignored during auto-balance probing")
            return

        now = time.time()
        with self._lock:
            if now - self._last_oom_time > self.SILENCE_TIMEOUT:
                self._consecutive_oom = 0
            self._consecutive_oom += 1
            self._last_oom_time = now
            consecutive = self._consecutive_oom

        logger.warning(f"OOM detected on port {port}! Consecutive: {consecutive}")
        pm = self.process_manager
        with pm._lock:
            req = pm._requests.get(port)
            if not req:
                return

        if consecutive >= self.MAX_CONSECUTIVE_OOM:
            pm.recovery_state = {
                "active": True,
                "failed": False,
                "message": f"OOM repetido na porta {port}. Divisao 50/50.",
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
                "message": f"OOM na porta {port}. Reduzindo carga...",
                "auto_balance": False,
            }
            weights = list(req.gpu_weights)
            active = [w for w in weights if w.active]
            if len(active) <= 1:
                pm.recovery_state = {
                    "active": False,
                    "failed": True,
                    "message": f"Single GPU ou sem pesos (porta {port}).",
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
                "auto_balance": False,
                "auto_balance_profile": False,
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
                port=port,
            )
            logger.info(f"OOM: Instancia na porta {port} reiniciada com novo split")
        except Exception as e:
            logger.error(f"OOM: Falha ao reiniciar instancia na porta {port}: {e}")

        time.sleep(3)
        if port == SERVER_PORT:
            pm.recovery_state = {
                "active": False,
                "failed": False,
                "message": f"Recuperado (OOM porta {port})",
                "auto_balance": False,
            }

    def stop(self) -> None:
        self._stopping = True
