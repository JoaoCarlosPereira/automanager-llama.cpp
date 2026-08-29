import logging
import os
import signal
import socket
import subprocess
import threading
import time
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from fastapi import HTTPException, Request
import psutil

from config_manager import ConfigManager, TokenManager
from log_manager import LogManager, logger
from llama_server_bin import (
    get_llama_server_bin,
    is_turboquant_bin,
    supports_cli_flag,
    validate_turboquant_cache_types,
)
from gpu_manager import reasoning_cli_args, mtp_cli_args
from schemas import StartRequest, GPUWeight, DEFAULT_PARALLEL_SLOTS, DEFAULT_BATCH_SIZE, DEFAULT_MTP_DRAFT_TOKENS, DEFAULT_CACHE_TYPE, TURBOQUANT_DEFAULT_CACHE_K, TURBOQUANT_DEFAULT_CACHE_V
from paths import INSTALL_ROOT

# Lazy import to avoid circular dependency with auto_balance
_AutoBalanceProber = None
_AutoBalanceCancelled = None


def _get_auto_balance_types():
    global _AutoBalanceProber, _AutoBalanceCancelled
    if _AutoBalanceProber is None:
        from auto_balance import AutoBalanceProber, AutoBalanceCancelled
        _AutoBalanceProber = AutoBalanceProber
        _AutoBalanceCancelled = AutoBalanceCancelled
    return _AutoBalanceProber, _AutoBalanceCancelled

SERVER_PORT = 8085


def compute_server_ctx_size(context_size: int, parallel_slots: int) -> int:
    """
    Compute total context size for the llama-server command.
    Matches llama-server behavior: if --parallel N is used, --ctx-size refers to
    the total buffer, but internally it often needs to be N * context_per_slot.
    
    Actually, in newer llama.cpp, --ctx-size is the total.
    We assume the UI 'Context' value is 'context per slot'.
    """
    return context_size * parallel_slots


def resolve_llama_server_bin() -> str:
    """Locate the llama-server binary.

    Delegates to the shared resolver in ``llama_server_bin`` so that start()
    uses the SAME discovery logic that runs at startup (env var, paths.json,
    PATH/``shutil.which`` and the common ``llama.cpp/build/bin`` locations).
    The previous local implementation only checked ``INSTALL_ROOT/bin`` and the
    cwd, so on hosts where the binary lives elsewhere (e.g.
    ``~/llama.cpp/build/bin/llama-server``) it fell back to the bare name and
    Popen failed with ``[Errno 2] No such file or directory: 'llama-server'``.
    """
    return get_llama_server_bin()


def resolve_numa_mode(gpu_weights: List[GPUWeight]) -> str:
    """Pick llama-server --numa TYPE for the active GPU layout."""
    active_gpus = 0
    for w in gpu_weights:
        if hasattr(w, "model_dump"):
            data = w.model_dump()
        elif isinstance(w, dict):
            data = w
        else:
            data = {"active": w.active, "device": w.device, "weight": w.weight}
        if (
            data.get("active")
            and data.get("device") == "gpu"
            and float(data.get("weight", 0) or 0) > 0
        ):
            active_gpus += 1
    # Multi-GPU tensor-split often spans PCIe roots / NUMA nodes.
    if active_gpus > 1:
        return "distribute"
    # Single GPU: stay on the NUMA node where execution starts (near --main-gpu).
    return "isolate"


class ProcessManager:
    def __init__(
        self,
        config: ConfigManager,
        token_mgr: TokenManager,
        gpu_manager: "GPUManager",
        log_manager: LogManager,
    ):
        self.config = config
        self.token_mgr = token_mgr
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
        self.processes: Dict[int, subprocess.Popen] = {}
        self._requests: Dict[int, StartRequest] = {}
        self._lock = threading.Lock()
        self._auto_balance_active = False
        self._auto_balance_cancel = False
        self._auto_balance_port = SERVER_PORT
        self._auto_balance_run_id = 0
        self.recovery_state = {
            "active": False,
            "failed": False,
            "message": "",
            "auto_balance": False,
        }

    @property
    def auto_balance_cancel_requested(self) -> bool:
        return self._auto_balance_cancel

    @auto_balance_cancel_requested.setter
    def auto_balance_cancel_requested(self, value: bool) -> None:
        self._auto_balance_cancel = value

    def _is_port_free(self, port: int) -> bool:
        """Check if a TCP port is free to bind."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    def _wait_port_released(self, port: int, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self._is_port_free(port):
                return True
            time.sleep(0.5)
        return False

    def stop(self, port: Optional[int] = None) -> dict:
        """Stop one or all llama-server processes."""
        with self._lock:
            if port is None:
                ports = list(self.processes.keys())
            else:
                ports = [port] if port in self.processes else []

            for p in ports:
                proc = self.processes.pop(p)
                try:
                    # Try SIGINT first for graceful exit
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass
                if p in self._requests:
                    del self._requests[p]

            if not self.processes and not self._auto_balance_active:
                self.recovery_state = {
                    "active": False,
                    "failed": False,
                    "message": "",
                    "auto_balance": False,
                }

        # Wait outside the lock for each port to be released so the next
        # start() can bind it. When nothing matched, still wait on the
        # resolved target (default SERVER_PORT) to honor the contract.
        wait_ports = ports if ports else [port if port is not None else SERVER_PORT]
        for p in wait_ports:
            self._wait_port_released(p)

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
            self._auto_balance_run_id += 1
            run_id = self._auto_balance_run_id
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
            "smart_calibration": request.smart_calibration,
            "smart_proposal": None,
            "attempt": 0,
            "model": request.path,
            "run_id": run_id,
        }
        return {
            "message": "Auto-balance em andamento",
            "probing": True,
            "run_id": run_id,
        }

    def _run_auto_balance(self, request: StartRequest) -> None:
        Prober, Cancelled = _get_auto_balance_types()
        AutoBalanceProber = Prober
        AutoBalanceCancelled = Cancelled

        with self._lock:
            self._auto_balance_active = True

        run_id = self.recovery_state.get("run_id")

        # Multi-instância: se o modelo a calibrar já está carregado, sua
        # instância antiga distorceria a leitura de VRAM da sondagem (nas
        # versões single-instance o probe sempre partia de VRAM livre).
        with self._lock:
            same_model_ports = [
                p for p, req in self._requests.items()
                if req is not None and req.path == request.path
            ]
        for p in same_model_ports:
            logger.info(
                "Auto-balance: parando instancia existente do modelo na porta %s "
                "antes da calibracao", p,
            )
            self.stop(p)
        if same_model_ports:
            time.sleep(3.0)  # aguarda o driver liberar a VRAM antes de sondar

        try:
            prober = AutoBalanceProber(
                self, self.config, self.gpu_manager, self.log_manager
            )
            # Monkeypatch request to use our allocated port
            request.port = self._auto_balance_port

            # Discover returns (success, gpu_weights, message, result_data)
            result = prober.discover(request)
            success, gpu_weights, message, result_data = result
            failure = result_data if not success else None
            proposal = result_data.get("proposal") if result_data else None

            if success:
                saved_weights = [w.model_dump() for w in gpu_weights]
                
                if request.smart_calibration:
                    self.stop(self._auto_balance_port)
                    self.recovery_state = {
                        "active": False,
                        "failed": False,
                        "message": "Calibração concluída. Proposta gerada.",
                        "auto_balance": False,
                        "smart_calibration": True,
                        "smart_proposal": proposal,
                        "gpu_weights": saved_weights,
                        "pinned_fields": request.pinned_fields or {},
                        "model": request.path,
                        "run_id": run_id,
                    }
                    return

                # Normal Auto-Balance: Auto-save and Auto-reload
                self.config.update_model_settings(
                    request.path,
                    {
                        "context_size": request.context_size,
                        "parallel_slots": request.parallel_slots,
                        "batch_size": request.batch_size,
                        "ubatch_size": request.ubatch_size,
                        "cache_type_k": request.cache_type_k,
                        "cache_type_v": request.cache_type_v,
                        "numa_enabled": request.numa_enabled,
                        "flash_attn_enabled": request.flash_attn_enabled,
                        "threads": request.threads,
                        "threads_batch": request.threads_batch,
                        "mmproj_path": request.mmproj_path,
                        "mmproj_disabled": request.mmproj_disabled,
                        "vision_enabled": request.vision_enabled,
                        "split_mode": request.split_mode,
                        "thinking_enabled": request.thinking_enabled,
                        "mtp_enabled": request.mtp_enabled,
                        "mtp_draft_tokens": request.mtp_draft_tokens,
                        "mtp_model_path": request.mtp_model_path,
                        "gpu_weights": saved_weights,
                        "pinned_fields": request.pinned_fields or {},
                        "llama_server_bin": request.llama_server_bin,
                        "turboquant_preset": request.turboquant_preset,
                        "auto_balance": False,
                        "auto_balance_profile": True,
                        "manual_gpu_override": False,
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
                        mmproj_disabled=request.mmproj_disabled
                        or request.vision_enabled is False,
                        vision_enabled=request.vision_enabled,
                        split_mode=request.split_mode,
                        parallel_slots=request.parallel_slots,
                        batch_size=request.batch_size,
                        ubatch_size=request.ubatch_size,
                        cache_type_k=request.cache_type_k,
                        cache_type_v=request.cache_type_v,
                        numa_enabled=request.numa_enabled,
                        flash_attn_enabled=request.flash_attn_enabled,
                        threads=request.threads,
                        threads_batch=request.threads_batch,
                        thinking_enabled=request.thinking_enabled,
                        mtp_enabled=request.mtp_enabled,
                        mtp_draft_tokens=request.mtp_draft_tokens,
                        mtp_model_path=request.mtp_model_path,
                        total_layers=request.total_layers,
                        cpu_enabled=final_cpu_enabled,
                        port=self._auto_balance_port,
                        llama_server_bin=request.llama_server_bin,
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
                    "failed": not model_loaded,
                    "message": message,
                    "auto_balance": False,
                    "auto_balance_completed": True,
                    "model_loaded": model_loaded,
                    "gpu_weights": saved_weights,
                    "model": request.path,
                    "run_id": run_id,
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
                    "ubatch_size": request.ubatch_size,
                    "cache_type_k": request.cache_type_k,
                    "cache_type_v": request.cache_type_v,
                    "numa_enabled": request.numa_enabled,
                    "flash_attn_enabled": request.flash_attn_enabled,
                    "threads": request.threads,
                    "threads_batch": request.threads_batch,
                    "mmproj_path": request.mmproj_path
                    or existing.get("mmproj_path"),
                    "mmproj_disabled": request.mmproj_disabled
                    or existing.get("mmproj_disabled", False),
                    "vision_enabled": request.vision_enabled
                    if request.vision_enabled is not None
                    else existing.get("vision_enabled", True),
                    "split_mode": request.split_mode,
                    "thinking_enabled": request.thinking_enabled,
                    "mtp_enabled": request.mtp_enabled,
                    "mtp_draft_tokens": request.mtp_draft_tokens,
                    "mtp_model_path": request.mtp_model_path
                    or existing.get("mtp_model_path"),
                    "gpu_weights": existing.get("gpu_weights")
                    or [w.model_dump() for w in request.gpu_weights],
                    "llama_server_bin": request.llama_server_bin
                    or existing.get("llama_server_bin"),
                    "turboquant_preset": request.turboquant_preset
                    or existing.get("turboquant_preset"),
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
                    "model": request.path,
                    "run_id": run_id,
                }
        except AutoBalanceCancelled:
            self.stop(self._auto_balance_port)
            self.recovery_state = {
                "active": False,
                "failed": False,
                "cancelled": True,
                "message": "Auto-balance cancelado.",
                "auto_balance": False,
                "model": request.path,
                "run_id": run_id,
            }
        except Exception as exc:
            logger.exception(f"Auto-balance error: {exc}")
            self.stop(self._auto_balance_port)
            self.recovery_state = {
                "active": False,
                "failed": True,
                "message": f"Erro no auto-balance: {exc}",
                "auto_balance": False,
                "hardware_capacity_exceeded": False,
                "model": request.path,
                "run_id": run_id,
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
        mmproj_disabled: bool = False,
        split_mode: str = "layer",
        parallel_slots: int = DEFAULT_PARALLEL_SLOTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ubatch_size: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        numa_enabled: bool = False,
        flash_attn_enabled: bool = True,
        threads: int = 0,
        threads_batch: int = 0,
        vision_enabled: Optional[bool] = None,
        thinking_enabled: bool = True,
        mtp_enabled: bool = False,
        mtp_draft_tokens: int = DEFAULT_MTP_DRAFT_TOKENS,
        mtp_model_path: Optional[str] = None,
        total_layers: int = 0,
        cpu_enabled: Optional[bool] = None,
        port: Optional[int] = None,
        llama_server_bin: Optional[str] = None,
        manual_gpu_override: bool = False,
      ) -> dict:
        """Start a llama-server instance."""
        # Vision can be disabled independently of the remembered projector.
        # Keep the projector in config for a future re-enable, but never pass
        # it (or --mmproj-auto) to this process while disabled.
        mmproj_disabled = mmproj_disabled or vision_enabled is False
        # Validate inputs before acquiring lock
        gpu_weights = self.gpu_manager.normalize_gpu_weights(gpu_weights)
        valid, err = self.gpu_manager.validate_gpu_weights(gpu_weights)
        if not valid:
            raise HTTPException(status_code=400, detail=err)

        visible = self.gpu_manager.get_visible_devices(gpu_weights)

        # Build offload plan
        if not total_layers or total_layers <= 0:
            total_layers = self.gpu_manager.detect_model_layers(model_path)
        total_layers = max(1, total_layers)

        if cpu_enabled is True:
            try:
                from auto_balance import AutoBalancePlanner

                est = AutoBalancePlanner.estimate_model_vram_mb(
                    model_path,
                    context_size,
                    parallel_slots,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                )
                self.gpu_manager._cached_model_vram_mb = est["weights_mb"]
            except Exception as exc:
                logger.warning(
                    "Não foi possível estimar VRAM do modelo para CPU offload: %s",
                    exc,
                )

        plan = self.gpu_manager.compute_offload_plan(
            gpu_weights, total_layers, cpu_enabled
        )
        split = plan.tensor_split
        n_gpu_layers = plan.n_gpu_layers
        main_gpu = self.gpu_manager.resolve_main_gpu_index(gpu_weights)

        llama_bin = llama_server_bin or resolve_llama_server_bin()
        if llama_server_bin and not os.path.isfile(llama_server_bin):
            raise HTTPException(
                status_code=400,
                detail=f"Binário llama-server não encontrado: {llama_server_bin}",
            )
        if mtp_enabled and mtp_model_path and not os.path.isfile(mtp_model_path):
            raise HTTPException(
                status_code=400,
                detail=f"Modelo draft MTP não encontrado: {mtp_model_path}",
            )
        turbo_err = validate_turboquant_cache_types(cache_type_k, cache_type_v, llama_bin)
        if turbo_err:
            raise HTTPException(status_code=400, detail=turbo_err)
        cache_type_k = (cache_type_k or "").strip() or (
            TURBOQUANT_DEFAULT_CACHE_K if is_turboquant_bin(llama_bin) else DEFAULT_CACHE_TYPE
        )
        cache_type_v = (cache_type_v or "").strip() or (
            TURBOQUANT_DEFAULT_CACHE_V if is_turboquant_bin(llama_bin) else DEFAULT_CACHE_TYPE
        )
        api_token = self.token_mgr.get_or_create()
        server_ctx_size = compute_server_ctx_size(context_size, parallel_slots)

        # Allocate port and acquire exclusive access under lock
        with self._lock:
            if port is None:
                port = SERVER_PORT
                while port in self.processes or not self._is_port_free(port):
                    port += 1

        cmd = [
            llama_bin,
            "-m",
            model_path,
            "-ngl",
            str(n_gpu_layers),
            "--cache-type-k",
            cache_type_k,
            "--cache-type-v",
            cache_type_v,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            str(parallel_slots),
            "--ctx-size",
            str(server_ctx_size),
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(ubatch_size),
            "--mlock",
        ]

        if split:
            cmd.extend([
                "--main-gpu",
                main_gpu,
                "--split-mode",
                split_mode,
                "--tensor-split",
                ",".join(split),
            ])

        cmd.extend(["--api-key", api_token])

        if flash_attn_enabled:
            cmd.extend(["--flash-attn", "on"])

        if numa_enabled:
            cmd.extend(["--numa", resolve_numa_mode(gpu_weights)])

        # Performance: Use physical cores if not specified
        cpu_info = self.gpu_manager.detect_cpu_info()
        final_threads = threads
        if final_threads <= 0 and cpu_info.physical_cores > 0:
            final_threads = cpu_info.physical_cores

        if final_threads > 0:
            cmd.extend(["--threads", str(final_threads)])
        if threads_batch > 0:
            cmd.extend(["--threads-batch", str(threads_batch)])

        if supports_cli_flag("--pinned-memory", llama_bin):
            cmd.append("--pinned-memory")

        if supports_cli_flag("--kv-unified", llama_bin):
            cmd.append("--kv-unified")

        if mmproj_path and os.path.exists(mmproj_path) and not mmproj_disabled:
            cmd.extend(["--mmproj", mmproj_path])
        elif not mmproj_disabled:
            cmd.append("--mmproj-auto")

        # Logic for reasoning and mtp
        reasoning_args = reasoning_cli_args(thinking_enabled)
        cmd.extend(reasoning_args)

        mtp_args, mtp_applied, mtp_reason = mtp_cli_args(
            mtp_enabled,
            mtp_draft_tokens,
            model_path,
            self.gpu_manager,
            mtp_model_path=mtp_model_path,
        )
        cmd.extend(mtp_args)

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible if visible is not None else ""

        logger.info(f"Starting llama-server on port {port} for model {os.path.basename(model_path)}")
        try:
            # Stop any existing process on this port under lock
            with self._lock:
                if port in self.processes:
                    existing_proc = self.processes.pop(port)
                    try:
                        existing_proc.send_signal(signal.SIGINT)
                        try:
                            existing_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            existing_proc.kill()
                    except Exception:
                        pass
                    self._requests.pop(port, None)

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            proc._start_time = time.time()
            with self._lock:
                self.processes[port] = proc
                self._requests[port] = StartRequest(
                    path=model_path,
                    mmproj_path=mmproj_path,
                    mmproj_disabled=mmproj_disabled,
                    gpu_weights=gpu_weights,
                    context_size=context_size,
                    parallel_slots=parallel_slots,
                    batch_size=batch_size,
                    ubatch_size=ubatch_size,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    numa_enabled=numa_enabled,
                    flash_attn_enabled=flash_attn_enabled,
                    threads=threads,
                    threads_batch=threads_batch,
                    vision_enabled=vision_enabled,
                    split_mode=split_mode,
                    thinking_enabled=thinking_enabled,
                    mtp_enabled=mtp_enabled,
                    mtp_draft_tokens=mtp_draft_tokens,
                    mtp_model_path=mtp_model_path,
                    manual_gpu_override=manual_gpu_override,
                    port=port,
                )
            
            # Watch stderr/stdout
            self.log_manager.start_streaming(port, proc, cmd=cmd)
            return {
                "message": "Servidor iniciado",
                "port": port,
                "start_time": proc._start_time,
            }
        except Exception as e:
            logger.exception("Failed to start llama-server")
            raise HTTPException(status_code=500, detail=str(e))

    def get_status(self) -> dict:
        instances = []
        with self._lock:
            dead_ports = [
                port for port, proc in self.processes.items()
                if proc.poll() is not None
            ]
            for port in dead_ports:
                self.processes.pop(port, None)
                self._requests.pop(port, None)

            for port, proc in self.processes.items():
                req = self._requests.get(port)
                status = "running" if proc.poll() is None else "stopped"
                instances.append({
                    "port": port,
                    "status": status,
                    "model": os.path.basename(req.path) if req else "unknown",
                    "model_path": req.path if req else None,
                    "start_time": getattr(proc, "_start_time", None),
                    "config": req.model_dump() if req else None,
                })
        return {
            "instances": instances,
            "recovery": self.recovery_state,
        }


class OOMWatchdog:
    OOM_PATTERNS = re.compile(
        r"(?i)(out of memory|cuda error|failed to allocate|malloc failed|c10\.Error)"
    )
    READY_PATTERNS = re.compile(
        r"(?i)(model loaded|server is listening|listening on http://)"
    )

    def __init__(
        self,
        process_manager: ProcessManager,
        config_manager: Optional["ConfigManager"] = None,
        gpu_manager: Optional["GPUManager"] = None,
        log_manager: Optional[LogManager] = None,
    ):
        self.pm = process_manager
        self.config = config_manager
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
        self._stop = threading.Event()
        self._thread = None
        self._consecutive_oom = 0
        self._last_oom_time = 0
        self._oom_cooldown = 30  # seconds
        self._log_offsets: Dict[int, int] = {}
        self._ready_ports = set()
        self._pending_oom_ports = set()
        self._process_identities: Dict[int, object] = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="oom-watchdog")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _read_new_log(self, port: int) -> str:
        """Read the per-port server log since the last watchdog scan.

        The process' stdout/stderr are already consumed by the log-streaming
        thread, so OOM detection must read the log file on disk (server.log for
        the default port, server_{port}.log otherwise).
        """
        if self.log_manager is None:
            return ""
        try:
            path = self.log_manager.get_server_log_path(port)
        except Exception:
            return ""
        offset = self._log_offsets.get(port, 0)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                self._log_offsets[port] = f.tell()
                return chunk
        except OSError:
            return ""

    def _run(self):
        """Main loop for log-based OOM detection.

        Periodically scans each active llama-server's log file for OOM patterns
        and triggers recovery logic when detected.
        """
        while not self._stop.is_set():
            try:
                with self.pm._lock:
                    active_ports = [p for p, proc in self.pm.processes.items()
                                    if proc is not None and proc.poll() is None]
                    known_ports = [p for p, proc in self.pm.processes.items()
                                   if proc is not None]
                    process_identities = {
                        p: getattr(proc, "pid", None) or id(proc)
                        for p, proc in self.pm.processes.items()
                        if proc is not None
                    }

                # A reused port is a new server instance.  Do not carry the
                # previous instance's readiness/OOM state into its log scan.
                for port, identity in process_identities.items():
                    if self._process_identities.get(port) != identity:
                        self._process_identities[port] = identity
                        self._log_offsets.pop(port, None)
                        self._ready_ports.discard(port)
                        self._pending_oom_ports.discard(port)

                # Drop offsets for ports that are no longer running so a
                # restarted instance on the same port re-scans from the top.
                for stale in [p for p in self._log_offsets if p not in active_ports]:
                    self._log_offsets.pop(stale, None)
                for stale in [p for p in self._ready_ports if p not in known_ports]:
                    self._ready_ports.discard(stale)
                for stale in [p for p in self._pending_oom_ports if p not in known_ports]:
                    self._pending_oom_ports.discard(stale)
                for stale in [p for p in self._process_identities if p not in known_ports]:
                    self._process_identities.pop(stale, None)

                # An OOM before readiness can be recoverable: llama.cpp may
                # retry buffer allocation without pipeline parallelism and
                # continue loading.  Only handle it if that process later
                # exits; this avoids killing a server that successfully
                # reached its listening state.
                for port in list(self._pending_oom_ports):
                    with self.pm._lock:
                        proc = self.pm.processes.get(port)
                    if proc is not None and proc.poll() is not None:
                        self._pending_oom_ports.discard(port)
                        logger.warning(
                            "OOM-affected llama-server on port %s exited before "
                            "becoming ready", port
                        )
                        self._handle_oom(port)

                for port in active_ports:
                    try:
                        chunk = self._read_new_log(port)
                        if chunk and self.READY_PATTERNS.search(chunk):
                            self._ready_ports.add(port)
                            self._pending_oom_ports.discard(port)

                        if chunk and self.OOM_PATTERNS.search(chunk):
                            logger.warning(f"OOM detected on port {port} from server log")
                            if port in self._ready_ports:
                                self._handle_oom(port)
                            else:
                                self._pending_oom_ports.add(port)
                                logger.info(
                                    "Deferring OOM recovery on port %s until "
                                    "llama-server readiness is known",
                                    port,
                                )
                    except Exception:
                        logger.exception("Watchdog OOM scan error on port %s", port)
            except Exception:
                logger.exception("Watchdog loop error")
            time.sleep(5)

    def _handle_oom(self, port: int) -> None:
        """Handle OOM signal for a specific port."""
        now = time.time()
        # Só o estado compartilhado do ProcessManager precisa do lock. stop()/
        # start() readquirem self.pm._lock (não reentrante); segurá-lo aqui
        # causaria deadlock, então lemos o necessário e liberamos antes.
        with self.pm._lock:
            if (now - self._last_oom_time) > self._oom_cooldown:
                self._consecutive_oom = 0
            self._consecutive_oom += 1
            self._last_oom_time = now

            if getattr(self.pm, "_auto_balance_active", False) is True:
                logger.info("Watchdog: Skipping OOM handle during active auto-balance")
                return

            # Robust request retrieval (handle both dicts and MagicMocks)
            reqs = getattr(self.pm, "_requests", {})
            req = reqs.get(port) if hasattr(reqs, "get") else None

        if not req:
            return

        logger.warning(f"OOM detected on port {port}! Consecutive: {self._consecutive_oom}")

        # A manually selected split is an explicit user contract.  Automatic
        # recovery used to rewrite it (and persist the rewritten values), so
        # the next launch appeared with GPUs swapped or with a different
        # distribution.  Leave manual layouts untouched and let the UI/log
        # report the real OOM so the user can choose a new split/context.
        if getattr(req, "manual_gpu_override", False):
            logger.error(
                "OOM on port %s with manual GPU split; preserving configured "
                "weights and not restarting automatically",
                port,
            )
            self.pm.stop(port)
            if self.config:
                self.config.update_model_settings(req.path, {
                    "last_failure": "OOM",
                    "last_failure_time": now,
                })
            return

        self.pm.stop(port)

        # Recovery logic
        weights = getattr(req, "gpu_weights", [])
        active_weights = [w for w in weights if w.active and w.device == "gpu"]

        if self._consecutive_oom >= 3:
            # Fallback: reduce active to 50%, ensure inactive are 0
            for w in weights:
                if w.active:
                    w.weight = 50.0
                else:
                    w.weight = 0.0
        else:
            # Conservative recovery: reduce main and redistribute
            if active_weights:
                # Find main (highest weight for now)
                main_w = max(active_weights, key=lambda x: x.weight)
                reduction = 10.0
                main_w.weight -= reduction

                others = [w for w in active_weights if w != main_w]
                if others:
                    boost = reduction / len(others)
                    for w in others:
                        w.weight += boost

            # Always ensure inactive GPUs have 0 weight during recovery
            for w in weights:
                if not w.active:
                    w.weight = 0.0

        # Save and restart
        if self.config:
            # Use model_dump if available (Pydantic V2), fallback to dict()
            weights_data = []
            for w in weights:
                if hasattr(w, "model_dump"):
                    weights_data.append(w.model_dump())
                else:
                    weights_data.append(w.dict())

            self.config.update_model_settings(req.path, {
                "gpu_weights": weights_data,
                "last_failure": "OOM",
                "last_failure_time": now
            })

        # Restart after a brief pause
        def _restart():
            time.sleep(2)
            self.pm.start(
                model_path=req.path,
                gpu_weights=req.gpu_weights,
                context_size=req.context_size,
                mmproj_path=getattr(req, "mmproj_path", None),
                mmproj_disabled=getattr(req, "mmproj_disabled", False),
                vision_enabled=getattr(req, "vision_enabled", None),
                split_mode=getattr(req, "split_mode", "layer"),
                parallel_slots=getattr(req, "parallel_slots", 1),
                batch_size=getattr(req, "batch_size", 2048),
                ubatch_size=getattr(req, "ubatch_size", 512),
                cache_type_k=getattr(req, "cache_type_k", "f16"),
                cache_type_v=getattr(req, "cache_type_v", "f16"),
                numa_enabled=getattr(req, "numa_enabled", False),
                flash_attn_enabled=getattr(req, "flash_attn_enabled", True),
                threads=getattr(req, "threads", 0),
                threads_batch=getattr(req, "threads_batch", 0),
                thinking_enabled=getattr(req, "thinking_enabled", True),
                mtp_enabled=getattr(req, "mtp_enabled", False),
                mtp_draft_tokens=getattr(req, "mtp_draft_tokens", 3),
                mtp_model_path=getattr(req, "mtp_model_path", None),
                manual_gpu_override=getattr(req, "manual_gpu_override", False),
                port=port,
                llama_server_bin=getattr(req, "llama_server_bin", None),
            )

        threading.Thread(target=_restart, daemon=True).start()
