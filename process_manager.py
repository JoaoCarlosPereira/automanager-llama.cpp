import logging
import os
import signal
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
from schemas import StartRequest, GPUWeight, DEFAULT_PARALLEL_SLOTS, DEFAULT_BATCH_SIZE, DEFAULT_MTP_DRAFT_TOKENS, DEFAULT_CACHE_TYPE, TURBOQUANT_DEFAULT_CACHE_K, TURBOQUANT_DEFAULT_CACHE_V
from paths import INSTALL_ROOT

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
        gpu_manager: any,
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
        import socket
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
        from auto_balance import AutoBalanceCancelled, AutoBalanceProber

        with self._lock:
            self._auto_balance_active = True

        run_id = self.recovery_state.get("run_id")

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
                        "threads": request.threads,
                        "threads_batch": request.threads_batch,
                        "mmproj_path": request.mmproj_path,
                        "split_mode": request.split_mode,
                        "thinking_enabled": request.thinking_enabled,
                        "mtp_enabled": request.mtp_enabled,
                        "mtp_draft_tokens": request.mtp_draft_tokens,
                        "gpu_weights": saved_weights,
                        "pinned_fields": request.pinned_fields or {},
                        "llama_server_bin": request.llama_server_bin,
                        "turboquant_preset": request.turboquant_preset,
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
                        ubatch_size=request.ubatch_size,
                        cache_type_k=request.cache_type_k,
                        cache_type_v=request.cache_type_v,
                        numa_enabled=request.numa_enabled,
                        threads=request.threads,
                        threads_batch=request.threads_batch,
                        thinking_enabled=request.thinking_enabled,
                        mtp_enabled=request.mtp_enabled,
                        mtp_draft_tokens=request.mtp_draft_tokens,
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
                    "threads": request.threads,
                    "threads_batch": request.threads_batch,
                    "mmproj_path": request.mmproj_path
                    or existing.get("mmproj_path"),
                    "split_mode": request.split_mode,
                    "thinking_enabled": request.thinking_enabled,
                    "mtp_enabled": request.mtp_enabled,
                    "mtp_draft_tokens": request.mtp_draft_tokens,
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
        split_mode: str = "layer",
        parallel_slots: int = DEFAULT_PARALLEL_SLOTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ubatch_size: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        numa_enabled: bool = False,
        threads: int = 0,
        threads_batch: int = 0,
        thinking_enabled: bool = True,
        mtp_enabled: bool = False,
        mtp_draft_tokens: int = DEFAULT_MTP_DRAFT_TOKENS,
        total_layers: int = 0,
        cpu_enabled: Optional[bool] = None,
        port: Optional[int] = None,
        llama_server_bin: Optional[str] = None,
    ) -> dict:
        """Start a llama-server instance."""
        if port is None:
            port = SERVER_PORT
            while not self._is_port_free(port):
                port += 1

        self.stop(port)

        gpu_weights = self.gpu_manager.normalize_gpu_weights(gpu_weights)
        valid, err = self.gpu_manager.validate_gpu_weights(gpu_weights)
        if not valid:
            raise HTTPException(status_code=400, detail=err)

        visible = self.gpu_manager.get_visible_devices(gpu_weights)
        
        # Build offload plan
        if not total_layers or total_layers <= 0:
            total_layers = self.gpu_manager.detect_model_layers(model_path)
        total_layers = max(1, total_layers)

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
        cmd = [
            llama_bin,
            "-m",
            model_path,
            "-ngl",
            str(n_gpu_layers),
            "--flash-attn",
            "on",
            "--cache-type-k",
            cache_type_k,
            "--cache-type-v",
            cache_type_v,
            "--host",
            "127.0.0.1",
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
            "--ubatch-size",
            str(ubatch_size),
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

        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])
        else:
            cmd.append("--mmproj-auto")

        # Logic for reasoning and mtp
        from gpu_manager import reasoning_cli_args, mtp_cli_args
        reasoning_args = reasoning_cli_args(thinking_enabled)
        cmd.extend(reasoning_args)
        
        mtp_args, mtp_applied, mtp_reason = mtp_cli_args(
            mtp_enabled, mtp_draft_tokens, model_path, self.gpu_manager
        )
        cmd.extend(mtp_args)

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible

        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        try:
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
                    gpu_weights=gpu_weights,
                    context_size=context_size,
                    parallel_slots=parallel_slots,
                    batch_size=batch_size,
                    ubatch_size=ubatch_size,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    numa_enabled=numa_enabled,
                    threads=threads,
                    threads_batch=threads_batch,
                    split_mode=split_mode,
                    thinking_enabled=thinking_enabled,
                    mtp_enabled=mtp_enabled,
                    mtp_draft_tokens=mtp_draft_tokens,
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
        r"(?:CUDA out of memory|failed to allocate CUDA buffer|malloc failed|torch runtime raised c10.Error)",
        re.IGNORECASE
    )

    def __init__(
        self,
        process_manager: ProcessManager,
        config_manager: ConfigManager = None,
        gpu_manager: any = None,
        log_manager: LogManager = None,
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

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="oom-watchdog")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        """Main loop for log-based OOM detection (placeholder for now)."""
        while not self._stop.is_set():
            time.sleep(5)

    def _handle_oom(self, port: int) -> None:
        """Handle OOM signal for a specific port."""
        now = time.time()
        with self.pm._lock:
            # Update consecutive counter (global to watchdog)
            if (now - self._last_oom_time) > self._oom_cooldown:
                self._consecutive_oom = 0
            self._consecutive_oom += 1
            self._last_oom_time = now

            # Check if auto-balance is currently running to avoid race conditions
            is_active = getattr(self.pm, "_auto_balance_active", False)
            if is_active is True:
                logger.info("Watchdog: Skipping OOM handle during active auto-balance")
                return
            
            # Robust request retrieval (handle both dicts and MagicMocks)
            reqs = getattr(self.pm, "_requests", {})
            req = None
            if isinstance(reqs, dict):
                req = reqs.get(port)
            elif hasattr(reqs, "get"):
                req = reqs.get(port)
            
            if not req:
                return
                
            logger.warning(f"OOM detected on port {port}! Consecutive: {self._consecutive_oom}")
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
                self.pm.start(req)
            
            threading.Thread(target=_restart, daemon=True).start()
