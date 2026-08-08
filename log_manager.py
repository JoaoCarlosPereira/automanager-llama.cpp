"""Log file management, rotation, and SSE streaming."""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import StreamingResponse

from paths import (
    LOGS_DIR,
    MANAGER_LOG_PATH,
    SERVER_LOG_PATH,
)

MAX_LOG_SIZE = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3
MAX_PROXY_REQUEST_LOG_SIZE = 50 * 1024 * 1024
PROXY_REQUEST_LOG_BACKUP_COUNT = 5

logger = logging.getLogger("automanager")


class MetricsService:
    """Métricas thread-safe em memória com histogramas e percentis (p50, p95, p99).

    Armazena valores em listas por-métrica dentro de um dict protegido por
    ``threading.Lock``.  Suporta consulta paginada que **nunca** vaz payloads.
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def observe(self, metric: str, value: float) -> None:
        """Registra um ponto de dado na métrica nomeada."""
        with self._lock:
            if metric not in self._data:
                self._data[metric] = []
            self._data[metric].append(value)

    def _percentile(self, metric: str, p: float) -> Optional[float]:
        """Retorna o percentil *p* (0-100) para a métrica."""
        values = self._get_sorted(metric)
        if not values:
            return None
        k = (len(values) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < len(values) else f
        d = k - f
        return values[f] + d * (values[c] - values[f])

    def _get_sorted(self, metric: str) -> List[float]:
        """Retorna uma lista ordenada (cópia segura)."""
        values = self._data.get(metric, [])
        return sorted(values)

    def percentile(self, metric: str, p: float) -> Optional[float]:
        """Percentil público."""
        with self._lock:
            return self._percentile(metric, p)

    def summary(self, metric: str) -> Dict[str, Any]:
        """Retorna resumo metadata-only da métrica (count, p50, p95, p99, min, max)."""
        with self._lock:
            values = self._data.get(metric, [])
            if not values:
                return {"metric": metric, "count": 0}
            s = sorted(values)
            return {
                "metric": metric,
                "count": len(s),
                "min": s[0],
                "max": s[-1],
                "mean": sum(s) / len(s),
                "p50": self._percentile(metric, 50),
                "p95": self._percentile(metric, 95),
                "p99": self._percentile(metric, 99),
            }

    def all_summaries(self) -> List[Dict[str, Any]]:
        """Retorna summaries de todas as métricas."""
        with self._lock:
            keys = list(self._data.keys())
        return [self.summary(k) for k in keys]

    def query(
        self,
        page: int = 1,
        per_page: int = 50,
        metric_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Consulta paginada de métricas sem vazar payloads.

        Retorna **apenas** summaries (count, percentis, min, max, mean).
        """
        if page < 1:
            page = 1
        start = (page - 1) * per_page
        end = start + per_page

        if metric_filter:
            items = [self.summary(metric_filter)]
        else:
            with self._lock:
                keys = list(self._data.keys())
            items = [self.summary(k) for k in keys]

        total = len(items)
        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, (total + per_page - 1) // per_page),
            "items": items[start:end],
        }

    def reset(self, metric: Optional[str] = None) -> None:
        """Limpa dados de uma métrica ou de todas."""
        with self._lock:
            if metric:
                self._data.pop(metric, None)
            else:
                self._data.clear()


class LogManager:
    """Project-local logs with rotation."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        server_log_path: Optional[str] = None,
        manager_log_path: Optional[str] = None,
    ):
        self._project_root = project_root or os.path.dirname(os.path.abspath(__file__))
        self._logs_dir = os.path.join(self._project_root, "logs")
        self._default_server_log_path = server_log_path or os.path.join(
            self._logs_dir, "server.log"
        )
        self._manager_log_path = manager_log_path or os.path.join(
            self._logs_dir, "manager.log"
        )
        os.makedirs(self._logs_dir, exist_ok=True)
        self._stream_threads: Dict[int, threading.Thread] = {}
        self._stream_lock = threading.Lock()
        self._proxy_request_log_lock = threading.Lock()
        self._setup_manager_logging()
        self._setup_server_log_rotation(self._default_server_log_path)

    def record_proxy_request(
        self,
        *,
        path: str,
        received_payload: Dict[str, Any],
        forwarded_payload: Dict[str, Any],
        backend: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        stream: bool = False,
    ) -> None:
        """Persist a complete proxy payload for post-request diagnostics.

        Authentication headers are intentionally not included. The request
        body is preserved because fields such as ``reasoning_effort`` are
        needed to diagnose provider compatibility.
        """
        record = {
            "id": uuid.uuid4().hex,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "path": path,
            "stream": bool(stream),
            "status_code": status_code,
            "duration_ms": duration_ms,
            "backend": backend or {},
            "received_payload": received_payload,
            "forwarded_payload": forwarded_payload,
        }
        path = os.path.join(self._logs_dir, "proxy_requests.jsonl")
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._proxy_request_log_lock:
                self._rotate_proxy_request_log(path)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist proxy request log: %s", exc)

    def _rotate_proxy_request_log(self, path: str) -> None:
        """Rotate the diagnostic request log before it exceeds its limit."""
        try:
            if not os.path.exists(path):
                return
            if os.path.getsize(path) + 4096 <= MAX_PROXY_REQUEST_LOG_SIZE:
                return
        except OSError:
            return
        for index in range(PROXY_REQUEST_LOG_BACKUP_COUNT - 1, 0, -1):
            source = f"{path}.{index}"
            target = f"{path}.{index + 1}"
            try:
                if os.path.exists(source):
                    os.replace(source, target)
            except OSError:
                pass
        try:
            os.replace(path, f"{path}.1")
        except OSError:
            pass

    def _setup_manager_logging(self) -> None:
        root_logger = logging.getLogger("automanager")
        if root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        if not any(
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == self._manager_log_path
            for h in root_logger.handlers
        ):
            try:
                rfh = RotatingFileHandler(
                    self._manager_log_path,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=LOG_BACKUP_COUNT,
                )
                rfh.setFormatter(formatter)
                root_logger.addHandler(rfh)
            except OSError:
                pass

    def _setup_server_log_rotation(self, path: str) -> None:
        """Ensure server.log has rotation to prevent disk exhaustion."""
        try:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > MAX_LOG_SIZE:
                    self._rotate_server_log(path)
        except OSError:
            pass

    def _rotate_server_log(self, path: str) -> None:
        """Manually rotate server.log using the same pattern as manager.log."""
        for i in range(LOG_BACKUP_COUNT - 1, 0, -1):
            src = f"{path}.{i}"
            dst = f"{path}.{i + 1}"
            try:
                if os.path.exists(src):
                    os.replace(src, dst)
            except OSError:
                pass
        # Move current log to .1
        try:
            if os.path.exists(path):
                os.replace(path, f"{path}.1")
        except OSError:
            pass

    def _get_path(self, port: Optional[int] = None) -> str:
        if port is None or port == 8085:
            return self._default_server_log_path
        return os.path.join(self._logs_dir, f"server_{port}.log")

    def get_server_log_path(self, port: Optional[int] = None) -> str:
        return self._get_path(port)

    def clear_server_log(self, port: Optional[int] = None) -> None:
        path = self._get_path(port)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass

    def open_server_log_append(self, port: Optional[int] = None):
        """Open server log for subprocess stdout (append) with rotation check."""
        path = self._get_path(port)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Check if rotation is needed before opening
        self._setup_server_log_rotation(path)
        return open(path, "a", encoding="utf-8")

    def start_streaming(
        self,
        port: int,
        proc: subprocess.Popen,
        cmd: Optional[list] = None,
    ) -> None:
        """Read subprocess stdout in a background thread and write to server log."""
        self.clear_server_log(port)
        path = self._get_path(port)
        try:
            with open(path, "a", encoding="utf-8") as f:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"=== llama-server :{port} {stamp} ===\n")
                if cmd:
                    f.write(f"CMD: {' '.join(cmd)}\n")
                f.write("\n")
        except OSError:
            pass

        def _pump() -> None:
            log_file = None
            try:
                log_file = self.open_server_log_append(port)
                stdout = proc.stdout
                if stdout is None:
                    return
                for line in stdout:
                    log_file.write(line)
                    log_file.flush()
                    try:
                        if os.path.getsize(path) > MAX_LOG_SIZE:
                            log_file.close()
                            self._rotate_server_log(path)
                            log_file = self.open_server_log_append(port)
                    except OSError:
                        pass
            except Exception as exc:
                logger.debug("Log streaming ended for port %s: %s", port, exc)
            finally:
                if log_file is not None and not log_file.closed:
                    log_file.close()
                with self._stream_lock:
                    self._stream_threads.pop(port, None)

        thread = threading.Thread(
            target=_pump,
            name=f"log-stream-{port}",
            daemon=True,
        )
        with self._stream_lock:
            self._stream_threads[port] = thread
        thread.start()

    async def _stream_should_stop(
        self,
        stop_event: Optional[threading.Event],
        request: Optional[Request],
    ) -> bool:
        if stop_event and stop_event.is_set():
            return True
        if request is not None and await request.is_disconnected():
            return True
        return False

    def _format_sse_line(self, line: str) -> str:
        return f"data: {line.rstrip(chr(10))}\n\n"

    def stream_logs(
        self,
        stop_event: Optional[threading.Event] = None,
        request: Optional[Request] = None,
        port: Optional[int] = None,
    ) -> StreamingResponse:
        path = self._get_path(port)

        async def generate():
            if not os.path.exists(path):
                yield self._format_sse_line("Arquivo de log nao encontrado.")
                return

            f = open(path, "r", encoding="utf-8", errors="replace")
            try:
                existing = f.readlines()
                for line in existing[-500:]:
                    if await self._stream_should_stop(stop_event, request):
                        return
                    yield self._format_sse_line(line)

                try:
                    last_size = os.path.getsize(path)
                except OSError:
                    last_size = 0

                while True:
                    if await self._stream_should_stop(stop_event, request):
                        return
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        size = last_size
                    if size < last_size:
                        f.close()
                        f = open(path, "r", encoding="utf-8", errors="replace")
                        try:
                            last_size = os.path.getsize(path)
                        except OSError:
                            last_size = 0
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            yield self._format_sse_line(line)
                        continue
                    last_size = size
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.15)
                        continue
                    yield self._format_sse_line(line)
            finally:
                if f and not f.closed:
                    f.close()

        return StreamingResponse(generate(), media_type="text/event-stream")
