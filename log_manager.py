"""Log file management, rotation, and SSE streaming."""

import asyncio
import os
import subprocess
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

from paths import (
    LOGS_DIR,
    MANAGER_LOG_PATH,
    SERVER_LOG_PATH,
)

MAX_LOG_SIZE = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3

logger = logging.getLogger("automanager")


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
        self._setup_manager_logging()
        self._setup_server_log_rotation(self._default_server_log_path)

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

    def start_streaming(self, port: int, proc: subprocess.Popen) -> None:
        """Read subprocess stdout in a background thread and write to server log."""
        self.clear_server_log(port)
        path = self._get_path(port)

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

    def stream_logs(
        self,
        stop_event: Optional[threading.Event] = None,
        request: Optional[Request] = None,
        port: Optional[int] = None,
    ) -> StreamingResponse:
        path = self._get_path(port)

        async def generate():
            if not os.path.exists(path):
                yield "data: Arquivo de log nao encontrado.\n\n"
                return
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                existing = f.readlines()
                for line in existing[-500:]:
                    if await self._stream_should_stop(stop_event, request):
                        return
                    yield f"data: {line}"
                while True:
                    if await self._stream_should_stop(stop_event, request):
                        return
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue
                    yield f"data: {line}"

        return StreamingResponse(generate(), media_type="text/event-stream")
