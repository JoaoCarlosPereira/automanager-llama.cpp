"""Log file management, rotation, and SSE streaming."""

import asyncio
import os
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

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
        self._server_log_path = server_log_path or os.path.join(
            self._logs_dir, "server.log"
        )
        self._manager_log_path = manager_log_path or os.path.join(
            self._logs_dir, "manager.log"
        )
        os.makedirs(self._logs_dir, exist_ok=True)
        self._setup_manager_logging()
        self._setup_server_log_rotation()

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

    def _setup_server_log_rotation(self) -> None:
        """Ensure server.log has rotation to prevent disk exhaustion."""
        try:
            if os.path.exists(self._server_log_path):
                size = os.path.getsize(self._server_log_path)
                if size > MAX_LOG_SIZE:
                    self._rotate_server_log()
        except OSError:
            pass

    def _rotate_server_log(self) -> None:
        """Manually rotate server.log using the same pattern as manager.log."""
        for i in range(LOG_BACKUP_COUNT - 1, 0, -1):
            src = f"{self._server_log_path}.{i}"
            dst = f"{self._server_log_path}.{i + 1}"
            try:
                if os.path.exists(src):
                    os.replace(src, dst)
            except OSError:
                pass
        # Move current log to .1
        try:
            if os.path.exists(self._server_log_path):
                os.replace(self._server_log_path, f"{self._server_log_path}.1")
        except OSError:
            pass

    def get_server_log_path(self) -> str:
        return self._server_log_path

    def clear_server_log(self) -> None:
        try:
            with open(self._server_log_path, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass

    def open_server_log_append(self):
        """Open server log for subprocess stdout (append) with rotation check."""
        os.makedirs(os.path.dirname(self._server_log_path), exist_ok=True)
        # Check if rotation is needed before opening
        try:
            if os.path.exists(self._server_log_path):
                size = os.path.getsize(self._server_log_path)
                if size > MAX_LOG_SIZE:
                    self._rotate_server_log()
        except OSError:
            pass
        return open(self._server_log_path, "a", encoding="utf-8")

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
    ) -> StreamingResponse:
        path = self._server_log_path

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
