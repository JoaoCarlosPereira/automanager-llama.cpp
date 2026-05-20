"""Log file management, rotation, and SSE streaming."""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from fastapi.responses import StreamingResponse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SERVER_LOG_PATH = os.path.join(LOGS_DIR, "server.log")
MANAGER_LOG_PATH = os.path.join(LOGS_DIR, "manager.log")
LEGACY_SERVER_LOG_PATH = "/root/llama_server.log"
LEGACY_MANAGER_LOG_PATH = "/root/manager.log"
MAX_LOG_SIZE = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3

logger = logging.getLogger("automanager")


class LogManager:
    """Project-local logs with rotation; dual-write to legacy paths when possible."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        server_log_path: Optional[str] = None,
        manager_log_path: Optional[str] = None,
    ):
        self._project_root = project_root or PROJECT_ROOT
        self._logs_dir = os.path.join(self._project_root, "logs")
        self._server_log_path = server_log_path or os.path.join(
            self._logs_dir, "server.log"
        )
        self._manager_log_path = manager_log_path or os.path.join(
            self._logs_dir, "manager.log"
        )
        os.makedirs(self._logs_dir, exist_ok=True)
        self._setup_manager_logging()

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
            rfh = RotatingFileHandler(
                self._manager_log_path,
                maxBytes=MAX_LOG_SIZE,
                backupCount=LOG_BACKUP_COUNT,
            )
            rfh.setFormatter(formatter)
            root_logger.addHandler(rfh)

        try:
            os.makedirs(os.path.dirname(LEGACY_MANAGER_LOG_PATH), exist_ok=True)
            if not any(
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", "") == LEGACY_MANAGER_LOG_PATH
                for h in root_logger.handlers
            ):
                legacy = logging.FileHandler(LEGACY_MANAGER_LOG_PATH)
                legacy.setFormatter(formatter)
                root_logger.addHandler(legacy)
        except OSError:
            pass

    def get_server_log_path(self) -> str:
        return self._server_log_path

    def clear_server_log(self) -> None:
        for path in (self._server_log_path, LEGACY_SERVER_LOG_PATH):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
            except OSError:
                pass

    def open_server_log_append(self):
        """Open server log for subprocess stdout (append)."""
        os.makedirs(os.path.dirname(self._server_log_path), exist_ok=True)
        return open(self._server_log_path, "a", encoding="utf-8")

    def stream_logs(self) -> StreamingResponse:
        path = self._server_log_path

        def generate():
            if not os.path.exists(path):
                yield "data: Arquivo de log nao encontrado.\n\n"
                return
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                existing = f.readlines()
                for line in existing[-500:]:
                    yield f"data: {line}"
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    yield f"data: {line}"

        return StreamingResponse(generate(), media_type="text/event-stream")
