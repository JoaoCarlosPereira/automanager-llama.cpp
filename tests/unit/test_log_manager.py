"""Unit tests for log SSE shutdown behavior."""

import asyncio
import threading

import pytest
from starlette.requests import Request

from log_manager import LogManager


@pytest.mark.asyncio
async def test_stream_logs_stops_when_shutdown_event_set(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text("line one\nline two\n", encoding="utf-8")
    manager = LogManager(
        project_root=str(tmp_path),
        server_log_path=str(log_path),
        manager_log_path=str(tmp_path / "manager.log"),
    )
    stop_event = threading.Event()

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/logs",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "http_version": "1.1",
    }
    request = Request(scope)

    response = manager.stream_logs(stop_event=stop_event, request=request)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
        stop_event.set()
    assert chunks
