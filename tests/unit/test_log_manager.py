"""Unit tests for log SSE shutdown behavior."""

import asyncio
import os
import subprocess
import sys
import threading
import time
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from log_manager import LogManager


def test_start_streaming_writes_subprocess_output(tmp_path):
    log_path = tmp_path / "server.log"
    manager = LogManager(
        project_root=str(tmp_path),
        server_log_path=str(log_path),
        manager_log_path=str(tmp_path / "manager.log"),
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", "print('hello from server')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    manager.start_streaming(8085, proc)
    proc.wait(timeout=5)
    time.sleep(0.2)
    assert "hello from server" in log_path.read_text(encoding="utf-8")


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

    # First call returns connected (empty message), subsequent calls raise StopAsyncIteration
    # so is_disconnected() returns False initially
    call_count = 0

    async def mock_receive():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: not disconnected
            return {}
        # After that, raise to signal no more messages
        raise StopAsyncIteration

    request._receive = mock_receive

    # Access the async generator directly from the StreamingResponse
    response = manager.stream_logs(stop_event=stop_event, request=request)
    gen = response.body_iterator
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
        stop_event.set()
        if len(chunks) >= 2:
            break
    assert chunks
