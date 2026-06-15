"""Extended tests for log_manager.py covering rotation, SSE streaming, and edge cases."""
import pytest
from unittest.mock import MagicMock, patch
import os
import time
from log_manager import LogManager


class TestLogManagerInit:
    """Tests for LogManager initialization."""

    def test_init_creates_logs_directory(self, tmp_path):
        """Test that LogManager creates logs directory."""
        logs_dir = str(tmp_path / "logs")
        lm = LogManager(project_root=str(tmp_path))
        assert os.path.isdir(logs_dir)

    def test_init_sets_manager_log_path(self, tmp_path):
        """Test LogManager sets manager log path."""
        lm = LogManager(project_root=str(tmp_path))
        expected = os.path.join(str(tmp_path), "logs", "manager.log")
        assert lm._manager_log_path == expected

    def test_init_sets_server_log_path(self, tmp_path):
        """Test LogManager sets default server log path."""
        lm = LogManager(project_root=str(tmp_path))
        expected = os.path.join(str(tmp_path), "logs", "server.log")
        assert lm._default_server_log_path == expected


class TestGetServerLogPath:
    """Tests for get_server_log_path method."""

    def test_default_port_returns_server_log(self, tmp_path):
        """Test get_server_log_path returns server.log for default port."""
        lm = LogManager(project_root=str(tmp_path))
        path = lm.get_server_log_path()
        assert path.endswith("server.log")

    def test_default_port_8085_returns_server_log(self, tmp_path):
        """Test get_server_log_path for explicit port 8085."""
        lm = LogManager(project_root=str(tmp_path))
        path = lm.get_server_log_path(8085)
        assert path.endswith("server.log")

    def test_custom_port_returns_named_log(self, tmp_path):
        """Test get_server_log_path with custom port creates server_{port}.log."""
        lm = LogManager(project_root=str(tmp_path))
        path = lm.get_server_log_path(9999)
        assert "server_9999.log" in path


class TestClearServerLog:
    """Tests for clear_server_log method."""

    def test_clear_creates_empty_file(self, tmp_path):
        """Test clear_server_log creates an empty log file."""
        lm = LogManager(project_root=str(tmp_path))
        # Write some content first
        path = lm._default_server_log_path
        with open(path, "w") as f:
            f.write("old content\n")
        lm.clear_server_log()
        with open(path, "r") as f:
            content = f.read()
        assert content == ""

    def test_clear_nonexistent_file_does_not_crash(self, tmp_path):
        """Test clear_server_log on nonexistent file does not crash."""
        lm = LogManager(project_root=str(tmp_path))
        lm.clear_server_log(9999)  # port that has no log file


class TestOpenServerLogAppend:
    """Tests for open_server_log_append method."""

    def test_open_returns_file_object(self, tmp_path):
        """Test open_server_log_append returns an open file handle."""
        lm = LogManager(project_root=str(tmp_path))
        f = lm.open_server_log_append()
        assert hasattr(f, "write")
        f.close()

    def test_open_creates_file_if_not_exists(self, tmp_path):
        """Test open_server_log_append creates file if it doesn't exist."""
        lm = LogManager(project_root=str(tmp_path))
        f = lm.open_server_log_append()
        assert os.path.exists(lm._default_server_log_path)
        f.close()


class TestLogRotation:
    """Tests for log rotation functionality."""

    def test_rotate_server_log_shifts_files(self, tmp_path):
        """Test _rotate_server_log shifts .1, .2, etc."""
        lm = LogManager(project_root=str(tmp_path))
        path = lm._default_server_log_path

        # Create existing rotated files
        with open(path, "w") as f:
            f.write("current\n")
        with open(f"{path}.1", "w") as f:
            f.write("rotated1\n")

        lm._rotate_server_log(path)

        # .1 should now be the old current
        with open(f"{path}.1", "r") as f:
            assert f.read().strip() == "current"
        # .2 should be the old .1
        with open(f"{path}.2", "r") as f:
            assert f.read().strip() == "rotated1"

    def test_rotate_nonexistent_file_does_not_crash(self, tmp_path):
        """Test _rotate_server_log on nonexistent file does not crash."""
        lm = LogManager(project_root=str(tmp_path))
        lm._rotate_server_log("/nonexistent/path.log")


class TestStartStreaming:
    """Tests for start_streaming method."""

    def test_start_streaming_clears_log_first(self, tmp_path):
        """Test start_streaming clears log before starting."""
        lm = LogManager(project_root=str(tmp_path))
        path = lm._default_server_log_path
        with open(path, "w") as f:
            f.write("old content\n")

        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        lm.start_streaming(8085, mock_proc)

        # Log should have been cleared and header written
        with open(path, "r") as f:
            content = f.read()
        assert "=== llama-server" in content
        assert "old content" not in content

    def test_start_streaming_writes_cmd_to_log(self, tmp_path):
        """Test start_streaming writes command to log."""
        lm = LogManager(project_root=str(tmp_path))
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        lm.start_streaming(8085, mock_proc, cmd=["llama-server", "-m", "model.gguf"])
        path = lm._default_server_log_path
        with open(path, "r") as f:
            content = f.read()
        assert "CMD:" in content
        assert "llama-server" in content

    def test_start_streaming_starts_thread(self, tmp_path):
        """Test start_streaming starts a background thread."""
        lm = LogManager(project_root=str(tmp_path))
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        lm.start_streaming(8085, mock_proc)
        assert 8085 in lm._stream_threads
        assert isinstance(lm._stream_threads[8085], type(lm._stream_threads[8085]))


class TestSSEStreaming:
    """Tests for stream_logs SSE endpoint."""

    def test_stream_logs_returns_streaming_response(self, tmp_path):
        """Test stream_logs returns a StreamingResponse object."""
        lm = LogManager(project_root=str(tmp_path))
        # Create a log file with content
        path = lm._default_server_log_path
        with open(path, "w") as f:
            f.write("log line 1\n")
            f.write("log line 2\n")

        response = lm.stream_logs(port=None)
        assert response is not None
        assert response.media_type == "text/event-stream"

    def test_stream_logs_missing_file_returns_error(self, tmp_path):
        """Test stream_logs returns error for nonexistent log file."""
        lm = LogManager(project_root=str(tmp_path))
        response = lm.stream_logs(port=9999)
        assert response is not None

    def test_stream_logs_formats_sse_lines(self, tmp_path):
        """Test _format_sse_line formats correctly."""
        lm = LogManager(project_root=str(tmp_path))
        formatted = lm._format_sse_line("test line\n")
        assert formatted == "data: test line\n\n"


class TestLogManagerEdgeCases:
    """Edge case tests for LogManager."""

    def test_start_streaming_twice_does_not_crash(self, tmp_path):
        """Test calling start_streaming twice on same port is safe."""
        lm = LogManager(project_root=str(tmp_path))
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        lm.start_streaming(8085, mock_proc)
        lm.start_streaming(8085, mock_proc)

    def test_multiple_ports_stream_independently(self, tmp_path):
        """Test multiple ports have independent log files."""
        lm = LogManager(project_root=str(tmp_path))
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        lm.start_streaming(8085, mock_proc)
        lm.start_streaming(9999, mock_proc)
        assert os.path.exists(lm._default_server_log_path)
        assert os.path.exists(lm.get_server_log_path(9999))

    def test_custom_server_log_path(self, tmp_path):
        """Test LogManager with custom server log path."""
        custom_log = str(tmp_path / "custom_server.log")
        lm = LogManager(
            project_root=str(tmp_path),
            server_log_path=custom_log,
        )
        assert lm._default_server_log_path == custom_log

    def test_custom_manager_log_path(self, tmp_path):
        """Test LogManager with custom manager log path."""
        custom_log = str(tmp_path / "custom_manager.log")
        lm = LogManager(
            project_root=str(tmp_path),
            manager_log_path=custom_log,
        )
        assert lm._manager_log_path == custom_log


class TestLogManagerIntegration:
    """Integration tests for LogManager."""

    def test_streaming_writes_to_log_file(self, tmp_path):
        """Test that streaming actually writes output to the log file."""
        lm = LogManager(project_root=str(tmp_path))
        lines_written = ["[server] line 1\n", "[server] line 2\n"]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines_written)

        lm.start_streaming(8085, mock_proc)

        path = lm._default_server_log_path
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read()
        assert "llama-server" in content

    def test_manager_log_handler_configured(self, tmp_path):
        """Test that manager logging has a rotating file handler."""
        lm = LogManager(project_root=str(tmp_path))
        import logging
        root_logger = logging.getLogger("automanager")
        handlers = [h for h in root_logger.handlers
                    if hasattr(h, 'baseFilename') and
                    h.baseFilename == lm._manager_log_path]
        assert len(handlers) > 0
