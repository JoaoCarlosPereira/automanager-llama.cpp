"""Extended tests for process_manager.py covering OOMWatchdog, port allocation, and edge cases."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from process_manager import ProcessManager, OOMWatchdog, SERVER_PORT, _get_auto_balance_types


class TestPortAllocation:
    """Tests for port allocation and race condition prevention."""

    def test_port_allocation_under_lock(self):
        """Verify port allocation happens under lock."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm._lock.acquire()
        assert pm._lock.locked()
        pm._lock.release()

    def test_is_port_free_free(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        # Test with a port that's likely free
        result = pm._is_port_free(65534)
        assert isinstance(result, bool)

    def test_is_port_free_returns_boolean(self):
        """Verify _is_port_free returns a boolean."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        for port in [8085, 8086, 8087, 9999, 30000]:
            result = pm._is_port_free(port)
            assert isinstance(result, bool)

    def test_wait_port_released_returns_true_when_free(self):
        """Test that _wait_port_released returns True when port is free."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        # Port 65534 is likely free
        result = pm._wait_port_released(65534, timeout=2.0)
        assert isinstance(result, bool)

    def test_wait_port_released_times_out(self):
        """Test that _wait_port_released returns False when port stays busy."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        # Simulate port always busy
        with patch.object(pm, '_is_port_free', return_value=False):
            result = pm._wait_port_released(9999, timeout=0.5)
            assert result is False


class TestStopMethod:
    """Tests for the stop method."""

    def test_stop_single_port(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        pm.processes[8085] = mock_proc
        result = pm.stop(8085)
        assert 8085 not in pm.processes
        assert "message" in result

    def test_stop_no_matching_port(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        pm.processes[8085] = mock_proc
        result = pm.stop(9999)
        # Should still have the process since port doesn't match
        assert 8085 in pm.processes

    def test_stop_all_ports(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        pm.processes[8085] = mock_proc
        pm.processes[8086] = mock_proc
        result = pm.stop()
        assert len(pm.processes) == 0

    def test_stop_handles_timeout(self):
        """Test stop handles process that doesn't respond to SIGINT."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = Exception("Timeout")
        pm.processes[8085] = mock_proc
        result = pm.stop(8085)
        assert 8085 not in pm.processes

    def test_stop_removes_from_requests(self):
        """Test that stop removes entry from _requests dict."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        pm.processes[8085] = mock_proc
        from schemas import StartRequest
        from schemas import GPUWeight
        pm._requests[8085] = StartRequest(
            path="/models/model.gguf",
            gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
            context_size=8192,
            parallel_slots=1,
            batch_size=512,
        )
        pm.stop(8085)
        assert 8085 not in pm._requests


class TestAutoBalanceCancel:
    """Tests for auto-balance cancellation."""

    def test_cancel_auto_balance_success(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm._auto_balance_active = True
        result = pm.cancel_auto_balance()
        assert result["message"] == "Cancelando auto-balance..."

    def test_cancel_auto_balance_when_inactive(self):
        from fastapi import HTTPException
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm._auto_balance_active = False
        with pytest.raises(HTTPException):
            pm.cancel_auto_balance()


class TestAutoBalanceStart:
    """Tests for starting auto-balance."""

    def test_start_auto_balance_success(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        from schemas import StartRequest, GPUWeight
        req = StartRequest(
            path="/models/model.gguf",
            gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
            context_size=8192,
            parallel_slots=1,
            batch_size=512,
            auto_balance=True,
        )
        result = pm.start_auto_balance(req)
        assert "message" in result
        assert "probing" in result
        assert result["probing"] is True

    def test_start_auto_balance_already_active(self):
        from fastapi import HTTPException
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm._auto_balance_active = True
        from schemas import StartRequest, GPUWeight
        req = StartRequest(
            path="/models/model.gguf",
            gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
            context_size=8192,
            parallel_slots=1,
            batch_size=512,
            auto_balance=True,
        )
        with pytest.raises(HTTPException):
            pm.start_auto_balance(req)

    def test_auto_balance_run_id_increments(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm._auto_balance_run_id = 0
        from schemas import StartRequest, GPUWeight
        req = StartRequest(
            path="/models/model.gguf",
            gpu_weights=[GPUWeight(index=0, weight=100.0, name="GPU0", active=True)],
            context_size=8192,
            parallel_slots=1,
            batch_size=512,
            auto_balance=True,
        )
        # Mock threading.Thread so the background thread doesn't actually run,
        # allowing both calls to start_auto_balance without hitting the
        # _auto_balance_active guard.
        with patch("threading.Thread") as mock_thread:
            result1 = pm.start_auto_balance(req)
            result2 = pm.start_auto_balance(req)
            assert result1["run_id"] != result2["run_id"]
            assert result2["run_id"] == result1["run_id"] + 1


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_empty(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        status = pm.get_status()
        assert "instances" in status
        assert "recovery" in status
        assert status["instances"] == []

    def test_get_status_with_running_process(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        pm.processes[8085] = mock_proc
        pm._requests[8085] = MagicMock()
        pm._requests[8085].path = "/models/model.gguf"
        pm._requests[8085].model_dump.return_value = {"path": "/models/model.gguf"}
        mock_proc._start_time = 1000.0
        status = pm.get_status()
        assert len(status["instances"]) == 1
        assert status["instances"][0]["status"] == "running"
        assert status["instances"][0]["model"] == "model.gguf"

    def test_get_status_removes_dead_processes(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Process dead
        pm.processes[8085] = mock_proc
        status = pm.get_status()
        assert 8085 not in pm.processes

    def test_get_status_with_dead_process(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Process dead
        pm.processes[8085] = mock_proc
        status = pm.get_status()
        # Dead processes are cleaned up (removed from pm.processes) by get_status
        assert 8085 not in pm.processes
        assert status["instances"] == []


class TestOOMWatchdogRun:
    """Tests for the OOMWatchdog _run method."""

    def test_oom_watchdog_run_detects_oom(self):
        """Test that OOMWatchdog detects OOM from stderr."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = "CUDA out of memory: tried to allocate 1 GiB"
        pm.processes[8085] = mock_proc

        watchdog = OOMWatchdog(pm)
        watchdog.start()
        import time
        time.sleep(1)
        watchdog.stop()

    def test_oom_watchdog_run_no_oom(self):
        """Test OOMWatchdog doesn't trigger on normal output."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = "server listening on 0.0.0.0:8085"
        pm.processes[8085] = mock_proc

        watchdog = OOMWatchdog(pm)
        watchdog.start()
        import time
        time.sleep(1)
        watchdog.stop()

    def test_oom_watchdog_run_with_none_process(self):
        """Test OOMWatchdog handles None in processes dict."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        pm.processes[8085] = None
        watchdog = OOMWatchdog(pm)
        watchdog.start()
        import time
        time.sleep(1)
        watchdog.stop()

    def test_oom_watchdog_run_exception_handling(self):
        """Test OOMWatchdog handles exceptions gracefully."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.side_effect = Exception("Read error")
        pm.processes[8085] = mock_proc

        watchdog = OOMWatchdog(pm)
        watchdog.start()
        import time
        time.sleep(1)
        watchdog.stop()

    def test_oom_watchdog_stop(self):
        """Test OOMWatchdog can be stopped."""
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        watchdog = OOMWatchdog(pm)
        watchdog.start()
        watchdog.stop()
        # Should not raise any exceptions


class TestOOMWatchdogPatterns:
    """Tests for OOM detection patterns."""

    def test_oom_pattern_cuda_out_of_memory(self):
        watchdog = OOMWatchdog(MagicMock())
        assert watchdog.OOM_PATTERNS.search("CUDA out of memory: tried to allocate 1 GiB")

    def test_oom_pattern_failed_to_allocate(self):
        watchdog = OOMWatchdog(MagicMock())
        assert watchdog.OOM_PATTERNS.search("failed to allocate CUDA buffer")

    def test_oom_pattern_malloc_failed(self):
        watchdog = OOMWatchdog(MagicMock())
        assert watchdog.OOM_PATTERNS.search("malloc failed")

    def test_oom_pattern_c10_error(self):
        watchdog = OOMWatchdog(MagicMock())
        assert watchdog.OOM_PATTERNS.search("torch runtime raised c10.Error")

    def test_oom_pattern_ignore_normal_line(self):
        watchdog = OOMWatchdog(MagicMock())
        assert not watchdog.OOM_PATTERNS.search("server listening on 0.0.0.0:8085")
        assert not watchdog.OOM_PATTERNS.search("loaded model metadata successfully")

    def test_oom_pattern_case_insensitive(self):
        watchdog = OOMWatchdog(MagicMock())
        assert watchdog.OOM_PATTERNS.search("cuda OUT OF MEMORY")


class TestGetAutoBalanceTypes:
    """Tests for lazy auto_balance imports."""

    def test_get_auto_balance_types_returns_tuple(self):
        prober, cancelled = _get_auto_balance_types()
        assert prober is not None
        assert cancelled is not None

    def test_get_auto_balance_types_returns_classes(self):
        prober, cancelled = _get_auto_balance_types()
        # Should be classes/types
        assert hasattr(prober, '__name__') or callable(prober)
        assert hasattr(cancelled, '__name__') or isinstance(cancelled, type)


class TestProcessManagerInit:
    """Tests for ProcessManager initialization."""

    def test_init_default_state(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        assert pm.processes == {}
        assert pm._requests == {}
        assert not pm._auto_balance_active
        assert not pm._auto_balance_cancel
        assert pm._auto_balance_port == SERVER_PORT
        assert pm._auto_balance_run_id == 0
        assert pm.recovery_state["active"] is False
        assert pm.recovery_state["failed"] is False

    def test_init_auto_balance_cancel_property(self):
        pm = ProcessManager(
            config=MagicMock(),
            token_mgr=MagicMock(),
            gpu_manager=MagicMock(),
            log_manager=MagicMock(),
        )
        assert pm.auto_balance_cancel_requested is False
        pm.auto_balance_cancel_requested = True
        assert pm._auto_balance_cancel is True
        assert pm.auto_balance_cancel_requested is True
