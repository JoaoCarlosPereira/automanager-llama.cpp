"""Unit tests for OOMWatchdog OOM detection and recovery."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from schemas import GPUWeight, StartRequest
from process_manager import OOMWatchdog, SERVER_PORT


def _make_watchdog(process_manager=None, config_manager=None):
    process_manager = process_manager or MagicMock()
    process_manager._lock = threading.Lock()
    process_manager._processes = {SERVER_PORT: MagicMock()}
    process_manager._requests = {SERVER_PORT: None}
    process_manager.recovery_state = {
        "active": False,
        "failed": False,
        "message": "",
    }
    process_manager.auto_balance_active = False

    config_manager = config_manager or MagicMock()

    return OOMWatchdog(
        process_manager=process_manager,
        config_manager=config_manager,
        gpu_manager=MagicMock(),
        log_manager=MagicMock(),
    )


def _make_request(gpu_weights):
    return StartRequest(
        path="/models/test.gguf",
        gpu_weights=gpu_weights,
        context_size=4096,
        mmproj_path=None,
    )


@pytest.mark.parametrize(
    "line",
    [
        "CUDA out of memory: tried to allocate 2.00 GiB",
        "backend allocator: malloc failed",
        "torch runtime raised c10.Error while evaluating",
    ],
)
def test_oom_patterns_match_expected_errors(line):
    assert OOMWatchdog.OOM_PATTERNS.search(line)


@pytest.mark.parametrize(
    "line",
    [
        "server listening on 0.0.0.0:8085",
        "loaded model metadata successfully",
        "prompt processed in 120 ms",
        "GPU memory available: 24192 MB",
    ],
)
def test_oom_patterns_ignore_normal_lines(line):
    assert OOMWatchdog.OOM_PATTERNS.search(line) is None


def test_consecutive_oom_count_increments_within_timeout():
    watchdog = _make_watchdog()

    with patch("process_manager.time.time", return_value=1000.0):
        watchdog._handle_oom(SERVER_PORT)
    with patch("process_manager.time.time", return_value=1005.0):
        watchdog._handle_oom(SERVER_PORT)

    assert watchdog._consecutive_oom == 2
    assert watchdog._last_oom_time == 1005.0


def test_consecutive_oom_count_resets_after_timeout():
    watchdog = _make_watchdog()

    with patch("process_manager.time.time", return_value=1000.0):
        watchdog._handle_oom(SERVER_PORT)
    with patch("process_manager.time.time", return_value=1031.0):
        watchdog._handle_oom(SERVER_PORT)

    assert watchdog._consecutive_oom == 1
    assert watchdog._last_oom_time == 1031.0


def test_conservative_recovery_reduces_main_gpu_and_redistributes_weight():
    process_manager = MagicMock()
    config_manager = MagicMock()
    watchdog = _make_watchdog(process_manager, config_manager)
    request = _make_request(
        [
            GPUWeight(index=0, weight=80.0, name="main", active=True),
            GPUWeight(index=1, weight=10.0, name="secondary-1", active=True),
            GPUWeight(index=2, weight=10.0, name="secondary-2", active=True),
        ]
    )
    process_manager._requests = {SERVER_PORT: request}
    process_manager._processes = {SERVER_PORT: MagicMock()}

    with patch("process_manager.time.time", return_value=1000.0), patch(
        "process_manager.time.sleep", return_value=None
    ):
        watchdog._handle_oom(SERVER_PORT)

    assert [w.weight for w in request.gpu_weights] == pytest.approx(
        [70.0, 15.0, 15.0]
    )
    config_manager.update_model_settings.assert_called_once()
    saved_path, saved_settings = config_manager.update_model_settings.call_args.args
    assert saved_path == request.path
    assert saved_settings["gpu_weights"] == [
        {
            "index": 0,
            "weight": 70.0,
            "name": "main",
            "active": True,
            "is_main": False,
            "pinned": False,
            "device": "gpu",
        },
        {
            "index": 1,
            "weight": 15.0,
            "name": "secondary-1",
            "active": True,
            "is_main": False,
            "pinned": False,
            "device": "gpu",
        },
        {
            "index": 2,
            "weight": 15.0,
            "name": "secondary-2",
            "active": True,
            "is_main": False,
            "pinned": False,
            "device": "gpu",
        },
    ]


def test_fallback_after_three_consecutive_ooms_sets_active_gpus_to_50():
    process_manager = MagicMock()
    config_manager = MagicMock()
    watchdog = _make_watchdog(process_manager, config_manager)
    request = _make_request(
        [
            GPUWeight(index=0, weight=70.0, name="main", active=True),
            GPUWeight(index=1, weight=20.0, name="secondary", active=True),
            GPUWeight(index=2, weight=10.0, name="inactive", active=False),
        ]
    )
    process_manager._requests = {SERVER_PORT: request}
    process_manager._processes = {SERVER_PORT: MagicMock()}
    watchdog._consecutive_oom = 2
    watchdog._last_oom_time = 1000.0

    with patch("process_manager.time.time", return_value=1005.0), patch(
        "process_manager.time.sleep", return_value=None
    ):
        watchdog._handle_oom(SERVER_PORT)

    # In Python, modifying mutable objects in-place won't be reflected in request if new objects were created
    # but ProcessManager OOM logic for >= MAX_CONSECUTIVE_OOM creates a NEW list of new GPUWeight objects
    # and doesn't modify req.gpu_weights in-place.
    
    config_manager.update_model_settings.assert_called_once()
    final_weights = config_manager.update_model_settings.call_args.args[1]["gpu_weights"]
    assert final_weights[0]["weight"] == 50.0
    assert final_weights[1]["weight"] == 50.0
    assert final_weights[2]["weight"] == 0.0


def test_manual_gpu_override_preserves_weights_and_does_not_restart():
    process_manager = MagicMock()
    config_manager = MagicMock()
    watchdog = _make_watchdog(process_manager, config_manager)
    request = _make_request(
        [
            GPUWeight(index=0, weight=70.0, name="GPU0", active=True),
            GPUWeight(index=1, weight=30.0, name="GPU1", active=True),
        ]
    )
    request.manual_gpu_override = True
    process_manager._requests = {SERVER_PORT: request}
    process_manager._processes = {SERVER_PORT: MagicMock()}

    with patch("process_manager.time.time", return_value=1000.0):
        watchdog._handle_oom(SERVER_PORT)

    assert [w.weight for w in request.gpu_weights] == [70.0, 30.0]
    process_manager.start.assert_not_called()
    config_manager.update_model_settings.assert_called_once_with(
        request.path,
        {"last_failure": "OOM", "last_failure_time": 1000.0},
    )
