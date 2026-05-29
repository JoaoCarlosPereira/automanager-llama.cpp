"""Unit tests for auto-balance weight planning and VRAM maximization helpers."""

import threading
from unittest.mock import MagicMock

import pytest

from auto_balance import (
    TARGET_VRAM_PCT,
    AutoBalancePlanner,
    AutoBalanceProber,
    FAILURE_HARDWARE_CAPACITY,
    MIN_MAIN_WEIGHT,
    READY_PATTERNS,
)
from schemas import GPUWeight
from tests.unit.test_oom_watchdog import _make_request, _make_watchdog


class TestAutoBalancePlanner:
    def test_spill_order_puts_main_first(self):
        order = AutoBalancePlanner.spill_order(2, [0, 1, 2])
        assert order == [2, 0, 1]

    def test_weights_single_gpu_is_100_percent(self):
        weights = AutoBalancePlanner.weights_for_active_count(
            [0], {0: 24000}, 1
        )
        assert weights == {0: 100}

    def test_weights_two_gpus_sum_to_100(self):
        weights = AutoBalancePlanner.weights_for_active_count(
            [0, 1], {0: 24000, 1: 16000}, 2
        )
        assert sum(weights.values()) == 100
        assert weights[0] > weights[1]

    def test_set_target_weight_increases_main_from_secondary(self):
        base = {0: 70, 1: 30}
        spill = [0, 1]
        trial = AutoBalancePlanner.set_target_weight(base, spill, 0, 85)
        assert trial == {0: 85, 1: 15}

    def test_set_target_weight_cannot_steal_from_earlier_spill_gpu(self):
        base = {0: 80, 1: 20}
        spill = [0, 1]
        trial = AutoBalancePlanner.set_target_weight(base, spill, 1, 40)
        assert trial is None

    def test_distribute_unpinned_respects_pinned_sum(self):
        pinned = {0: 60}
        result = AutoBalancePlanner.distribute_unpinned(
            pinned, [1, 2], {0: 24000, 1: 16000, 2: 8000}, [0, 1, 2]
        )
        assert result is not None
        assert result[0] == 60
        assert sum(result.values()) == 100

    def test_set_target_weight_skips_pinned_donor(self):
        base = {0: 50, 1: 30, 2: 20}
        spill = [0, 1, 2]
        pinned = {2: 20}
        trial = AutoBalancePlanner.set_target_weight(
            base, spill, 0, 70, pinned_map=pinned
        )
        assert trial == {0: 70, 1: 10, 2: 20}

    def test_max_weight_for_gpu_returns_pinned_value(self):
        assert AutoBalancePlanner.max_weight_for_gpu(
            {0: 40, 1: 60}, [0, 1], 1, {1: 60}
        ) == 60

    def test_max_weight_for_secondary_respects_main_lock(self):
        weights = {0: 85, 1: 15}
        spill = [0, 1]
        assert AutoBalancePlanner.max_weight_for_gpu(weights, spill, 1) == 15

    def test_reduce_main_returns_none_at_floor(self):
        weights = AutoBalancePlanner.reduce_main_weight(
            {0: MIN_MAIN_WEIGHT, 1: 90},
            [0, 1],
            {0: 24000, 1: 16000},
        )
        assert weights is None


@pytest.mark.parametrize(
    "line",
    [
        "server listening on 0.0.0.0:8085",
        "HTTP server listening on port 8085",
        "main loop started",
    ],
)
def test_ready_patterns(line):
    assert READY_PATTERNS.search(line)


def test_target_vram_threshold():
    assert TARGET_VRAM_PCT >= 90


def test_build_hardware_capacity_failure_includes_context():
    request = _make_request(
        [GPUWeight(index=0, weight=100, name="RTX", active=True, is_main=True)]
    )
    request.path = "/models/huge-model.gguf"
    request.context_size = 32768
    request.parallel_slots = 2

    msg, failure = AutoBalanceProber.build_hardware_capacity_failure(
        request,
        [{"index": 0, "name": "RTX 3090", "vram": 24576}],
        [0],
        {0: 24576},
    )

    assert "huge-model.gguf" in msg
    assert failure["code"] == FAILURE_HARDWARE_CAPACITY
    assert failure["total_vram_mb"] == 24576
    assert failure["context_size"] == 32768
    assert len(failure["suggestions"]) >= 2


def test_oom_watchdog_skips_during_auto_balance():
    process_manager = MagicMock()
    process_manager._lock = threading.Lock()
    process_manager._last_request = _make_request(
        [GPUWeight(index=0, weight=100, name="a", active=True)]
    )
    process_manager.auto_balance_active = True
    config_manager = MagicMock()
    watchdog = _make_watchdog(process_manager, config_manager)

    watchdog._handle_oom()

    config_manager.update_model_settings.assert_not_called()
    process_manager.start.assert_not_called()
