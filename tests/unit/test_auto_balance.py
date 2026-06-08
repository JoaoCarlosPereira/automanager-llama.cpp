"""Unit tests for auto-balance weight planning and VRAM maximization helpers."""

import threading
from unittest.mock import MagicMock

import pytest

from auto_balance import (
    CPU_OFFLOAD_STEP,
    TARGET_VRAM_PCT,
    AutoBalanceCancelled,
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


class TestAutoBalanceCpu:
    def test_should_add_cpu_false_when_disabled(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert prober._should_add_cpu([0, 1], {0: 50, 1: 50}, False) is False

    def test_should_add_cpu_true_when_all_gpus_active(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert prober._should_add_cpu([0, 1], {0: 60, 1: 40}, True) is True

    def test_should_add_cpu_false_when_gpu_missing(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert prober._should_add_cpu([0, 1], {0: 100}, True) is False

    def test_should_add_cpu_ignores_unselected_gpus(self):
        """Only selected GPUs matter — not every card detectada."""
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert prober._should_add_cpu([0, 1], {0: 50, 1: 50}, True) is True

    def test_algorithmic_gpu_map_ignores_ui_percentages(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        weight_map = prober._algorithmic_gpu_map(
            [0, 1],
            [0, 1],
            {0: 24000, 1: 16000},
            {},
            active_count=2,
            cpu_weight=0,
        )
        assert sum(weight_map.values()) == 100
        assert weight_map[0] > weight_map[1]

    def test_cpu_config_from_request_disabled(self):
        request = _make_request(
            [GPUWeight(index=0, weight=100, name="GPU0", active=True)]
        )
        cfg = AutoBalanceProber._cpu_config_from_request(request)
        assert cfg["enabled"] is False

    def test_cpu_config_from_request_enabled(self):
        request = _make_request(
            [
                GPUWeight(index=0, weight=70, name="GPU0", active=True),
                GPUWeight(
                    index=-1, weight=30, name="CPU", active=True, device="cpu"
                ),
            ]
        )
        cfg = AutoBalanceProber._cpu_config_from_request(request)
        assert cfg == {"enabled": True, "pinned": False, "weight": 0}

    def test_cpu_config_from_request_enabled_disables_pinning(self):
        """CPU weight is dynamic — no pinning in new design."""
        request = _make_request(
            [
                GPUWeight(index=0, weight=70, name="GPU0", active=True),
                GPUWeight(
                    index=-1,
                    weight=25,
                    name="CPU",
                    active=True,
                    pinned=True,
                    device="cpu",
                ),
            ]
        )
        cfg = AutoBalanceProber._cpu_config_from_request(request)
        # Pinned is always False now — weight is calculated by LoadDistributor
        assert cfg == {"enabled": True, "pinned": False, "weight": 0}

    def test_finalize_cpu_split_no_cap(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        gpu_map, cpu_w = prober._finalize_cpu_split(
            {0: 20, 1: 10},
            {"enabled": True, "pinned": False, "weight": 0},
        )
        # No 70% cap — raw spill-over (100 - 30 = 70)
        assert cpu_w == 70
        assert sum(gpu_map.values()) == 30

    def test_finalize_cpu_split_respects_pinned_cpu(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        gpu_map, cpu_w = prober._finalize_cpu_split(
            {0: 80, 1: 20},
            {"enabled": True, "pinned": True, "weight": 30},
        )
        assert cpu_w == 30
        assert sum(gpu_map.values()) == 70

    def test_resolve_probe_cpu_weight_is_minimum_spillover(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        cpu_w = prober._resolve_probe_cpu_weight(
            {0: 85, 1: 10},
            {"enabled": True, "pinned": False, "weight": 0},
        )
        assert cpu_w == 5

    def test_resolve_probe_cpu_weight_disabled_returns_zero(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        cpu_w = prober._resolve_probe_cpu_weight(
            {0: 60, 1: 40},
            {"enabled": False, "pinned": False, "weight": 0},
        )
        assert cpu_w == 0

    def test_phase2_gpu_target_allows_higher_main_than_phase1_cpu_budget(self):
        """Phase 2 can reclaim CPU budget to raise the main GPU weight."""
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = AutoBalancePlanner.spill_order(0, [0, 1])
        weight_map = {0: 75, 1: 10}
        cpu_config = {"enabled": True, "pinned": False, "weight": 0}
        trial = prober._adjust_target_weight_for_maximize(
            weight_map, spill, 0, 90, {}, cpu_config
        )
        assert trial is not None
        assert trial[0] == 90
        assert sum(trial.values()) == 100
        assert prober._resolve_probe_cpu_weight(trial, cpu_config) == 0

    def test_max_gpu_weight_for_maximize_includes_cpu_budget(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = AutoBalancePlanner.spill_order(0, [0, 1])
        weight_map = {0: 70, 1: 10}
        cpu_config = {"enabled": True, "pinned": False, "weight": 0}
        hi = prober._max_gpu_weight_for_maximize(
            weight_map, spill, 0, {}, cpu_config
        )
        capped = AutoBalancePlanner.max_weight_for_gpu(
            weight_map, spill, 0, {}, target_total=80
        )
        assert hi > capped

    def test_adjust_target_weight_reclaims_cpu_for_last_spill_gpu(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = AutoBalancePlanner.spill_order(0, [0, 1])
        weight_map = {0: 90, 1: 0}
        cpu_config = {"enabled": True, "pinned": False, "weight": 0}
        trial = prober._adjust_target_weight_for_maximize(
            weight_map, spill, 1, 5, {}, cpu_config
        )
        assert trial == {0: 90, 1: 5}
        assert prober._resolve_probe_cpu_weight(trial, cpu_config) == 5

    def test_to_gpu_weights_includes_cpu_entry(self):
        weights = AutoBalancePlanner.to_gpu_weights(
            [{"index": 0, "name": "GPU0"}],
            {0: 70},
            0,
            cpu_weight=30,
        )
        cpu = next(w for w in weights if w.device == "cpu")
        assert cpu.index == -1
        assert cpu.weight == 30.0
        assert cpu.active is True

    def test_scale_weight_map_targets_budget(self):
        scaled = AutoBalancePlanner.scale_weight_map({0: 60, 1: 40}, 70)
        assert sum(scaled.values()) == 70
        assert scaled[0] > scaled[1]


def test_prober_raises_when_cancel_requested():
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = True
    prober = AutoBalanceProber(
        process_manager, MagicMock(), MagicMock(), MagicMock()
    )
    with pytest.raises(AutoBalanceCancelled):
        prober._raise_if_cancelled()
