"""Unit tests for auto-balance weight planning and VRAM maximization helpers."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from auto_balance import (
    CPU_OFFLOAD_STEP,
    DEVICE_BUDGET_TOTAL,
    DEVICE_BUDGET_TOLERANCE,
    MIN_SPILL_GPU_WEIGHT,
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

THREE_GPU_HARDWARE = [
    {"index": 0, "name": "Tesla P100-PCIE-16GB", "vram": 16384},
    {"index": 1, "name": "Tesla P100-PCIE-16GB", "vram": 16384},
    {"index": 2, "name": "NVIDIA GeForce RTX 3090", "vram": 24576},
]
MAIN_GPU_INDEX = 2
SPILL_ORDER_3090_MAIN = [2, 0, 1]

# Fase 1 cascade (3090 principal): main max, demais GPUs com fatia mínima (10%).
CASCADE_GPU_ONLY_MAP = {2: 80, 0: 10, 1: 10}
CASCADE_CPU_OFFLOAD_MAP = {2: 72, 0: 9, 1: 9}
VALID_GPU_ONLY_CPU = 0
VALID_CPU_OFFLOAD_CPU = 10

# Estado inválido legado (reduce_main antigo) — regressão 110%.
LEGACY_INVERTED_GPU_MAP = {0: 43, 1: 47, 2: 10}

CPU_VALVE_ON = {"enabled": True, "pinned": False, "weight": 0}


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

    def test_cascade_spill_single_gpu_is_100_on_main(self):
        weights = AutoBalancePlanner.weights_for_cascade_spill([2, 0, 1], 1)
        assert weights == {2: 100}

    def test_cascade_spill_two_gpus_main_gets_90(self):
        weights = AutoBalancePlanner.weights_for_cascade_spill([2, 0, 1], 2)
        assert weights == {2: 90, 0: 10}

    def test_cascade_spill_three_gpus_main_gets_80(self):
        weights = AutoBalancePlanner.weights_for_cascade_spill([2, 0, 1], 3)
        assert weights == {2: 80, 0: 10, 1: 10}
        assert sum(weights.values()) == 100

    def test_cascade_spill_respects_target_total_with_cpu_budget(self):
        weights = AutoBalancePlanner.weights_for_cascade_spill(
            [2, 0, 1], 3, target_total=90
        )
        assert weights == {2: 70, 0: 10, 1: 10}
        assert sum(weights.values()) == 90

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
    assert TARGET_VRAM_PCT >= 95


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

    def test_algorithmic_gpu_map_uses_cascade_priority_fill(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        weight_map = prober._algorithmic_gpu_map(
            SPILL_ORDER_3090_MAIN,
            [0, 1, 2],
            {0: 16384, 1: 16384, 2: 24576},
            {},
            active_count=3,
            cpu_weight=0,
        )
        assert weight_map == CASCADE_GPU_ONLY_MAP
        assert weight_map[MAIN_GPU_INDEX] > weight_map[0]
        assert weight_map[MAIN_GPU_INDEX] > weight_map[1]

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

    def test_to_gpu_weights_includes_cpu_valve_at_zero(self):
        weights = AutoBalancePlanner.to_gpu_weights(
            [{"index": 0, "name": "GPU0"}],
            {0: 100},
            0,
            cpu_weight=0,
            cpu_valve_enabled=True,
        )
        cpu = next(w for w in weights if w.device == "cpu")
        assert cpu.weight == 0.0
        assert cpu.active is True
        ok, _ = AutoBalancePlanner.validate_device_budget_from_weights(weights)
        assert ok

    def test_scale_weight_map_targets_budget(self):
        scaled = AutoBalancePlanner.scale_weight_map({0: 60, 1: 40}, 70)
        assert sum(scaled.values()) == 70
        assert scaled[0] > scaled[1]


class TestCanonicalAutoBalanceStates:
    """Estados cascade: main max na Fase 1; offload CPU 10% escala GPUs para 90%."""

    @staticmethod
    def _assert_canonical_export(gpu_map, cpu_weight, *, cpu_valve_enabled=True):
        ok, err = AutoBalancePlanner.validate_device_budget(gpu_map, cpu_weight)
        assert ok, err

        weights = AutoBalancePlanner.to_gpu_weights(
            THREE_GPU_HARDWARE,
            gpu_map,
            MAIN_GPU_INDEX,
            cpu_weight=cpu_weight,
            cpu_valve_enabled=cpu_valve_enabled,
        )
        ok, err = AutoBalancePlanner.validate_device_budget_from_weights(weights)
        assert ok, err

        assert weights[0].weight == float(gpu_map.get(0, 0))
        assert weights[1].weight == float(gpu_map.get(1, 0))
        assert weights[2].weight == float(gpu_map.get(2, 0))
        assert weights[2].is_main is True
        assert gpu_map.get(MAIN_GPU_INDEX, 0) >= gpu_map.get(0, 0)
        assert gpu_map.get(MAIN_GPU_INDEX, 0) >= gpu_map.get(1, 0)

        cpu_entries = [w for w in weights if w.device == "cpu"]
        assert len(cpu_entries) == 1
        cpu = cpu_entries[0]
        assert cpu.weight == float(cpu_weight)
        assert cpu.active is (cpu_valve_enabled or cpu_weight > 0)
        assert sum(w.weight for w in weights if w.active) == pytest.approx(
            DEVICE_BUDGET_TOTAL, abs=DEVICE_BUDGET_TOLERANCE
        )

    def test_valid_state_gpu_only_cascade_phase1(self):
        """Sem offload: 3090=80%, P100s=10% cada (Fase 1 cascade)."""
        gpu_map = dict(CASCADE_GPU_ONLY_MAP)
        normalized, enforced_cpu = AutoBalancePlanner.enforce_device_budget(
            gpu_map, CPU_VALVE_ON
        )
        assert normalized == CASCADE_GPU_ONLY_MAP
        assert enforced_cpu == VALID_GPU_ONLY_CPU
        self._assert_canonical_export(normalized, enforced_cpu)

    def test_valid_state_cpu_offload_cascade_scaled(self):
        """Com offload CPU 10%: GPUs escaladas para 90% (72/9/9 + CPU 10%)."""
        gpu_map = dict(CASCADE_CPU_OFFLOAD_MAP)
        normalized, enforced_cpu = AutoBalancePlanner.enforce_device_budget(
            gpu_map, CPU_VALVE_ON
        )
        assert normalized == CASCADE_CPU_OFFLOAD_MAP
        assert enforced_cpu == VALID_CPU_OFFLOAD_CPU
        self._assert_canonical_export(normalized, enforced_cpu)

    def test_escalate_cpu_offload_from_cascade_produces_scaled_map(self):
        """Fase 1: CPU +10% a partir do cascade 80/10/10 -> 72/9/9 + CPU 10%."""
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        vram = {g["index"]: g["vram"] for g in THREE_GPU_HARDWARE}

        new_map, new_cpu = prober._escalate_cpu_offload(
            dict(CASCADE_GPU_ONLY_MAP),
            0,
            SPILL_ORDER_3090_MAIN,
            [0, 1, 2],
            vram,
            {},
        )

        assert new_map == CASCADE_CPU_OFFLOAD_MAP
        assert new_cpu == VALID_CPU_OFFLOAD_CPU
        self._assert_canonical_export(new_map, new_cpu)

    def test_scale_weight_map_derives_offload_from_cascade_reference(self):
        scaled = AutoBalancePlanner.scale_weight_map(
            CASCADE_GPU_ONLY_MAP,
            DEVICE_BUDGET_TOTAL - VALID_CPU_OFFLOAD_CPU,
        )
        assert scaled == CASCADE_CPU_OFFLOAD_MAP


class TestDeviceBudgetInvariants:
    """Production contract: active GPU + CPU weights must always sum to 100%."""

    CPU_CONFIG_ON = CPU_VALVE_ON
    CPU_CONFIG_OFF = {"enabled": False, "pinned": False, "weight": 0}

    @staticmethod
    def _assert_budget(gpu_map, cpu_weight):
        ok, err = AutoBalancePlanner.validate_device_budget(gpu_map, cpu_weight)
        assert ok, err

    def test_enforce_rejects_gpu100_plus_stale_cpu10(self):
        """Regression: mapa invertido legado + CPU 10% => 110%."""
        gpu_map = dict(LEGACY_INVERTED_GPU_MAP)
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            gpu_map, self.CPU_CONFIG_ON
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == 0
        assert normalized == gpu_map

    def test_enforce_cpu_offload_10_splits_gpu_to_90(self):
        scaled = AutoBalancePlanner.scale_weight_map(
            CASCADE_GPU_ONLY_MAP,
            DEVICE_BUDGET_TOTAL - VALID_CPU_OFFLOAD_CPU,
        )
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            scaled, self.CPU_CONFIG_ON
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == VALID_CPU_OFFLOAD_CPU
        assert normalized == CASCADE_CPU_OFFLOAD_MAP

    def test_enforce_cpu_disabled_scales_gpus_to_100(self):
        gpu_map = {0: 38, 1: 42, 2: 9}
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            gpu_map, self.CPU_CONFIG_OFF
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == 0
        assert sum(normalized.values()) == DEVICE_BUDGET_TOTAL

    def test_enforce_respects_pinned_cpu(self):
        gpu_map = {0: 80, 1: 20}
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            gpu_map,
            {"enabled": True, "pinned": True, "weight": 30},
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == 30
        assert sum(normalized.values()) == 70

    def test_validate_device_budget_from_weights_rejects_110(self):
        weights = [
            GPUWeight(index=0, weight=43, name="P100", active=True, device="gpu"),
            GPUWeight(index=1, weight=47, name="P100", active=True, device="gpu"),
            GPUWeight(index=2, weight=10, name="3090", active=True, device="gpu"),
            GPUWeight(index=-1, weight=10, name="CPU", active=True, device="cpu"),
        ]
        ok, err = AutoBalancePlanner.validate_device_budget_from_weights(weights)
        assert not ok
        assert "110" in err

    def test_production_scenario_phase2_reclaim_cpu_budget(self):
        """Main floor 10% + P100 split; phase 2 reclaims CPU -> GPU 100%, CPU 0."""
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        gpu_map = dict(CASCADE_GPU_ONLY_MAP)
        optimized, cpu_w = prober._finalize_cpu_split(gpu_map, self.CPU_CONFIG_ON)
        self._assert_budget(optimized, cpu_w)
        assert cpu_w == VALID_GPU_ONLY_CPU
        assert optimized == CASCADE_GPU_ONLY_MAP
        weights = AutoBalancePlanner.to_gpu_weights(
            THREE_GPU_HARDWARE,
            optimized,
            MAIN_GPU_INDEX,
            cpu_weight=cpu_w,
            cpu_valve_enabled=True,
        )
        ok, _ = AutoBalancePlanner.validate_device_budget_from_weights(weights)
        assert ok
        cpu = next(w for w in weights if w.device == "cpu")
        assert cpu.weight == float(VALID_GPU_ONLY_CPU)
        assert cpu.active is True

    def test_find_feasible_never_reduces_main_before_cpu(self):
        """Cláusula pétrea: com todas GPUs ativas, OOM vai para CPU — não reduz main."""
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        prober = AutoBalanceProber(
            process_manager, MagicMock(), MagicMock(), log_manager=MagicMock()
        )
        request = _make_request(
            [
                GPUWeight(index=0, weight=10, name="P0", active=True, device="gpu"),
                GPUWeight(index=1, weight=10, name="P1", active=True, device="gpu"),
                GPUWeight(
                    index=2, weight=80, name="3090", active=True,
                    is_main=True, device="gpu",
                ),
                GPUWeight(index=-1, weight=0, name="CPU", active=True, device="cpu"),
            ]
        )
        all_gpus = THREE_GPU_HARDWARE
        spill = SPILL_ORDER_3090_MAIN
        vram = {g["index"]: g["vram"] for g in all_gpus}
        cpu_config = CPU_VALVE_ON

        with patch.object(prober, "_probe_start", side_effect=["oom"] * 20):
            with patch.object(
                AutoBalancePlanner, "reduce_main_weight"
            ) as mock_reduce:
                prober._find_feasible_split(
                    request, all_gpus, MAIN_GPU_INDEX, spill, vram,
                    {}, [0, 1, 2], 0, cpu_config,
                )
                mock_reduce.assert_not_called()

    def test_find_feasible_ready_syncs_cpu_with_gpu_map(self):
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        gpu_manager = MagicMock()
        log_manager = MagicMock()
        log_manager.get_server_log_path.return_value = "/tmp/server.log"
        prober = AutoBalanceProber(
            process_manager, MagicMock(), gpu_manager, log_manager
        )

        request = _make_request(
            [
                GPUWeight(index=0, weight=43, name="P100", active=True, device="gpu"),
                GPUWeight(index=1, weight=47, name="P100", active=True, device="gpu"),
                GPUWeight(
                    index=2,
                    weight=10,
                    name="3090",
                    active=True,
                    is_main=True,
                    device="gpu",
                ),
                GPUWeight(index=-1, weight=0, name="CPU", active=True, device="cpu"),
            ]
        )
        all_gpus = [
            {"index": 0, "name": "P100", "vram": 16384},
            {"index": 1, "name": "P100", "vram": 16384},
            {"index": 2, "name": "3090", "vram": 24576},
        ]
        spill = AutoBalancePlanner.spill_order(2, [0, 1, 2])
        vram = {g["index"]: g["vram"] for g in all_gpus}
        cpu_config = AutoBalanceProber._cpu_config_from_request(request)

        with patch.object(prober, "_probe_start", return_value="ready"):
            feasible, _, _, cpu_w = prober._find_feasible_split(
                request,
                all_gpus,
                2,
                spill,
                vram,
                {},
                [0, 1, 2],
                0,
                cpu_config,
            )

        assert feasible is not None
        self._assert_budget(feasible, cpu_w)

    def test_probe_start_syncs_stale_cpu_weight(self):
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        gpu_manager = MagicMock()
        log_manager = MagicMock()
        prober = AutoBalanceProber(
            process_manager, MagicMock(), gpu_manager, log_manager
        )

        request = MagicMock()
        request.path = "/models/big.gguf"
        request.context_size = 205000
        request.parallel_slots = 1
        request.mmproj_path = None
        request.split_mode = "layer"
        request.batch_size = 16384
        request.thinking_enabled = True
        request.mtp_enabled = True
        request.mtp_draft_tokens = 4
        request.total_layers = 80

        all_gpus = [{"index": 0, "name": "GPU0"}]
        cpu_config = {"enabled": True, "pinned": False, "weight": 0}
        weight_map = {0: 100}

        with patch.object(prober, "_wait_for_outcome", return_value="ready"):
            prober._probe_start(
                request,
                weight_map,
                0,
                all_gpus,
                1,
                cpu_weight=10,
                cpu_config=cpu_config,
            )

        start_call = process_manager.start.call_args
        sent = start_call.kwargs["gpu_weights"]
        ok, err = AutoBalancePlanner.validate_device_budget_from_weights(sent)
        assert ok, err
        cpu = next(w for w in sent if w.device == "cpu")
        assert cpu.weight == 0.0

    def test_adjust_target_weight_delta_rejects_invalid_budget(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = AutoBalancePlanner.spill_order(0, [0, 1])
        weight_map = {0: 100, 1: 0}
        cpu_config = self.CPU_CONFIG_ON
        trial = prober._adjust_target_weight_for_maximize(
            weight_map, spill, 1, 20, {}, cpu_config
        )
        assert trial is None

    def test_apply_pins_scales_to_gpu_target_without_pins(self):
        scaled = AutoBalancePlanner.apply_pins(
            CASCADE_GPU_ONLY_MAP,
            {},
            SPILL_ORDER_3090_MAIN,
            [0, 1, 2],
            {0: 16384, 1: 16384, 2: 24576},
            target_total=90,
        )
        assert scaled == CASCADE_CPU_OFFLOAD_MAP

    def test_escalate_cpu_offload_maintains_budget(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        new_map, new_cpu = prober._escalate_cpu_offload(
            dict(CASCADE_GPU_ONLY_MAP),
            0,
            SPILL_ORDER_3090_MAIN,
            [0, 1, 2],
            {0: 16384, 1: 16384, 2: 24576},
            {},
        )
        assert new_map is not None
        self._assert_budget(new_map, new_cpu)
        assert new_cpu == CPU_OFFLOAD_STEP
        assert sum(new_map.values()) == DEVICE_BUDGET_TOTAL - CPU_OFFLOAD_STEP


def test_prober_raises_when_cancel_requested():
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = True
    prober = AutoBalanceProber(
        process_manager, MagicMock(), MagicMock(), MagicMock()
    )
    with pytest.raises(AutoBalanceCancelled):
        prober._raise_if_cancelled()
