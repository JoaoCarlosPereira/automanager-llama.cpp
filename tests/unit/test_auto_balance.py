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

# Fase 1 cascade (3090 principal): main max, demais GPUs com fatia mínima por VRAM.
CASCADE_GPU_ONLY_MAP = {2: 80, 0: 10, 1: 10}
# CPU +10%: tira da principal por último; todas as GPUs selecionadas mantêm fatia mínima.
CASCADE_CPU_OFFLOAD_MAP = {2: 70, 0: 10, 1: 10}
VALID_GPU_ONLY_CPU = 0
VALID_CPU_OFFLOAD_CPU = 10

# Estado inválido legado (reduce_main antigo) — regressão 110%.
LEGACY_INVERTED_GPU_MAP = {0: 43, 1: 47, 2: 10}

CPU_VALVE_ON = {"enabled": True, "pinned": False, "weight": 0}
CPU_VALVE_ON_SPILL = {**CPU_VALVE_ON, "cpu_spill_allowed": True}


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
        vram = {0: 16384, 1: 16384, 2: 24576}
        weights = AutoBalancePlanner.weights_for_cascade_spill(
            [2, 0, 1], 3, vram
        )
        assert weights == {2: 80, 0: 10, 1: 10}
        assert sum(weights.values()) == 100

    def test_cascade_spill_secondary_flat_min_slice(self):
        """Secundárias recebem fatia mínima plana — nunca split proporcional.

        VRAM bem desigual entre as secundárias não deve enviesar o template:
        a main fica com a maior fatia e cada secundária com MIN_SPILL_GPU_WEIGHT.
        O preenchimento real por VRAM acontece na Fase 2.
        """
        vram = {0: 24000, 1: 8000, 2: 24576, 3: 8000}
        weights = AutoBalancePlanner.weights_for_cascade_spill(
            [2, 0, 1, 3], 4, vram
        )
        assert weights[2] == 70
        assert weights[2] > max(weights[0], weights[1], weights[3])
        # Sem proporcionalidade: todas as secundárias com a mesma fatia mínima.
        assert weights[0] == weights[1] == weights[3] == MIN_SPILL_GPU_WEIGHT
        assert sum(weights.values()) == 100

    def test_shift_gpu_budget_takes_from_last_spill_gpu_first(self):
        shifted = AutoBalancePlanner.shift_gpu_budget_for_cpu(
            {2: 80, 0: 10, 1: 10},
            [2, 0, 1],
            {},
            90,
        )
        assert shifted == {2: 80, 0: 10, 1: 0}

    def test_shift_gpu_budget_preserves_all_selected_gpus(self):
        shifted = AutoBalancePlanner.shift_gpu_budget_for_cpu(
            {2: 80, 0: 10, 1: 10},
            SPILL_ORDER_3090_MAIN,
            {},
            90,
            [0, 1, 2],
        )
        assert shifted == {2: 70, 0: 10, 1: 10}

    def test_shift_gpu_budget_main_is_last_resort(self):
        shifted = AutoBalancePlanner.shift_gpu_budget_for_cpu(
            {2: 80, 0: 10, 1: 0},
            [2, 0, 1],
            {},
            70,
        )
        assert shifted.get(0, 0) == 0
        assert shifted.get(1, 0) == 0
        assert shifted[2] == 70

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

    def test_model_weights_mb_from_disk(self, tmp_path):
        model = tmp_path / "test.gguf"
        model.write_bytes(b"\0" * (50 * 1024 * 1024))
        assert AutoBalancePlanner.model_weights_mb_from_disk(str(model)) == 50
        assert AutoBalancePlanner.model_weights_mb_from_disk("/missing.gguf") is None

    def test_estimate_model_vram_uses_disk_file_size(self, tmp_path):
        model = tmp_path / "qwen-49b.gguf"
        model.write_bytes(b"\0" * (100 * 1024 * 1024))
        est = AutoBalancePlanner.estimate_model_vram_mb(str(model), 65536, 1)
        weights_mb = est["weights_mb"]
        kv_mb = est["kv_cache_mb"]
        total = est["total_mb"]
        assert total >= 100 + 6553  # weights + ctx overhead (65536*0.1)
        assert kv_mb == 6553
        # runtime_overhead = max(256, 5% de 100) = max(256, 5) = 256
        assert weights_mb == 100 + 256  # disk + runtime_overhead
        assert total == weights_mb + kv_mb

    def test_plan_min_gpu_count_for_large_model(self):
        spill = SPILL_ORDER_3090_MAIN
        vram = {0: 16384, 1: 16384, 2: 24576}
        assert AutoBalancePlanner.plan_min_gpu_count(spill, vram, 4 * 1024) == 1
        assert AutoBalancePlanner.plan_min_gpu_count(spill, vram, 30 * 1024) == 2
        assert AutoBalancePlanner.plan_min_gpu_count(spill, vram, 80 * 1024) == 3

    def test_estimate_cpu_spill_weight_gpu_only_when_fits(self):
        total = 24576 + 16384 + 16384
        assert AutoBalancePlanner.estimate_cpu_spill_weight(total, 50 * 1024) == 0

    def test_estimate_cpu_spill_weight_when_model_exceeds_vram(self):
        total = 24576 + 16384 + 16384
        cpu = AutoBalancePlanner.estimate_cpu_spill_weight(total, 70 * 1024)
        assert 0 < cpu <= 90

    def test_align_cpu_weight_step(self):
        assert AutoBalancePlanner.align_cpu_weight_step(0) == 0
        assert AutoBalancePlanner.align_cpu_weight_step(1) == 10
        assert AutoBalancePlanner.align_cpu_weight_step(22) == 30
        assert AutoBalancePlanner.align_cpu_weight_step(95) == 90


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
            CPU_VALVE_ON_SPILL,
            cpu_spill_allowed=True,
        )
        # No 70% cap — raw spill-over (100 - 30 = 70)
        assert cpu_w == 70
        assert sum(gpu_map.values()) == 30

    def test_finalize_cpu_split_valve_on_without_spill_scales_gpu(self):
        """Válvula CPU ligada não preenche gap até offload ser confirmado."""
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        gpu_map, cpu_w = prober._finalize_cpu_split(
            {2: 82, 0: 14, 1: 0},
            CPU_VALVE_ON,
            cpu_spill_allowed=False,
            spill_order=SPILL_ORDER_3090_MAIN,
            selected_indices=[0, 2],
        )
        assert cpu_w == 0
        assert sum(gpu_map.values()) == DEVICE_BUDGET_TOTAL
        assert gpu_map[2] >= 82
        assert gpu_map[0] >= 14

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

    def test_resolve_probe_cpu_weight_without_spill_scales_gpu(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        cpu_w = prober._resolve_probe_cpu_weight(
            {0: 85, 1: 10},
            CPU_VALVE_ON,
            cpu_spill_allowed=False,
            spill_order=[0, 1],
            selected_indices=[0, 1],
        )
        assert cpu_w == 0

    def test_resolve_probe_cpu_weight_with_spill_is_minimum_spillover(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        cpu_w = prober._resolve_probe_cpu_weight(
            {0: 85, 1: 10},
            CPU_VALVE_ON_SPILL,
            cpu_spill_allowed=True,
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
        cpu_config = {**CPU_VALVE_ON, "cpu_spill_allowed": True}
        trial = prober._adjust_target_weight_for_maximize(
            weight_map, spill, 0, 90, {}, cpu_config
        )
        assert trial is not None
        assert trial[0] == 90
        assert sum(trial.values()) == 100
        assert prober._resolve_probe_cpu_weight(
            trial, cpu_config, cpu_spill_allowed=True
        ) == 0

    def test_max_gpu_weight_for_maximize_includes_cpu_budget(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = AutoBalancePlanner.spill_order(0, [0, 1])
        weight_map = {0: 70, 1: 10}
        cpu_config = CPU_VALVE_ON_SPILL
        hi = prober._max_gpu_weight_for_maximize(
            weight_map, spill, 0, {}, cpu_config, selected_indices=[0, 1]
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
        cpu_config = CPU_VALVE_ON_SPILL
        trial = prober._adjust_target_weight_for_maximize(
            weight_map, spill, 1, 5, {}, cpu_config, selected_indices=[0, 1]
        )
        assert trial == {0: 90, 1: 5}
        assert prober._resolve_probe_cpu_weight(
            trial, cpu_config, cpu_spill_allowed=True
        ) == 5

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
        """Com offload CPU 10%: principal cede 10%%, secundárias mantêm 10%% cada."""
        gpu_map = dict(CASCADE_CPU_OFFLOAD_MAP)
        normalized, enforced_cpu = AutoBalancePlanner.enforce_device_budget(
            gpu_map, CPU_VALVE_ON_SPILL
        )
        assert normalized == CASCADE_CPU_OFFLOAD_MAP
        assert enforced_cpu == VALID_CPU_OFFLOAD_CPU
        self._assert_canonical_export(normalized, enforced_cpu)

    def test_escalate_cpu_offload_from_cascade_produces_scaled_map(self):
        """Fase 1: CPU +10% a partir do cascade 80/10/10 -> 70/10/10 + CPU 10%."""
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

    def test_shift_gpu_budget_derives_offload_from_cascade_reference(self):
        shifted = AutoBalancePlanner.shift_gpu_budget_for_cpu(
            CASCADE_GPU_ONLY_MAP,
            SPILL_ORDER_3090_MAIN,
            {},
            DEVICE_BUDGET_TOTAL - VALID_CPU_OFFLOAD_CPU,
            [0, 1, 2],
        )
        assert shifted == CASCADE_CPU_OFFLOAD_MAP


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
        shifted = AutoBalancePlanner.shift_gpu_budget_for_cpu(
            CASCADE_GPU_ONLY_MAP,
            SPILL_ORDER_3090_MAIN,
            {},
            DEVICE_BUDGET_TOTAL - VALID_CPU_OFFLOAD_CPU,
            [0, 1, 2],
        )
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            shifted, CPU_VALVE_ON_SPILL
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == VALID_CPU_OFFLOAD_CPU
        assert normalized == CASCADE_CPU_OFFLOAD_MAP

    def test_enforce_valve_on_without_spill_never_adds_cpu_gap(self):
        gpu_map = {2: 82, 0: 14, 1: 0}
        normalized, cpu_w = AutoBalancePlanner.enforce_device_budget(
            gpu_map,
            CPU_VALVE_ON,
            spill_order=SPILL_ORDER_3090_MAIN,
            selected_indices=[0, 2],
        )
        self._assert_budget(normalized, cpu_w)
        assert cpu_w == 0
        assert sum(normalized.values()) == DEVICE_BUDGET_TOTAL

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


class TestBudgetSelectedIndices:
    def test_single_gpu_probe_not_inflated_to_cascade(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = SPILL_ORDER_3090_MAIN
        budget = prober._budget_selected_indices({2: 100}, spill, [0, 1, 2])
        assert budget == [2]

        synced, cpu_w = AutoBalancePlanner.enforce_device_budget(
            {2: 100},
            CPU_VALVE_ON,
            spill_order=spill,
            selected_indices=budget,
        )
        assert synced == {2: 100}
        assert cpu_w == 0

    def test_two_gpu_trial_keeps_only_active_pair(self):
        prober = AutoBalanceProber(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        spill = SPILL_ORDER_3090_MAIN
        weight_map = {2: 90, 0: 10, 1: 0}
        budget = prober._budget_selected_indices(weight_map, spill, [0, 1, 2])
        assert budget == [2, 0]


class TestPhase2MaximizeFloors:
    def test_maximize_floors_allow_secondary_down_to_zero(self):
        spill = SPILL_ORDER_3090_MAIN
        weight_map = {2: 90, 0: 10}
        floors = AutoBalancePlanner.gpu_weight_floors_for_maximize(
            weight_map, spill
        )
        assert floors[2] == MIN_MAIN_WEIGHT
        assert floors[0] == 0

    def test_set_target_weight_can_reach_93_7_split(self):
        spill = SPILL_ORDER_3090_MAIN
        weight_map = {2: 90, 0: 10}
        floors = AutoBalancePlanner.gpu_weight_floors_for_maximize(
            weight_map, spill
        )
        trial = AutoBalancePlanner.set_target_weight(
            weight_map,
            spill,
            2,
            93,
            {},
            target_total=100,
            weight_floors=floors,
        )
        assert trial == {2: 93, 0: 7}

    def test_phase2_maximizes_each_gpu_in_spill_order(self):
        """Fase 2 enche cada GPU na ordem de prioridade — não só a main.

        Regressão da cláusula pétrea: antes, ``_maximize_vram_per_gpu`` só tinha
        a main em ``active_ordered``; agora itera todas as GPUs ativas em ordem
        de spill. Espionamos o ajuste por-alvo (retornando None p/ encerrar a
        busca binária de imediato) e conferimos que a main é o 1º alvo e que as
        secundárias também são maximizadas.
        """
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        prober = AutoBalanceProber(
            process_manager, MagicMock(), MagicMock(), MagicMock()
        )
        spill = SPILL_ORDER_3090_MAIN  # [2, 0, 1]
        weight_map = {2: 80, 0: 10, 1: 10}
        gpu_only = {
            "enabled": False, "pinned": False, "weight": 0,
            "cpu_spill_allowed": False,
        }
        targets = []

        def spy(_wm, _spill, target_idx, *args, **kwargs):
            targets.append(target_idx)
            return None  # encerra a busca binária deste alvo de imediato

        with patch.object(
            prober, "_adjust_target_weight_for_maximize", side_effect=spy
        ):
            prober._maximize_vram_per_gpu(
                MagicMock(),
                THREE_GPU_HARDWARE,
                MAIN_GPU_INDEX,
                spill,
                weight_map,
                {0: 16384, 1: 16384, 2: 24576},
                {},
                0,
                gpu_only,
                0,
                active_indices=[0, 1, 2],
            )

        assert targets, "Fase 2 não tentou maximizar nenhuma GPU"
        assert targets[0] == MAIN_GPU_INDEX  # main tem prioridade absoluta
        # Secundárias agora também são maximizadas (antes: só a main).
        assert {MAIN_GPU_INDEX, 0, 1} <= set(targets)
        # A main é sempre tentada antes de qualquer secundária.
        assert targets.index(MAIN_GPU_INDEX) < targets.index(0)
        assert targets.index(0) < targets.index(1)


class TestTrimTrailingSpillGpus:
    def test_removes_trailing_gpu_when_probe_ready(self):
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        prober = AutoBalanceProber(
            process_manager, MagicMock(), MagicMock(), MagicMock()
        )
        spill = SPILL_ORDER_3090_MAIN
        weight_map = dict(CASCADE_GPU_ONLY_MAP)
        all_gpus = THREE_GPU_HARDWARE
        cpu_config = CPU_VALVE_ON

        with patch.object(
            prober, "_probe_start", side_effect=["ready", "oom"]
        ):
            trimmed, _, cpu_w = prober._trim_trailing_spill_gpus(
                _make_request([]),
                all_gpus,
                MAIN_GPU_INDEX,
                spill,
                weight_map,
                {},
                [0, 1, 2],
                0,
                cpu_config,
                0,
            )

        assert trimmed == {2: 90, 0: 10, 1: 0}
        assert cpu_w == 0

    def test_keeps_trailing_gpu_when_probe_oom(self):
        process_manager = MagicMock()
        process_manager.auto_balance_cancel_requested = False
        prober = AutoBalanceProber(
            process_manager, MagicMock(), MagicMock(), MagicMock()
        )
        spill = SPILL_ORDER_3090_MAIN
        weight_map = dict(CASCADE_GPU_ONLY_MAP)
        all_gpus = THREE_GPU_HARDWARE
        cpu_config = CPU_VALVE_ON

        with patch.object(prober, "_probe_start", return_value="oom"):
            trimmed, _, _ = prober._trim_trailing_spill_gpus(
                _make_request([]),
                all_gpus,
                MAIN_GPU_INDEX,
                spill,
                weight_map,
                {},
                [0, 1, 2],
                0,
                cpu_config,
                0,
            )

        assert trimmed == CASCADE_GPU_ONLY_MAP


class TestCpuNotDominant:
    """CPU weight must not exceed total GPU weight."""

    def test_valid_equal_gpu_and_cpu(self):
        """GPU sum == CPU weight is valid (boundary case)."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 50, 1: 50}, 100
        )
        assert ok
        assert err == ""

    def test_valid_gpu_exceeds_cpu(self):
        """GPU sum > CPU weight is valid."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 70, 1: 30}, 50
        )
        assert ok
        assert err == ""

    def test_valid_gpu_only_no_cpu(self):
        """GPU-only configuration is valid."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 100}, 0
        )
        assert ok
        assert err == ""

    def test_valid_cpu_zero_with_gpus(self):
        """CPU weight 0 is always valid when GPUs active."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 50, 1: 50}, 0
        )
        assert ok
        assert err == ""

    def test_rejected_cpu_exceeds_gpus(self):
        """CPU weight > GPU sum is rejected."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 30, 1: 20}, 60
        )
        assert not ok
        assert "CPU=60% > GPU total=50%" in err

    def test_rejected_cpu_only_no_gpu(self):
        """CPU with no GPUs is rejected."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {}, 100
        )
        assert not ok
        assert "nenhuma GPU ativa" in err

    def test_rejected_cpu_dominant_single_gpu(self):
        """Single GPU with CPU > GPU weight is rejected."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 40}, 60
        )
        assert not ok
        assert "CPU=60% > GPU total=40%" in err

    def test_valid_boundary_cpu_equals_gpu(self):
        """CPU == GPU is the boundary — valid."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 50}, 50
        )
        assert ok
        assert err == ""

    def test_valid_cascade_with_cpu_offload(self):
        """Cascade 80/10/10 + CPU 10 is valid (100 > 10)."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {2: 80, 0: 10, 1: 10}, 10
        )
        assert ok
        assert err == ""

    def test_rejected_cascade_heavy_cpu(self):
        """Cascade 80/10/10 + CPU 100 is rejected (100 > 100)."""
        # Actually 100 == 100, so it's valid (boundary)
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {2: 80, 0: 10, 1: 10}, 101
        )
        assert not ok
        assert "CPU=101% > GPU total=100%" in err

    def test_rejected_high_cpu_weight(self):
        """High CPU weight with low GPU weights is rejected."""
        ok, err = AutoBalancePlanner.validate_cpu_not_dominant(
            {0: 20, 1: 10}, 50
        )
        assert not ok
        assert "CPU=50% > GPU total=30%" in err


class TestDiscoverPhaseHandoff:
    """End-to-end discover(): Phase 3 must continue from Phase 2's result and
    the CPU valve must stay unmarked while every probe is GPU-only."""

    class _PMStub:
        """Process-manager double that records every progress (recovery) state."""

        def __init__(self):
            self.auto_balance_cancel_requested = False
            self._lock = threading.Lock()
            self._current_process = None
            self.recovery_states = []

        @property
        def recovery_state(self):
            return self.recovery_states[-1] if self.recovery_states else None

        @recovery_state.setter
        def recovery_state(self, value):
            self.recovery_states.append(value)

        def stop(self):
            pass

        def start(self, **kwargs):
            pass

    def _build_prober(self, ready_rule, vram_pct_for_main):
        """Wire a prober whose probes/metrics derive from the live trial map.

        *ready_rule(weight_map) -> bool* decides ready vs OOM.
        *vram_pct_for_main(main_weight) -> float* feeds Phase 2/3 settle reads.
        """
        process_manager = self._PMStub()
        gpu_manager = MagicMock()
        gpu_manager.detect_gpus.return_value = THREE_GPU_HARDWARE
        log_manager = MagicMock()
        log_manager.get_server_log_path.return_value = "/tmp/server.log"
        prober = AutoBalanceProber(
            process_manager, MagicMock(), gpu_manager, log_manager
        )

        state = {"map": {MAIN_GPU_INDEX: 100}, "probes": []}

        def fake_probe(request, weight_map, *args, **kwargs):
            state["map"] = dict(weight_map)
            state["probes"].append(dict(weight_map))
            return "ready" if ready_rule(weight_map) else "oom"

        def fake_metrics():
            main_w = state["map"].get(MAIN_GPU_INDEX, 0)
            gpus = []
            for g in THREE_GPU_HARDWARE:
                if g["index"] == MAIN_GPU_INDEX:
                    pct = vram_pct_for_main(main_w)
                else:
                    pct = float(state["map"].get(g["index"], 0))
                gpus.append({
                    "index": g["index"],
                    "vram_pct": pct,
                    "mem_used": g["vram"] * pct / 100.0,
                    "mem_total": g["vram"],
                    "util": 0,
                })
            return {"gpus": gpus, "cpu": 0}

        gpu_manager.get_metrics.side_effect = fake_metrics
        prober._probe_start = MagicMock(side_effect=fake_probe)
        return prober, state, process_manager

    def test_phase3_continues_from_phase2_and_cpu_unmarked(self):
        # Model fits on GPU2 + GPU0 only; GPU2 must carry <= 92%.
        def ready_rule(weight_map):
            return weight_map.get(MAIN_GPU_INDEX, 0) <= 92

        def vram_pct_for_main(main_weight):
            # 92% weight -> ~96% VRAM, lands inside the [95, 99] target window.
            return min(99.0, float(main_weight) + 4.0)

        prober, state, pm = self._build_prober(ready_rule, vram_pct_for_main)
        request = _make_request(
            [
                GPUWeight(index=0, weight=10, name="P100", active=True, device="gpu"),
                GPUWeight(index=1, weight=10, name="P100", active=True, device="gpu"),
                GPUWeight(
                    index=2, weight=80, name="3090", active=True,
                    is_main=True, device="gpu",
                ),
                GPUWeight(index=-1, weight=0, name="CPU", active=True, device="cpu"),
            ]
        )

        with patch.object(AutoBalancePlanner, "model_weights_mb_from_disk",
                          return_value=21109), \
             patch("auto_balance.time.sleep", return_value=None):
            ok, weights, msg, failure = prober.discover(request)

        assert ok, msg
        by_idx = {w.index: w for w in weights}
        # Phase 3 kept Phase 2's maximized 2-GPU split — NOT a fresh 3-GPU cascade.
        assert by_idx[2].weight == 92.0
        assert by_idx[0].weight == 8.0
        assert by_idx[1].weight == 0.0
        assert not by_idx[1].active
        # GPU1 was dropped in Phase 2 and never re-probed in Phase 3.
        assert all(p.get(1, 0) == 0 for p in state["probes"][-5:])

        # Issue 2: during every GPU-only progress update the CPU valve must show
        # unmarked (it never carried load in this run).
        saw_progress_cpu = False
        for prog in pm.recovery_states:
            for entry in prog.get("gpu_weights", []):
                if entry["device"] == "cpu":
                    saw_progress_cpu = True
                    assert entry["weight"] == 0.0 and not entry["active"], (
                        f"CPU marked during GPU-only phase: {entry}"
                    )
        # (CPU rows may be omitted entirely while unused — both are acceptable.)
        assert pm.recovery_states, "expected progress updates to be recorded"

        # Scope chosen by the user: the saved result preserves the CPU valve the
        # user enabled, so the final CPU entry stays present (spill still allowed).
        cpu = next((w for w in weights if w.device == "cpu"), None)
        assert cpu is not None and cpu.weight == 0.0
