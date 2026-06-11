"""Unit tests for GPUManager compute_n_gpu_layers() and validate_gpu_weights()."""

import pytest

from gpu_manager import GPUManager, ALL_GPU_LAYERS
from schemas import GPUWeight


@pytest.fixture
def gpu_mgr():
    return GPUManager()


# ── compute_n_gpu_layers ──────────────────────────────────────────────────


def test_compute_n_gpu_layers_single_gpu_100(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_single_gpu_70(gpu_mgr):
    weights = [GPUWeight(index=0, weight=70, name="A", device="gpu")]
    # Sem CPU ativa: 100% das layers vão para GPU
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_single_gpu_70_with_cpu_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu", active=True),
        GPUWeight(index=-1, weight=30, name="C", device="cpu", active=False),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_single_gpu_50(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_two_gpus_sum_100(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu"),
        GPUWeight(index=1, weight=30, name="B", device="gpu"),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_gpu_plus_cpu(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu"),
        GPUWeight(index=0, weight=30, name="C", device="cpu"),
    ]
    # Only GPU weight counts → 70% of 32 → 22
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 22


def test_compute_n_gpu_layers_all_cpu(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="C", device="cpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 0


def test_compute_n_gpu_layers_inactive_gpu_ignored(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=100, name="A", device="gpu", active=False),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 0


def test_compute_n_gpu_layers_zero_weight_ignored(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=0, name="A", device="gpu"),
        GPUWeight(index=0, weight=100, name="C", device="cpu"),
    ]
    # GPU weight=0 is active but weight=0 contributes nothing
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 0


def test_compute_n_gpu_layers_clamped_to_total(gpu_mgr):
    weights = [GPUWeight(index=0, weight=120, name="A", device="gpu")]
    # > 100% shouldn't exceed total_layers
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_small_model(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_large_model_70b_80_layers(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=40, name="A", device="gpu"),
        GPUWeight(index=1, weight=30, name="B", device="gpu"),
        GPUWeight(index=0, weight=30, name="C", device="cpu"),
    ]
    # GPU total = 70% of 80 layers = 56
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=80) == 56


def test_compute_n_gpu_layers_empty_list(gpu_mgr):
    assert gpu_mgr.compute_n_gpu_layers([], total_layers=32) == 0


def test_compute_n_gpu_layers_10_layers_rounding_up(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=33, name="A", device="gpu"),
        GPUWeight(index=-1, weight=67, name="C", device="cpu", active=True),
    ]
    # 33% of 10 = 3.3 → rounds to 3 (offload CPU ativo)
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=10) == 3


def test_compute_n_gpu_layers_10_layers_no_cpu_offload(gpu_mgr):
    weights = [GPUWeight(index=0, weight=33, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=10) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_10_layers_rounding_boundary(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="A", device="gpu"),
        GPUWeight(index=-1, weight=50, name="C", device="cpu", active=True),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=10) == 5


# ── validate_gpu_weights ──────────────────────────────────────────────────────


def test_validate_gpu_weights_gpu_only_100_ok(gpu_mgr):
    weights = [GPUWeight(index=0, weight=60, name="A", device="gpu"),
               GPUWeight(index=1, weight=40, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True
    assert msg == ""


def test_validate_gpu_weights_gpu_plus_cpu_ok(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu"),
        GPUWeight(index=0, weight=30, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_sum_too_low(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu"),
               GPUWeight(index=1, weight=30, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "somam" in msg.lower() or "100" in msg


def test_validate_gpu_weights_sum_too_high(gpu_mgr):
    weights = [GPUWeight(index=0, weight=70, name="A", device="gpu"),
               GPUWeight(index=1, weight=50, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "somam" in msg.lower() or "100" in msg


def test_validate_gpu_weights_cpu_any_weight_ok(gpu_mgr):
    # 20 + 80 = 100 (sum OK), CPU = 80% — no CPU cap anymore
    weights = [
        GPUWeight(index=0, weight=20, name="A", device="gpu"),
        GPUWeight(index=0, weight=80, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_cpu_at_70_ok(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=30, name="A", device="gpu"),
        GPUWeight(index=0, weight=70, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_no_active_devices(gpu_mgr):
    weights = [GPUWeight(index=0, weight=0, name="A", device="gpu", active=False)]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "ativo" in msg.lower() or "active" in msg.lower()


def test_validate_gpu_weights_cpu_only_rejected(gpu_mgr):
    weights = [GPUWeight(index=-1, weight=100, name="C", device="cpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "gpu" in msg.lower()


def test_validate_gpu_weights_inactive_excluded_from_sum(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="A", device="gpu"),
        GPUWeight(index=1, weight=50, name="B", device="gpu", active=False),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    # Only GPU 0 is active (50%), which fails sum != 100
    assert ok is False


def test_validate_gpu_weights_cpu_0_ok(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_single_gpu_100(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_multiple_gpus_sum_100(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=40, name="A", device="gpu"),
        GPUWeight(index=1, weight=35, name="B", device="gpu"),
        GPUWeight(index=2, weight=25, name="C", device="gpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


def test_validate_gpu_weights_three_gpus_plus_cpu(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=40, name="A", device="gpu"),
        GPUWeight(index=1, weight=30, name="B", device="gpu"),
        GPUWeight(index=2, weight=20, name="C", device="gpu"),
        GPUWeight(index=0, weight=10, name="D", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True


# ── compute_offload_plan via cascata estrita (task_02) ────────────────────


def _fake_metrics(vram_by_index):
    # get_metrics() emite "mem_total" (string MiB), não "vram_total_mb".
    return {"gpus": [
        {"index": i, "mem_total": str(v)} for i, v in vram_by_index.items()
    ]}


def _mgr_with_hw(gpu_mgr, vram_by_index, model_vram_mb):
    """Configura métricas de VRAM e tamanho estimado do modelo no manager."""
    gpu_mgr.get_metrics = lambda: _fake_metrics(vram_by_index)
    gpu_mgr._cached_model_vram_mb = model_vram_mb
    return gpu_mgr


def test_build_priority_order_main_first():
    weights = [
        GPUWeight(index=0, weight=10, name="A", device="gpu", is_main=False),
        GPUWeight(index=2, weight=10, name="C", device="gpu", is_main=True),
        GPUWeight(index=1, weight=10, name="B", device="gpu", is_main=False),
    ]
    assert GPUManager._build_priority_order(weights) == [2, 0, 1]


def test_offload_plan_cascata_modelo_cabe_na_principal(gpu_mgr):
    """Modelo cabe na 3090 (principal) → tudo na GPU, CPU 0%, viável."""
    _mgr_with_hw(gpu_mgr, {0: 24000, 1: 16000}, model_vram_mb=20000)
    weights = [
        GPUWeight(index=0, weight=50, name="3090", device="gpu", is_main=True),
        GPUWeight(index=1, weight=50, name="P100", device="gpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=80, cpu_enabled=True)
    assert plan.is_feasible
    assert plan.cpu_pct == 0.0
    assert plan.n_gpu_layers == 80  # 100% das camadas na GPU


def test_offload_plan_cascata_transborda_para_cpu(gpu_mgr):
    """Modelo > soma das GPUs com CPU ligada → CPU recebe sobra, viável."""
    _mgr_with_hw(gpu_mgr, {0: 24000, 1: 16000, 2: 16000}, model_vram_mb=70000)
    weights = [
        GPUWeight(index=0, weight=33, name="3090", device="gpu", is_main=True),
        GPUWeight(index=1, weight=33, name="P100a", device="gpu"),
        GPUWeight(index=2, weight=34, name="P100b", device="gpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=80, cpu_enabled=True)
    assert plan.is_feasible
    assert plan.cpu_pct > 0
    assert plan.n_gpu_layers < 80  # parte das camadas vai para a CPU


def test_offload_plan_cascata_cpu_off_infeasivel(gpu_mgr):
    """Modelo não cabe e CPU desligada → plano sinaliza is_feasible=False."""
    _mgr_with_hw(gpu_mgr, {0: 24000, 1: 16000}, model_vram_mb=70000)
    weights = [
        GPUWeight(index=0, weight=50, name="3090", device="gpu", is_main=True),
        GPUWeight(index=1, weight=50, name="P100", device="gpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=80, cpu_enabled=False)
    assert plan.is_feasible is False


def test_offload_plan_reads_mem_total_key_not_all_cpu(gpu_mgr):
    """Regressão: get_metrics usa 'mem_total'; VRAM deve ser lida (não 0).

    Antes da correção, a chave errada ('vram_total_mb') zerava a VRAM e jogava
    100% na CPU mesmo com GPUs com folga.
    """
    gpu_mgr.get_metrics = lambda: {
        "gpus": [
            {"index": 0, "mem_total": "24000"},
            {"index": 1, "mem_total": "16000"},
        ]
    }
    gpu_mgr._cached_model_vram_mb = 20000  # cabe na 3090
    weights = [
        GPUWeight(index=0, weight=50, name="3090", device="gpu", is_main=True),
        GPUWeight(index=1, weight=50, name="P100", device="gpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=80, cpu_enabled=True)
    assert plan.cpu_pct == 0.0          # nada na CPU
    assert plan.gpu_pct == 100.0        # tudo nas GPUs
    assert plan.n_gpu_layers == 80


def test_offload_plan_maxes_gpus_then_spills_remainder_to_cpu(gpu_mgr):
    """Modelo > soma das GPUs → GPUs no máximo (98%) e só o restante na CPU."""
    gpu_mgr.get_metrics = lambda: {
        "gpus": [
            {"index": 0, "mem_total": "24000"},
            {"index": 1, "mem_total": "16000"},
        ]
    }
    gpu_mgr._cached_model_vram_mb = 60000  # excede ~39200 de caps
    weights = [
        GPUWeight(index=0, weight=50, name="3090", device="gpu", is_main=True),
        GPUWeight(index=1, weight=50, name="P100", device="gpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=80, cpu_enabled=True)
    # GPUs absorvem o máximo (~39200/60000 ≈ 65%); CPU só o restante (~35%).
    assert plan.gpu_pct > 60.0
    assert 0 < plan.cpu_pct < 40.0
    assert 0 < plan.n_gpu_layers < 80
