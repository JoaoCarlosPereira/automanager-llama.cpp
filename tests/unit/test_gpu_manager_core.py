"""Unit tests for GPUManager compute_n_gpu_layers() and validate_weights()."""

import pytest

from gpu_manager import GPUManager
from schemas import GPUWeight


@pytest.fixture
def gpu_mgr():
    return GPUManager()


# ── compute_n_gpu_layers ──────────────────────────────────────────────────


def test_compute_n_gpu_layers_single_gpu_100(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


def test_compute_n_gpu_layers_single_gpu_70(gpu_mgr):
    weights = [GPUWeight(index=0, weight=70, name="A", device="gpu")]
    # Sem CPU ativa: 100% das layers vão para GPU
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


def test_compute_n_gpu_layers_single_gpu_70_with_cpu_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu", active=True),
        GPUWeight(index=-1, weight=30, name="C", device="cpu", active=False),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


def test_compute_n_gpu_layers_single_gpu_50(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


def test_compute_n_gpu_layers_two_gpus_sum_100(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu"),
        GPUWeight(index=1, weight=30, name="B", device="gpu"),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


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
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


def test_compute_n_gpu_layers_small_model(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu")]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == 32


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
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=10) == 10


def test_compute_n_gpu_layers_10_layers_rounding_boundary(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="A", device="gpu"),
        GPUWeight(index=-1, weight=50, name="C", device="cpu", active=True),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=10) == 5


# ── validate_weights ──────────────────────────────────────────────────────


def test_validate_weights_gpu_only_100_ok(gpu_mgr):
    weights = [GPUWeight(index=0, weight=60, name="A", device="gpu"),
               GPUWeight(index=1, weight=40, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True
    assert msg == ""


def test_validate_weights_gpu_plus_cpu_ok(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="A", device="gpu"),
        GPUWeight(index=0, weight=30, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_sum_too_low(gpu_mgr):
    weights = [GPUWeight(index=0, weight=50, name="A", device="gpu"),
               GPUWeight(index=1, weight=30, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "somam" in msg.lower() or "100" in msg


def test_validate_weights_sum_too_high(gpu_mgr):
    weights = [GPUWeight(index=0, weight=70, name="A", device="gpu"),
               GPUWeight(index=1, weight=50, name="B", device="gpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "somam" in msg.lower() or "100" in msg


def test_validate_weights_cpu_exceeds_70_limit(gpu_mgr):
    # 20 + 80 = 100 (sum OK) but CPU = 80 > 70 → fails CPU limit
    weights = [
        GPUWeight(index=0, weight=20, name="A", device="gpu"),
        GPUWeight(index=0, weight=80, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "70" in msg or "cpu" in msg.lower() or "CPU" in msg


def test_validate_weights_cpu_at_70_ok(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=30, name="A", device="gpu"),
        GPUWeight(index=0, weight=70, name="C", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_no_active_devices(gpu_mgr):
    weights = [GPUWeight(index=0, weight=0, name="A", device="gpu", active=False)]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "ativo" in msg.lower() or "active" in msg.lower()


def test_validate_weights_cpu_only_rejected(gpu_mgr):
    weights = [GPUWeight(index=-1, weight=100, name="C", device="cpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "gpu" in msg.lower()


def test_validate_weights_inactive_excluded_from_sum(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="A", device="gpu"),
        GPUWeight(index=1, weight=50, name="B", device="gpu", active=False),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    # Only GPU 0 is active (50%), which fails sum != 100
    assert ok is False


def test_validate_weights_cpu_0_ok(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_single_gpu_100(gpu_mgr):
    weights = [GPUWeight(index=0, weight=100, name="A", device="gpu")]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_multiple_gpus_sum_100(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=40, name="A", device="gpu"),
        GPUWeight(index=1, weight=35, name="B", device="gpu"),
        GPUWeight(index=2, weight=25, name="C", device="gpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_three_gpus_plus_cpu(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=40, name="A", device="gpu"),
        GPUWeight(index=1, weight=30, name="B", device="gpu"),
        GPUWeight(index=2, weight=20, name="C", device="gpu"),
        GPUWeight(index=0, weight=10, name="D", device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True
