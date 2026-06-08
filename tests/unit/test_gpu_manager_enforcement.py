"""Tests for GPU strict enforcement helpers."""

import pytest

from gpu_manager import GPUManager
from schemas import GPUWeight


@pytest.fixture
def gpu_mgr():
    return GPUManager()


def test_compute_tensor_split_two_gpus(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=80, name="a", active=True),
        GPUWeight(index=1, weight=20, name="b", active=True),
    ]
    assert gpu_mgr.compute_tensor_split(weights) == ["0.8000", "0.2000"]


def test_compute_tensor_split_excludes_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=0, name="a", active=True),
        GPUWeight(index=1, weight=100, name="b", active=True),
    ]
    assert gpu_mgr.compute_tensor_split(weights) == ["1.0000"]


def test_get_visible_devices(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="a", active=True),
        GPUWeight(index=2, weight=50, name="c", active=True),
        GPUWeight(index=1, weight=0, name="b", active=False),
    ]
    assert gpu_mgr.get_visible_devices(weights) == "0,2"


def test_validate_gpu_weights_rejects_empty(gpu_mgr):
    ok, msg = gpu_mgr.validate_gpu_weights(
        [GPUWeight(index=0, weight=0, name="a", active=True)]
    )
    assert ok is False
    assert "No active GPUs" in msg


def test_validate_gpu_weights_requires_100_when_cpu_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="a", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="cpu", active=False, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "100" in msg


def test_validate_gpu_weights_ok_gpu_only_100(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=100, name="a", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="cpu", active=False, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True
