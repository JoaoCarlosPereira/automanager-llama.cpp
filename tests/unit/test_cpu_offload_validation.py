"""Edge-case validation for CPU offload weights (task 12)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from gpu_manager import GPUManager
from process_manager import ProcessManager
from schemas import GPUWeight


@pytest.fixture
def gpu_mgr():
    return GPUManager()


@pytest.fixture
def pm():
    config = MagicMock()
    token = MagicMock()
    token.get_or_create.return_value = "sk-test"
    log_mgr = MagicMock()
    log_mgr.open_server_log_append.return_value = MagicMock()
    return ProcessManager(config, token, GPUManager(), log_mgr)


# ── validate_weights edge cases ───────────────────────────────────────────


def test_validate_weights_cpu_unchecked_weight_zero_ok(gpu_mgr):
    """CPU inactive (unchecked) with weight 0 — GPU-only 100% is valid."""
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True
    assert msg == ""


def test_validate_weights_only_gpu_selected_ok(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=60, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=40, name="GPU1", active=True, device="gpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


def test_validate_weights_cpu_only_rejected(gpu_mgr):
    weights = [
        GPUWeight(index=-1, weight=100, name="CPU", active=True, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "gpu" in msg.lower()


def test_validate_weights_cpu_exceeds_70_portuguese_message(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=20, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=80, name="CPU", active=True, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "70" in msg
    assert "cpu" in msg.lower()


def test_validate_weights_sum_not_100_portuguese_message(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is False
    assert "100" in msg


def test_validate_weights_config_without_device_defaults_gpu(gpu_mgr):
    """Backward compatibility: missing device field defaults to gpu."""
    weights = [GPUWeight(index=0, weight=100, name="GPU0", active=True)]
    assert weights[0].device == "gpu"
    ok, msg = gpu_mgr.validate_weights(weights)
    assert ok is True


# ── ProcessManager.start() validation wiring ──────────────────────────────


def _start_with_weights(pm, weights):
    import process_manager as pm_mod

    orig_setsid = getattr(pm_mod.os, "setsid", None)
    pm_mod.os.setsid = lambda: None
    try:
        with patch.object(pm, "stop"):
            with patch("subprocess.Popen") as popen:
                popen.return_value = MagicMock(pid=1)
                with patch.object(pm.gpu_manager, "get_visible_devices", return_value="0"):
                    with patch.object(
                        pm.gpu_manager, "compute_tensor_split", return_value=["1.0"]
                    ):
                        with patch.object(
                            pm.gpu_manager, "detect_model_layers", return_value=32
                        ):
                            pm.start(
                                model_path="/fake/model.gguf",
                                gpu_weights=weights,
                                context_size=8192,
                            )
    finally:
        if orig_setsid is not None:
            pm_mod.os.setsid = orig_setsid
        elif hasattr(pm_mod.os, "setsid"):
            delattr(pm_mod.os, "setsid")


def test_start_with_active_cpu_calls_validate_weights_sum_error(pm):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    with pytest.raises(HTTPException) as exc:
        _start_with_weights(pm, weights)
    assert exc.value.status_code == 400
    assert "100" in exc.value.detail


def test_start_with_active_cpu_rejects_cpu_over_70(pm):
    weights = [
        GPUWeight(index=0, weight=20, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=80, name="CPU", active=True, device="cpu"),
    ]
    with pytest.raises(HTTPException) as exc:
        _start_with_weights(pm, weights)
    assert exc.value.status_code == 400
    assert "70" in exc.value.detail


def test_start_with_active_cpu_rejects_cpu_only(pm):
    weights = [
        GPUWeight(index=0, weight=0, name="GPU0", active=False, device="gpu"),
        GPUWeight(index=-1, weight=100, name="CPU", active=True, device="cpu"),
    ]
    with pytest.raises(HTTPException) as exc:
        _start_with_weights(pm, weights)
    assert exc.value.status_code == 400
    assert "gpu" in exc.value.detail.lower()


def test_start_without_active_cpu_skips_full_validate_weights(pm):
    """GPU-only path still uses validate_gpu_weights (no sum enforcement)."""
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    _start_with_weights(pm, weights)


def test_start_with_active_cpu_valid_offload_ok(pm):
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    _start_with_weights(pm, weights)
