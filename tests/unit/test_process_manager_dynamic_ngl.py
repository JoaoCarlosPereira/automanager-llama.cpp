"""Unit tests for dynamic --n-gpu-layers in ProcessManager.start().

Tests verify that:
1. ProcessManager.start() calls gpu_manager.compute_n_gpu_layers() and uses the result as the -ngl value.
2. Backward compatibility: when all GPUWeight entries have device="gpu" (the default), the behaviour
   matches the pre-change "all layers on GPU" path — every active GPU with 100% weight => all layers.
3. Mixed GPU+CPU weights produce a reduced layer count proportional to the GPU weight sum.
4. detect_model_layers() reads total_layers from llama-server --model-info output.
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gpu_manager import GPUManager, DEFAULT_TOTAL_LAYERS, ALL_GPU_LAYERS
from schemas import GPUWeight


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def gpu_mgr():
    return GPUManager()


@pytest.fixture
def mock_config():
    cm = MagicMock()
    cm.get_or_create.return_value = "test-token"
    return cm


@pytest.fixture
def mock_token():
    tm = MagicMock()
    tm.get_or_create.return_value = "test-token"
    return tm


@pytest.fixture
def mock_log_mgr():
    lm = MagicMock()
    lm.open_server_log_append.return_value = MagicMock()
    lm.start_streaming = MagicMock()
    return lm


@pytest.fixture
def pm(mock_config, mock_token, mock_log_mgr, gpu_mgr):
    from process_manager import ProcessManager

    return ProcessManager(mock_config, mock_token, gpu_mgr, mock_log_mgr)


# ── detect_model_layers ───────────────────────────────────────────────────


def test_detect_model_layers_parses_n_layers(gpu_mgr):
    sample = (
        "llama model: n_embd        = 4096\n"
        "llama model: n_layer        = 32\n"
        "llama model: n_head         = 32\n"
    )
    with patch(
        "subprocess.check_output",
        return_value=sample.encode(),
    ):
        result = gpu_mgr.detect_model_layers("/fake/model.gguf")

    assert result == 32


def test_detect_model_layers_ignores_extra_output(gpu_mgr):
    sample = (
        "loading model model.gguf\n"
        "llama model: n_head         = 8\n"
        "llama model: n_layer        = 80\n"
        "llama model: n_embd        = 8192\n"
    )
    with patch(
        "subprocess.check_output",
        return_value=sample.encode(),
    ):
        result = gpu_mgr.detect_model_layers("/fake/model.gguf")

    assert result == 80


def test_detect_model_layers_fallback_on_error(gpu_mgr):
    with patch(
        "subprocess.check_output",
        side_effect=subprocess.SubprocessError("not found"),
    ):
        result = gpu_mgr.detect_model_layers("/fake/model.gguf")

    assert result == DEFAULT_TOTAL_LAYERS


def test_detect_model_layers_fallback_missing_pattern(gpu_mgr):
    sample = "some output without n_layer info\n"
    with patch(
        "subprocess.check_output",
        return_value=sample.encode(),
    ):
        result = gpu_mgr.detect_model_layers("/fake/model.gguf")

    assert result == DEFAULT_TOTAL_LAYERS


# ── ProcessManager.start() uses dynamic n_gpu_layers ──────────────────────


def _captured_cmd(pm, gpu_weights=None):
    """Helper: patch Popen and return the cmd list that was passed."""
    cmd = None

    def _capture(args, **kwargs):
        nonlocal cmd
        cmd = list(args)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        return mock_proc

    weights = gpu_weights or [
        GPUWeight(index=0, weight=100.0, name="GPU0", active=True, device="gpu"),
    ]

    # Patch os.setsid in the process_manager module for Windows compatibility
    import process_manager as pm_mod
    orig_setsid = getattr(pm_mod.os, "setsid", None)
    pm_mod.os.setsid = lambda: None

    try:
        with patch.object(pm, "stop"):
            with patch(
                "process_manager.resolve_llama_server_bin",
                return_value="/usr/bin/llama-server",
            ):
                with patch("subprocess.Popen", side_effect=_capture):
                    with patch.object(pm.gpu_manager, "validate_gpu_weights", return_value=(True, "")):
                        with patch.object(pm.gpu_manager, "get_visible_devices", return_value="0"):
                            with patch.object(pm.gpu_manager, "compute_tensor_split", return_value=["1.0"]):
                                with patch.object(pm.gpu_manager, "detect_model_layers", return_value=32):
                                    pm.start(
                                        model_path="/fake/model.gguf",
                                        gpu_weights=weights,
                                        context_size=8192,
                                    )
    finally:
        if orig_setsid is not None:
            pm_mod.os.setsid = orig_setsid
        else:
            delattr(pm_mod.os, "setsid")

    return cmd


def test_start_uses_dynamic_ngl_all_gpu_100(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """Single GPU at 100% with CPU offload off => -ngl is ALL_GPU_LAYERS."""
    cmd = _captured_cmd(pm)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == str(ALL_GPU_LAYERS)


def test_start_uses_dynamic_ngl_gpu_70_plus_cpu_30(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """GPU 70% + CPU 30% with 32 layers => n_gpu_layers = round(0.70 * 32) = 22."""
    weights = [
        GPUWeight(index=0, weight=70.0, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=30.0, name="CPU", active=True, device="cpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == "22"


def test_start_uses_dynamic_ngl_all_cpu(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """All CPU (0% GPU) => n_gpu_layers = 0."""
    weights = [
        GPUWeight(index=0, weight=0.0, name="GPU0", active=True, device="gpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == "0"


def test_start_backward_compat_no_device_field(
    pm, mock_config, mock_token, mock_log_mgr
):
    """When GPUWeight is created without device (default='gpu'), behaviour is identical to pre-change."""
    weights = [
        GPUWeight(index=0, weight=100.0, name="GPU0", active=True),
    ]
    # device should default to "gpu"
    assert weights[0].device == "gpu"

    cmd = _captured_cmd(pm)
    ngl_idx = cmd.index("-ngl")
    # 100% GPU on a 32-layer model => ALL_GPU_LAYERS
    assert cmd[ngl_idx + 1] == str(ALL_GPU_LAYERS)


def test_start_dynamic_ngl_two_gpus_50_50(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """Two GPUs both active at 50% each => total GPU weight = 100% => n_gpu_layers = 32."""
    weights = [
        GPUWeight(index=0, weight=50.0, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=50.0, name="GPU1", active=True, device="gpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == str(ALL_GPU_LAYERS)


def test_start_dynamic_ngl_large_model(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """Large model with 80 layers, GPU 70% => n_gpu_layers = round(0.70 * 80) = 56."""
    # Need to patch at module level since the helper patches detect_model_layers
    import process_manager as pm_mod
    orig_setsid = getattr(pm_mod.os, "setsid", None)
    pm_mod.os.setsid = lambda: None

    try:
        cmd = None

        def _capture(args, **kwargs):
            nonlocal cmd
            cmd = list(args)
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            return mock_proc

        weights = [
            GPUWeight(index=0, weight=70.0, name="GPU0", active=True, device="gpu"),
            GPUWeight(index=1, weight=30.0, name="CPU", active=True, device="cpu"),
        ]

        with patch.object(pm, "stop"):
            with patch(
                "process_manager.resolve_llama_server_bin",
                return_value="/usr/bin/llama-server",
            ):
                with patch("subprocess.Popen", side_effect=_capture):
                    with patch.object(pm.gpu_manager, "validate_gpu_weights", return_value=(True, "")):
                        with patch.object(pm.gpu_manager, "get_visible_devices", return_value="0"):
                            with patch.object(pm.gpu_manager, "compute_tensor_split", return_value=["1.0"]):
                                with patch.object(pm.gpu_manager, "detect_model_layers", return_value=80):
                                    pm.start(
                                        model_path="/fake/model.gguf",
                                        gpu_weights=weights,
                                        context_size=8192,
                                    )
    finally:
        if orig_setsid is not None:
            pm_mod.os.setsid = orig_setsid
        else:
            delattr(pm_mod.os, "setsid")

    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == "56"


def test_start_clamps_ngl_to_total_layers(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """GPU weight sum exceeds 100% — n_gpu_layers should be clamped to total_layers."""
    weights = [
        GPUWeight(index=0, weight=100.0, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=100.0, name="GPU1", active=True, device="gpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == str(ALL_GPU_LAYERS)


def test_start_inactive_gpu_weight_ignored(
    gpu_mgr, pm, mock_config, mock_token, mock_log_mgr
):
    """Inactive GPU weight is ignored; remaining active GPU at 100% => all layers."""
    weights = [
        GPUWeight(index=0, weight=100.0, name="GPU0", active=False, device="gpu"),
        GPUWeight(index=1, weight=100.0, name="GPU1", active=True, device="gpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == str(ALL_GPU_LAYERS)


def test_start_zero_gpu_weight(gpu_mgr, pm, mock_config, mock_token, mock_log_mgr):
    """GPU with 0 weight => n_gpu_layers = 0."""
    weights = [
        GPUWeight(index=0, weight=0.0, name="GPU0", active=True, device="gpu"),
    ]
    cmd = _captured_cmd(pm, weights)
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == "0"


# ── port release on stop() ─────────────────────────────────────────────────


def test_wait_port_released_returns_when_free(pm):
    """_wait_port_released returns True as soon as the port is free."""
    with patch.object(type(pm), "_is_port_free", return_value=True):
        assert pm._wait_port_released(port=8085, timeout=1.0) is True


def test_wait_port_released_times_out_when_bound(pm):
    """Still bound after timeout -> returns False (logged), does not hang."""
    with patch.object(type(pm), "_is_port_free", return_value=False), \
         patch("process_manager.time.sleep", return_value=None):
        assert pm._wait_port_released(port=8085, timeout=0.5) is False


def test_stop_waits_for_port_release(pm):
    """stop() must wait for the port to free so the next start() can bind it."""
    with patch("process_manager.subprocess.run"), \
         patch.object(pm, "_wait_port_released", return_value=True) as waited:
        pm.stop()
    waited.assert_called_once()
