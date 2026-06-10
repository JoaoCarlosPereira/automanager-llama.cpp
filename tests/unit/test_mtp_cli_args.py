"""Unit tests for MTP CLI args and model detection."""

from unittest.mock import MagicMock, patch

import pytest

from gpu_manager import GPUManager
from process_manager import mtp_cli_args


@pytest.fixture
def gpu_mgr():
    return GPUManager()


def test_mtp_cli_args_disabled():
    mgr = MagicMock()
    flags, applied, reason = mtp_cli_args(False, 3, "/models/a.gguf", mgr)
    assert flags == []
    assert applied is False
    mgr.detect_model_mtp.assert_not_called()


def test_mtp_cli_args_incompatible_model():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = False
    flags, applied, reason = mtp_cli_args(True, 3, "/models/a.gguf", mgr)
    assert flags == []
    assert applied is False
    assert "MTP" in reason


def test_mtp_cli_args_enabled_compatible():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = True
    flags, applied, reason = mtp_cli_args(True, 3, "/models/mtp.gguf", mgr)
    assert flags == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    assert applied is True
    assert reason == ""


def test_mtp_cli_args_clamps_draft_tokens_high():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = True
    flags, applied, reason = mtp_cli_args(True, 99, "/models/mtp.gguf", mgr)
    assert flags[-1] == "64"
    assert applied is True


def test_mtp_cli_args_draft_tokens_default():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = True
    flags, applied, reason = mtp_cli_args(True, None, "/models/mtp.gguf", mgr)
    assert flags[-1] == "3"
    assert applied is True


def test_detect_model_mtp_parses_nextn_predict_layers(gpu_mgr):
    output = "model info\nnextn_predict_layers = 1\nn_layer = 32\n"
    with patch("gpu_manager.subprocess.check_output", return_value=output.encode()):
        assert gpu_mgr.detect_model_mtp("/fake/model.gguf") is True


def test_detect_model_mtp_zero_layers_returns_false(gpu_mgr):
    output = "nextn_predict_layers = 0\n"
    with patch("gpu_manager.subprocess.check_output", return_value=output.encode()):
        assert gpu_mgr.detect_model_mtp("/fake/model.gguf") is False


def test_detect_model_mtp_fallback_on_error(gpu_mgr):
    with patch(
        "gpu_manager.subprocess.check_output",
        side_effect=RuntimeError("boom"),
    ):
        assert gpu_mgr.detect_model_mtp("/fake/model.gguf") is False
