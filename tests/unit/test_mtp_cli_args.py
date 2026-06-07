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
    assert mtp_cli_args(False, 3, "/models/a.gguf", mgr) == []
    mgr.detect_model_mtp.assert_not_called()


def test_mtp_cli_args_incompatible_model():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = False
    assert mtp_cli_args(True, 3, "/models/a.gguf", mgr) == []


def test_mtp_cli_args_enabled_compatible():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = True
    assert mtp_cli_args(True, 3, "/models/mtp.gguf", mgr) == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
    ]


def test_mtp_cli_args_clamps_draft_tokens():
    mgr = MagicMock()
    mgr.detect_model_mtp.return_value = True
    args = mtp_cli_args(True, 99, "/models/mtp.gguf", mgr)
    assert args[-1] == "6"


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
