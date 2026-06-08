"""Unit tests for MTP fields in StartRequest."""

import pytest
from pydantic import ValidationError

from schemas import (
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_MTP_ENABLED,
    GPUWeight,
    StartRequest,
)


def _minimal_gpu_weights():
    return [
        GPUWeight(index=0, weight=100.0, name="GPU0", active=True, is_main=True),
    ]


def test_start_request_mtp_defaults():
    req = StartRequest(path="/models/a.gguf", gpu_weights=_minimal_gpu_weights())
    assert req.mtp_enabled is DEFAULT_MTP_ENABLED
    assert req.mtp_draft_tokens == DEFAULT_MTP_DRAFT_TOKENS


def test_start_request_mtp_valid_values():
    req = StartRequest(
        path="/models/a.gguf",
        gpu_weights=_minimal_gpu_weights(),
        mtp_enabled=True,
        mtp_draft_tokens=2,
    )
    assert req.mtp_enabled is True
    assert req.mtp_draft_tokens == 2


@pytest.mark.parametrize("invalid_tokens", [0, 7, -1])
def test_start_request_mtp_draft_tokens_out_of_range(invalid_tokens):
    with pytest.raises(ValidationError):
        StartRequest(
            path="/models/a.gguf",
            gpu_weights=_minimal_gpu_weights(),
            mtp_draft_tokens=invalid_tokens,
        )
