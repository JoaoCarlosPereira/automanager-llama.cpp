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


def test_start_request_mtp_draft_tokens_any_value_accepted():
    """After removing the clamp, any integer value is accepted."""
    for val in [0, 1, 7, 50, 100, -5, 0]:
        req = StartRequest(
            path="/models/a.gguf",
            gpu_weights=_minimal_gpu_weights(),
            mtp_draft_tokens=val,
        )
        assert req.mtp_draft_tokens == val
