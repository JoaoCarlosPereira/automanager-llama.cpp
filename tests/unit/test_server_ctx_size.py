"""Unit tests for llama-server context size calculation."""

import pytest

from process_manager import compute_server_ctx_size, reasoning_cli_args


@pytest.mark.parametrize(
    ("context_size", "parallel_slots", "expected"),
    [
        (65536, 1, 65536),
        (524288, 1, 524288),
        (1048576, 1, 1048576),
        (1048576, 2, 2097152),
        (524288, 4, 2097152),
    ],
)
def test_compute_server_ctx_size_multiplies_by_parallel_slots(
    context_size, parallel_slots, expected
):
    assert compute_server_ctx_size(context_size, parallel_slots) == expected


def test_compute_server_ctx_size_clamps_invalid_inputs():
    assert compute_server_ctx_size(0, 0) == 1


def test_reasoning_cli_args_enabled():
    assert reasoning_cli_args(True) == ["--reasoning", "on"]


def test_reasoning_cli_args_disabled_forces_zero_budget():
    assert reasoning_cli_args(False) == [
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
