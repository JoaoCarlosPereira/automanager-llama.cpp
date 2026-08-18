"""Testes unitários do resolvedor de orçamento, limites e capacidades."""

import pytest
from context_optimizer import (
    LimitConfidence,
    ModelLimits,
    RequiredCapabilities,
    UNKNOWN_PLATFORM_CONTEXT_LIMIT,
    calculate_target_budget,
    derive_required_capabilities,
    derive_target_capabilities,
    resolve_model_limits,
)


def test_derive_required_capabilities():
    simple_payload = {"messages": [{"role": "user", "content": "Olá"}]}
    req1 = derive_required_capabilities(simple_payload)
    assert req1.as_set() == frozenset({"text"})

    complex_payload = {
        "tools": [{"type": "function"}],
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Veja esta foto e este arquivo"},
                    {"type": "image_url", "image_url": {"url": "http://img.jpg"}},
                    {"type": "file", "file_url": "http://file.pdf"},
                ],
            }
        ],
    }
    req2 = derive_required_capabilities(complex_payload)
    assert req2.as_set() == frozenset({"text", "vision", "tools", "structured_output", "files"})
    assert req2.is_subset_of(frozenset({"text", "vision", "tools", "structured_output", "files"}))
    assert not req2.is_subset_of(frozenset({"text", "tools"}))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.com/a.png"}
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg"},
                        }
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "image_url", "url": "data:image/png;base64,AA=="},
                }
            ]
        },
        {
            "messages": [{"role": "user", "images": ["AA=="]}],
        },
        {
            "input": [{"type": "inline_data", "mime_type": "image/png", "data": "AA=="}],
        },
    ],
)
def test_derive_required_capabilities_recognizes_image_variants(payload):
    assert derive_required_capabilities(payload).vision is True


def test_resolve_model_limits_local_single_slot():
    backend_info = {
        "backend_type": "local",
        "config": {"context_size": 32768, "parallel_slots": 1},
    }
    limits = resolve_model_limits(backend_info)
    assert limits.context_tokens == 32768
    assert limits.confidence == LimitConfidence.KNOWN_LOCAL
    assert limits.is_known is True


def test_resolve_model_limits_local_multi_slot_divided_once():
    backend_info = {
        "backend_type": "local",
        "config": {"context_size": 32768, "parallel_slots": 4},
    }
    limits = resolve_model_limits(backend_info)
    assert limits.context_tokens == 8192
    assert limits.confidence == LimitConfidence.KNOWN_LOCAL


def test_resolve_model_limits_platform_known():
    backend_info = {"backend_type": "platform", "provider": "codex"}
    metadata = {"context_length": 128000, "max_completion_tokens": 4096}
    limits = resolve_model_limits(backend_info, metadata)
    assert limits.context_tokens == 128000
    assert limits.input_tokens == 128000
    assert limits.max_output_tokens == 4096
    assert limits.confidence == LimitConfidence.KNOWN_PROVIDER
    assert limits.is_known is True


def test_platform_input_limit_does_not_reserve_output_twice():
    backend_info = {"backend_type": "platform", "provider": "codex"}
    metadata = {"context_length": 372000, "max_completion_tokens": 128000}
    limits = resolve_model_limits(backend_info, metadata)

    budget = calculate_target_budget(
        {}, limits, frozenset({"text"}), protocol_overhead=512, safety_margin=256
    )

    assert limits.input_tokens == 372000
    assert budget.output_reserve == 128000
    assert budget.input_budget == 372000 - 512 - 256


def test_explicit_platform_input_token_limit_is_input_only():
    backend_info = {"backend_type": "platform", "provider": "antigravity"}
    metadata = {"inputTokenLimit": 1048576, "outputTokenLimit": 65535}
    limits = resolve_model_limits(backend_info, metadata)

    budget = calculate_target_budget(
        {}, limits, frozenset({"text"}), protocol_overhead=512, safety_margin=256
    )

    assert limits.context_tokens == 1048576
    assert limits.input_tokens == 1048576
    assert budget.input_budget == 1048576 - 512 - 256


def test_resolve_model_limits_platform_unknown_or_sentinel():
    backend_info = {"backend_type": "platform"}
    metadata1 = {}
    limits1 = resolve_model_limits(backend_info, metadata1)
    assert limits1.context_tokens is None
    assert limits1.confidence == LimitConfidence.UNKNOWN
    assert limits1.is_known is False

    metadata2 = {"context_length": UNKNOWN_PLATFORM_CONTEXT_LIMIT}
    limits2 = resolve_model_limits(backend_info, metadata2)
    assert limits2.context_tokens is None
    assert limits2.confidence == LimitConfidence.UNKNOWN
    assert limits2.is_known is False


def test_derive_target_capabilities_local():
    b_no_vision = {"backend_type": "local", "config": {"mmproj_path": None}}
    caps1 = derive_target_capabilities(b_no_vision)
    assert caps1 == frozenset({"text", "tools", "structured_output", "files"})

    b_vision = {"backend_type": "local", "config": {"mmproj_path": "/path/mmproj.gguf", "mmproj_disabled": False}}
    caps2 = derive_target_capabilities(b_vision)
    assert caps2 == frozenset({"text", "vision", "tools", "structured_output", "files"})


def test_derive_target_capabilities_platform():
    b_platform = {"backend_type": "platform"}
    metadata = {"capabilities": ["vision", "tools"]}
    caps = derive_target_capabilities(b_platform, metadata)
    assert caps == frozenset({"text", "vision", "tools"})


def test_derive_target_capabilities_platform_assumes_vision_without_metadata():
    b_platform = {"backend_type": "platform"}

    caps = derive_target_capabilities(b_platform)

    assert caps == frozenset({"text", "vision"})


def test_derive_target_capabilities_platform_can_disable_vision():
    b_platform = {
        "backend_type": "platform",
        "config": {"vision_enabled": False},
    }
    metadata = {"capabilities": ["vision", "tools"], "supports_vision": True}

    caps = derive_target_capabilities(b_platform, metadata)

    assert caps == frozenset({"text", "tools"})


def test_calculate_target_budget_order_of_reserves():
    limits = ModelLimits(context_tokens=16384, max_output_tokens=2048, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    caps = frozenset({"text"})

    p1 = {"max_completion_tokens": 1000, "max_tokens": 500}
    b1 = calculate_target_budget(p1, limits, caps, protocol_overhead=100, safety_margin=50)
    assert b1.output_reserve == 1000
    assert b1.input_budget == 16384 - 1000 - 100 - 50

    p2 = {"max_tokens": 500}
    b2 = calculate_target_budget(p2, limits, caps, protocol_overhead=100, safety_margin=50)
    assert b2.output_reserve == 500
    assert b2.input_budget == 16384 - 500 - 100 - 50

    p3 = {}
    b3 = calculate_target_budget(p3, limits, caps, protocol_overhead=100, safety_margin=50)
    assert b3.output_reserve == 2048
    assert b3.input_budget == 16384 - 2048 - 100 - 50

    limits_no_out = ModelLimits(context_tokens=16384, max_output_tokens=None, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    b4 = calculate_target_budget(p3, limits_no_out, caps, protocol_overhead=100, safety_margin=50, default_output_reserve=1024)
    assert b4.output_reserve == 1024
    assert b4.input_budget == 16384 - 1024 - 100 - 50


def test_calculate_target_budget_unknown_confidence():
    limits = ModelLimits(context_tokens=None, max_output_tokens=2048, source="platform", confidence=LimitConfidence.UNKNOWN)
    caps = frozenset({"text"})
    budget = calculate_target_budget({}, limits, caps)
    assert budget.confidence == LimitConfidence.UNKNOWN
    assert budget.input_budget is None
