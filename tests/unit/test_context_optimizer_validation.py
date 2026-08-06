"""Invariantes do validador estrutural do Context Optimizer."""

import json

import pytest

from context_optimizer import (
    StructuralValidationError,
    parse_request_ir,
    validate_request_ir,
    validate_transformed_payload,
)


def _payload():
    return {
        "model": "validator-model",
        "messages": [
            {"role": "system", "content": "system-secret"},
            {"role": "user", "content": "preserve this"},
            {"role": "assistant", "content": "old answer"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "image input"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {"type": "file", "file_url": {"url": "https://example.test/a.pdf"}},
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_object"},
        "unknown_envelope_field": {"keep": True},
    }


def test_validation_accepts_deterministic_lossless_round_trip():
    payload = _payload()
    ir = parse_request_ir(payload)

    report = validate_request_ir(ir, ir.to_payload())

    assert report.valid is True
    assert report.retained_units == len(payload["messages"])
    assert json.loads(json.dumps(ir.to_payload(), sort_keys=True)) == payload


def test_validation_allows_only_non_protected_message_removal():
    payload = _payload()
    ir = parse_request_ir(payload)
    candidate = ir.to_payload()
    candidate["messages"] = [candidate["messages"][0], candidate["messages"][1], *candidate["messages"][3:]]

    report = validate_transformed_payload(ir, candidate)

    assert report.valid is True
    assert report.retained_units == len(candidate["messages"])


def test_validation_rejects_protected_removal_and_reordering():
    payload = _payload()
    ir = parse_request_ir(payload)

    removed = ir.to_payload()
    removed["messages"] = removed["messages"][1:]
    with pytest.raises(StructuralValidationError) as removed_error:
        validate_transformed_payload(ir, removed)
    assert removed_error.value.code == "protected_unit_removed"
    assert "system-secret" not in str(removed_error.value)

    reordered = ir.to_payload()
    reordered["messages"] = [reordered["messages"][0], reordered["messages"][2], reordered["messages"][1], *reordered["messages"][3:]]
    with pytest.raises(StructuralValidationError) as reordered_error:
        validate_transformed_payload(ir, reordered)
    assert reordered_error.value.code == "message_changed_or_reordered"


def test_validation_rejects_partial_or_changed_tool_group():
    payload = _payload()
    ir = parse_request_ir(payload)

    partial = ir.to_payload()
    partial["messages"].pop(4)
    with pytest.raises(StructuralValidationError) as partial_error:
        validate_transformed_payload(ir, partial)
    assert partial_error.value.code == "atomic_group_split"

    changed = ir.to_payload()
    changed["messages"][3]["tool_calls"][0]["function"]["arguments"] = '{"q":"changed"}'
    with pytest.raises(StructuralValidationError) as changed_error:
        validate_transformed_payload(ir, changed)
    assert changed_error.value.code == "message_changed_or_reordered"


def test_validation_rejects_media_unknown_fields_and_structured_output_changes():
    payload = _payload()
    ir = parse_request_ir(payload)

    for mutate, code in (
        (lambda candidate: candidate["messages"][5]["content"][1]["image_url"].update(url="data:image/png;base64,CHANGED"), "message_changed_or_reordered"),
        (lambda candidate: candidate.pop("unknown_envelope_field"), "envelope_field_changed"),
        (lambda candidate: candidate["response_format"].update(type="text"), "response_format_changed"),
    ):
        candidate = ir.to_payload()
        mutate(candidate)
        with pytest.raises(StructuralValidationError) as error:
            validate_transformed_payload(ir, candidate)
        assert error.value.code == code


def test_validation_rejects_invalid_serialization_and_increased_cost():
    payload = _payload()
    ir = parse_request_ir(payload)

    invalid = ir.to_payload()
    invalid["messages"][2]["content"] = object()
    with pytest.raises(StructuralValidationError) as serialization_error:
        validate_transformed_payload(ir, invalid)
    assert serialization_error.value.code == "not_json_serializable"
    assert "object at" not in str(serialization_error.value)

    candidate = ir.to_payload()
    with pytest.raises(StructuralValidationError) as budget_error:
        validate_transformed_payload(ir, candidate, original_cost=0)
    assert budget_error.value.code == "cost_increased"


def test_opaque_payload_must_remain_byte_for_byte_structurally_equal():
    payload = {"prompt": "opaque-secret", "custom": {"keep": True}}
    ir = parse_request_ir(payload)

    candidate = dict(payload)
    candidate["prompt"] = "changed"
    with pytest.raises(StructuralValidationError) as error:
        validate_request_ir(ir, candidate)

    assert error.value.code == "opaque_payload_changed"
    assert "opaque-secret" not in str(error.value)
