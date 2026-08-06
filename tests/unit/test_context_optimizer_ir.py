"""Testes unitários do IR estrutural, adaptadores conservadores e passagem opaca do Context Optimizer."""

import pytest
from context_optimizer import (
    AtomicGroup,
    ConversationBlock,
    RequestEnvelope,
    RequestIR,
    build_request_ir,
    parse_request_ir,
    reconstruct_payload,
)


def test_simple_chat_round_trip():
    payload = {
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": "Olá, tudo bem?"},
        ],
        "temperature": 0.7,
        "custom_param": "preservado",
    }
    ir = parse_request_ir(payload)
    assert not ir.is_opaque
    assert ir.structural_validity is True
    assert len(ir.ordered_units) == 2
    assert ir.ordered_units[0].kind == "system"
    assert ir.ordered_units[1].kind == "user_text"
    assert ir.ordered_units[0].protected is True
    assert ir.ordered_units[1].protected is True

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload
    assert reconstructed["custom_param"] == "preservado"


def test_parts_list_preserves_order_types_and_unknown_fields():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Instrução inicial", "unk_part_attr": 123},
                    {"type": "custom_part", "data": "xyz"},
                ],
                "unk_message_attr": "abc",
            }
        ],
        "envelope_extra": True,
    }
    ir = parse_request_ir(payload)
    assert not ir.is_opaque
    assert len(ir.ordered_units) == 1
    unit = ir.ordered_units[0]
    assert unit.kind == "user_text"
    assert unit.original_value["unk_message_attr"] == "abc"

    reconstructed = ir.to_payload()
    assert reconstructed == payload
    assert reconstructed["envelope_extra"] is True
    assert reconstructed["messages"][0]["unk_message_attr"] == "abc"
    assert reconstructed["messages"][0]["content"][0]["unk_part_attr"] == 123


def test_image_data_url_and_file_integrity():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise a imagem e o arquivo:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                    {"type": "file", "file_url": {"url": "https://example.com/doc.pdf"}},
                ],
            }
        ]
    }
    ir = parse_request_ir(payload)
    assert not ir.is_opaque
    assert ir.required_capabilities.vision is True
    assert ir.required_capabilities.files is True
    assert ir.ordered_units[0].kind == "multimodal_message"

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_tool_call_with_multiple_results_atomic_group():
    payload = {
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "messages": [
            {"role": "user", "content": "Como está o tempo em SP e RJ?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_sp", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SP"}'}},
                    {"id": "call_rj", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"RJ"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_sp", "content": '{"temp": "25C"}'},
            {"role": "tool", "tool_call_id": "call_rj", "content": '{"temp": "30C"}'},
        ],
    }
    ir = build_request_ir(payload)
    assert not ir.is_opaque
    assert len(ir.ordered_units) == 4
    assert len(ir.atomic_groups) == 1

    group = list(ir.atomic_groups.values())[0]
    assert set(group.tool_call_ids) == {"call_sp", "call_rj"}
    assert len(group.block_ids) == 3  # 1 assistant block + 2 tool result blocks
    assert ir.ordered_units[1].atomic_group_id == group.group_id
    assert ir.ordered_units[2].atomic_group_id == group.group_id
    assert ir.ordered_units[3].atomic_group_id == group.group_id

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_out_of_order_tool_results():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "f1"}},
                    {"id": "tc2", "type": "function", "function": {"name": "f2"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "res2"},
            {"role": "tool", "tool_call_id": "tc1", "content": "res1"},
        ]
    }
    ir = parse_request_ir(payload)
    assert not ir.is_opaque
    assert len(ir.ordered_units) == 3
    group = list(ir.atomic_groups.values())[0]
    assert set(group.tool_call_ids) == {"tc1", "tc2"}
    assert ir.ordered_units[1].original_index == 1
    assert ir.ordered_units[2].original_index == 2

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_orphan_tool_result_makes_payload_opaque():
    payload = {
        "messages": [
            {"role": "user", "content": "Oi"},
            {"role": "tool", "tool_call_id": "orphan_call_id", "content": "resultado sem call"},
        ]
    }
    ir = parse_request_ir(payload)
    assert ir.is_opaque is True
    assert ir.structural_validity is False

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_duplicate_tool_call_id_makes_payload_opaque():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_dup", "type": "function", "function": {"name": "f1"}},
                    {"id": "call_dup", "type": "function", "function": {"name": "f2"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_dup", "content": "res"},
        ]
    }
    ir = parse_request_ir(payload)
    assert ir.is_opaque is True
    assert ir.structural_validity is False

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_payload_without_messages_is_opaque_and_lossless():
    payload = {
        "prompt": "Complete esta frase...",
        "max_tokens": 100,
        "custom_completion_field": True,
    }
    ir = parse_request_ir(payload)
    assert ir.is_opaque is True

    reconstructed = reconstruct_payload(ir)
    assert reconstructed == payload


def test_non_dict_payload_or_messages():
    ir1 = parse_request_ir("str_payload")  # type: ignore
    assert ir1.is_opaque is True

    ir2 = parse_request_ir({"messages": "not_a_list"})
    assert ir2.is_opaque is True


def test_tools_response_format_and_total_token_calculation():
    payload = {
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "Busque novidades"}],
    }
    ir = parse_request_ir(payload)
    assert not ir.is_opaque
    assert ir.required_capabilities.tools is True
    assert ir.required_capabilities.structured_output is True

    tokens = ir.calculate_total_tokens()
    assert tokens > 0

    reconstructed = ir.to_payload()
    assert reconstructed == payload
