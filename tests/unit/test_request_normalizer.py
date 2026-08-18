"""Tests for malformed historical tool-call normalization."""

import json

from request_normalizer import (
    RECOVERED_ARGUMENTS_FIELD,
    normalize_tool_call_arguments,
)


def _payload(arguments):
    return {
        "model": "model.gguf",
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write", "arguments": arguments},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        "tools": [{
            "type": "function",
            "function": {"name": "write", "parameters": {"type": "object"}},
        }],
    }


def test_valid_json_arguments_are_untouched_without_copy():
    payload = _payload('{"path":"ok"}')

    normalized, repairs = normalize_tool_call_arguments(payload)

    assert normalized is payload
    assert repairs == 0


def test_truncated_arguments_are_repaired_and_raw_text_is_preserved():
    raw = '{"task":"unfinished'
    payload = _payload(raw)

    normalized, repairs = normalize_tool_call_arguments(payload)
    repaired = normalized["messages"][1]["tool_calls"][0]["function"]["arguments"]

    assert repairs == 1
    assert normalized is not payload
    assert json.loads(repaired) == {RECOVERED_ARGUMENTS_FIELD: raw}
    assert payload["messages"][1]["tool_calls"][0]["function"]["arguments"] == raw
    assert normalized["tools"] is payload["tools"]


def test_non_string_arguments_are_serialized_for_llama_cpp():
    payload = _payload({"path": "already-decoded"})

    normalized, repairs = normalize_tool_call_arguments(payload)
    repaired = normalized["messages"][1]["tool_calls"][0]["function"]["arguments"]

    assert repairs == 1
    assert json.loads(repaired) == {"path": "already-decoded"}


def test_user_content_and_top_level_tool_schema_are_not_modified():
    payload = _payload('{"broken":')
    original_tools = payload["tools"]
    original_user = payload["messages"][0]

    normalized, repairs = normalize_tool_call_arguments(payload)

    assert repairs == 1
    assert normalized["tools"] is original_tools
    assert normalized["messages"][0] is original_user
