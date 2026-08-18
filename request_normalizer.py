"""Safe normalization for malformed OpenAI-compatible request history."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple


RECOVERED_ARGUMENTS_FIELD = "_automanager_recovered_raw_arguments"


def _normalized_arguments(value: Any) -> tuple[str, bool]:
    """Return llama.cpp-compatible JSON arguments and whether they changed."""
    if isinstance(value, str):
        try:
            json.loads(value)
            return value, False
        except (json.JSONDecodeError, TypeError):
            recovered = {RECOVERED_ARGUMENTS_FIELD: value}
            return json.dumps(
                recovered, ensure_ascii=False, separators=(",", ":")
            ), True

    try:
        return json.dumps(
            value if value is not None else {},
            ensure_ascii=False,
            separators=(",", ":"),
        ), True
    except (TypeError, ValueError):
        recovered = {RECOVERED_ARGUMENTS_FIELD: str(value)}
        return json.dumps(
            recovered, ensure_ascii=False, separators=(",", ":")
        ), True


def normalize_tool_call_arguments(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Repair invalid historical ``function.arguments`` without mutating input.

    OpenAI-compatible clients store tool-call arguments as JSON strings. An
    interrupted stream can leave one of those strings truncated. llama.cpp's
    chat-template parser rejects the entire history in that case. Valid strings
    remain byte-for-byte unchanged; malformed raw text is preserved inside a
    valid JSON object so both tokenization and inference see identical input.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, 0

    normalized_payload = payload
    normalized_messages = messages
    repairs = 0

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or "arguments" not in function:
                continue

            arguments, changed = _normalized_arguments(function.get("arguments"))
            if not changed:
                continue

            if normalized_payload is payload:
                normalized_payload = dict(payload)
                normalized_messages = list(messages)
                normalized_payload["messages"] = normalized_messages

            current_message = normalized_messages[message_index]
            if current_message is message:
                current_message = dict(message)
                current_message["tool_calls"] = list(tool_calls)
                normalized_messages[message_index] = current_message

            current_calls = current_message["tool_calls"]
            current_call = dict(current_calls[call_index])
            current_function = dict(function)
            current_function["arguments"] = arguments
            current_call["function"] = current_function
            current_calls[call_index] = current_call
            repairs += 1

    return normalized_payload, repairs
