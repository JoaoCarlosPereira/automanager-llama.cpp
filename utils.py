"""Shared utility functions for automanager."""

import re
from typing import Any, Optional


def mask_api_key(api_key: Any) -> str:
    """Return a safe masked representation of *api_key*.

    - Keys starting with ``sk-`` show the first 6 characters then ``****...****``.
    - All other strings are returned unchanged (prefix too short or different format).
    - ``None`` / empty strings return ``""``.
    - Non-string truthy values are converted to string and processed.

    Examples:
        >>> mask_api_key("sk-12345")
        'sk-123****...****'
        >>> mask_api_key("outra-string")
        'outra-string'
        >>> mask_api_key("")
        ''
        >>> mask_api_key(None)
        ''
    """
    if not api_key:
        return ""
    # Coerce non-string types to str so callers don't get AttributeError.
    api_key = str(api_key)
    # Only mask keys that look like sk- prefixed tokens.
    if api_key.startswith("sk-"):
        if len(api_key) <= 6:
            return api_key + "****"
        return api_key[:6] + "****...****"
    # Keys from other providers (e.g. anthropic-..., hc-...) — do not mask;
    # they have no stable prefix convention we can safely expose.
    return api_key


def sanitize_for_log(value: Any, key_name: str = "credential") -> str:
    """Return a sanitized version of *value* for safe inclusion in logs.

    If *value* looks like an api_key (starts with ``sk-``) it is masked.
    Otherwise returns the string representation unchanged.
    """
    s = str(value) if value is not None else ""
    if s.startswith("sk-"):
        return mask_api_key(s)
    return s


def sanitize_dict_for_display(data: dict, sensitive_keys: list[str]) -> dict:
    """Return a copy of *data* with *sensitive_keys* masked in-place."""
    result = dict(data)
    for key in sensitive_keys:
        if key in result and isinstance(result[key], str):
            result[key] = mask_api_key(result[key])
    return result
