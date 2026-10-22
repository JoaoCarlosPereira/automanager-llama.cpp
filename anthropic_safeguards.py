"""Compatibilidade Claude Code auto mode (safeguards / safeguard_results).

Claude Code v2.1.278+ envia ``safeguards`` em ``POST /v1/messages`` e espera
``safeguard_results`` na resposta. Backends locais (llama-server) e a maioria
das plataformas não autores esses veredictos; sem eles o cliente exibe o aviso
de gateway incompatível e cai no classificador local cobrado.

Este módulo:
1. Detecta o envelope ``safeguards`` na requisição.
2. Remove-o antes de encaminhar a backends que não o entendem.
3. Sintetiza ``safeguard_results`` (JSON e SSE) no formato capturado pela API
   Anthropic / LiteLLM, com avaliação heurística conservadora.

Referências:
- https://code.claude.com/docs/en/auto-mode-classifier-billing
- https://code.claude.com/docs/en/llm-gateway-protocol
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("automanager")

DANGEROUS_TOOL_USE = "dangerous_tool_use"
SAFEGUARDS_BETA = "dangerous-tool-use-2026-09-03"

# Provedores que podem devolver safeguard_results nativos (Anthropic).
_NATIVE_SAFEGUARDS_PROVIDERS = frozenset({"claude", "anthropic"})

# Padrões Bash tipicamente perigosos fora do que o usuário pediu.
_BASH_DANGEROUS = (
    re.compile(r"(curl|wget)\s+[^\n|;]*\|\s*(?:ba)?sh\b", re.I),
    re.compile(r"\brm\s+(?:-[^\s]*\s+)*-(?:[^\s]*f|rf|fr)\b", re.I),
    re.compile(r"\bmkfs\b|\bdd\s+if=", re.I),
    re.compile(r"\bchmod\s+(?:-R\s+)?777\b", re.I),
    re.compile(r"(?:^|[;&|]\s*)shutdown\b|(?:^|[;&|]\s*)reboot\b", re.I),
    re.compile(r"\b(?:scp|rsync|curl|wget)\b[^\n]*(?:id_rsa|\.pem|\.env\b|aws/credentials)", re.I),
    re.compile(r"\bcat\s+[^\n]*(?:id_rsa|\.pem|/etc/shadow)", re.I),
    re.compile(r"\b(?:base64|xxd)\b[^\n]*\|\s*(?:ba)?sh\b", re.I),
)


def request_has_safeguards(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    safeguards = body.get("safeguards")
    return isinstance(safeguards, list) and len(safeguards) > 0


def peek_safeguards(body: Mapping[str, Any]) -> Optional[List[Any]]:
    if not request_has_safeguards(body):
        return None
    return list(body.get("safeguards") or [])


def backend_supports_native_safeguards(provider: Optional[str]) -> bool:
    return str(provider or "").strip().lower() in _NATIVE_SAFEGUARDS_PROVIDERS


def strip_safeguards(body: Dict[str, Any]) -> Dict[str, Any]:
    """Remove o campo safeguards para não quebrar backends estritos."""
    if "safeguards" not in body:
        return body
    cleaned = dict(body)
    cleaned.pop("safeguards", None)
    return cleaned


def prepare_forward_body(
    body: Dict[str, Any],
    *,
    provider: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[List[Any]]]:
    """Retorna (corpo a encaminhar, safeguards originais ou None)."""
    safeguards = peek_safeguards(body)
    if safeguards is None:
        return body, None
    if backend_supports_native_safeguards(provider):
        return body, safeguards
    return strip_safeguards(body), safeguards


def _classifier_context(safeguards: Optional[List[Any]]) -> Dict[str, Any]:
    if not safeguards:
        return {}
    for entry in safeguards:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != DANGEROUS_TOOL_USE:
            continue
        ctx = entry.get("classifier_context")
        return ctx if isinstance(ctx, dict) else {}
    return {}


def _trusted_paths(ctx: Mapping[str, Any]) -> List[str]:
    trusted = ctx.get("trusted_directories")
    paths: List[str] = []
    if isinstance(trusted, dict):
        primary = trusted.get("primary")
        if isinstance(primary, dict):
            for key in ("path", "resolved"):
                value = primary.get(key)
                if isinstance(value, str) and value:
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(str(item) for item in value if item)
        additional = trusted.get("additional")
        if isinstance(additional, list):
            for item in additional:
                if isinstance(item, str) and item:
                    paths.append(item)
                elif isinstance(item, dict):
                    for key in ("path", "resolved"):
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            paths.append(value)
    elif isinstance(trusted, list):
        paths.extend(str(item) for item in trusted if item)
    live_cwd = ctx.get("live_cwd") or ctx.get("cwd")
    if isinstance(live_cwd, str) and live_cwd:
        paths.append(live_cwd)
    # Dedup preservando ordem
    seen = set()
    ordered: List[str] = []
    for path in paths:
        norm = path.rstrip("/") or path
        if norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered


def _path_is_trusted(path: str, trusted: List[str]) -> bool:
    if not path or not trusted:
        return True  # sem contexto → não inventar deny
    candidate = path if path.startswith("/") else path
    for root in trusted:
        root_n = root.rstrip("/") or root
        if candidate == root_n or candidate.startswith(root_n + "/"):
            return True
        # caminho relativo: confiar se não sobe de diretório
        if not candidate.startswith("/") and ".." not in candidate.split("/"):
            return True
    return False


def _tool_input_as_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {}


def evaluate_tool_use(
    name: str,
    tool_input: Any,
    *,
    safeguards: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Avalia uma tool_use; fail-closed só em padrões claramente perigosos."""
    ctx = _classifier_context(safeguards)
    trusted = _trusted_paths(ctx)
    payload = _tool_input_as_dict(tool_input)
    lowered_name = str(name or "").strip()

    if lowered_name in {"Bash", "bash", "Shell", "shell"}:
        command = str(payload.get("command") or payload.get("_raw") or "")
        for pattern in _BASH_DANGEROUS:
            if pattern.search(command):
                return {
                    "type": "evaluated",
                    "outcome": "flagged",
                    "explanation": "Comando de shell potencialmente destrutivo ou de exfiltração.",
                }

    if lowered_name in {"Write", "Edit", "NotebookEdit", "write", "edit"}:
        path = str(
            payload.get("file_path")
            or payload.get("path")
            or payload.get("notebook_path")
            or ""
        )
        if path and not _path_is_trusted(path, trusted):
            return {
                "type": "evaluated",
                "outcome": "flagged",
                "explanation": "Escrita fora dos diretórios confiáveis da sessão.",
            }

    return {"type": "evaluated", "outcome": "not_flagged"}


def extract_tool_uses_from_message(content: Any) -> List[Dict[str, Any]]:
    """Extrai blocos tool_use de uma resposta Anthropic Messages."""
    tools: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return tools
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        tool_id = block.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        tools.append(
            {
                "id": tool_id,
                "name": str(block.get("name") or ""),
                "input": block.get("input"),
            }
        )
    return tools


def build_safeguard_results(
    tool_uses: List[Dict[str, Any]],
    *,
    safeguards: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Monta o envelope safeguard_results no formato capturado da API."""
    evaluations: Dict[str, Dict[str, Any]] = {}
    for tool in tool_uses:
        tool_id = tool.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        evaluations[tool_id] = evaluate_tool_use(
            str(tool.get("name") or ""),
            tool.get("input"),
            safeguards=safeguards,
        )
    return [
        {
            "type": DANGEROUS_TOOL_USE,
            "status": {
                "type": "available",
                "tool_uses": evaluations,
            },
        }
    ]


def _has_available_safeguard_results(obj: Mapping[str, Any]) -> bool:
    results = obj.get("safeguard_results")
    if not isinstance(results, list) or not results:
        return False
    for entry in results:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if isinstance(status, dict) and status.get("type") == "available":
            return True
    return False


def attach_safeguard_results_to_json(
    content: bytes,
    safeguards: Optional[List[Any]],
) -> bytes:
    """Injeta safeguard_results em resposta JSON não-stream, se ausentes."""
    if not safeguards:
        return content
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return content
    if not isinstance(obj, dict):
        return content
    if _has_available_safeguard_results(obj):
        return content
    tool_uses = extract_tool_uses_from_message(obj.get("content"))
    obj["safeguard_results"] = build_safeguard_results(
        tool_uses, safeguards=safeguards
    )
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _parse_sse_data_line(line: bytes) -> Tuple[Optional[dict], bytes, bool]:
    """Retorna (objeto|None, linha reemitível, tinha CR)."""
    has_cr = line.endswith(b"\r")
    stripped = line[:-1] if has_cr else line
    if not stripped.lstrip().startswith(b"data:"):
        return None, line, has_cr
    prefix_len = stripped.index(b"data:") + len(b"data:")
    payload = stripped[prefix_len:].strip()
    if not payload or payload == b"[DONE]":
        return None, line, has_cr
    try:
        obj = json.loads(payload)
    except (TypeError, ValueError):
        return None, line, has_cr
    if not isinstance(obj, dict):
        return None, line, has_cr
    return obj, line, has_cr


def _format_sse_data(obj: dict, *, has_cr: bool) -> bytes:
    encoded = b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return encoded + b"\r" if has_cr else encoded


async def attach_safeguard_results_to_sse(
    byte_iter: AsyncIterator[bytes],
    safeguards: Optional[List[Any]],
) -> AsyncIterator[bytes]:
    """Injeta safeguard_results no delta do message_delta final (LiteLLM/Anthropic)."""
    if not safeguards:
        async for chunk in byte_iter:
            yield chunk
        return

    line_buffer = b""
    pending_tools: Dict[str, Dict[str, Any]] = {}
    current_tool_id: Optional[str] = None

    async for chunk in byte_iter:
        line_buffer += chunk
        while b"\n" in line_buffer:
            line, line_buffer = line_buffer.split(b"\n", 1)
            obj, original, has_cr = _parse_sse_data_line(line)
            if obj is None:
                yield original + b"\n"
                continue

            event_type = obj.get("type")
            if event_type == "content_block_start":
                block = obj.get("content_block")
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if isinstance(tool_id, str) and tool_id:
                        pending_tools[tool_id] = {
                            "id": tool_id,
                            "name": str(block.get("name") or ""),
                            "input_parts": [],
                            "input": block.get("input")
                            if isinstance(block.get("input"), dict)
                            else {},
                        }
                        current_tool_id = tool_id
                yield original + b"\n"
                continue

            if event_type == "content_block_delta":
                delta = obj.get("delta")
                if (
                    isinstance(delta, dict)
                    and delta.get("type") == "input_json_delta"
                    and current_tool_id
                    and current_tool_id in pending_tools
                ):
                    partial = delta.get("partial_json")
                    if isinstance(partial, str) and partial:
                        pending_tools[current_tool_id]["input_parts"].append(partial)
                yield original + b"\n"
                continue

            if event_type == "content_block_stop":
                if current_tool_id and current_tool_id in pending_tools:
                    entry = pending_tools[current_tool_id]
                    joined = "".join(entry.get("input_parts") or [])
                    if joined:
                        try:
                            parsed = json.loads(joined)
                        except (TypeError, ValueError):
                            parsed = {"_raw": joined}
                        if isinstance(parsed, dict):
                            entry["input"] = parsed
                    current_tool_id = None
                yield original + b"\n"
                continue

            if event_type == "message_delta":
                if _has_available_safeguard_results(obj.get("delta") or {}):
                    yield original + b"\n"
                    continue
                # Alguns upstreams colocam safeguard_results no topo do evento.
                if _has_available_safeguard_results(obj):
                    yield original + b"\n"
                    continue
                tool_list = [
                    {
                        "id": tool_id,
                        "name": meta.get("name") or "",
                        "input": meta.get("input") or {},
                    }
                    for tool_id, meta in pending_tools.items()
                ]
                results = build_safeguard_results(
                    tool_list, safeguards=safeguards
                )
                delta = obj.get("delta")
                if not isinstance(delta, dict):
                    delta = {}
                    obj["delta"] = delta
                delta["safeguard_results"] = results
                yield _format_sse_data(obj, has_cr=has_cr) + b"\n"
                continue

            yield original + b"\n"

    if line_buffer:
        yield line_buffer
