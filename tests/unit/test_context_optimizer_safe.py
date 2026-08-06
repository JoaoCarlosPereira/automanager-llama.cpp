"""Testes unitários para o pipeline de otimização modo Safe (Task 6)."""

import pytest

from context_optimizer import (
    LimitConfidence,
    ModelLimits,
    OptimizationAudit,
    OptimizationResult,
    calculate_target_budget,
    optimize_request_ir_safe,
    parse_request_ir,
)


@pytest.mark.asyncio
async def test_safe_mode_returns_metadata_only_audit_and_payload():
    """Testa se optimize_request_ir_safe retorna audit metadata-only e o payload seguro."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": "Olá,   como   vai   você?  "},
            {"role": "assistant", "content": "Estou bem, obrigado!"},
            {"role": "user", "content": "Tudo bem!"},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=16384, max_output_tokens=2048, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}))

    result = await optimize_request_ir_safe(ir, budget)

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.audit, OptimizationAudit)
    assert result.audit.strategy == "safe"
    assert result.audit.original_cost >= result.audit.optimized_cost
    assert result.audit.savings_tokens >= 0
    assert result.audit.validation_passed is True
    assert isinstance(result.safe_payload, dict)


@pytest.mark.asyncio
async def test_safe_mode_preserves_opaque_payload():
    """Testa se o modo Safe não altera payloads opacos."""
    payload = {
        "prompt": "Complete a frase...",
        "max_tokens": 100,
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=16384, max_output_tokens=2048, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}))

    result = await optimize_request_ir_safe(ir, budget)

    assert result.audit.original_cost == result.audit.optimized_cost
    assert result.audit.savings_tokens == 0
    assert result.safe_payload == payload


@pytest.mark.asyncio
async def test_safe_mode_removes_empty_blocks():
    """Testa se o modo Safe remove blocos vazios não protegidos."""
    payload = {
        "messages": [
            {"role": "system", "content": "Instrução inicial"},
            {"role": "assistant", "content": ""},  # bloco vazio não protegido
            {"role": "user", "content": "Pergunta"},  # mensagem final protegida
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=16384, max_output_tokens=2048, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}))

    result = await optimize_request_ir_safe(ir, budget)

    assert result.audit.blocks_removed > 0
    assert len(result.safe_payload["messages"]) < len(payload["messages"])


@pytest.mark.asyncio
async def test_safe_mode_deduplicates_duplicate_consecutive_blocks():
    """Testa se o modo Safe remove blocos duplicados consecutivos desprotegidos."""
    payload = {
        "messages": [
            {"role": "system", "content": "Instrução"},
            {"role": "assistant", "content": "Ok"},
            {"role": "assistant", "content": "Ok"},  # duplicado consecutivo
            {"role": "user", "content": "Pergunta"},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=16384, max_output_tokens=2048, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}))

    result = await optimize_request_ir_safe(ir, budget)

    assert result.audit.blocks_deduplicated > 0 or result.audit.blocks_merged > 0
    assert len(result.safe_payload["messages"]) == 3
