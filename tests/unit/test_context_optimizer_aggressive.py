"""Unit tests for the Aggressive reduction stage and 413 context_too_large handling (Task 11)."""

import pytest

from context_optimizer import (
    ContextOptimizer,
    ContextTooLargeError,
    LimitConfidence,
    ModelLimits,
    OptimizationAudit,
    OptimizationResult,
    calculate_target_budget,
    optimize_request_ir_aggressive,
    parse_request_ir,
    validate_transformed_payload,
)


@pytest.mark.asyncio
async def test_aggressive_executes_only_when_limit_is_known():
    """Testa se Aggressive não executa quando o limite de contexto é UNKNOWN."""
    payload = {
        "messages": [
            {"role": "system", "content": "Sistema."},
            {"role": "user", "content": "Sim, usar PostgreSQL para o banco."},  # decisão técnica antiga
            {"role": "user", "content": "Pergunta do turno atual"},
        ],
    }

    ir = parse_request_ir(payload)
    unknown_limits = ModelLimits(context_tokens=None, max_output_tokens=None, source="platform", confidence=LimitConfidence.UNKNOWN)
    unknown_budget = calculate_target_budget(payload, unknown_limits, frozenset({"text"}))

    result = await optimize_request_ir_aggressive(ir, unknown_budget)

    # Nenhuma remoção ocorre com limite desconhecido
    assert result.safe_payload == payload
    assert result.audit.blocks_removed == 0


@pytest.mark.asyncio
async def test_aggressive_removes_old_technical_decisions_code_and_logs():
    """Testa se mensagens antigas contendo decisões técnicas, código e logs são removidas no Aggressive."""
    payload = {
        "messages": [
            {"role": "system", "content": "Sistema de ajuda."},
            {"role": "user", "content": "Sim, usar PostgreSQL para o banco principal."},  # decisão técnica antiga
            {"role": "assistant", "content": "PostgreSQL configurado."},
            {"role": "user", "content": "```python\ndef foo(): pass\n```"},  # código antigo
            {"role": "assistant", "content": "[ERROR] Fallback triggered"},  # log antigo
            {"role": "user", "content": "Como otimizar a consulta final?"},  # turno atual (protegido)
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    orig_cost = ir.calculate_total_tokens()

    # Budget ajustado para couber após remover mensagens antigas
    from context_optimizer import TargetBudget
    small_budget = TargetBudget(
        context_limit=limits.context_tokens,
        output_reserve=limits.max_output_tokens or 200,
        protocol_overhead=480,
        safety_margin=300,
        input_budget=50,  # força remoção das mensagens antigas desprotegidas
        confidence=LimitConfidence.KNOWN_LOCAL,
        source="local",
        capabilities=frozenset({"text"}),
    )

    result = await optimize_request_ir_aggressive(ir, small_budget)

    assert result.audit.strategy == "aggressive"
    assert result.audit.blocks_removed > 0

    messages = result.safe_payload["messages"]
    contents = [m["content"] for m in messages]

    # Decisão técnica, código e logs antigos devem ter sido removidos
    assert "Sim, usar PostgreSQL para o banco principal." not in contents
    assert "```python\ndef foo(): pass\n```" not in contents
    assert "[ERROR] Fallback triggered" not in contents

    # System e turno atual obrigatoriamente mantidos
    assert contents[0] == "Sistema de ajuda."
    assert contents[-1] == "Como otimizar a consulta final?"


@pytest.mark.asyncio
async def test_aggressive_preserves_system_developer_current_turn_and_tool_dependencies():
    """Testa se o conjunto mínimo obrigatório (system, developer, turno atual e tools atômicas acopladas) é preservado."""
    payload = {
        "tools": [{"type": "function", "function": {"name": "get_status"}}],
        "messages": [
            {"role": "system", "content": "System prompt intocável."},
            {"role": "developer", "content": "Developer prompt intocável."},
            {"role": "user", "content": "Sim, usar PostgreSQL para o banco."},  # antigo (removível no Aggressive)
            {"role": "user", "content": "Executar verificação no sistema"},  # turno atual
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_curr_1", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_curr_1", "content": '{"status": "ok"}'},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    orig_cost = ir.calculate_total_tokens()

    from context_optimizer import TargetBudget
    small_budget = TargetBudget(
        context_limit=limits.context_tokens,
        output_reserve=limits.max_output_tokens or 200,
        protocol_overhead=400,
        safety_margin=300,
        input_budget=orig_cost - 10,
        confidence=LimitConfidence.KNOWN_LOCAL,
        source="local",
        capabilities=frozenset({"text", "tools"}),
    )

    result = await optimize_request_ir_aggressive(ir, small_budget)

    # Validar estruturalmente
    min_protected_ids = {b.block_id for b in ir.ordered_units if b.role in ("system", "developer") or b.original_index >= 5}
    report = validate_transformed_payload(ir, result.safe_payload, override_protected_ids=min_protected_ids)
    assert report.valid is True

    messages = result.safe_payload["messages"]
    contents = [m.get("content") for m in messages]

    # System e Developer intocados
    assert contents[0] == "System prompt intocável."
    assert contents[1] == "Developer prompt intocável."

    # Antigo removido
    assert "Sim, usar PostgreSQL para o banco." not in contents

    # Turno atual e tool_calls/results acoplados mantidos
    assert "Executar verificação no sistema" in contents
    tool_call_ids = [tc["id"] for m in messages if "tool_calls" in m and m["tool_calls"] for tc in m["tool_calls"]]
    tool_result_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]

    assert set(tool_call_ids) == {"call_curr_1"}
    assert set(tool_result_ids) == {"call_curr_1"}


@pytest.mark.asyncio
async def test_aggressive_never_truncates_partially_nor_adds_artificial_markers():
    """Testa se o Aggressive nunca trunca textos parcialmente nem adiciona sentinelas/marcadores artificiais."""
    payload = {
        "messages": [
            {"role": "system", "content": "Sistema de teste de integridade."},
            {"role": "user", "content": "Mensagem antiga com texto longo para ser descartada por inteiro se necessário."},
            {"role": "user", "content": "Mensagem do turno atual com texto longo e preciso sem cortes."},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    small_budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=450, safety_margin=300)

    result = await optimize_request_ir_aggressive(ir, small_budget)

    for msg in result.safe_payload["messages"]:
        content = msg["content"]
        assert "...[truncated]..." not in content
        assert "[truncated]" not in content
        assert "..." not in content or "..." in payload["messages"][0]["content"] or "..." in payload["messages"][2]["content"]

    # Conteúdo do turno atual deve ser byte-idêntico ao original
    assert result.safe_payload["messages"][-1]["content"] == payload["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_aggressive_exceeding_minimum_protected_set_triggers_context_too_large_413():
    """Testa se quando o conjunto mínimo protegido isoladamente excede o orçamento, ContextOptimizer lança ContextTooLargeError 413."""
    long_system = "Instrução de sistema extremamente longa. " * 100
    long_user = "Turno atual com payload muito grande. " * 100

    payload = {
        "messages": [
            {"role": "system", "content": long_system},
            {"role": "user", "content": long_user},
        ],
    }

    # Contexto tão pequeno que nem o conjunto mínimo protegido consegue caber
    backend_info = {"backend_type": "local", "context_size": 100, "parallel_slots": 1}
    optimizer = ContextOptimizer()

    with pytest.raises(ContextTooLargeError) as exc_info:
        await optimizer.optimize(payload, backend_info)

    err = exc_info.value
    assert err.status_code == 413
    assert err.code == "context_too_large"
    assert "Prompt excede o limite de contexto" in err.message


@pytest.mark.asyncio
async def test_aggressive_stops_as_soon_as_budget_is_met():
    """Testa se a remoção no Aggressive é interrompida imediatamente assim que o orçamento couber."""
    payload = {
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Antigo 1 - texto removível 1"},
            {"role": "user", "content": "Antigo 2 - texto removível 2"},
            {"role": "user", "content": "Antigo 3 - texto removível 3"},
            {"role": "user", "content": "Turno atual"},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    total_tokens = ir.calculate_total_tokens()

    # Budget dimensionado para caber assim que 1 mensagem antiga for removida
    from context_optimizer import TargetBudget
    custom_budget = TargetBudget(
        context_limit=1000,
        output_reserve=200,
        protocol_overhead=0,
        safety_margin=0,
        input_budget=total_tokens - 10,
        confidence=LimitConfidence.KNOWN_LOCAL,
        source="local",
        capabilities=frozenset({"text"}),
    )

    result = await optimize_request_ir_aggressive(ir, custom_budget)

    assert result.audit.strategy == "aggressive"
    assert result.audit.blocks_removed < 3
    assert result.audit.optimized_cost <= total_tokens - 10
