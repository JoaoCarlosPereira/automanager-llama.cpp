"""Testes unitários para a redução Moderate do Context Optimizer (Task 10)."""

import pytest

from context_optimizer import (
    LimitConfidence,
    ModelLimits,
    OptimizationAudit,
    OptimizationResult,
    calculate_target_budget,
    optimize_request_ir_moderate,
    parse_request_ir,
    validate_transformed_payload,
)


@pytest.mark.asyncio
async def test_moderate_removes_old_isolated_social_noise():
    """Testa se 'Obrigado, pode continuar' antigo é removido quando necessário."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": "Primeira instrução do usuário."},
            {"role": "assistant", "content": "Ok, entendi."},
            {"role": "user", "content": "Obrigado, pode continuar"},  # ruído social isolado
            {"role": "assistant", "content": "Continuando a tarefa com os detalhes..."},
            {"role": "user", "content": "Qual é a resposta final?"},  # turno atual (protegido)
        ],
    }

    ir = parse_request_ir(payload)
    # Definir um orçamento pequeno o suficiente para exigir remoção
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}))

    # Forçar input_budget baixo para acionar Moderate
    small_budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=400, safety_margin=300)

    result = await optimize_request_ir_moderate(ir, small_budget)

    assert result.audit.strategy == "moderate"
    assert result.audit.blocks_removed > 0
    messages = result.safe_payload["messages"]
    # Garante que "Obrigado, pode continuar" foi removido
    contents = [m["content"] for m in messages]
    assert "Obrigado, pode continuar" not in contents
    # System e pergunta final devem ser preservados
    assert contents[0] == "Você é um assistente útil."
    assert contents[-1] == "Qual é a resposta final?"


@pytest.mark.asyncio
async def test_moderate_preserves_technical_decision_and_subsequent_changes():
    """Testa se 'Sim, usar PostgreSQL' e mudanças posteriores são preservadas."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Assistente de arquitetura."},
            {"role": "user", "content": "Obrigado, entendi."},  # ruído
            {"role": "user", "content": "Sim, usar PostgreSQL para o banco de dados principal."},  # decisão técnica
            {"role": "assistant", "content": "PostgreSQL configurado."},
            {"role": "user", "content": "Mudar banco para MySQL ao invés de PostgreSQL."},  # mudança de requisito
            {"role": "user", "content": "Mostre a configuração atual."},  # turno atual
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=450, safety_margin=300)

    result = await optimize_request_ir_moderate(ir, budget)

    contents = [m["content"] for m in result.safe_payload["messages"]]
    assert "Sim, usar PostgreSQL para o banco de dados principal." in contents
    assert "Mudar banco para MySQL ao invés de PostgreSQL." in contents
    assert "Obrigado, entendi." not in contents


@pytest.mark.asyncio
async def test_moderate_never_removes_system_developer_or_current_turn():
    """Testa se system, developer e turno atual são imutavelmente protegidos."""
    payload = {
        "messages": [
            {"role": "system", "content": "System prompt intocável."},
            {"role": "developer", "content": "Developer instruction intocável."},
            {"role": "user", "content": "Obrigado"},
            {"role": "assistant", "content": "De nada"},
            {"role": "user", "content": "Solicitação do turno atual do usuário."},
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=500, max_output_tokens=100, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=250, safety_margin=100)

    result = await optimize_request_ir_moderate(ir, budget)

    messages = result.safe_payload["messages"]
    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]

    assert "system" in roles
    assert "developer" in roles
    assert contents[0] == "System prompt intocável."
    assert contents[1] == "Developer instruction intocável."
    assert contents[-1] == "Solicitação do turno atual do usuário."


@pytest.mark.asyncio
async def test_moderate_removes_or_preserves_tool_groups_atomically():
    """Testa se grupos de chamadas de ferramenta são removidos ou preservados integralmente."""
    payload = {
        "tools": [{"type": "function", "function": {"name": "query_db"}}],
        "messages": [
            {"role": "system", "content": "Assistente."},
            {"role": "user", "content": "Obrigado, entendi."},
            # Grupo atômico antigo 1
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_old_1", "type": "function", "function": {"name": "query_db", "arguments": '{"table":"users"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_old_1", "content": '{"rows": 50}'},
            # Grupo atômico antigo 2
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_old_2", "type": "function", "function": {"name": "query_db", "arguments": '{"table":"logs"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_old_2", "content": '{"rows": 100}'},
            {"role": "user", "content": "Qual o total de registros?"},  # turno atual
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=800, max_output_tokens=100, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text", "tools"}), protocol_overhead=400, safety_margin=200)

    result = await optimize_request_ir_moderate(ir, budget)

    # Validar estruturalmente o resultado
    report = validate_transformed_payload(ir, result.safe_payload)
    assert report.valid is True

    # Verificar integridade atômica dos tool_calls
    tool_call_ids = []
    tool_result_ids = []
    for msg in result.safe_payload["messages"]:
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                tool_call_ids.append(tc["id"])
        if msg.get("role") == "tool":
            tool_result_ids.append(msg["tool_call_id"])

    # Todos os tool_calls que permaneceram devem ter seus tool_results e vice-versa
    assert set(tool_call_ids) == set(tool_result_ids)


@pytest.mark.asyncio
async def test_moderate_preserves_code_sql_urls_ids_and_filenames_without_rewriting():
    """Testa se código, SQL, URLs, IDs e nomes de arquivos não são reescritos nem truncados."""
    code_snippet = "```python\ndef process_data(x):\n    return x * 2\n```"
    sql_snippet = "SELECT id, name FROM users WHERE active = true"
    url_str = "https://api.example.com/v1/resource"
    uuid_str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    file_path = "src/utils/config_manager.py"

    payload = {
        "messages": [
            {"role": "system", "content": "Você é um assistente de código."},
            {"role": "user", "content": f"Confira o arquivo {file_path} com UUID {uuid_str}: {url_str}"},
            {"role": "assistant", "content": f"Use o SQL: {sql_snippet} e o código:\n{code_snippet}"},
            {"role": "user", "content": "Execute a migração."},  # turno atual
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=400, safety_margin=200)

    result = await optimize_request_ir_moderate(ir, budget)

    contents = [m["content"] for m in result.safe_payload["messages"]]
    joined_contents = "\n".join(contents)
    assert code_snippet in joined_contents
    assert sql_snippet in joined_contents
    assert url_str in joined_contents
    assert uuid_str in joined_contents
    assert file_path in joined_contents


@pytest.mark.asyncio
async def test_moderate_preserves_important_logs():
    """Testa se logs contendo erros, warnings e stack traces são preservados."""
    log_text = """[2026-08-06 12:00:00] INFO Starting service
[2026-08-06 12:00:01] ERROR Database connection failed
Traceback (most recent call last):
  File "db.py", line 45, in connect
    raise ConnectionError("Timeout")
[2026-08-06 12:00:02] WARN Retrying connection..."""

    payload = {
        "messages": [
            {"role": "system", "content": "Analisador de logs."},
            {"role": "assistant", "content": log_text},
            {"role": "user", "content": "Como resolver este erro?"},  # turno atual
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=400, safety_margin=200)

    result = await optimize_request_ir_moderate(ir, budget)

    contents = [m["content"] for m in result.safe_payload["messages"]]
    assert log_text in contents


@pytest.mark.asyncio
async def test_moderate_does_not_execute_when_limit_is_unknown_and_is_idempotent():
    """Testa se Moderate não executa com limite desconhecido e é idempotente."""
    payload = {
        "messages": [
            {"role": "system", "content": "Sistema."},
            {"role": "user", "content": "Obrigado, pode continuar"},
            {"role": "user", "content": "Pergunta final"},
        ],
    }

    ir = parse_request_ir(payload)

    # Limite desconhecido
    unknown_limits = ModelLimits(context_tokens=None, max_output_tokens=None, source="platform", confidence=LimitConfidence.UNKNOWN)
    unknown_budget = calculate_target_budget(payload, unknown_limits, frozenset({"text"}))

    result_unknown = await optimize_request_ir_moderate(ir, unknown_budget)

    # Não deve alterar o payload quando o limite é desconhecido
    assert result_unknown.safe_payload == payload
    assert result_unknown.audit.blocks_removed == 0

    # Teste de Idempotência com limite conhecido
    known_limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)
    small_budget = calculate_target_budget(payload, known_limits, frozenset({"text"}), protocol_overhead=450, safety_margin=300)

    res1 = await optimize_request_ir_moderate(ir, small_budget)
    ir2 = parse_request_ir(res1.safe_payload)
    res2 = await optimize_request_ir_moderate(ir2, small_budget)

    assert res1.safe_payload == res2.safe_payload


@pytest.mark.asyncio
async def test_moderate_stops_as_soon_as_budget_is_met():
    """Testa se a remoção interrompe imediatamente assim que o orçamento couber."""
    payload = {
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Obrigado, pode continuar 1"},  # ruído 1
            {"role": "user", "content": "Obrigado, pode continuar 2"},  # ruído 2
            {"role": "user", "content": "Obrigado, pode continuar 3"},  # ruído 3
            {"role": "user", "content": "Pergunta final"},  # turno atual
        ],
    }

    ir = parse_request_ir(payload)
    limits = ModelLimits(context_tokens=1000, max_output_tokens=200, source="local", confidence=LimitConfidence.KNOWN_LOCAL)

    # Ajustar orçamento para caber após remover apenas 1 ou 2 itens de ruído
    total_tokens = ir.calculate_total_tokens()
    target_budget_tokens = total_tokens - 15  # Remover 1 ruído já deve bastar

    budget = calculate_target_budget(payload, limits, frozenset({"text"}), protocol_overhead=0, safety_margin=0)
    # Criar budget customizado com input_budget específico
    from context_optimizer import TargetBudget
    custom_budget = TargetBudget(
        context_limit=1000,
        output_reserve=200,
        protocol_overhead=0,
        safety_margin=0,
        input_budget=target_budget_tokens,
        confidence=LimitConfidence.KNOWN_LOCAL,
        source="local",
        capabilities=frozenset({"text"}),
    )

    result = await optimize_request_ir_moderate(ir, custom_budget)

    assert result.audit.blocks_removed < 3
    assert result.audit.optimized_cost <= target_budget_tokens
