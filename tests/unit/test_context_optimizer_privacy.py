"""Testes de privacidade do Context Optimizer (Task 15).

Confirma zero vazamento de sentinelas operacionais em logs, auditoria,
métricas, preview e respostas 413.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict

import pytest

from context_optimizer import (
    AUDIT_ALLOWLIST_FIELDS,
    AuditRecorder,
    ConservativeEstimator,
    ContextOptimizer,
    ContextTooLargeError,
    LimitConfidence,
    ModelLimits,
    OptimizationAudit,
    OptimizationResult,
    calculate_target_budget,
    optimize_request_ir_aggressive,
    optimize_request_ir_moderate,
    optimize_request_ir_safe,
    parse_request_ir,
    resolve_model_limits,
    validate_transformed_payload,
)

# ---------------------------------------------------------------------------
# Sentinelas operacionais — palavras-chave que NUNCA devem aparecer em
# outputs expostos (logs, auditoria, métricas, preview, erros 413).
# ---------------------------------------------------------------------------

OPERATIONAL_SENTINELS = frozenset({
    # Nomes de classes internas
    "ContextOptimizer",
    "OptimizationAudit",
    "OptimizationResult",
    "RequestIR",
    "RequestEnvelope",
    "ConversationBlock",
    "AtomicGroup",
    "TargetBudget",
    "ModelLimits",
    "RequiredCapabilities",
    "TokenCount",
    "TokenCountSource",
    "LimitConfidence",
    "TokenzierRegistry",
    "StructuralValidationError",
    "StructuralValidationReport",
    # Nomes de métodos internos
    "_extract_text_content",
    "_classify_block_retention",
    "_normalize_whitespace",
    "_remove_empty_blocks",
    "_merge_duplicate_blocks",
    "_merge_consecutive_identical_blocks",
    "_rebuild_with_blocks",
    "_canonical_json",
    "_same_json",
    "_raise_if_unknown_fields_changed",
    "_validate_message_subsequence",
    # Padrões de block_id
    "block_",
    "tool_group_asst_",
    # Nomes de pipeline interno
    "optimize_request_ir_safe",
    "optimize_request_ir_moderate",
    "optimize_request_ir_aggressive",
    "parse_request_ir",
    "build_request_ir",
    "reconstruct_payload",
    "validate_transformed_payload",
    "validate_request_ir",
    "resolve_model_limits",
    "derive_required_capabilities",
    "derive_target_capabilities",
    "calculate_target_budget",
    # Nomes de transformação interna
    "empty_blocks_removed",
    "dedup_iter",
    "merge_iter",
    "moderate_reduction",
    "aggressive_reduction",
    "moderate_already_fits",
    "aggressive_already_fits",
    # Tags internas de retenção
    "social_noise",
    "technical_decision",
    "code_media_file",
    "log_important",
    "multimodal_message",
    "file_content",
    "assistant_tool_calls",
    "tool_results",
    "assistant_text",
    "user_text",
    # Fallback interno
    "fail_open",
    "fail_open_on_error",
})

# Expressões regulares para detectar vazamento de estrutura interna.
_INTERNAL_PATTERNS = [
    # block_id patterns
    re.compile(r"block_\d+", re.IGNORECASE),
    # tool_group ID patterns
    re.compile(r"tool_group_asst_\d+", re.IGNORECASE),
]


def _contains_sentinels(text: str) -> list:
    """Retorna lista de sentinelas operacionais encontradas no texto."""
    found = []
    text_lower = text.lower()
    for sentinel in OPERATIONAL_SENTINELS:
        if sentinel.lower() in text_lower:
            found.append(sentinel)
    return found


def _contains_internal_patterns(text: str) -> list:
    """Retorna padrões internos encontrados no texto."""
    found = []
    for pattern in _INTERNAL_PATTERNS:
        if pattern.search(text):
            found.append(pattern.pattern)
    return found


# ---------------------------------------------------------------------------
# 1. Audit log privacy
# ---------------------------------------------------------------------------


class TestAuditLogPrivacy:
    """Logs de auditoria não vazam sentinelas operacionais."""

    @pytest.mark.asyncio
    async def test_audit_log_no_sentinels(self, tmp_path):
        """Audit log não contém sentinelas operacionais."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": "Explique como funciona"},
            ]
        }
        await optimizer.optimize(payload, {"backend_type": "local", "config": {"context_size": 8192}})
        recorder.close()

        audit_file = os.path.join(log_dir, "audit.jsonl")
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 1
        for line in lines:
            record = json.loads(line)
            text_content = json.dumps(record, ensure_ascii=False)
            sentinels = _contains_sentinels(text_content)
            internal = _contains_internal_patterns(text_content)
            assert len(sentinels) == 0, f"Sentinela vazada no audit: {sentinels}"
            assert len(internal) == 0, f"Padrão interno vazado no audit: {internal}"

    @pytest.mark.asyncio
    async def test_audit_allowlist_filtering_strict(self, tmp_path):
        """Auditoria filtra estritamente via allowlist."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)

        audit = OptimizationAudit(
            strategy="safe",
            original_cost=500,
            optimized_cost=400,
            savings_tokens=100,
            transformations_applied=["test"],
            protected_units_preserved=2,
            blocks_removed=1,
        )

        # extra com campos sensíveis
        recorder.record(audit, extra={
            "model": "test-model",
            "conversation_history": "SECRET USER DATA",
            "raw_payload": {"messages": [{"role": "user", "content": "SECRETS"}]},
        })
        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 1
        record = json.loads(lines[0])

        # Verificar allowlist
        for key in record:
            assert key in AUDIT_ALLOWLIST_FIELDS, f"Campo não autorizado: {key}"

        # Dados sensíveis NÃO devem aparecer
        assert "SECRET USER DATA" not in lines[0]
        assert "SECRETS" not in lines[0]

    @pytest.mark.asyncio
    async def test_audit_query_no_leak(self, tmp_path):
        """Query de auditoria não vaza dados."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {"messages": [{"role": "user", "content": "Data here"}]}
        await optimizer.optimize(payload, {"backend_type": "local", "config": {"context_size": 8192}})

        res = optimizer.query_audit_logs()
        for item in res["items"]:
            text = json.dumps(item, ensure_ascii=False)
            sentinels = _contains_sentinels(text)
            internal = _contains_internal_patterns(text)
            assert len(sentinels) == 0, f"Sentinela vazada na query: {sentinels}"
            assert len(internal) == 0, f"Padrão interno na query: {internal}"

        recorder.close()


# ---------------------------------------------------------------------------
# 2. 413 error privacy
# ---------------------------------------------------------------------------


class Test413ErrorPrivacy:
    """Respostas 413 não revelam sentinelas operacionais ou estrutura interna."""

    @pytest.mark.asyncio
    async def test_413_error_no_sentinels(self, tmp_path):
        """ContextTooLargeError não vaza sentinelas."""
        big = "x " * 1000
        payload = {
            "messages": [
                {"role": "system", "content": big},
                {"role": "user", "content": big},
            ]
        }
        try:
            await ContextOptimizer().optimize(
                payload,
                {"backend_type": "local", "config": {"context_size": 512}},
            )
        except ContextTooLargeError as e:
            error_text = json.dumps(e.payload(), ensure_ascii=False)
            sentinels = _contains_sentinels(error_text)
            assert len(sentinels) == 0, f"Sentinela no erro 413: {sentinels}"
            # A mensagem de erro é genérica
            assert "context_too_large" in error_text
            assert "prompt" in error_text.lower() or "limite" in error_text.lower()

    @pytest.mark.asyncio
    async def test_413_error_structure(self):
        """Estrutura do erro 413 é padronizada."""
        err = ContextTooLargeError(message="Test error", code="custom_code")
        payload = err.payload()
        assert "error" in payload
        assert payload["error"]["type"] == "proxy_error"
        assert payload["error"]["code"] == "custom_code"


# ---------------------------------------------------------------------------
# 3. Metrics privacy
# ---------------------------------------------------------------------------


class TestMetricsPrivacy:
    """Métricas de otimização não revelam dados sensíveis."""

    @pytest.mark.asyncio
    async def test_optimization_result_no_payload_leak(self):
        """OptimizationResult.safe_payload é o payload otimizado, não o original."""
        payload = {
            "messages": [
                {"role": "system", "content": "Secret system prompt"},
                {"role": "user", "content": "User message with sensitive data"},
            ],
            "api_key": "sk-secret-12345",
            "private_field": "SHOULD_NOT_BE_IN_AUDIT",
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits({"backend_type": "local", "config": {"context_size": 8192}}),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # safe_payload deve conter o payload otimizado (esperado)
        assert isinstance(result.safe_payload, dict)

        # audit NÃO deve conter payload
        audit_text = json.dumps(result.audit.__dict__, ensure_ascii=False)
        assert "sk-secret-12345" not in audit_text
        assert "SHOULD_NOT_BE_IN_AUDIT" not in audit_text
        assert "Secret system prompt" not in audit_text

    @pytest.mark.asyncio
    async def test_audit_does_not_contain_original_payload_fields(self, tmp_path):
        """Audit não contém campos do payload original."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [{"role": "user", "content": "X"}],
            "api_key": "secret-key-123",
            "password": "hunter2",
            "credit_card": "4111111111111111",
        }
        await optimizer.optimize(payload, {"backend_type": "local", "config": {"context_size": 8192}})
        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            content = f.read()

        assert "secret-key-123" not in content
        assert "hunter2" not in content
        assert "4111111111111111" not in content

    @pytest.mark.asyncio
    async def test_conversative_estimator_does_not_store_data(self):
        """ConservativeEstimator não armazena estado."""
        payload = {"secret": "data"}
        tokens = ConservativeEstimator.estimate_payload(payload)
        assert tokens > 0
        # Verificar que não há estado armazenado
        assert not hasattr(ConservativeEstimator, "_cache") or not getattr(ConservativeEstimator, "_cache", None)


# ---------------------------------------------------------------------------
# 4. Preview privacy (if exposed via API)
# ---------------------------------------------------------------------------


class TestPreviewPrivacy:
    """Dados de preview/visualização não expõem conteúdo sensível."""

    @pytest.mark.asyncio
    async def test_optimization_audit_summary_no_sensitive_data(self):
        """Resumo da otimização (audit) não contém dados sensíveis."""
        payload = {
            "messages": [
                {"role": "system", "content": "SECRET INSTRUCTIONS HERE"},
                {"role": "user", "content": "User sensitive text"},
            ],
            "token": "tok_12345secret",
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits({"backend_type": "local", "config": {"context_size": 8192}}),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        audit_dict = result.audit.__dict__
        audit_text = json.dumps(audit_dict, ensure_ascii=False)

        assert "SECRET INSTRUCTIONS HERE" not in audit_text
        assert "User sensitive text" not in audit_text
        assert "tok_12345secret" not in audit_text

        # Campos permitidos
        for key in audit_dict:
            assert key in AUDIT_ALLOWLIST_FIELDS or key == "transformations_applied"

    @pytest.mark.asyncio
    async def test_audit_extra_fields_filtered(self, tmp_path):
        """Campos extras passados ao recorder são filtrados pela allowlist."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)

        audit = OptimizationAudit(
            strategy="safe",
            original_cost=100,
            optimized_cost=80,
            savings_tokens=20,
            transformations_applied=[],
            protected_units_preserved=1,
        )

        recorder.record(audit, extra={
            "conversation": "full conversation history with secrets",
            "user_input": "sensitive user input",
            "model": "gpt-4o",  # permitido
        })
        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            line = f.readline()

        record = json.loads(line)
        assert "conversation" not in record
        assert "user_input" not in record
        assert record.get("model") == "gpt-4o"


# ---------------------------------------------------------------------------
# 5. Fail-open privacy
# ---------------------------------------------------------------------------


class TestFailOpenPrivacy:
    """Caminho fail-open não vaza dados."""

    @pytest.mark.asyncio
    async def test_fail_open_audit_no_error_details(self, tmp_path):
        """Fail-open registra erro sem exporar dados sensíveis do payload."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [
                {"role": "user", "content": "Secret message that should not appear in audit"},
            ],
            "password": "admin123",
        }
        # Forçar erro passando backend inválido
        result = await optimizer.optimize(payload, "not_a_dict")  # type: ignore

        assert result.audit.strategy == "fail_open"
        assert result.audit.validation_passed is True

        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            content = f.read()

        # Dados sensíveis não devem aparecer
        assert "Secret message that should not appear in audit" not in content
        assert "admin123" not in content
        assert "password" not in content


# ---------------------------------------------------------------------------
# 6. Structured validation error privacy
# ---------------------------------------------------------------------------


class TestValidationPrivacy:
    """Erros de validação estrutural não expõem internals."""

    @pytest.mark.asyncio
    async def test_structural_validation_error_no_payload_content(self):
        """StructuralValidationError não ecoa conteúdo do payload."""
        ir = parse_request_ir({
            "messages": [
                {"role": "user", "content": "Sensitive data here!!!"},
            ],
        })
        bad_payload = {"messages": [{"role": "assistant", "content": "CHANGED"}]}

        try:
            validate_transformed_payload(ir, bad_payload, original_cost=ir.calculate_total_tokens())
            pytest.fail("Deveria ter levantado StructuralValidationError")
        except Exception as e:
            error_str = str(e)
            assert "Sensitive data here" not in error_str
            assert "Sensitive" not in error_str
            assert "STRUCTURAL_VALIDATION_FAILED" in error_str or "structural_validation" in error_str.lower()

    @pytest.mark.asyncio
    async def test_structural_validation_error_code_only(self):
        """StructuralValidationError expõe apenas código, sem detalhes."""
        ir = parse_request_ir({
            "messages": [
                {"role": "user", "content": "Test"},
            ],
        })
        bad_payload = {"messages": [{"role": "assistant", "content": "WRONG"}]}

        try:
            validate_transformed_payload(ir, bad_payload, original_cost=ir.calculate_total_tokens())
            pytest.fail("Deveria ter levantado StructuralValidationError")
        except Exception as e:
            # Apenas o código deve estar no error
            assert hasattr(e, "code")
            assert hasattr(e, "stage")


# ---------------------------------------------------------------------------
# 7. Comprehensive privacy scan
# ---------------------------------------------------------------------------


class TestComprehensivePrivacy:
    """Varredura completa: nenhum output do pipeline contém sentinelas."""

    @pytest.mark.asyncio
    async def test_full_pipeline_no_sentinel_leak(self, tmp_path):
        """Pipeline completo (Safe->Moderate->Aggressive) não vaza sentinelas."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
                {"role": "user", "content": "E"},
            ],
            "api_key": "pk-live-xxxxxxxxxxxxxxxx",
        }

        backend = {"backend_type": "local", "config": {"context_size": 8192, "parallel_slots": 1}}
        await optimizer.optimize(payload, backend)
        recorder.close()

        # Verificar audit log
        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            audit_content = f.read()

        sentinels = _contains_sentinels(audit_content)
        assert len(sentinels) == 0, f"Sentinela vazada no pipeline completo: {sentinels}"

        # Verificar que dados sensíveis não vazaram
        assert "pk-live-xxxxxxxxxxxxxxxx" not in audit_content

    @pytest.mark.asyncio
    async def test_all_audit_fields_in_allowlist(self):
        """Todos os campos de OptimizationAudit estão na allowlist."""
        audit = OptimizationAudit(
            strategy="safe",
            original_cost=100,
            optimized_cost=80,
            savings_tokens=20,
            transformations_applied=["test"],
            protected_units_preserved=1,
            blocks_removed=0,
            blocks_merged=0,
            blocks_deduplicated=0,
            validation_passed=True,
            duration_ms=5.0,
        )
        audit_dict = audit.__dict__
        for key in audit_dict:
            assert key in AUDIT_ALLOWLIST_FIELDS, f"Campo '{key}' fora da allowlist"

    @pytest.mark.asyncio
    async def test_no_conversation_block_id_in_audit(self, tmp_path):
        """IDs de bloco não aparecem em nenhum registro de auditoria."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
            ],
        }
        await optimizer.optimize(payload, {"backend_type": "local", "config": {"context_size": 8192}})
        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            content = f.read()

        # block_XX patterns não devem aparecer
        block_ids = re.findall(r"block_\d+", content)
        assert len(block_ids) == 0, f"block_ids vazados: {block_ids}"

    @pytest.mark.asyncio
    async def test_no_atomic_group_id_in_audit(self, tmp_path):
        """IDs de grupo atômico não aparecem em auditoria."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [
                {"role": "user", "content": "Call"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "user", "content": "Final"},
            ],
        }
        await optimizer.optimize(payload, {"backend_type": "local", "config": {"context_size": 8192}})
        recorder.close()

        with open(os.path.join(log_dir, "audit.jsonl"), "r", encoding="utf-8") as f:
            content = f.read()

        assert "tool_group_asst_" not in content

    @pytest.mark.asyncio
    async def test_optimization_result_safe_payload_not_audit(self):
        """safe_payload é separada do audit — audit é metadata-only."""
        payload = {"messages": [{"role": "user", "content": "Test"}]}
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits({"backend_type": "local", "config": {"context_size": 8192}}),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # safe_payload contém o payload otimizado
        assert "messages" in result.safe_payload

        # audit contém apenas metadados
        audit_fields = set(result.audit.__dict__.keys())
        leaked = audit_fields - AUDIT_ALLOWLIST_FIELDS
        assert len(leaked) == 0, f"Campos vazados no audit: {leaked}"
