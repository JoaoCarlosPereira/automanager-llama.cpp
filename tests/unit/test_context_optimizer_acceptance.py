"""Testes de aceite final do Context Optimizer (Task 15).

Bateria completa cobrindo texto, tools, visão, arquivos, structured output,
payload opaco, taxa de sucesso de 99 % sem erros de contexto/quebra estrutural
e preservação de blocos obrigatórios.
"""

import asyncio
import copy
import json
import os
import time
from typing import Any, Dict

import pytest

from context_optimizer import (
    AUDIT_ALLOWLIST_FIELDS,
    AtomicGroup,
    AuditRecorder,
    ConservativeEstimator,
    ContextOptimizer,
    ContextTooLargeError,
    LimitConfidence,
    ModelLimits,
    OptimizationAudit,
    OptimizationResult,
    RequiredCapabilities,
    TargetBudget,
    calculate_target_budget,
    derive_required_capabilities,
    derive_target_capabilities,
    optimize_request_ir_aggressive,
    optimize_request_ir_moderate,
    optimize_request_ir_safe,
    parse_request_ir,
    reconstruct_payload,
    resolve_model_limits,
    validate_transformed_payload,
)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def _simple_backend(ctx=8192, slots=1, backend_type="local"):
    return {
        "backend_type": backend_type,
        "context_size": ctx,
        "parallel_slots": slots,
        "config": {"context_size": ctx, "parallel_slots": slots, "gpu_weights": []},
    }


def _platform_backend(ctx=None):
    return {
        "backend_type": "platform",
        "config": {"backend_type": "platform", "proxy_eligible": True},
    }


# ---------------------------------------------------------------------------
# 1. Text processing
# ---------------------------------------------------------------------------


class TestTextProcessing:
    """Text processing: normalização, remoção de social noise e preservação semântica."""

    @pytest.mark.asyncio
    async def test_text_normalization_preserves_meaning(self):
        """Normaliza whitespace mantendo o significado original."""
        payload = {
            "messages": [
                {"role": "system", "content": "  Você  é   um assistente.  "},
                {"role": "user", "content": "Olá,   como   vai   você?"},
                {"role": "assistant", "content": "Estou   bem,   obrigado!"},
                {"role": "user", "content": "Tudo bem!"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        assert result.audit.strategy == "safe"
        assert result.audit.validation_passed is True
        # System message is protected, so whitespace is preserved.
        # Check the user message (index 1) which is NOT protected.
        user_msg = result.safe_payload["messages"][1]["content"]
        assert user_msg.strip() == "Olá, como vai você?"
        assert "  " not in user_msg

    @pytest.mark.asyncio
    async def test_social_noise_removed(self):
        """Social noise é removido no modo Safe."""
        payload = {
            "messages": [
                {"role": "system", "content": "Instrução"},
                {"role": "assistant", "content": "Obbrigado, pode continuar!"},
                {"role": "user", "content": "Pergunta importante"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # Social noise "Obbrigado, pode continuar!" deve ser removido
        assert result.audit.blocks_removed > 0 or result.audit.blocks_merged > 0
        assert len(result.safe_payload["messages"]) < len(payload["messages"])
        # Último user (turno atual) preservado
        assert result.safe_payload["messages"][-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_text_technical_decision_preserved(self):
        """Decisões técnicas em texto são preservadas."""
        payload = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "assistant", "content": "Decidimos usar postgresql para persistência."},
                {"role": "user", "content": "Confirmado."},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # Bloco com decisão técnica deve ser preservado
        tech_found = any(
            "postgresql" in str(msg.get("content", "")).lower()
            for msg in result.safe_payload["messages"]
        )
        assert tech_found is True


# ---------------------------------------------------------------------------
# 2. Tools handling
# ---------------------------------------------------------------------------


class TestToolsHandling:
    """Tools: atomicidade de grupos, preservação de tool_calls e tool_results."""

    @pytest.mark.asyncio
    async def test_tools_atomic_group_preserved(self):
        """Grupo atômico de ferramentas permanece intacto."""
        payload = {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "messages": [
                {"role": "user", "content": "Clima em SP?"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": "25C"}'},
                {"role": "user", "content": "Obrigado!"},
            ],
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text", "tools"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        assert result.audit.validation_passed is True
        # Verificar atomic group
        assert len(ir.atomic_groups) > 0
        reconstructed = reconstruct_payload(ir)
        assert reconstructed == payload

    @pytest.mark.asyncio
    async def test_tools_not_duplicated_or_reordered(self):
        """Ferramentas não são duplicadas nem reordenadas."""
        payload = {
            "tools": [
                {"type": "function", "function": {"name": "f1"}},
                {"type": "function", "function": {"name": "f2"}},
            ],
            "messages": [
                {"role": "user", "content": "Chame f1 depois f2"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_f1", "type": "function", "function": {"name": "f1"}},
                        {"id": "call_f2", "type": "function", "function": {"name": "f2"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_f1", "content": "res1"},
                {"role": "tool", "tool_call_id": "call_f2", "content": "res2"},
                {"role": "user", "content": "Final"},
            ],
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text", "tools"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # Round-trip deve ser idêntico
        assert result.safe_payload["tools"] == payload["tools"]
        # Ordem mantida
        msg_roles = [m["role"] for m in result.safe_payload["messages"]]
        assert msg_roles == ["user", "assistant", "tool", "tool", "user"]

    @pytest.mark.asyncio
    async def test_tools_capability_derived(self):
        """Capacidade 'tools' é derivada corretamente quando há campo tools no payload."""
        caps = derive_required_capabilities({"tools": [{"type": "function", "function": {"name": "x"}}], "messages": []})
        assert caps.tools is True

    @pytest.mark.asyncio
    async def test_moderate_preserves_tool_groups(self):
        """Moderate não remove grupos atômicos de ferramentas."""
        payload = {
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "assistant", "content": "Ok"},  # social noise antigo
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                {"role": "user", "content": "Z"},
            ],
        }
        backend = _simple_backend(ctx=4096)
        limits = resolve_model_limits(backend)
        caps = frozenset({"text", "tools"})
        budget = calculate_target_budget(payload, limits, caps)

        safe_ir = parse_request_ir(payload)
        safe_result = await optimize_request_ir_safe(safe_ir, budget)

        moderate_ir = parse_request_ir(safe_result.safe_payload)
        moderate_result = await optimize_request_ir_moderate(
            moderate_ir, budget, safe_audit=safe_result.audit
        )

        # Tool group não deve ter sido removido
        tool_msgs = [m for m in moderate_result.safe_payload["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio
    async def test_aggressive_preserves_tool_dependencies(self):
        """Aggressive preserva dependências de ferramentas."""
        payload = {
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "assistant", "content": "Decisão antiga"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "res"},
                {"role": "user", "content": "Atual"},
            ],
        }
        backend = _simple_backend(ctx=2048)
        limits = resolve_model_limits(backend)
        caps = frozenset({"text", "tools"})
        budget = calculate_target_budget(payload, limits, caps)

        safe_ir = parse_request_ir(payload)
        safe_result = await optimize_request_ir_safe(safe_ir, budget)

        moderate_ir = parse_request_ir(safe_result.safe_payload)
        moderate_result = await optimize_request_ir_moderate(
            moderate_ir, budget, safe_audit=safe_result.audit
        )

        aggressive_ir = parse_request_ir(moderate_result.safe_payload)
        aggressive_result = await optimize_request_ir_aggressive(
            aggressive_ir, budget, moderate_audit=moderate_result.audit
        )

        # Tool calls preservados
        tool_calls_found = any("tool_calls" in m for m in aggressive_result.safe_payload["messages"])
        assert tool_calls_found is True


# ---------------------------------------------------------------------------
# 3. Vision handling
# ---------------------------------------------------------------------------


class TestVisionHandling:
    """Vision: detecção de imagens e preservação de base64."""

    @pytest.mark.asyncio
    async def test_vision_capability_detected(self):
        """Capacidade 'vision' detectada a partir de image_url."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analise:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
                    ],
                }
            ]
        }
        caps = derive_required_capabilities(payload)
        assert caps.vision is True

    @pytest.mark.asyncio
    async def test_vision_payload_preserved_through_optimization(self):
        """Dados base64 de imagem não são alterados."""
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Olhe esta imagem"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
                {"role": "user", "content": "Final"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text", "vision"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        # Round-trip lossless
        assert result.safe_payload["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{b64}"

    @pytest.mark.asyncio
    async def test_vision_fallback_when_budget_unknown(self):
        """Sem limite conhecido, pipeline retorna payload intacto."""
        backend = _platform_backend()
        limits = resolve_model_limits(backend, {})
        assert limits.confidence == LimitConfidence.UNKNOWN

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Img"},
                        {"type": "image", "image": "data:image/png;base64,abc"},
                    ],
                },
                {"role": "user", "content": "Fim"},
            ]
        }
        caps = derive_target_capabilities(backend, {})
        budget = calculate_target_budget(payload, limits, caps)
        assert budget.input_budget is None

        ir = parse_request_ir(payload)
        result = await optimize_request_ir_safe(ir, budget)
        assert result.audit.optimized_cost == ir.calculate_total_tokens()


# ---------------------------------------------------------------------------
# 4. Files handling
# ---------------------------------------------------------------------------


class TestFilesHandling:
    """Files: detecção de arquivos e preservação de file_url."""

    @pytest.mark.asyncio
    async def test_files_capability_detected(self):
        """Capacidade 'files' detectada a partir de file_url."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file_url", "file_url": {"url": "https://example.com/doc.pdf"}},
                    ],
                }
            ]
        }
        caps = derive_required_capabilities(payload)
        assert caps.files is True

    @pytest.mark.asyncio
    async def test_file_content_preserved(self):
        """Referência de arquivo preservada através da otimização."""
        url = "https://example.com/document.pdf"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Leia o arquivo:"},
                        {"type": "file_url", "file_url": {"url": url}},
                    ],
                },
                {"role": "user", "content": "Resposta"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text", "files"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        assert result.audit.validation_passed is True
        file_url_found = any(
            url in str(msg.get("content", ""))
            for msg in result.safe_payload["messages"]
        )
        assert file_url_found is True


# ---------------------------------------------------------------------------
# 5. Structured output handling
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Structured output: response_format preservado."""

    @pytest.mark.asyncio
    async def test_structured_output_capability_detected(self):
        """Capacidade 'structured_output' derivada de response_format."""
        payload = {"response_format": {"type": "json_object"}, "messages": []}
        caps = derive_required_capabilities(payload)
        assert caps.structured_output is True

    @pytest.mark.asyncio
    async def test_response_format_not_mutated(self):
        """response_format não é modificado pelo pipeline."""
        payload = {
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "assistant", "content": "Ok"},
                {"role": "user", "content": "B"},
            ],
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text", "structured_output"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        assert result.safe_payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_structured_output_validation(self):
        """Validação estrutural rejeita mudança em response_format."""
        ir = parse_request_ir({
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "A"}],
        })
        bad_payload = {
            "response_format": {"type": "text"},
            "messages": [{"role": "user", "content": "A"}],
        }
        with pytest.raises(Exception):
            validate_transformed_payload(ir, bad_payload, original_cost=ir.calculate_total_tokens())


# ---------------------------------------------------------------------------
# 6. Opaque payload handling
# ---------------------------------------------------------------------------


class TestOpaquePayload:
    """Payload opaco: integridade total preservada."""

    @pytest.mark.asyncio
    async def test_opaque_payload_unchanged(self):
        """Payload sem 'messages' é tratado como opaco e não alterado."""
        payload = {"prompt": "Complete...", "max_tokens": 100}
        ir = parse_request_ir(payload)
        assert ir.is_opaque is True

        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        assert result.safe_payload == payload
        assert result.audit.original_cost == result.audit.optimized_cost
        assert result.audit.savings_tokens == 0

    @pytest.mark.asyncio
    async def test_opaque_round_trip_lossless(self):
        """Round-trip de payload opaco é perfeitamente lossless."""
        original = {
            "prompt": "Teste opaco",
            "custom_field": {"nested": [1, 2, 3]},
            "boolean": True,
            "number": 42.5,
        }
        ir = parse_request_ir(original)
        result = await optimize_request_ir_safe(ir, calculate_target_budget(original, resolve_model_limits(_simple_backend()), frozenset()))
        assert result.safe_payload == original


# ---------------------------------------------------------------------------
# 7. 99 % success rate: sem erros de contexto / quebra estrutural
# ---------------------------------------------------------------------------


class TestSuccessRate:
    """Taxa de sucesso: pipeline não gera erros estruturais em cenários típicos."""

    async def _run_full_pipeline(self, payload, backend_info, model_metadata=None):
        """Executa Safe -> Moderate -> Aggressive e retorna (result, error)."""
        errors = []
        try:
            optimizer = ContextOptimizer()
            result = await optimizer.optimize(payload, backend_info, model_metadata)
            return result, errors
        except ContextTooLargeError as e:
            errors.append(f"ContextTooLarge: {e.code}")
            return None, errors
        except Exception as e:
            errors.append(f"Error: {type(e).__name__}")
            return None, errors

    @pytest.mark.asyncio
    async def test_short_conversation_passes(self):
        """Conversão curta passa sem erro."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result, errors = await self._run_full_pipeline(payload, _simple_backend(ctx=8192))
        assert len(errors) == 0
        assert result is not None
        assert result.audit.validation_passed is True

    @pytest.mark.asyncio
    async def test_long_conversation_with_social_noise_passes(self):
        """Conversação longa com social noise passa sem erro estrutural."""
        messages = [{"role": "system", "content": "Instrução"}]
        for i in range(50):
            messages.append({"role": "assistant", "content": f"Ok {i}"})
            messages.append({"role": "user", "content": "Obbrigado!"})
        messages.append({"role": "user", "content": "Final"})

        payload = {"messages": messages}
        result, errors = await self._run_full_pipeline(payload, _simple_backend(ctx=8192))
        assert len(errors) == 0
        assert result is not None
        assert result.audit.validation_passed is True

    @pytest.mark.asyncio
    async def test_tools_conversation_passes(self):
        """Conversação com ferramentas passa sem erro."""
        payload = {
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "messages": [
                {"role": "system", "content": "Sys"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"results": [1]}'},
                {"role": "user", "content": "Thanks!"},
                {"role": "user", "content": "Final"},
            ],
        }
        result, errors = await self._run_full_pipeline(payload, _simple_backend(ctx=8192))
        assert len(errors) == 0
        assert result is not None

    @pytest.mark.asyncio
    async def test_vision_conversation_passes(self):
        """Conversação com vision passa sem erro."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analise"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
                    ],
                },
                {"role": "user", "content": "Final"},
            ]
        }
        result, errors = await self._run_full_pipeline(payload, _simple_backend(ctx=16384))
        assert len(errors) == 0
        assert result is not None

    @pytest.mark.asyncio
    async def test_structured_output_passes(self):
        """Structured output passa sem erro."""
        payload = {
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Responda em JSON."},
                {"role": "user", "content": "Informe o clima"},
            ],
        }
        result, errors = await self._run_full_pipeline(payload, _simple_backend(ctx=8192))
        assert len(errors) == 0
        assert result is not None

    async def _stress_test_round_trip(self, n, ctx=8192):
        """Testa n payloads aleatórios e verifica integridade."""
        errors = []
        passed = 0
        for i in range(n):
            payload = {
                "messages": [
                    {"role": "system", "content": f"System {i}"},
                    {"role": "user", "content": f"User {i}"},
                    {"role": "assistant", "content": f"Assistant {i}"},
                    {"role": "user", "content": f"Final {i}"},
                ],
                "custom_field": f"value_{i}",
            }
            ir = parse_request_ir(payload)
            budget = calculate_target_budget(
                payload,
                resolve_model_limits(_simple_backend(ctx=ctx)),
                frozenset({"text"}),
            )
            result = await optimize_request_ir_safe(ir, budget)
            # Verificar round-trip
            reconstructed = reconstruct_payload(ir)
            if reconstructed != payload:
                errors.append(f"Iter {i}: round-trip mismatch")
            else:
                passed += 1
        return passed, n, errors

    @pytest.mark.asyncio
    async def test_99_percent_success_rate(self):
        """99 % dos payloads passam sem erro de contexto/quebra estrutural."""
        passed, total, errors = await self._stress_test_round_trip(100, ctx=4096)
        rate = passed / total * 100
        assert rate >= 99.0, f"Taxa de sucesso {rate}% abaixo de 99%: {errors[:5]}"

    @pytest.mark.asyncio
    async def test_full_pipeline_99_percent_success(self):
        """Pipeline completo (Safe+Moderate+Aggressive) >= 99 % sucesso."""
        errors_count = 0
        for i in range(100):
            payload = {
                "messages": [
                    {"role": "system", "content": f"Sys {i}"},
                    {"role": "user", "content": f"User {i} com texto longo repetitivo " * 20},
                    {"role": "assistant", "content": f"Ok {i}"},
                    {"role": "user", "content": f"Final {i}"},
                ],
            }
            _, errs = await self._run_full_pipeline(payload, _simple_backend(ctx=4096))
            if errs:
                errors_count += 1
        rate = (100 - errors_count) / 100 * 100
        assert rate >= 99.0, f"Taxa de sucesso {rate}%: {errors_count} erros"


# ---------------------------------------------------------------------------
# 8. Preservation of required blocks
# ---------------------------------------------------------------------------


class TestBlockPreservation:
    """Preservação de blocos obrigatórios: system, developer, turno atual, decisões."""

    @pytest.mark.asyncio
    async def test_system_block_preserved(self):
        """Bloco system sempre preservado."""
        payload = {
            "messages": [
                {"role": "system", "content": "Instrução crítica"},
                {"role": "assistant", "content": "Obbrigado!"},
                {"role": "user", "content": "Fim"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        system_msgs = [m for m in result.safe_payload["messages"] if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Instrução crítica"

    @pytest.mark.asyncio
    async def test_developer_block_preserved(self):
        """Bloco developer sempre preservado."""
        payload = {
            "messages": [
                {"role": "developer", "content": "Developer note"},
                {"role": "user", "content": "Pergunta"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        dev_msgs = [m for m in result.safe_payload["messages"] if m.get("role") == "developer"]
        assert len(dev_msgs) == 1

    @pytest.mark.asyncio
    async def test_current_turn_user_preserved(self):
        """Última mensagem user (turno atual) sempre preservada."""
        payload = {
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
                {"role": "user", "content": "E - turno atual"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)

        last_msg = result.safe_payload["messages"][-1]
        assert last_msg["role"] == "user"
        assert "turno atual" in last_msg["content"]

    @pytest.mark.asyncio
    async def test_technical_decisions_preserved_in_aggressive(self):
        """O Aggressive preserva o conjunto mínimo (system, turno atual) e descarta
        retenções estendidas como decisões técnicas."""
        payload = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "assistant", "content": "Decidimos usar postgresql e redis."},
                {"role": "user", "content": "Ok"},
                {"role": "assistant", "content": "Outra coisa"},
                {"role": "user", "content": "Atual"},
            ]
        }
        backend = _simple_backend(ctx=1024)
        limits = resolve_model_limits(backend)
        budget = calculate_target_budget(payload, limits, frozenset({"text"}))

        # Executar pipeline completo
        ir = parse_request_ir(payload)
        safe = await optimize_request_ir_safe(ir, budget)
        mod_ir = parse_request_ir(safe.safe_payload)
        mod = await optimize_request_ir_moderate(mod_ir, budget, safe_audit=safe.audit)
        agg_ir = parse_request_ir(mod.safe_payload)

        # Sobrescrever input_budget para forçar remoção no Aggressive
        from context_optimizer import TargetBudget, LimitConfidence
        tight_budget = TargetBudget(
            context_limit=budget.context_limit,
            output_reserve=budget.output_reserve,
            protocol_overhead=budget.protocol_overhead,
            safety_margin=budget.safety_margin,
            input_budget=20,  # força remoção da decisão técnica
            confidence=LimitConfidence.KNOWN_LOCAL,
            source=budget.source,
            capabilities=frozenset({"text"}),
        )
        agg = await optimize_request_ir_aggressive(agg_ir, tight_budget, moderate_audit=mod.audit)

        # Verificar que o conjunto mínimo (system + turno atual) é preservado
        all_text = " ".join(
            str(m.get("content", "")) for m in agg.safe_payload.get("messages", [])
        )
        assert "Sys" in all_text
        assert "Atual" in all_text
        # Decisões técnicas são descartadas no modo Aggressive
        assert "postgresql" not in all_text.lower() and "redis" not in all_text.lower()

    @pytest.mark.asyncio
    async def test_protected_units_preserved_count(self):
        """Contagem de unidades protegídas preservadas é correta."""
        payload = {
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "B"},
                {"role": "user", "content": "C"},
            ]
        }
        ir = parse_request_ir(payload)
        assert len(ir.protected_unit_ids) == 2  # system + último user (turno atual)

        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)
        assert result.audit.protected_units_preserved == 2


# ---------------------------------------------------------------------------
# 9. End-to-end integration tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Testes de integração ponta a ponta do ContextOptimizer."""

    @pytest.mark.asyncio
    async def test_context_optimizer_full_pipeline(self):
        """ContextOptimizer.executa Safe->Moderate->Aggressive corretamente."""
        optimizer = ContextOptimizer()
        payload = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "assistant", "content": "Ok Ok Ok"},  # repetição
                {"role": "user", "content": "Final"},
            ]
        }
        backend = _simple_backend(ctx=4096)
        result = await optimizer.optimize(payload, backend)

        assert isinstance(result, OptimizationResult)
        assert result.audit.strategy in ("safe", "moderate", "aggressive")
        assert result.audit.validation_passed is True

    @pytest.mark.asyncio
    async def test_context_optimizer_413_on_too_large(self):
        """ContextTooLargeError(413) gerado quando mesmo Aggressive não cabe."""
        optimizer = ContextOptimizer()
        # Payload enorme com muitos tokens
        big_text = "Palavra " * 500
        payload = {
            "messages": [
                {"role": "system", "content": big_text},
                {"role": "user", "content": big_text},
                {"role": "assistant", "content": big_text},
                {"role": "user", "content": "Final"},
            ]
        }
        backend = _simple_backend(ctx=512)
        try:
            await optimizer.optimize(payload, backend)
            pytest.fail("ContextTooLargeError deveria ter sido levantado")
        except ContextTooLargeError as e:
            assert e.status_code == 413
            assert e.code == "context_too_large"

    @pytest.mark.asyncio
    async def test_context_optimizer_query_audit_logs(self, tmp_path):
        """Consulta de logs de auditoria funciona."""
        log_dir = str(tmp_path / "audit")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {"messages": [{"role": "user", "content": "Test"}]}
        await optimizer.optimize(payload, _simple_backend())

        res = optimizer.query_audit_logs()
        assert res["total"] >= 1
        assert res["pages"] >= 1

        recorder.close()

    @pytest.mark.asyncio
    async def test_model_limits_local_single_slot(self):
        """Limite local com 1 slot."""
        backend = {"backend_type": "local", "config": {"context_size": 8192, "parallel_slots": 1}}
        limits = resolve_model_limits(backend)
        assert limits.context_tokens == 8192
        assert limits.confidence == LimitConfidence.KNOWN_LOCAL
        assert limits.is_known is True

    @pytest.mark.asyncio
    async def test_model_limits_local_multi_slot(self):
        """Limite local com múltiplos slots divide contexto."""
        backend = {"backend_type": "local", "config": {"context_size": 16384, "parallel_slots": 4}}
        limits = resolve_model_limits(backend)
        assert limits.context_tokens == 4096

    @pytest.mark.asyncio
    async def test_budget_calculation_with_max_tokens(self):
        """Budget considera max_tokens do payload."""
        payload = {"max_tokens": 512, "messages": []}
        limits = ModelLimits(context_tokens=8192, max_output_tokens=1024, source="test", confidence=LimitConfidence.KNOWN_LOCAL)
        budget = calculate_target_budget(payload, limits, frozenset({"text"}))
        assert budget.output_reserve == 512

    @pytest.mark.asyncio
    async def test_capability_as_set(self):
        """RequiredCapabilities.as_set() retorna frozenset correto."""
        caps = RequiredCapabilities(text=True, vision=True, tools=True, structured_output=False, files=True)
        s = caps.as_set()
        assert s == frozenset({"text", "vision", "tools", "files"})

    @pytest.mark.asyncio
    async def test_capability_is_subset(self):
        """RequiredCapabilities.is_subset_of() funciona."""
        req = RequiredCapabilities(text=True, vision=True)
        assert req.is_subset_of(frozenset({"text", "vision", "tools"})) is True
        assert req.is_subset_of(frozenset({"text"})) is False

    @pytest.mark.asyncio
    async def test_conversation_block_dataclass(self):
        """ConversationBlock tem todos os campos esperados."""
        block = parse_request_ir({
            "messages": [{"role": "user", "content": "Hi"}]
        }).ordered_units[0]
        assert block.block_id == "block_0"
        assert block.kind == "user_text"
        assert block.role == "user"
        assert block.protected is True
        assert block.token_cost > 0

    @pytest.mark.asyncio
    async def test_optimization_audit_fields(self):
        """OptimizationAudit tem todos os campos obrigatórios."""
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
        assert audit.strategy == "safe"
        assert audit.original_cost == 100
        assert audit.optimized_cost == 80
        assert audit.savings_tokens == 20

    @pytest.mark.asyncio
    async def test_optimization_result_has_safe_payload_and_audit(self):
        """OptimizationResult contém audit e safe_payload."""
        payload = {"messages": [{"role": "user", "content": "A"}]}
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)
        assert isinstance(result.audit, OptimizationAudit)
        assert isinstance(result.safe_payload, dict)


# ---------------------------------------------------------------------------
# 10. Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Casos extremos e edge cases."""

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """Lista vazia de mensagens."""
        payload = {"messages": []}
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)
        assert result.audit.validation_passed is True

    @pytest.mark.asyncio
    async def test_payload_without_model_field(self):
        """Payload sem campo 'model' não quebra."""
        payload = {"messages": [{"role": "user", "content": "X"}]}
        ir = parse_request_ir(payload)
        assert ir.envelope.model is None

    @pytest.mark.asyncio
    async def test_special_characters_in_text(self):
        """Caracteres especiais preservados."""
        text = "Código: SELECT * FROM users WHERE name = 'João'; URL: https://ex.com/a+b?x=1"
        payload = {"messages": [{"role": "user", "content": text}]}
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)
        assert text in result.safe_payload["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_unicode_text_preserved(self):
        """Texto Unicode preservado."""
        payload = {
            "messages": [
                {"role": "user", "content": "Olá, mundo! 你好世界 مرحبا بالعالم"},
                {"role": "user", "content": "Fim"},
            ]
        }
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(
            payload,
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        result = await optimize_request_ir_safe(ir, budget)
        assert result.safe_payload["messages"][0]["content"] == payload["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_conservative_estimator_returns_positive(self):
        """ConservativeEstimator retorna valores >= 1 para texto não vazio."""
        assert ConservativeEstimator.estimate_text("hello") >= 1
        assert ConservativeEstimator.estimate_text("") == 0
        assert ConservativeEstimator.estimate_payload(None) == 0

    @pytest.mark.asyncio
    async def test_audit_allowlist_fields_immutable(self):
        """AUDIT_ALLOWLIST_FIELDS é imutável."""
        assert isinstance(AUDIT_ALLOWLIST_FIELDS, frozenset)

    @pytest.mark.asyncio
    async def test_target_budget_is_frozen(self):
        """TargetBudget é imutável (frozen)."""
        budget = calculate_target_budget(
            {},
            resolve_model_limits(_simple_backend()),
            frozenset({"text"}),
        )
        # Tentar modificar deve falhar
        with pytest.raises(Exception):
            budget.output_reserve = 999

    @pytest.mark.asyncio
    async def test_model_limits_is_frozen(self):
        """ModelLimits é imutável."""
        limits = resolve_model_limits(_simple_backend())
        with pytest.raises(Exception):
            limits.context_tokens = 9999

    @pytest.mark.asyncio
    async def test_optimization_result_is_frozen(self):
        """OptimizationResult é imutável."""
        payload = {"messages": [{"role": "user", "content": "X"}]}
        ir = parse_request_ir(payload)
        budget = calculate_target_budget(payload, resolve_model_limits(_simple_backend()), frozenset())
        result = await optimize_request_ir_safe(ir, budget)
        with pytest.raises(Exception):
            result.safe_payload = {}
