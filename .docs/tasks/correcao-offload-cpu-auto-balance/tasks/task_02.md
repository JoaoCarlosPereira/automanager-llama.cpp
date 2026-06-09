---
status: pending
title: Atualizar schemas.py - adicionar campo cpu_enabled ao StartRequest
type: backend
complexity: low
dependencies: []
---

# Task 02: Atualizar schemas.py - adicionar campo cpu_enabled

## Visao Geral

Adicionar o campo `cpu_enabled: bool = False` ao schema `StartRequest` em `schemas.py`. Este campo representa o estado do checkbox de CPU na interface e sera usado pelo `ProcessManager` para determinar se o CPU pode ser usado como fallback (spill-over) durante o carregamento do modelo.

<critical>
- O campo DEVE ter default `False` para compatibilidade com requisições existentes
- Manter a ordem dos campos existentes para evitar breaking changes
- Adicionar docstring explicando que é a válvula on/off do checkbox de CPU
</critical>

<requirements>
1. O campo `cpu_enabled: bool = False` DEVE ser adicionado à classe `StartRequest`
2. O campo DEVE vir após `gpu_weights` ou após `auto_balance` para manter consistência
3. O campo DEVE ter docstring: "Checkbox de CPU - valve on/off. Se False, nenhum layer vai para CPU"
4. Nenhuma outra alteração nos schemas existentes é permitida
</requirements>

## Subtarefas

- [ ] Adicionar campo `cpu_enabled: bool = False` à classe `StartRequest`
- [ ] Adicionar docstring explicativa ao novo campo
- [ ] Verificar que o Pydantic v2 não gera warnings de campo opcional
- [ ] Rodar `python -c "from schemas import StartRequest; print(StartRequest.model_fields['cpu_enabled'])"` para validar

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `schemas.py` | Adicionar campo `cpu_enabled` ao `StartRequest` (linha ~33) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `llama_manager.py` | Endpoint `start_model` precisará ler `request.cpu_enabled` |
| `process_manager.py` | Passará `cpu_enabled` para `compute_offload_plan()` |
| `gpu.js` | Frontend precisará enviar `cpu_enabled` no payload de start |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- Campo `cpu_enabled: bool = False` adicionado à classe `StartRequest`
- Schema validado via import e `model_fields`

## Testes

- [ ] `test_start_request_with_cpu_enabled_true` - criar `StartRequest` com `cpu_enabled=True` e validar
- [ ] `test_start_request_with_cpu_enabled_default` - criar `StartRequest` sem `cpu_enabled` e verificar default `False`
- [ ] `test_start_request_all_fields` - criar `StartRequest` com todos os campos e validar schema completo

## Critérios de Sucesso

- Campo `cpu_enabled` presente em `StartRequest.model_fields`
- Default `False` funciona corretamente
- Nenhum breaking change em schemas existentes
- Schema passa em todos os testes de validação
