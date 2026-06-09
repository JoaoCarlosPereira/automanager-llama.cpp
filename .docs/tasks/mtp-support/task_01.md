---
status: completed
title: Extender schemas.py com campos MTP
type: backend
complexity: low
dependencies: []
---

# Extender schemas.py com campos MTP

## Visão Geral

Adiciona os campos `mtp_enabled` e `mtp_draft_tokens` ao contrato de requisição `StartRequest`, com constantes de default e validação Pydantic. Esta tarefa estabelece a base tipada que todos os demais componentes backend e frontend consumirão.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE definir `DEFAULT_MTP_ENABLED = False` e `DEFAULT_MTP_DRAFT_TOKENS = 3` em `schemas.py`
- DEVE adicionar `mtp_enabled: bool` com default `False` em `StartRequest`
- DEVE adicionar `mtp_draft_tokens: int` com `Field(default=3, ge=1, le=6)` em `StartRequest`
- DEVE rejeitar POST `/start` com `mtp_draft_tokens` fora do intervalo 1–6 (422 Pydantic)
- DEVE manter compatibilidade retroativa — payloads sem campos MTP usam defaults
</requirements>

## Subtarefas

- [ ] 1.1 Adicionar constantes `DEFAULT_MTP_*` ao lado das constantes existentes (`DEFAULT_CONTEXT_SIZE`, etc.)
- [ ] 1.2 Estender `StartRequest` com os dois campos MTP
- [ ] 1.3 Criar testes unitários de validação Pydantic para defaults e limites
- [ ] 1.4 Verificar que imports existentes de `StartRequest` não quebram

## Detalhes de Implementação

Seguir o pattern de `thinking_enabled` e `parallel_slots` já presentes em `StartRequest`. Ver seção **Interfaces Principais** e **Modelos de Dados** do TechSpec.

### Arquivos Relevantes

- `schemas.py` — definição de `StartRequest` e constantes de default

### Arquivos Dependentes

- `llama_manager.py` — importa `StartRequest` no endpoint `/start`
- `process_manager.py` — usa `StartRequest` em `_last_request` e auto-balance
- `auto_balance.py` — recebe `StartRequest` no prober

### ADRs Relacionados

- [ADR-004: Helpers em gpu_manager + process_manager](../adrs/adr-004.md) — define nomes `mtp_enabled` / `mtp_draft_tokens`

## Entregáveis

- Constantes e campos MTP em `schemas.py`
- Testes unitários de validação Pydantic com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] `StartRequest()` sem campos MTP usa `mtp_enabled=False` e `mtp_draft_tokens=3`
  - [ ] `StartRequest(mtp_enabled=True, mtp_draft_tokens=2)` aceita valor válido
  - [ ] `StartRequest(mtp_draft_tokens=0)` rejeita com ValidationError
  - [ ] `StartRequest(mtp_draft_tokens=7)` rejeita com ValidationError
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% nos campos MTP de `schemas.py`
- `StartRequest` serializa/deserializa `mtp_enabled` e `mtp_draft_tokens` corretamente
