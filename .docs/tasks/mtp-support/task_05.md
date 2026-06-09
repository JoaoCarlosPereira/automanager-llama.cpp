---
status: completed
title: Propagar MTP em llama_manager (/start, /status, auto-start)
type: backend
complexity: medium
dependencies:
  - task_01
  - task_02
  - task_04
---

# Propagar MTP em llama_manager (/start, /status, auto-start)

## Visão Geral

Propaga campos MTP do endpoint POST `/start` até `process_manager.start()` e `config_manager`, incluindo persistência em `base_settings` e repasse no evento de auto-start do modelo padrão.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE incluir `mtp_enabled` e `mtp_draft_tokens` em `base_settings` do endpoint `/start`
- DEVE repassar campos ao `process_manager.start()` em todos os caminhos (manual, manual_gpu_override)
- DEVE persistir campos via `config_manager.update_model_settings()` junto com demais settings
- DEVE ler e repassar campos MTP no `startup_event()` auto-start do modelo padrão
- DEVE usar defaults seguros quando config legada não contém campos MTP
</requirements>

## Subtarefas

- [ ] 5.1 Estender `base_settings` e chamadas a `process_manager.start()` em `/start`
- [ ] 5.2 Propagar campos no fluxo `manual_gpu_override`
- [ ] 5.3 Estender `startup_event()` para ler `mtp_*` de saved_cfg
- [ ] 5.4 Adicionar/estender testes de integração em `tests/integration/test_api_endpoints.py`

## Detalhes de Implementação

Seguir pattern de `thinking_enabled` em `llama_manager.py` (~linhas 231–283 e 1186–1218). Ver seção **Endpoints de API** do TechSpec.

### Arquivos Relevantes

- `llama_manager.py` — endpoint `/start`, `startup_event()`

### Arquivos Dependentes

- `process_manager.py` — recebe args MTP (task_04)
- `config_manager.py` — persiste campos (task_02)
- `schemas.py` — StartRequest (task_01)

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — config por modelo

## Entregáveis

- Propagação MTP completa em `llama_manager.py`
- Testes de integração POST `/start` com campos MTP, cobertura >= 80% nos caminhos alterados **(OBRIGATÓRIO)**

## Testes

- Testes de integração:
  - [ ] POST `/start` com `mtp_enabled=true, mtp_draft_tokens=2` persiste config e chama start com args MTP
  - [ ] POST `/start` sem campos MTP usa defaults (`false`, `3`)
  - [ ] Auto-start no startup lê `mtp_*` de config salva
- Testes unitários:
  - [ ] Payload inválido `mtp_draft_tokens=0` retorna 422
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% nos caminhos MTP de `llama_manager.py`
- `/start` e auto-start propagam campos MTP end-to-end no backend
