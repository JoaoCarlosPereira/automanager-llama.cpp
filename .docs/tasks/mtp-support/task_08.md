---
status: completed
title: Atualizar models.js (applyModelConfig, startModel)
type: frontend
complexity: medium
dependencies:
  - task_07
---

# Atualizar models.js (applyModelConfig, startModel)

## Visão Geral

Estende `models.js` para restaurar config MTP ao selecionar modelo, incluir campos no payload POST `/start` e registrar listener do toggle MTP no `initDashboard()`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE restaurar `mtp_enabled` e `mtp_draft_tokens` em `applyModelConfig()` a partir de `window.modelConfigs[path]`
- DEVE tratar ausência de campos como `mtp_enabled=false` e `mtp_draft_tokens=3`
- DEVE incluir `mtp_enabled` e `mtp_draft_tokens` no JSON body de `startModel()`
- DEVE clampar `mtp_draft_tokens` entre 1 e 6 antes do POST
- DEVE registrar listener `change` no `mtp-toggle` em `initDashboard()` chamando `updateMtpBadge()`
- DEVE importar `updateMtpBadge` de `gpu.js`
</requirements>

## Subtarefas

- [ ] 8.1 Estender `applyModelConfig()` para campos MTP
- [ ] 8.2 Estender `startModel()` para coletar e enviar campos MTP
- [ ] 8.3 Adicionar listener do toggle em `initDashboard()`
- [ ] 8.4 Adicionar testes Jest em `static/js/models.test.js`

## Detalhes de Implementação

Referência: pattern `thinking_enabled` em `applyModelConfig()` (~linha 95) e `startModel()` (~linha 320) em `static/js/models.js`. Ver TechSpec componente `models.js`.

### Arquivos Relevantes

- `static/js/models.js` — apply, start, init

### Arquivos Dependentes

- `static/js/gpu.js` — `updateMtpBadge`, `resetToDefaults` (task_07)
- `llama_manager.py` — endpoint `/start` (task_05)

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — restore e start sempre leem campos

## Entregáveis

- Integração MTP completa em `models.js`
- Testes Jest em `models.test.js`, cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários (Jest):
  - [ ] `applyModelConfig()` com `{ mtp_enabled: true, mtp_draft_tokens: 5 }` atualiza toggle e input
  - [ ] `applyModelConfig()` sem campos MTP usa defaults (off, 3)
  - [ ] `startModel()` POST body inclui `mtp_enabled` e `mtp_draft_tokens`
  - [ ] Valor de input 99 é clampado para 6 no payload
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura Jest >= 80% nos caminhos MTP de `models.js`
- Seleção de modelo restaura config MTP; start envia campos corretos
