---
status: completed
title: Atualizar metrics.js (sync config running)
type: frontend
complexity: low
dependencies:
  - task_07
---

# Atualizar metrics.js (sync config running)

## Visão Geral

Estende `metrics.js` para sincronizar campos MTP do painel quando um modelo está em execução, espelhando o comportamento existente de `thinking_enabled`, `parallel_slots` e `batch_size`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE atualizar `mtp-toggle` e `mtp-draft-tokens` quando `data.config` contém campos MTP e modelo está running
- DEVE chamar `updateMtpBadge()` após sync do toggle
- DEVE mergear campos MTP em `window.modelConfigs[data.config.path]` junto com demais config
- DEVE tratar `mtp_enabled !== undefined` explicitamente (não confundir `false` com ausência)
</requirements>

## Subtarefas

- [ ] 9.1 Estender bloco de sync running em `updateStatus()` / handler de metrics
- [ ] 9.2 Importar e chamar `updateMtpBadge` de `gpu.js`
- [ ] 9.3 Adicionar testes Jest em `static/js/metrics.test.js`

## Detalhes de Implementação

Referência: sync de `thinking_enabled` (~linhas 41–44) em `static/js/metrics.js`. Ver TechSpec componente `metrics.js` e endpoint GET `/status`.

### Arquivos Relevantes

- `static/js/metrics.js` — sync de config running

### Arquivos Dependentes

- `static/js/gpu.js` — `updateMtpBadge` (task_07)
- `process_manager.py` — `get_status()["config"]` com campos MTP (task_04)

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — UI reflete config ativa

## Entregáveis

- Sync MTP em `metrics.js`
- Testes Jest em `metrics.test.js`, cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários (Jest):
  - [ ] Response `/status` com `config.mtp_enabled=true, mtp_draft_tokens=4` atualiza DOM
  - [ ] `updateMtpBadge` chamado com valor correto após sync
  - [ ] `window.modelConfigs[path]` recebe merge dos campos MTP
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura Jest >= 80% nos caminhos MTP de `metrics.js`
- Painel reflete config MTP do processo em execução
