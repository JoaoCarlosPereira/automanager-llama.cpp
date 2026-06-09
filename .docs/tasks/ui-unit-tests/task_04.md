---
status: completed
title: Extrair módulo `metrics.js` e testes Jest
type: frontend
complexity: medium
dependencies: [task_03]
---

# Tarefa 4: Extrair módulo `metrics.js` e testes Jest

## Visão Geral

Extrair as funções de métricas e monitoramento (polling de status/métricas, SSE log streaming, controle de timers) para `static/js/metrics.js`, e criar testes Jest correspondentes. Este módulo depende de `apiFetch` do auth.js para tratamento de 401.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O arquivo `static/js/metrics.js` DEVE conter exports para: `updateMetrics`, `updateStatus`, `updateUptime`, `startLogs`, `stopDashboardPolling`, `startDashboardPolling`, `ensureStatusPolling`, `cancelAutoBalance`, `hideAutoBalanceCapacityAlert`, `showAutoBalanceCapacityAlert`
- `updateMetrics` DEVE atualizar DOM elements com dados do /metrics endpoint
- `updateStatus` DEVE atualizar status badge, active model card, model list buttons, GPU weights
- `startLogs` DEVE iniciar SSE stream de /logs
- `startDashboardPolling` e `stopDashboardPolling` DEVEM gerenciar timers (setInterval/clearInterval)
- O arquivo `static/js/metrics.test.js` DEVE existir com cobertura ≥ 90%
- `apiFetch` do auth.js DEVE ser importado para tratamento de 401

</requirements>

## Subtarefas

- [ ] 4.1 Extrair `updateMetrics`, `updateStatus`, `updateUptime`, `startLogs` de `_build_html()` para `metrics.js`
- [ ] 4.2 Extrair `startDashboardPolling`, `stopDashboardPolling`, `ensureStatusPolling`, `cancelAutoBalance`, `hideAutoBalanceCapacityAlert`, `showAutoBalanceCapacityAlert`
- [ ] 4.3 Importar `apiFetch` do auth.js para tratamento de 401
- [ ] 4.4 Exportar variáveis de estado: `logStream`, `startTime`, `statusPollTimer`, `metricsTimer`, `downloadsTimer`, `modelsTimer`, `statusPollIntervalMs`
- [ ] 4.5 Criar `metrics.test.js` com testes Jest para cada função
- [ ] 4.6 Mockar `setInterval`/`clearInterval` para evitar timers reais nos testes
- [ ] 4.7 Validar cobertura Jest ≥ 90% para metrics.js

## Detalhes de Implementação

Referencie a seção "Extração de JavaScript para Módulos" do TechSpec.

Funções a extrair:
- `updateMetrics()` — GET /metrics, atualiza barras e valores de CPU/RAM/GPU
- `updateStatus()` — GET /status, atualiza badge, active card, model buttons, GPU weights, auto-balance UI
- `updateUptime(serverStartTime)` — calcula e exibe tempo de execução
- `startLogs()` — SSE stream de /logs, formata linhas com highlight de error/warn/info
- `stopDashboardPolling()` — clearInterval de todos os timers
- `startDashboardPolling()` — setInterval para metrics/status/downloads/models
- `ensureStatusPolling(fast)` — ajusta intervalo de polling de status
- `cancelAutoBalance()` — POST /auto-balance/cancel
- `hideAutoBalanceCapacityAlert()` — esconde alert de capacidade
- `showAutoBalanceCapacityAlert(recovery)` — exibe alert de capacidade

Variáveis de estado exportadas: `logStream`, `startTime`, `statusPollTimer`, `metricsTimer`, `downloadsTimer`, `modelsTimer`, `statusPollIntervalMs`

### Arquivos Relevantes

- `llama_manager.py` (linhas 1040-1070, 1229-1260, 1446-1454, 1586-1648, 1662-1821) — funções de metrics no JS embutido
- `static/js/metrics.js` — novo
- `static/js/metrics.test.js` — novo

### Arquivos Dependentes

- `static/js/auth.js` — importa `apiFetch`
- `static/js/index.js` — importa variáveis de estado

### ADRs Relacionados

- [ADR-002: Estrutura modular dos arquivos JavaScript](../adrs/adr-002.md) — Define metrics.js como módulo de métricas

## Entregáveis

- `static/js/metrics.js` com 10 exports + 6 variáveis de estado
- `static/js/metrics.test.js` com testes para todas as funções
- Cobertura Jest ≥ 90% para metrics.js
- Dashboard funcional: métricas atualizam, logs SSE funcionam

## Testes

- Testes unitários Jest:
  - [ ] `updateMetrics` com dados válidos: barras CPU/RAM/GPU atualizadas corretamente
  - [ ] `updateMetrics` com erro de rede: catch ignora sem crash
  - [ ] `updateStatus` com running=true: badge mostra ONLINE, active card visível
  - [ ] `updateStatus` com running=false: badge mostra OFFLINE, active card oculto
  - [ ] `updateStatus` com autoBalance=true: badge mostra AUTO BALANCE, cancel button visível
  - [ ] `updateUptime` com start_time: calcula e exibe formato "Xh Ym Zs"
  - [ ] `startDashboardPolling` cria timers (setInterval chamado 4x)
  - [ ] `stopDashboardPolling` limpa todos os timers (clearInterval chamado)
  - [ ] `ensureStatusPolling(fast=true)` ajusta intervalo para 1000ms
  - [ ] `ensureStatusPolling(fast=false)` ajusta intervalo para 3000ms
  - [ ] `startLogs` inicia SSE stream e formata linhas (error → ERRO, warn → AVISO)
  - [ ] `startLogs` limita a 500 linhas no DOM
  - [ ] `cancelAutoBalance` faz POST e desabilita botão
  - [ ] `hideAutoBalanceCapacityAlert` adiciona classe hidden
  - [ ] `showAutoBalanceCapacityAlert` preenche e mostra alert
- Meta de cobertura: >= 90%
- Todos os testes devem passar

## Critérios de Sucesso

- metrics.js extraído e servido via /static/js/metrics.js
- metrics.test.js passa todos os testes com cobertura ≥ 90%
- `npm test -- metrics.test.js` passa
- Dashboard funcional: métricas atualizam, polling funciona, logs SSE funcionam
