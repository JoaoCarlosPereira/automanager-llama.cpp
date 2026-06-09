---
status: completed
title: Extrair módulo `index.js` e testes Jest
type: frontend
complexity: medium
dependencies: [task_03, task_04, task_05, task_06]
---

# Tarefa 7: Extrair módulo `index.js` e testes Jest

## Visão Geral

Criar `static/js/index.js` — o arquivo principal que importa os 4 módulos, declara estado global compartilhado, injeta variáveis Python injetadas no HTML, e inicializa a dashboard. Também criar `index.test.js` para validar inicialização e gerenciamento de timers.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O arquivo `static/js/index.js` DEVE importar os 4 módulos: auth.js, models.js, metrics.js, gpu.js
- O arquivo `static/js/index.js` DEVE declarar e exportar variáveis de estado globais: `statusPollTimer`, `metricsTimer`, `downloadsTimer`, `modelsTimer`, `logStream`, `startTime`, `currentSelectedModel`, `currentRunningModelPath`, `manualGpuOverride`, `autoBalancePending`, `sessionExpiredHandled`
- O arquivo `static/js/index.js` DEVE injetar variáveis Python: `window.fixedIp`, `window.CONTEXT_PRESET_VALUES`, `window.DEFAULT_CONTEXT_SIZE`, `window.CONTEXT_K_MULTIPLIER`, `window.DEFAULT_PARALLEL_SLOTS`, `window.DEFAULT_BATCH_SIZE`
- A função `initDashboard` DEVE chamar `bindGpuManualListeners`, `syncContextSizeCustomVisibility`, e iniciar polling
- O arquivo `static/js/index.test.js` DEVE existir com cobertura ≥ 90%

</requirements>

## Subtarefas

- [x] 7.1 Criar `index.js` com imports dos 4 módulos (auth, models, metrics, gpu)
- [x] 7.2 Declarar e exportar todas as variáveis de estado global (estado em `state.js`; index faz wiring)
- [x] 7.3 Injetar variáveis Python via `window.*` (fixedIp, context presets, defaults — injetados no HTML pelo FastAPI)
- [x] 7.4 Implementar `initDashboard()` com chamadas a bindGpuManualListeners, syncContextSizeCustomVisibility, updateStatus, updateMetrics, etc. (em `models.js`)
- [x] 7.5 Implementar `startDashboardPolling()` e `stopDashboardPolling()` (em `metrics.js`, expostos em `window`)
- [x] 7.6 Configurar `document.getElementById('chat-link').href` e `document.getElementById('api-link').innerText`
- [x] 7.7 Implementar auto-init se dashboard estiver visível
- [x] 7.8 Criar `index.test.js` com testes Jest
- [x] 7.9 Validar cobertura Jest ≥ 90% para index.js

## Detalhes de Implementação

Referencie a seção "Extração de JavaScript para Módulos — Módulo: index.js" do TechSpec.

Variáveis de estado exportadas:
- `statusPollTimer`, `metricsTimer`, `downloadsTimer`, `modelsTimer`, `logStream`, `startTime`
- `currentSelectedModel`, `currentRunningModelPath`, `manualGpuOverride`, `autoBalancePending`, `sessionExpiredHandled`
- `statusPollIntervalMs`

Variáveis window injetadas:
- `window.fixedIp`
- `window.modelConfigs = {}`
- `window.CONTEXT_PRESET_VALUES`
- `window.DEFAULT_CONTEXT_SIZE`
- `window.CONTEXT_K_MULTIPLIER`

### Arquivos Relevantes

- `llama_manager.py` (linhas 970-1038, 2070-2073) — estado global e inicialização no JS embutido
- `static/js/index.js` — novo
- `static/js/index.test.js` — novo

### Arquivos Dependentes

- `static/js/auth.js` — importado
- `static/js/models.js` — importado
- `static/js/metrics.js` — importado
- `static/js/gpu.js` — importado

### ADRs Relacionados

- [ADR-002: Estrutura modular dos arquivos JavaScript](../adrs/adr-002.md) — Define index.js como módulo principal

## Entregáveis

- `static/js/index.js` com imports, estado global, inicialização
- `static/js/index.test.js` com testes para initDashboard, start/stop polling
- Cobertura Jest ≥ 90% para index.js
- Dashboard funcional: inicializa corretamente após load dos 4 módulos

## Testes

- Testes unitários Jest:
  - [x] `window.*` expõe funções críticas após `import('./index.js')` (auth, gpu, metrics, models)
  - [x] Links `chat-link` / `api-link` usam `window.fixedIp`
  - [x] Auto-init chama `initDashboard` + `startDashboardPolling` quando `#dashboard` visível (via `state.metricsTimer` e fetch `/status`)
  - [x] Auto-init omitido quando `#dashboard` está `display:none`
  - [x] `window.modelConfigs` inicializado ou preservado
  - [ ] `startDashboardPolling` / `stopDashboardPolling` — cobertos em `metrics.test.js` (não duplicados em `index.test.js`)
- Meta de cobertura: >= 90%
- Todos os testes devem passar

## Critérios de Sucesso

- index.js extraído e servido via /static/js/index.js
- index.test.js passa todos os testes com cobertura ≥ 90%
- `npm test -- index.test.js` passa
- Dashboard inicializa corretamente após load dos 5 módulos
- Todos os imports dos 4 módulos resolvem sem erro

## Verificação

- `npm test -- --testPathPattern=index.test.js --coverage --collectCoverageFrom=static/js/index.js` — 8 testes, 100% statements/branches/functions/lines em `index.js`
- Correção durante testes: `window.updateDownloads` adicionado ao wiring (requerido por `initDashboard` em `models.js`)
