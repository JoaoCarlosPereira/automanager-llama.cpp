---
status: completed
title: Extrair módulo `gpu.js` e testes Jest
type: frontend
complexity: medium
dependencies: [task_04]
---

# Tarefa 6: Extrair módulo `gpu.js` e testes Jest

## Visão Geral

Extrair as funções de configuração de GPU e auto-balance para `static/js/gpu.js`, e criar testes Jest correspondentes. Este módulo contém a lógica mais complexa do frontend: cálculo de pesos de GPU, redistribuição com pinning, validação de contexto e controle do auto-balance.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O arquivo `static/js/gpu.js` DEVE conter exports para: `getContextSize`, `setContextSize`, `tokensToContextK`, `syncContextSizeCustomVisibility`, `onContextSizePresetChange`, `onContextSizeCustomInput`, `updateTotal`, `balanceWeights`, `redistributeUnpinnedWeights`, `onGpuPinToggle`, `bindGpuManualListeners`, `applyGpuWeightsToUI`, `markManualGpuChange`, `hideAutoBalanceCapacityAlert`, `showAutoBalanceCapacityAlert`, `modelIncapableBadgeHtml`, `modelIncapableRowClass`, `isModelHardwareIncapable`, `updateAutoBalanceProfileBadge`, `resetToDefaults`
- `redistributeUnpinnedWeights` DEVE implementar a lógica completa de redistribuição (pinned/unpinned, last-one-gets-remainder)
- `getContextSize` DEVE retornar tokens válidos ou null para contexto inválido
- `updateTotal` DEVE calcular soma dos pesos e atualizar classe CSS (blue para 100%, red para != 100%)
- O arquivo `static/js/gpu.test.js` DEVE existir com cobertura ≥ 90%

</requirements>

## Subtarefas

- [x] 6.1 Extrair `getContextSize`, `setContextSize`, `tokensToContextK`, `syncContextSizeCustomVisibility`, `onContextSizePresetChange`, `onContextSizeCustomInput`
- [x] 6.2 Extrair `updateTotal`, `balanceWeights`, `redistributeUnpinnedWeights`
- [x] 6.3 Extrair `onGpuPinToggle`, `bindGpuManualListeners`
- [x] 6.4 Extrair `applyGpuWeightsToUI`, `markManualGpuChange`, `updateAutoBalanceProfileBadge`
- [x] 6.5 Extrair `hideAutoBalanceCapacityAlert`, `showAutoBalanceCapacityAlert`, `modelIncapableBadgeHtml`, `modelIncapableRowClass`, `isModelHardwareIncapable`
- [x] 6.6 Extrair `resetToDefaults`
- [x] 6.7 Criar `gpu.test.js` com testes Jest para cada função
- [x] 6.8 Validar cobertura Jest ≥ 90% para gpu.js

## Detalhes de Implementação

Referencie a seção "Extração de JavaScript para Módulos" do TechSpec.

Funções a extrair:
- `getContextSize()` — retorna contexto em tokens (preset ou custom * 1000) ou null
- `setContextSize(value)` — define preset ou custom no select/input
- `tokensToContextK(tokens)` — converte tokens para formato K (ex: 100000 → "100")
- `syncContextSizeCustomVisibility()` — mostra/esconde campo custom
- `updateTotal()` — soma pesos e atualiza badge (100% = blue, != 100% = red)
- `balanceWeights(changedInput)` — delega para redistributeUnpinnedWeights
- `redistributeUnpinnedWeights(changedInput)` — redistribui 100% entre unpinned GPUs
- `onGpuPinToggle(pinCheckbox)` — toggle pin, recalcula pesos
- `bindGpuManualListeners()` — attach event listeners a GPU rows
- `applyGpuWeightsToUI(weights, duringAutoBalance)` — sincroniza weights para inputs
- `markManualGpuChange()` — seta flag manualGpuOverride, esconde badge
- `hideAutoBalanceCapacityAlert()` — esconde alert
- `showAutoBalanceCapacityAlert(recovery)` — preenche e mostra alert
- `modelIncapableBadgeHtml(incapable)` — gera HTML do badge
- `modelIncapableRowClass(incapable)` — retorna classe CSS
- `isModelHardwareIncapable(path)` — verifica window.modelConfigs
- `updateAutoBalanceProfileBadge(hasProfile)` — mostra/esconde badge "Salvo"
- `resetToDefaults()` — restaura valores padrão em todos os inputs

### Arquivos Relevantes

- `llama_manager.py` (linhas 978-1030, 1190-1217, 1306-1443, 1476-1494) — funções de GPU no JS embutido
- `static/js/gpu.js` — novo
- `static/js/gpu.test.js` — novo

### Arquivos Dependentes

- `static/js/index.js` — importa variáveis de estado

### ADRs Relacionados

- [ADR-002: Estrutura modular dos arquivos JavaScript](../adrs/adr-002.md) — Define gpu.js como módulo de GPU

## Entregáveis

- `static/js/gpu.js` com 20 exports
- `static/js/gpu.test.js` com testes para todas as funções
- Cobertura Jest ≥ 90% para gpu.js
- Dashboard funcional: GPU weights, context size, auto-balance funcionam

## Testes

- Testes unitários Jest:
  - [x] `getContextSize` com preset: retorna valor do select
  - [x] `getContextSize` com custom: retorna custom * 1000
  - [x] `getContextSize` com custom inválido: retorna null
  - [x] `setContextSize` com preset válido: define select value
  - [x] `setContextSize` com custom: define select como "custom", preenche input
  - [x] `tokensToContextK(100000)` retorna "100"
  - [x] `tokensToContextK(150000)` retorna "150"
  - [x] `updateTotal` com soma 100%: badge azul (texto "CARGA TOTAL: 100%")
  - [x] `updateTotal` com soma 90%: badge vermelho
  - [x] `updateTotal` com soma 0%: badge vermelho
  - [x] `redistributeUnpinnedWeights` com 1 unpinned: atribui 100 - pinnedSum
  - [x] `redistributeUnpinnedWeights` com 2+ unpinned: distribui igualmente
  - [x] `redistributeUnpinnedWeights` com pinned: respeita soma pinned
  - [x] `applyGpuWeightsToUI` com weights válidos: inputs atualizados
  - [x] `resetToDefaults` restaura valores padrão
  - [x] `bindGpuManualListeners` attach listeners a inputs
  - [x] `isModelHardwareIncapable` retorna true para caminho em modelConfigs
  - [x] `modelIncapableBadgeHtml(true)` retorna badge HTML
  - [x] `modelIncapableBadgeHtml(false)` retorna string vazia
  - [x] `showAutoBalanceCapacityAlert` preenche e exibe alert
  - [x] `hideAutoBalanceCapacityAlert` adiciona classe hidden
- Meta de cobertura: >= 90%
- Todos os testes devem passar

## Critérios de Sucesso

- gpu.js extraído e servido via /static/js/gpu.js
- gpu.test.js passa todos os testes com cobertura ≥ 90%
- `npm test -- gpu.test.js` passa
- Dashboard funcional: GPU weights, context size, auto-balance funcionam
