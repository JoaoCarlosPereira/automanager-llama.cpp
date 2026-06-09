---
status: pending
title: Bind de eventos CPU em gpu.js + redistribuição de pesos
type: frontend
complexity: medium
dependencies:
  - task_07
---

# Tarefa 08: Bind de eventos CPU em gpu.js + redistribuição de pesos

## Visão Geral

Esta tarefa estende o frontend em `static/js/gpu.js` para lidar com os controles da CPU: bind de event listeners para checkbox, weight input e pin checkbox da CPU, e estende as funções de redistribuição de pesos (`redistributeUnpinnedWeights()`, `updateTotal()`) para incluir a CPU no cálculo de 100%.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- `bindCpuListeners()` DEVE ser implementada e chamada durante a inicialização
- O input `.cpu-weight` DEVE validar valores entre 0 e 70 (clamp) no evento `change`
- Ao alterar o peso da CPU, `redistributeWeights('cpu')` DEVE ser chamada
- `redistributeUnpinnedWeights()` DEVE considerar a CPU: peso restante = `100 - cpuWeight`
- Se CPU está desativada (checkbox desmarcado), seu peso DEVE ser considerado 0
- A CPU DEVE ser tratada como pinned por padrão na redistribuição (peso não é redistribuído para GPUs)
- `updateTotal()` DEVE incluir o peso da CPU na soma total
- Todos os seletores DEVE verificar `null` antes de acessar propriedades (defesa contra elementos não-existentes)

</requirements>

## Subtarefas
- [ ] 08.1 Implementar função `bindCpuListeners()` com event listeners para checkbox, weight e pin
- [ ] 08.2 Clamping do input de peso: `Math.min(70, Math.max(0, val))`
- [ ] 08.3 Chamar `bindCpuListeners()` dentro ou após `bindGpuManualListeners()`
- [ ] 08.4 Estender `redistributeUnpinnedWeights()` para subtrair peso da CPU do total
- [ ] 08.5 Estender `updateTotal()` para incluir peso da CPU na soma
- [ ] 08.6 Adicionar defesa null-check em todos os seletores de elementos CPU
- [ ] 08.7 Testar manualmente no dashboard: alterar peso da CPU e verificar redistribuição

## Detalhes de Implementação

**Arquivo principal:** `static/js/gpu.js` (linhas 64-237)

A função `bindGpuManualListeners()` (linha 142-162) é estendida para incluir bindings de CPU. A função `redistributeUnpinnedWeights()` (linha 64-140) é modificada para considerar o peso da CPU.

### Arquivos Relevantes
- `static/js/gpu.js` — `bindGpuManualListeners()` (linha 142-162), `redistributeUnpinnedWeights()` (linha 64-140), `updateTotal()` (linha 187-199)
- `d:\dsv-git\automanager-llama.cpp\static\js\gpu.js` — todo o arquivo

### Arquivos Dependentes
- `llama_manager.py` — HTML com linha CPU precisa estar injetado (tarefa 07)
- `static/js/metrics.js` — métricas da CPU são atualizadas separadamente (tarefa 09)
- `static/js/models.js` — coleta de peso da CPU (tarefa 10)

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado na Tabela de Recursos

## Entregáveis
- `bindCpuListeners()` implementada e funcionando
- Peso da CPU clamped entre 0 e 70
- `redistributeUnpinnedWeights()` considera CPU no cálculo
- `updateTotal()` inclui peso da CPU
- Testes manuais no dashboard funcionando
- Testes unitários para funções modificadas com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `bindCpuListeners()` anexa listeners aos elementos `.cpu-checkbox`, `.cpu-weight`, `.cpu-pin`
  - [ ] Input de peso com valor 80 é clamped para 70
  - [ ] Input de peso com valor -5 é clamped para 0
  - [ ] `redistributeUnpinnedWeights()` com CPU a 20% distribui 80% entre GPUs unpinned
  - [ ] `redistributeUnpinnedWeights()` com CPU desativada distribui 100% entre GPUs
  - [ ] `updateTotal()` inclui peso da CPU na soma total
  - [ ] `updateTotal()` com CPU desativada não inclui peso da CPU
- Testes manuais:
  - [ ] Alterar peso da CPU no dashboard e verificar redistribuição nas GPUs
  - [ ] Desativar CPU e verificar que pesos GPU somam 100%
  - [ ] Fixar peso da CPU e verificar que Auto Balance não o altera
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Redistribuição de pesos funciona corretamente com CPU + GPUs
- Total sempre mostra 100% quando pesos estão equilibrados
- Defesas null-check previnem erros quando elementos CPU não existem
