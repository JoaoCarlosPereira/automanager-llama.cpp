---
status: pending
title: Estender models.js para coletar peso da CPU no startModel()
type: frontend
complexity: medium
dependencies:
  - task_08
---

# Tarefa 10: Estender models.js para coletar peso da CPU no `startModel()`

## Visão Geral

Esta tarefa estende a função `startModel()` em `static/js/models.js` para coletar o peso da CPU e incluí-lo no payload enviado ao endpoint `/start`. A função já coleta pesos das GPUs — agora precisa também coletar o peso da CPU (se ativada) e incluí-lo na lista `gpu_weights` com `device: "cpu"` e `index: -1`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- `startModel()` DEVE coletar o peso da CPU se `.cpu-checkbox` estiver checked
- O objeto de peso da CPU DEVE ter: `index: -1`, `device: "cpu"`, `active: true`, `is_main: false`, `pinned: <valor>`, `weight: <valor>`, `name: <texto>`
- A CPU DEVE ser adicionada ao array `gpu_weights` junto com as GPUs
- O payload enviado ao POST `/start` DEVE incluir `gpu_weights` com GPUs + CPU
- Se `.cpu-row` não existe no DOM, a função DEVE continuar funcionando sem erro
- A coleta DEVE usar os mesmos seletores e padrões que a coleta de GPUs

</requirements>

## Subtarefas
- [ ] 10.1 Adicionar coleta de peso da CPU após a coleta de GPUs em `startModel()`
- [ ] 10.2 Verificar se `.cpu-row` existe e se `.cpu-checkbox` está checked
- [ ] 10.3 Criar objeto de peso com `device: "cpu"`, `index: -1`, e campos corretos
- [ ] 10.4 Adicionar objeto CPU ao array `gpu_weights`
- [ ] 10.5 Adicionar null-check para `.cpu-row` inexistente
- [ ] 10.6 Verificar que o payload JSON enviado ao `/start` inclui `device` em todos os pesos
- [ ] 10.7 Testar manualmente: iniciar modelo com CPU ativada e verificar payload

## Detalhes de Implementação

**Arquivo principal:** `static/js/models.js` — função `startModel()` (linhas 256-344)

A função já itera sobre `.gpu-row` elements. Adiciona-se iteração sobre `#cpu-row` com condições semelhantes.

O payload JSON enviado ao `/start` é construído nas linhas 308-315. A propriedade `device` precisa ser incluída em todos os objetos `gpu_weights`.

### Arquivos Relevantes
- `static/js/models.js` — `startModel()` (linha 256-344), coleta de pesos (linha 272-285)
- `d:\dsv-git\automanager-llama.cpp\static\js\models.js` — todo o arquivo

### Arquivos Dependentes
- `llama_manager.py` — endpoint `/start` recebe `gpu_weights` com `device` (schemas task_01)
- `schemas.py` — `GPUWeight` com campo `device` (tarefa 01)
- `static/js/gpu.js` — pesos são modificados por `redistributeUnpinnedWeights()` (tarefa 08)

### ADRs Relacionados
- [ADR-002](adrs/adr-002.md) — Extensão do Schema GPUWeight com Campo Device

## Entregáveis
- `startModel()` coleta peso da CPU quando ativada
- Payload `/start` inclui `device: "cpu"` e `index: -1` para a CPU
- Null-safe: funciona sem erro quando linha CPU não existe
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `startModel()` inclui objeto CPU com `device: "cpu"` quando checkbox checked
  - [ ] `startModel()` não inclui objeto CPU quando checkbox deschecked
  - [ ] `startModel()` não inclui objeto CPU quando `.cpu-row` não existe no DOM
  - [ ] Objeto CPU tem `index: -1`, `is_main: false`, `active: true`
  - [ ] Objeto CPU tem `pinned` correto e `weight` do input
  - [ ] Payload JSON inclui `device` em todos os objetos `gpu_weights`
- Testes manuais:
  - [ ] Iniciar modelo com CPU ativada: payload contém GPU weights + CPU weight
  - [ ] Iniciar modelo com CPU desativada: payload contém apenas GPU weights
  - [ ] Backend aceita payload com `device: "cpu"` sem erro
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Peso da CPU é coletado e enviado corretamente ao backend
- Backend aceita e processa payload com `device: "cpu"`
- Fluxo completo funciona: UI → backend → llama-server com `--n-gpu-layers` correto
