---
status: pending
title: Atualizar metrics.js para renderizar CPU usage e RAM
type: frontend
complexity: low
dependencies:
  - task_07
---

# Tarefa 09: Atualizar metrics.js para renderizar CPU usage e RAM

## Visão Geral

Esta tarefa estende a função `updateMetrics()` em `static/js/metrics.js` para atualizar os elementos da linha da CPU no dashboard: CPU usage %, barra de progresso, e RAM usada/total em MB. O backend já envia esses dados no endpoint `/metrics` (tarefa 03).

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- `updateMetrics()` DEVE atualizar `.cpu-row .device-util-val` com `Math.round(data.cpu) + '%'`
- `updateMetrics()` DEVE atualizar `.cpu-row .device-util-bar` com `width: data.cpu + '%'`
- `updateMetrics()` DEVE atualizar `.cpu-row .device-vram-text` com `{ram_used_mb} / {ram_total_mb} MB`
- `updateMetrics()` DEVE atualizar `.cpu-row .device-vram-bar` com `width: data.ram + '%'`
- Todos os seletores DEVE verificar `null` antes de acessar propriedades
- A atualização DEVE ser polida a cada 2 segundos (mesmo intervalo do metrics polling)
- Métricas de GPU existentes DEVE permanecer inalteradas

</requirements>

## Subtarefas
- [ ] 09.1 Adicionar atualização de CPU usage (`device-util-val` e `device-util-bar`) em `updateMetrics()`
- [ ] 09.2 Adicionar atualização de RAM (`device-vram-text` e `device-vram-bar`) em `updateMetrics()`
- [ ] 09.3 Adicionar null-checks em todos os seletores de elementos CPU
- [ ] 09.4 Verificar que métricas de GPU existentes permanecem inalteradas
- [ ] 09.5 Testar manualmente no dashboard: verificar atualização em tempo real

## Detalhes de Implementação

**Arquivo principal:** `static/js/metrics.js` — função `updateMetrics()` (linhas 329-355)

A função `updateMetrics()` já atualiza métricas de GPU. Novos blocos de código são adicionados para os elementos da CPU, seguindo o mesmo padrão.

### Arquivos Relevantes
- `static/js/metrics.js` — `updateMetrics()` (linha 329-355), `startDashboardPolling()` (linha 273-279)
- `d:\dsv-git\automanager-llama.cpp\static\js\metrics.js` — todo o arquivo

### Arquivos Dependentes
- `llama_manager.py` — endpoint `/metrics` deve retornar `cpu`, `ram`, `ram_used_mb`, `ram_total_mb` (tarefa 03)
- `llama_manager.py` — HTML deve conter elementos `.cpu-row .device-util-val`, etc. (tarefa 07)
- `gpu_manager.py` — `get_metrics()` retorna os dados brutos

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado na Tabela de Recursos

## Entregáveis
- `updateMetrics()` atualiza CPU usage e RAM na linha da CPU
- Atualização em tempo real a cada 2 segundos
- Métricas de GPU inalteradas
- Testes unitários para função modificada com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `updateMetrics()` atualiza `.cpu-row .device-util-val` com valor correto
  - [ ] `updateMetrics()` atualiza `.cpu-row .device-util-bar` com largura correta
  - [ ] `updateMetrics()` atualiza `.cpu-row .device-vram-text` com formato correto "X / Y MB"
  - [ ] `updateMetrics()` atualiza `.cpu-row .device-vram-bar` com largura correta
  - [ ] `updateMetrics()` não lança erro quando elementos CPU não existem (null-safe)
  - [ ] `updateMetrics()` continua atualizando métricas de GPU corretamente
- Testes manuais:
  - [ ] Dashboard mostra CPU usage atualizando em tempo real
  - [ ] Dashboard mostra RAM usada/total atualizando em tempo real
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- CPU usage e RAM atualizam em tempo real no dashboard
- Métricas de GPU permanecem funcionando corretamente
- Zero erros no console quando elementos CPU não estão presentes
