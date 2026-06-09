---
status: pending
title: Integrar CPU ao Auto Balance com priorização GPU e limite 70%
type: backend
complexity: high
dependencies:
  - task_01
  - task_03
---

# Tarefa 05: Integrar CPU ao Auto Balance com priorização GPU e limite 70%

## Visão Geral

Esta tarefa estende a classe `AutoBalanceProber` em `auto_balance.py` para considerar a CPU como dispositivo de compute. O algoritmo DEVE priorizar GPUs: tenta distribuir 100% entre as GPUs primeiro, e só adiciona CPU quando todas as GPUs estão ativas e ainda há OOM. O peso da CPU é limitado a 70% e o excedente é redistribuído para as GPUs.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O Auto Balance DEVE sempre priorizar GPUs sobre CPU
- A CPU DEVE ser adicionada apenas quando todas as GPUs já estão ativas e ainda há OOM
- O peso da CPU DEVE ser limitado a no máximo 70%
- Se o cálculo resulta em CPU > 70%, o excedente DEVE ser redistribuído para as GPUs
- `_should_add_cpu()` DEVE retornar `True` somente quando todas as GPUs estão em `weight_map`
- A spill order DEVE preservar o comportamento existente (spills da GPU principal para as demais)
- Se o modelo não cabe com CPU a 70%, o algoritmo DEVE retornar erro "Modelo além da capacidade"
- O peso da CPU calculado DEVE ser `100 - sum(gpu_weights)` após otimização das GPUs

</requirements>

## Subtarefas
- [ ] 05.1 Importar/adicionar constante `MAX_CPU_WEIGHT = 70` (reutilizar de task_04)
- [ ] 05.2 Implementar método `_should_add_cpu(all_gpus, weight_map)` na classe `AutoBalanceProber`
- [ ] 05.3 Modificar `_find_feasible_split()` para adicionar CPU quando todas GPUs ativas + OOM
- [ ] 05.4 Modificar `_maximize_vram_per_gpu()` para recalcular CPU weight após otimização GPU
- [ ] 05.5 Implementar lógica de redistribuição: se CPU > 70%, redistribuir excedente para GPUs
- [ ] 05.6 Atualizar `_to_gpu_weights()` para incluir `device="cpu"` com `index=-1`
- [ ] 05.7 Escrever testes unitários para cada cenário

## Detalhes de Implementação

**Arquivo principal:** `auto_balance.py` (classe `AutoBalanceProber`, linhas 274-725)

As modificações são localizadas nos métodos `_find_feasible_split()` (linha 442-515) e `_maximize_vram_per_gpu()` (linha 517-614). O fluxo é:

1. **Fase 1** — após tentar todas as GPUs, se ainda há OOM: adiciona CPU com `weight = 100 - sum(gpu_weights)`, limitado a 70%
2. **Fase 2** — após binary search em cada GPU: recalcula `cpu_weight = 100 - sum(gpu_weights)`; se > 70%, limita e redistribui

### Arquivos Relevantes
- `auto_balance.py` — classe `AutoBalanceProber`, métodos `discover()`, `_find_feasible_split()`, `_maximize_vram_per_gpu()`, `_to_gpu_weights()`
- `d:\dsv-git\automanager-llama.cpp\auto_balance.py` — linha 274 (classe), 336-440 (`discover`), 442-515 (`_find_feasible_split`), 517-614 (`_maximize_vram`)

### Arquivos Dependentes
- `process_manager.py` — usa os pesos calculados para `compute_n_gpu_layers()` (tarefa 04)
- `process_manager.py` — OOM Watchdog usa pesos recalculados (tarefa 06)
- `tests/unit/test_auto_balance.py` — padrão de teste existente
- `llama_manager.py` — endpoint `/start` aciona Auto Balance

### ADRs Relacionados
- [ADR-004](adrs/adr-004.md) — Priorização GPU no Auto Balance com Limite de 70%

## Entregáveis
- `_should_add_cpu()` implementado e funcional
- `_find_feasible_split()` integra CPU após esgotar GPUs
- `_maximize_vram_per_gpu()` recalcula e limita CPU weight
- Redistribuição de excedente implementada
- `_to_gpu_weights()` inclui `device="cpu"` e `index=-1`
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: Auto Balance com CPU gera pesos válidos **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `_should_add_cpu()` retorna `False` quando alguma GPU não está em `weight_map`
  - [ ] `_should_add_cpu()` retorna `True` quando todas as GPUs estão ativas
  - [ ] `_find_feasible_split()` adiciona CPU com peso correto quando todas GPUs + OOM
  - [ ] `_find_feasible_split()` limita CPU a 70% quando necessário
  - [ ] `_maximize_vram_per_gpu()` recalcula CPU weight após otimização GPU
  - [ ] CPU weight > 70% é limitado e excedente redistribuído para GPUs
  - [ ] Auto Balance retorna erro quando modelo não cabe com CPU a 70%
  - [ ] `_to_gpu_weights()` gera objeto CPU com `device="cpu"`, `index=-1`
  - [ ] Auto Balance com todas GPUs cabendo retorna CPU weight = 0%
- Testes de integração:
  - [ ] Fluxo completo: `AutoBalanceProber.discover()` com múltiplas GPUs + OOM retorna pesos válidos com CPU
  - [ ] Fluxo completo: `discover()` com GPUs suficientes retorna pesos sem CPU
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Auto Balance sempre prioriza GPUs — CPU só recebe carga quando VRAM insuficiente
- CPU weight nunca excede 70% no resultado do Auto Balance
- Redistribuição de excedente funciona corretamente
