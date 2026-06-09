---
status: completed
title: Backend ignora pinned quando auto_balance=true
type: backend
complexity: low
dependencies:
  - task_02
---

# Backend ignora pinned quando auto_balance=true

## Visão Geral
Garantir que, quando `auto_balance=true`, o backend desconsidere o campo `pinned` de cada `GPUWeight` antes de calcular a distribuição, deixando a cascata controlar 100% da alocação mesmo em chamadas diretas à API.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- Quando `StartRequest.auto_balance` for `true`, o backend DEVE tratar todos os `pinned` como `false` antes do cálculo da distribuição.
- O comportamento com `auto_balance=false` DEVE permanecer inalterado (pins respeitados no modo manual, se aplicável).
- A normalização DEVE ocorrer em ponto único no fluxo de `/start`, antes de `compute_offload_plan`.
</requirements>

## Subtarefas
- [x] 04.1 Localizar o ponto do fluxo `/start` onde os `gpu_weights` são recebidos (`start_model` em `llama_manager.py`).
- [x] 04.2 Quando `auto_balance=true`, normalizar `pinned=false` em todos os dispositivos (ponto único de entrada).
- [x] 04.3 Confirmar que o modo manual (`auto_balance=false`) não é afetado (preservado).
- [x] 04.4 Adicionar teste de integração cobrindo a limpeza de pins via `/start`.

## Notas de Execução
- Defesa em profundidade: além da normalização no `/start`, o `discover()` analítico (task_03) também ignora pins (passa `set()` a `to_gpu_weights`).

## Detalhes de Implementação
Aplicar a normalização no handler de `/start` (em `llama_manager.py`) ou em `ProcessManager.start`, antes de `compute_offload_plan`. Ver seção "Endpoints de API" do TechSpec. Mudança pequena e isolada.

### Arquivos Relevantes
- `process_manager.py` — `start()` monta o plano; ponto natural para a normalização.
- `llama_manager.py` — rota `/start` que recebe `StartRequest`.
- `schemas.py` — `StartRequest.auto_balance` e `GPUWeight.pinned`.

### Arquivos Dependentes
- `gpu_manager.py` — `compute_offload_plan` recebe os pesos já normalizados.

### ADRs Relacionados
- [ADR-001: Cascata estrita por prioridade como contrato único](../adrs/adr-001.md) — Decisão de Auto-Balance assumir o controle e ignorar pins.

## Entregáveis
- Normalização de `pinned=false` quando `auto_balance=true` em ponto único do `/start`.
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**.
- Teste de integração do `/start` cobrindo o comportamento **(OBRIGATÓRIO)**.

## Testes
- Testes unitários:
  - [ ] `auto_balance=true` com `GPUWeight.pinned=true` → tratado como `false` no cálculo.
  - [ ] `auto_balance=false` com `pinned=true` → pin preservado (modo manual inalterado).
  - [ ] Normalização não altera demais campos (`weight`, `is_main`, `device`).
- Testes de integração:
  - [ ] `POST /start` com `auto_balance=true` e pins marcados produz distribuição da cascata ignorando pins.
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Pins não afetam a distribuição quando `auto_balance=true`, inclusive via API direta
- Modo manual permanece inalterado
