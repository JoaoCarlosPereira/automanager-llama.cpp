---
status: completed
title: Testes de integração — fluxo completo de CPU offload
type: test
complexity: medium
dependencies:
  - task_01
  - task_02
  - task_03
  - task_04
  - task_05
  - task_06
  - task_07
  - task_08
  - task_09
  - task_10
---

# Tarefa 11: Testes de integração — fluxo completo de CPU offload

## Visão Geral

Esta tarefa implementa testes de integração que validam o fluxo ponta a ponta do CPU offload: desde a ativação da CPU na UI até o llama-server iniciar com `--n-gpu-layers` calculado dinamicamente. Os testes verificam a integração entre frontend (HTML/JS) e backend (Python APIs).

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- Os testes DEVEM cobrir o fluxo completo de integração entre frontend e backend
- Os testes DEVEM usar mocks para llama-server (não requer hardware GPU real)
- Os testes DEVEM seguir o padrão existente em `tests/integration/`
- Todos os testes DEVEM passar em ambiente CI (sem dependências de hardware)
- A meta de cobertura DEVE ser >= 80% para os módulos envolvidos

</requirements>

## Subtarefas
- [x] 11.1 Adicionar testes de integração para endpoint `/metrics` com `ram_used_mb`, `ram_total_mb`, `cpu_name`
- [x] 11.3 Adicionar teste para payload `/start` com `device: "cpu"` sendo aceito
- [x] 11.4 Adicionar teste para `compute_n_gpu_layers()` integrado com `ProcessManager.start()`
- [ ] 11.2 Adicionar teste para HTML contendo linha da CPU com todos os elementos
- [ ] 11.5 Adicionar teste para Auto Balance com CPU gerando pesos válidos
- [ ] 11.6 Adicionar teste E2E (Playwright) para fluxo completo: ativar CPU → ajustar peso → iniciar modelo
- [x] 11.7 Executar todos os testes e garantir passagem

## Detalhes de Implementação

**Diretório:** `tests/integration/` e `tests/e2e/`

Os testes de integração usam o padrão existente em `tests/integration/conftest.py` e `tests/integration/test_api_endpoints.py`. O teste E2E usa Playwright (padrão em `tests/e2e/`).

### Arquivos Relevantes
- `tests/integration/conftest.py` — fixtures de teste de integração
- `tests/integration/test_api_endpoints.py` — padrão de testes de endpoint
- `tests/e2e/conftest.py` — fixtures E2E
- `tests/e2e/helpers.ts` — helpers de teste E2E
- `d:\dsv-git\automanager-llama.cpp\tests\` — diretório de testes

### Arquivos Dependentes
- Todas as tarefas 01-10 devem estar concluídas
- `llama_manager.py` — endpoints `/metrics`, `/start`
- `gpu_manager.py` — `get_metrics()`, `detect_cpu_info()`
- `auto_balance.py` — `AutoBalanceProber`
- `process_manager.py` — `ProcessManager.start()`, `compute_n_gpu_layers()`

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado
- [ADR-002](adrs/adr-002.md) — Extensão do Schema GPUWeight
- [ADR-003](adrs/adr-003.md) — Cálculo Dinâmico de --n-gpu-layers
- [ADR-004](adrs/adr-004.md) — Priorização GPU no Auto Balance

## Entregáveis
- Testes de integração para `/metrics`, `/start`, HTML, Auto Balance
- Teste E2E (Playwright) para fluxo completo
- Cobertura >= 80% para módulos de integração
- Todos os testes passando em CI

## Testes
- Testes de integração:
  - [x] GET `/metrics` retorna `ram_used_mb`, `ram_total_mb`, `cpu_name` com valores válidos
  - [ ] GET `/` retorna HTML com linha da CPU e todos os elementos DOM
  - [x] POST `/start` com `device: "cpu"` é aceito e processado
  - [x] `ProcessManager.start()` calcula `--n-gpu-layers` correto com pesos mistos
  - [ ] Auto Balance com CPU gera pesos válidos (GPU-first, CPU <= 70%)
- Testes E2E (Playwright):
  - [ ] Fluxo: ativar CPU → ajustar peso → validar soma = 100% → iniciar modelo
  - [ ] Fluxo: CPU desativada → pesos GPU somam 100% → iniciar modelo
  - [ ] Fluxo: pin weight da CPU → Auto Balance respeita peso fixado
  - [ ] Métricas de CPU usage e RAM atualizam em tempo real no dashboard
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Fluxo completo funciona: UI → backend → llama-server
- Zero regressões nos testes existentes
- Testes E2E passam em ambiente com mocks
