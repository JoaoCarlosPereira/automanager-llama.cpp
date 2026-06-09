---
status: completed
title: Propagar MTP no fluxo de auto-balance
type: backend
complexity: medium
dependencies:
  - task_04
  - task_05
---

# Propagar MTP no fluxo de auto-balance

## Visão Geral

Garante que configurações MTP informadas pelo operador percorrem todo o fluxo de auto-balance: persistência após sucesso/falha, repasse ao `process_manager.start()` durante sondagem e inclusão no `StartRequest` do prober.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE repassar `mtp_enabled` e `mtp_draft_tokens` em `process_manager._run_auto_balance()` nos dicts de settings (sucesso e falha)
- DEVE repassar campos em `auto_balance.py` na chamada a `process_manager.start()` (~linha 841)
- DEVE incluir campos MTP no restart OOM watchdog se aplicável (~linha 574)
- DEVE manter campos MTP no `StartRequest` usado pelo prober durante toda a sondagem
- DEVE incluir testes que verificam propagação com mock de start
</requirements>

## Subtarefas

- [ ] 10.1 Estender `_run_auto_balance()` success/failure settings com `mtp_*`
- [ ] 10.2 Estender `auto_balance.py` probe start com args MTP
- [ ] 10.3 Verificar OOM recovery restart repassa campos MTP
- [ ] 10.4 Adicionar testes unitários/integração para auto-balance com MTP enabled

## Detalhes de Implementação

Referência: propagação de `thinking_enabled` em `process_manager._run_auto_balance()` (~linhas 247, 276) e `auto_balance.py` (~linha 849). Ver TechSpec **Sequenciamento** passo 8 e PRD F3.

### Arquivos Relevantes

- `process_manager.py` — `_run_auto_balance()`, OOM watchdog restart
- `auto_balance.py` — `_try_start_with_weights()` ou equivalente start durante probe

### Arquivos Dependentes

- `schemas.py` — StartRequest com campos MTP (task_01)
- `config_manager.py` — persistência (task_02)
- `llama_manager.py` — `/start` com auto_balance=true (task_05)

### ADRs Relacionados

- [ADR-004: Helpers em gpu_manager + process_manager](../adrs/adr-004.md) — flags aplicadas durante probe

## Entregáveis

- Propagação MTP completa em auto-balance e OOM recovery
- Testes unitários/integração, cobertura >= 80% nos caminhos alterados **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] `_run_auto_balance` success persiste `mtp_enabled` e `mtp_draft_tokens` via config_manager
  - [ ] Auto-balance probe chama `start()` com `mtp_enabled=True` quando request indica
  - [ ] OOM recovery restart repassa campos MTP do `_last_request`
- Testes de integração:
  - [ ] POST `/start` com `auto_balance=true` e MTP enabled propaga args ao prober (mock)
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% nos caminhos MTP de auto-balance
- Sondagem auto-balance usa mesma config MTP informada pelo operador (PRD F3)
