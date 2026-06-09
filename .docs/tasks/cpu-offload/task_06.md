---
status: pending
title: Tornar --n-gpu-layers dinâmico no process_manager
type: backend
complexity: medium
dependencies:
  - task_04
  - task_05
---

# Tarefa 06: Tornar `--n-gpu-layers` dinâmico no process_manager

## Visão Geral

Esta tarefa substitui o valor hardcoded `-ngl 99` no método `ProcessManager.start()` pelo cálculo dinâmico de `compute_n_gpu_layers()` (tarefa 04). Também integra a validação de pesos via `validate_weights()` antes de iniciar o llama-server.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O valor `"-ngl", "99"` hardcoded DEVE ser substituído por `"-ngl", str(compute_n_gpu_layers(gpu_weights, total_layers=999))`
- A função `validate_weights()` DEVE ser chamada antes de iniciar o llama-server
- Em caso de pesos inválidos, o sistema DEVE retornar erro descritivo ao usuário
- `--tensor-split` DEVE continuar sendo GPU-only (CPU não participa de tensor split)
- A CPU NÃO DEVE entrar em `CUDA_VISIBLE_DEVICES`
- O OOM Watchdog DEVE continuar funcionando com o valor dinâmico de `--n-gpu-layers`

</requirements>

## Subtarefas
- [ ] 06.1 Importar `compute_n_gpu_layers` e `validate_weights` do process_manager (ou do mesmo módulo)
- [ ] 06.2 Chamar `validate_weights(gpu_weights)` antes de construir o comando do llama-server
- [ ] 06.3 Calcular `n_gpu_layers = compute_n_gpu_layers(gpu_weights, total_layers=999)`
- [ ] 06.4 Substituir `"-ngl", "99"` por `"-ngl", str(n_gpu_layers)` na lista de comandos
- [ ] 06.5 Verificar que `--tensor-split` e `CUDA_VISIBLE_DEVICES` são GPU-only (CPU não incluída)
- [ ] 06.6 Escrever testes para o fluxo de início com pesos válidos e inválidos

## Detalhes de Implementação

**Arquivo principal:** `process_manager.py` — método `start()` (linhas 322-438)

A substituição é direta: na linha 363-364, trocar `"-ngl", "99"` por um cálculo dinâmico. A validação deve ser feita antes da construção do comando.

### Arquivos Relevantes
- `process_manager.py` — classe `ProcessManager`, método `start()` (linha 322-438), hardcoded na linha 363-364
- `d:\dsv-git\automanager-llama.cpp\process_manager.py` — linha 345 (tensor split), linha 399 (CUDA_VISIBLE_DEVICES)

### Arquivos Dependentes
- `llama_manager.py` — endpoint `/start` chama `process_manager.start()` (linha 233, 252)
- `auto_balance.py` — gera pesos que serão validados
- `tests/unit/test_oom_watchdog.py` — padrão de teste existente

### ADRs Relacionados
- [ADR-003](adrs/adr-003.md) — Cálculo Dinâmico de --n-gpu-layers

## Entregáveis
- Hardcoded `-ngl 99` substituído por cálculo dinâmico
- Validação de pesos executada antes de iniciar llama-server
- CPU não participa de tensor split nem CUDA_VISIBLE_DEVICES
- OOM Watchdog funciona com valor dinâmico
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: llama-server inicia com `--n-gpu-layers` correto **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `ProcessManager.start()` calcula `--n-gpu-layers` correto com 100% GPU
  - [ ] `ProcessManager.start()` calcula `--n-gpu-layers` correto com 50% GPU
  - [ ] `ProcessManager.start()` calcula `--n-gpu-layers` correto com 0% GPU (tudo na CPU)
  - [ ] `ProcessManager.start()` rejeita pesos inválidos (soma != 100%)
  - [ ] `ProcessManager.start()` rejeita pesos com CPU > 70%
  - [ ] `--tensor-split` não inclui CPU (index=-1)
  - [ ] `CUDA_VISIBLE_DEVICES` não inclui CPU
- Testes de integração:
  - [ ] llama-server inicia com `--n-gpu-layers` dinâmico e correto
  - [ ] OOM Watchdog detecta e recalcula pesos com `--n-gpu-layers` dinâmico
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- `--n-gpu-layers` reflete corretamente a proporção GPU/CPU
- llama-server inicia com sucesso em todos os cenários de peso válidos
- Validação previne início com pesos inválidos
