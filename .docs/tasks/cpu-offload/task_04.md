---
status: pending
title: Implementar compute_n_gpu_layers() e validate_weights()
type: backend
complexity: medium
dependencies:
  - task_01
---

# Tarefa 04: Implementar `compute_n_gpu_layers()` e `validate_weights()`

## Visão Geral

Esta tarefa implementa duas funções críticas em `process_manager.py`: `compute_n_gpu_layers()` calcula o número de camadas do modelo que devem ir para a GPU baseado no peso total das GPUs ativas, e `validate_weights()` valida que os pesos somam 100% e que a CPU não excede 70%. Essas funções substituem o valor hardcoded `-ngl 99` e introduzem validação de pesos com CPU.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- `compute_n_gpu_layers(gpu_weights, total_model_layers)` DEVE retornar `round(total_layers * gpu_weight_sum / 100)`
- Se GPU weight soma 100%: retorna `total_model_layers` (todas as camadas na GPU)
- Se GPU weight soma 0%: retorna `0` (todas as camadas na CPU)
- O valor retornado DEVE ser no mínimo `0` (nunca negativo)
- `validate_weights(gpu_weights)` DEVE retornar `True` se soma = 100% e CPU <= 70%
- `validate_weights()` DEVE retornar `False` se soma dos pesos != 100% (tolerância de 0.01%)
- `validate_weights()` DEVE retornar `False` se qualquer dispositivo CPU tem peso > 70%
- A constante `MAX_CPU_WEIGHT = 70` DEVE ser definida como constante global ou módulo-level
- O total de layers DEVE usar `999` como valor padrão (llama-server clampa internamente)

</requirements>

## Subtarefas
- [ ] 04.1 Definir constante `MAX_CPU_WEIGHT = 70` no módulo
- [ ] 04.2 Implementar `compute_n_gpu_layers()` — soma pesos das GPUs, calcula proporção
- [ ] 04.3 Implementar `validate_weights()` — valida soma = 100% e CPU <= 70%
- [ ] 04.4 Adicionar docstrings e type hints para ambas as funções
- [ ] 04.5 Escrever testes unitários abrangentes

## Detalhes de Implementação

**Arquivo principal:** `process_manager.py` (ProcessManager class, linhas 65-438)

As funções são implementadas como funções standalone (não métodos da classe) para facilitar teste unitário. O hardcoded `"-ngl", "99"` está nas linhas 363-364 e será substituído posteriormente na tarefa 06.

### Arquivos Relevantes
- `process_manager.py` — classe `ProcessManager`, método `start()` (linha 363-364 para hardcoded)
- `d:\dsv-git\automanager-llama.cpp\process_manager.py` — linhas 65 (classe), 322-438 (`start()`)

### Arquivos Dependentes
- `process_manager.py` — `start()` chamará `compute_n_gpu_layers()` (tarefa 06)
- `llama_manager.py` — endpoint `/start` pode chamar `validate_weights()` para validação prévia
- `auto_balance.py` — usa `MAX_CPU_WEIGHT` para limitar CPU no auto balance (tarefa 05)
- `tests/unit/test_auto_balance.py` — padrão de teste existente

### ADRs Relacionados
- [ADR-003](adrs/adr-003.md) — Cálculo Dinâmico de --n-gpu-layers
- [ADR-004](adrs/adr-004.md) — Priorização GPU no Auto Balance com Limite de 70%

## Entregáveis
- Funções `compute_n_gpu_layers()` e `validate_weights()` implementadas
- Constante `MAX_CPU_WEIGHT = 70` definida
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: funções retornam valores corretos para diversos cenários **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `compute_n_gpu_layers()` com 100% GPU retorna `total_layers`
  - [ ] `compute_n_gpu_layers()` com 0% GPU retorna `0`
  - [ ] `compute_n_gpu_layers()` com 50% GPU retorna `round(total_layers / 2)`
  - [ ] `compute_n_gpu_layers()` com GPU + CPU pesos mistos considera só GPUs
  - [ ] `validate_weights()` retorna `True` quando soma = 100% e CPU <= 70%
  - [ ] `validate_weights()` retorna `False` quando soma != 100%
  - [ ] `validate_weights()` retorna `False` quando CPU weight > 70%
  - [ ] `validate_weights()` retorna `True` quando CPU weight = 70% (limite)
  - [ ] `validate_weights()` retorna `True` com múltiplas GPUs e sem CPU
  - [ ] `validate_weights()` retorna `False` com pesos negativos
- Testes de integração:
  - [ ] `compute_n_gpu_layers()` com lista de `GPUWeight` objects funciona corretamente
  - [ ] `validate_weights()` com lista de `GPUWeight` objects funciona corretamente
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Funções determinísticas e puras (fáceis de testar)
- `compute_n_gpu_layers()` retorna valores inteiros válidos para qualquer entrada
