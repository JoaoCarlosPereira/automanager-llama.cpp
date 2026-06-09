---
status: pending
title: Atualizar auto_balance.py - remover MAX_CPU_WEIGHT_PCT, atualizar escalonamento de CPU
type: backend
complexity: medium
dependencies: [task_01]
---

# Task 04: Atualizar auto_balance.py

## Visao Geral

Remover o limite rigido de 70% no CPU do módulo `auto_balance.py` e atualizar todas as referencias a `MAX_CPU_WEIGHT_PCT` e `CPU_OFFLOAD_STEP`. O escalonamento de CPU deve agora escalar sem limite superior fixo (apenas ate 100% - min_gpu_budget), e a válvula de CPU deve funcionar como boolean on/off via checkbox.

<critical>
- Ler o TechSpec secoes 4.3.1 a 4.3.5 antes de implementar
- TODAS as referencias a `MAX_CPU_WEIGHT_PCT` DEVEM ser removidas
- O metodo `_escalate_cpu_offload()` DEVE escalar CPU sem cap maximo (max 100% - min_gpu_budget)
- O `_cpu_config_from_request()` DEVE ser simplificado para boolean valve
- Manter a logica de binary search do Phase 2 inalterada
</critical>

<requirements>
1. Remover a constante `MAX_CPU_WEIGHT_PCT = 70` (linha 33)
2. Remover a constante `CPU_OFFLOAD_STEP = 10` (linha 34)
3. `_escalate_cpu_offload()` DEVE escalonar CPU em 10% sem cap maximo: `new_cpu = min(cpu_weight + 10, 100 - min_gpu_budget)`
4. `_finalize_cpu_split()` DEVE retornar `raw_cpu` sem cap: `return gpu_map, raw_cpu`
5. `compute_cpu_offload_weights()` DEVE remover os caps de `gpu_fraction` e `cpu_weight`
6. `_cpu_config_from_request()` DEVE ser simplificado: checkbox = válvula, peso é calculado pelo LoadDistributor
7. Import de `LoadDistributor` DEVE ser adicionado no topo do arquivo
</requirements>

## Subtarefas

- [ ] Remover constante `MAX_CPU_WEIGHT_PCT = 70` (linha 33)
- [ ] Remover constante `CPU_OFFLOAD_STEP = 10` (linha 34)
- [ ] Adicionar `from load_distributor import LoadDistributor` no topo do arquivo
- [ ] Atualizar `_escalate_cpu_offload()`: remover `min(..., MAX_CPU_WEIGHT_PCT)`, usar `min(cpu_weight + 10, 100 - min_gpu_budget)`
- [ ] Atualizar `_finalize_cpu_split()`: remover verificacao `if raw_cpu <= MAX_CPU_WEIGHT_PCT`
- [ ] Atualizar `compute_cpu_offload_weights()`: remover `min(gpu_fraction, MAX_CPU_WEIGHT_PCT / 100.0)` e `min(cpu_weight, MAX_CPU_WEIGHT_PCT)`
- [ ] Simplificar `_cpu_config_from_request()`: checkbox ativo = enabled=True, peso=0 (calculado pelo LoadDistributor)
- [ ] Verificar que `_gpu_budget()` retorna `max(0, 100 - cpu_weight)` (ja está correto, apenas confirmar)
- [ ] Verificar que `compute_cpu_offload_weights()` calcula `cpu_weight = 100 - int(round(gpu_fraction * 100))` sem cap

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `auto_balance.py` | Remover caps de 70% em multiplicas localizacoes (linhas 33-34, 359, 362, 491-507, 552-575, 588-611) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `load_distributor.py` (task_01) | Fornece `LoadDistributor` para calculo de spill-over |
| `llama_manager.py` | Endpoint de auto-balance consome `AutoBalanceProber.discover()` |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- `auto_balance.py` sem nenhuma referencia a `MAX_CPU_WEIGHT_PCT`
- `_escalate_cpu_offload()` escalando sem cap de 70%
- `_finalize_cpu_split()` retornando valor de CPU sem cap
- `_cpu_config_from_request()` simplificado para boolean valve
- Import de `LoadDistributor` adicionado

## Testes

- [ ] `test_escalate_cpu_without_70_cap` - verificar que CPU pode ir acima de 70%
- [ ] `test_finalize_cpu_split_no_cap` - GPU map somando 30% -> CPU recebe 70% sem ser truncado
- [ ] `test_compute_cpu_offload_weights_no_cap` - vram=4GB, model=10GB -> GPU=40%, CPU=60% (sem cap)
- [ ] `test_cpu_config_from_request_checkbox_off` - checkbox inativo -> `{"enabled": False}`
- [ ] `test_cpu_config_from_request_checkbox_on` - checkbox ativo -> `{"enabled": True, "weight": 0}`
- [ ] `test_maximize_vram_per_gpu_still_works` - binary search Phase 2 inalterado

## Critérios de Sucesso

- Zero referencias a `MAX_CPU_WEIGHT_PCT` no arquivo
- CPU pode escalar acima de 70% quando necessario
- Binary search Phase 2 inalterado
- `compute_cpu_offload_weights()` calcula spill-over sem truncamento
