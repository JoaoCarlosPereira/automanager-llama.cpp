---
status: pending
title: Atualizar gpu_manager.py - delegar calculo ao LoadDistributor e remover cap de 70%
type: backend
complexity: medium
dependencies: [task_01, task_02]
---

# Task 03: Atualizar gpu_manager.py

## Visao Geral

Atualizar o `GPUManager` em `gpu_manager.py` para integrar o novo `LoadDistributor` e remover as verificacoes de cap de 70% que estao obsoletas. A principal mudança é na metodologia `compute_offload_plan()`, que deve delegar o calculo de distribuicao ao `LoadDistributor.distribute()`, e na metodologia `validate_weights()`, que deve remover a verificacao de limite de CPU.

<critical>
- Ler o TechSpec secao 4.2 antes de implementar
- O metodo `compute_offload_plan()` DEVE aceitar o novo parametro opcional `cpu_enabled`
- A validacao de cap de 70% DEVE ser completamente removida de `validate_weights()`
- Manter a compatibilidade: se `cpu_enabled=None`, inferir de `gpu_weights` via `cpu_offload_active()`
- Nao alterar `compute_tensor_split()`, `detect_model_layers()`, ou outros metodos nao listados no TechSpec
</critical>

<requirements>
1. `compute_offload_plan()` DEVE aceitar um parametro opcional `cpu_enabled: Optional[bool] = None`
2. Quando `cpu_enabled=None`, o metodo DEVE inferir via `self.cpu_offload_active(gpu_weights)`
3. `compute_offload_plan()` DEVE chamar `LoadDistributor.distribute()` com os parametros corretos
4. O resultado do `LoadDistributor` DEVE ser convertido em `OffloadPlan` (NamedTuple existente)
5. `validate_weights()` DEVE REMOVER completamente a verificacao `if cpu_weight > 70.0` (linhas 575-580)
6. `validate_gpu_weights()` DEVE MANTER a validacao de soma ≈100% entre GPUs
7. O import de `LoadDistributor` DEVE vir no topo do arquivo
8. A logica de `compute_n_gpu_layers()` existente DEVE continuar funcionando
</requirements>

## Subtarefas

- [ ] Adicionar `from load_distributor import LoadDistributor` no topo do arquivo
- [ ] Atualizar assinatura de `compute_offload_plan()` para aceitar `cpu_enabled: Optional[bool] = None`
- [ ] Implementar logica de inferencia: `cpu_enabled if cpu_enabled is not None else self.cpu_offload_active(gpu_weights)`
- [ ] Chamar `LoadDistributor.distribute()` com gpu_vram, gpu_weights, total_layers, estimated_model_vram_mb, cpu_enabled
- [ ] Converter `DistributionResult` para `OffloadPlan` (mapeando `n_gpu_layers`, `n_cpu_layers`, `gpu_pct`, `cpu_pct`, `tensor_split`)
- [ ] Remover verificacao de cap de 70% em `validate_weights()` (linhas 572-580)
- [ ] Verificar que `validate_gpu_weights()` continua inalterado
- [ ] Verificar que `compute_n_gpu_layers()` continua funcionando
- [ ] Manter `compute_tensor_split()` inalterado (ele ainda é usado para `--tensor-split` flag)

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `gpu_manager.py` | Modificar `compute_offload_plan()` (linhas 415-450) e `validate_weights()` (linhas 575-580) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `load_distributor.py` (task_01) | Fornece `LoadDistributor.distribute()` |
| `process_manager.py` (task_05) | Chamador de `compute_offload_plan()` |
| `schemas.py` (task_02) | Fornece `StartRequest` com `cpu_enabled` |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- `gpu_manager.py` com `compute_offload_plan()` delegando ao `LoadDistributor`
- `validate_weights()` sem verificacao de cap de 70%
- Todos os metodos existentes continuando funcionais
- Import de `LoadDistributor` adicionado

## Testes

- [ ] `test_compute_offload_plan_cpu_disabled` - weights com cpu inativo -> `n_cpu_layers=0`
- [ ] `test_compute_offload_plan_cpu_enabled_with_spillover` - weights com cpu ativo e vram insuficiente -> CPU recebe spill-over
- [ ] `test_compute_offload_plan_cpu_enabled_inferred` - `cpu_enabled=None` com peso cpu>0 -> infere `True`
- [ ] `test_validate_weights_no_cpu_cap` - `validate_weights` com CPU=80% DEVE passar (antes falhava)
- [ ] `test_validate_weights_sum_check` - weights somando != 100% DEVE falhar
- [ ] `test_compute_n_gpu_layers_still_works` - chamada direta a `compute_n_gpu_layers()` retorna valor correto

## Critérios de Sucesso

- `compute_offload_plan()` delega calculo ao `LoadDistributor`
- `validate_weights()` nao mais rejeita CPU > 70%
- Todos os testes de gpu_manager existentes continuam passando
- Import de `LoadDistributor` não causa erros
