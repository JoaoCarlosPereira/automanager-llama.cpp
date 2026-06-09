---
status: pending
title: Criar módulo load_distributor.py - motor unificado de distribuiçao GPU/CPU
type: backend
complexity: high
dependencies: []
---

# Task 01: Criar módulo load_distributor.py

## Visao Geral

Criar o módulo `load_distributor.py` com a classe `LoadDistributor`, um engine stateless que calcula a distribuicao ideal de camadas do modelo entre GPUs e CPU seguindo a politica GPU-first, CPU minimo. Este sera o ponto unico de verdade para todo o sistema de distribuicao de carga.

<critical>
- Ler o TechSpec secao 4.1 e ADR-001 antes de implementar
- O módulo DEVE ser 100% stateless e independentemente testavel
- Focar no O QUÊ (politica GPU-first, válvula on/off) nao no COMO
- Incluir testes unitários inline no docstring como exemplos de uso
</critical>

<requirements>
1. O módulo DEVE implementar a classe `LoadDistributor` com os seguintes metodos estaticos:
   - `distribute(gpu_vram, gpu_weights, total_layers, estimated_model_vram_mb, cpu_enabled)` -> `DistributionResult`
   - `is_feasible(gpu_vram, estimated_model_vram_mb, cpu_enabled)` -> `bool`
   - `compute_n_gpu_layers(total_layers, gpu_weight_pct)` -> `int`
2. O dataclass `DistributionResult` DEVE conter: `gpu_weights`, `cpu_weight`, `total_gpu_pct`, `is_feasible`
3. A politica GPU-first DEVE ser implementada: se `total_gpu_vram >= estimated_model_vram_mb`, CPU recebe 0
4. O parametro `cpu_enabled` DEVE funcionar como válvula on/off: se `False`, `cpu_weight` sempre 0
5. NENHUMA constante `MAX_CPU_WEIGHT_PCT` DEVE existir no módulo
6. O metodo `distribute()` DEVE preservar pesos do usuario no modo manual (escalonamento proporcional)
7. O metodo DEVE retornar `is_feasible=False` quando o modelo nao cabe e `cpu_enabled=False`
</requirements>

## Subtarefas

- [ ] Definir dataclass `DistributionResult` com campos `gpu_weights`, `cpu_weight`, `total_gpu_pct`, `is_feasible`
- [ ] Implementar `LoadDistributor.distribute()` com as 4 regras de decisao (cpu_enabled=False, vram suficiente, spill-over, inviavel)
- [ ] Implementar `LoadDistributor.is_feasible()` verificando capacidade total do hardware
- [ ] Implementar `LoadDistributor.compute_n_gpu_layers()` convertendo % para numero de camadas
- [ ] Implementar escalonamento proporcional de pesos GPU (preservando proporcoes do usuario)
- [ ] Adicionar docstrings explicativos e type hints em todos os metodos
- [ ] Incluir exemplos de uso no docstring do metodo `distribute()`

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `load_distributor.py` (NOVO) | Novo módulo stateless - unico ponto de calculo de distribuicao |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `gpu_manager.py` | Consumira `LoadDistributor.distribute()` em `compute_offload_plan()` |
| `auto_balance.py` | Consumira `LoadDistributor.is_feasible()` e `distribute()` |
| `tests/test_load_distributor.py` (NOVO) | Testes unitários do novo módulo |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- Módulo `load_distributor.py` com classe `LoadDistributor` e dataclass `DistributionResult`
- Todos os metodos estaticos implementados com docstrings e type hints
- Módulo testável unitariamente sem dependencias externas
- Cobertura >= 80% para o novo módulo

## Testes

- [ ] `test_gpu_only_model_fits` - vram=12GB, model=8GB, cpu_enabled=False -> gpu={0:100}, cpu=0, feasible=True
- [ ] `test_cpu_valve_off_model_doesnt_fit` - vram=4GB, model=8GB, cpu_enabled=False -> feasible=False
- [ ] `test_cpu_valve_on_spillover` - vram=4GB, model=8GB, cpu_enabled=True -> gpu={0:50}, cpu=50
- [ ] `test_multi_gpu_cpu_valve_on` - vram={0:4GB, 1:4GB}, model=10GB, cpu_enabled=True -> gpu={0:50, 1:50}, cpu=20
- [ ] `test_manual_weights_preserved` - weights={0:70, 1:30}, vram=12GB, model=8GB -> gpu={0:70, 1:30}, cpu=0
- [ ] `test_compute_n_gpu_layers` - total_layers=32, gpu_pct=62.5 -> n_gpu_layers=20
- [ ] `test_is_feasible_all_gpus_exhausted` - vram=2GB, model=8GB, cpu_enabled=True -> feasible=True
- [ ] `test_edge_case_no_active_gpus` - gpu_vram={} -> is_feasible=False

## Critérios de Sucesso

- Módulo `load_distributor.py` criado na raiz do projeto
- Todos os testes unitários passando
- Cobertura >= 80% para o novo módulo
- NENHUM lint ou erro de type hint
- Politica GPU-first implementada corretamente em todos os casos de teste
