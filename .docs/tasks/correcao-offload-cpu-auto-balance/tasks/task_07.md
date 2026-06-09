---
status: pending
title: Criar testes unitários para LoadDistributor
type: test
complexity: medium
dependencies: [task_01]
---

# Task 07: Criar testes unitarios para LoadDistributor

## Visao Geral

Criar o arquivo `tests/unit/test_load_distributor.py` com testes unitarios abrangentes para a classe `LoadDistributor`. Os testes devem cobrir todos os cenarios definidos no TechSpec secao 6.1, incluindo GPU-only, CPU valve, spill-over, multi-GPU, pesos preservados e edge cases.

<critical>
- Ler o TechSpec secao 6.1 antes de implementar
- Os testes DEVEM ser 100% independentes - sem mocks de hardware, sem nvidia-smi, sem subprocess
- `LoadDistributor.distribute()` é stateless e puro, testes devem refletir isso
- Seguir o padrao de naming do pytest existente no projeto (`test_*` functions)
- Usar `pytest` fixtures para dados comuns (gpu_vram, weights)
</critical>

<requirements>
1. O arquivo DEVE estar em `tests/unit/test_load_distributor.py`
2. O modulo DEVE ser importavel via `from load_distributor import LoadDistributor, DistributionResult`
3. TODOS os 8 casos de teste do TechSpec secao 6.1 DEVEM ser implementados
4. Os testes DEVEM usar `assert` direto ou `pytest` fixtures conforme padrao do projeto
5. A cobertura DEVE atingir >= 80% do modulo `load_distributor.py`
6. NENHUM teste DEVE depender de nvidia-smi, subprocess, ou hardware real
</requirements>

## Subtarefas

- [ ] Criar `tests/unit/test_load_distributor.py`
- [ ] Implementar `test_gpu_only_model_fits` - vram=12GB, model=8GB, cpu_enabled=False -> gpu={0:100}, cpu=0, feasible=True
- [ ] Implementar `test_cpu_valve_off_model_doesnt_fit` - vram=4GB, model=8GB, cpu_enabled=False -> feasible=False
- [ ] Implementar `test_cpu_valve_on_spillover` - vram=4GB, model=8GB, cpu_enabled=True -> gpu={0:50}, cpu=50
- [ ] Implementar `test_multi_gpu_cpu_valve_on` - vram={0:4GB, 1:4GB}, model=10GB -> gpu={0:50, 1:50}, cpu=20
- [ ] Implementar `test_manual_weights_preserved` - weights={0:70, 1:30}, vram=12GB, model=8GB -> gpu={0:70, 1:30}, cpu=0
- [ ] Implementar `test_compute_n_gpu_layers` - total_layers=32, gpu_pct=62.5 -> n_gpu_layers=20
- [ ] Implementar `test_is_feasible_all_gpus_exhausted` - vram=2GB, model=8GB, cpu_enabled=True -> feasible=True
- [ ] Implementar `test_edge_case_no_active_gpus` - gpu_vram={} -> is_feasible=False
- [ ] Rodar `pytest tests/unit/test_load_distributor.py -v` e confirmar todos passando
- [ ] Verificar cobertura: `pytest --cov=load_distributor tests/unit/test_load_distributor.py`

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `tests/unit/test_load_distributor.py` (NOVO) | Testes unitarios para `LoadDistributor` |
| `load_distributor.py` (task_01) | Modulo alvo dos testes |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `tests/unit/conftest.py` | Se existir fixtures globais, usar como base |
| `pytest.ini` | Configuracao pytest do projeto |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- Arquivo `tests/unit/test_load_distributor.py` com 8+ casos de teste
- Todos os testes passando via `pytest`
- Cobertura >= 80% do modulo `load_distributor.py`
- NENHUM lint ou erro de tipo nos testes

## Testes

### Testes Unitarios (8+ casos)
- [ ] GPU-only, modelo cabe - vram=12GB, model=8GB, cpu_enabled=False
- [ ] CPU valve OFF, modelo nao cabe - vram=4GB, model=8GB, cpu_enabled=False
- [ ] CPU valve ON, spill-over - vram=4GB, model=8GB, cpu_enabled=True
- [ ] Multi-GPU, CPU valve ON - vram={0:4GB, 1:4GB}, model=10GB, cpu_enabled=True
- [ ] Manual pesos preservados - weights={0:70, 1:30}, vram=12GB, model=8GB
- [ ] n_gpu_layers calculation - total_layers=32, gpu_pct=62.5 -> n_gpu_layers=20
- [ ] is_feasible all gpus exhausted - vram=2GB, model=8GB, cpu_enabled=True
- [ ] Edge case no active GPUs - gpu_vram={} -> is_feasible=False

## Critérios de Sucesso

- Arquivo `tests/unit/test_load_distributor.py` criado com 8+ casos de teste
- Todos os testes passando (`pytest` exit code 0)
- Cobertura >= 80% do modulo `load_distributor.py`
- NENHUM lint ou erro de type hint
- Testes executam em < 2 segundos
