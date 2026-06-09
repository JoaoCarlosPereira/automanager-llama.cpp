---
status: completed
title: Adaptar compute_offload_plan para delegar à cascata e remover caminho proporcional/legado
type: refactor
complexity: medium
dependencies:
  - task_01
---

# Adaptar compute_offload_plan para delegar à cascata e remover caminho proporcional/legado

## Visão Geral
Fazer `GPUManager.compute_offload_plan` sempre delegar à cascata reescrita em `LoadDistributor`, derivando a ordem de prioridade a partir da GPU principal e das GPUs ativas, e removendo o caminho proporcional/legado. Garante que o modo manual e o Auto-Balance usem o mesmo motor.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- `compute_offload_plan` DEVE derivar `priority_order` da GPU `is_main` seguida das demais GPUs ativas por índice crescente.
- `compute_offload_plan` DEVE sempre chamar a cascata de `LoadDistributor` e converter `DistributionResult` em `OffloadPlan` (`n_gpu_layers`, `tensor_split`, `gpu_pct`, `cpu_pct`).
- O caminho proporcional/legado e `_compute_offload_plan_with_lu` DEVEM ser removidos.
- Quando `is_feasible=False`, o plano DEVE sinalizar inviabilidade para o chamador acionar o bloqueio com alerta.
- A estimativa de VRAM do modelo DEVE ser obtida via `AutoBalancePlanner.estimate_model_vram_mb` e repassada à cascata.

</requirements>

## Subtarefas
- [x] 02.1 Derivar `priority_order` a partir de `is_main` + índice das GPUs ativas (`_build_priority_order`).
- [x] 02.2 Obter `estimated_model_vram_mb` e a VRAM por GPU e chamar a cascata.
- [x] 02.3 Converter `DistributionResult` em `OffloadPlan` (mapear % → `n_gpu_layers`/`tensor_split`).
- [x] 02.4 Propagar `is_feasible` no `OffloadPlan` (campo novo) para o `/start` poder bloquear.
- [x] 02.5 Refatorar `_compute_offload_plan_with_lu` para delegar 100% à cascata (sem matemática proporcional).
- [x] 02.6 Adicionar testes da cascata em `tests/unit/test_gpu_manager_core.py` (prioridade, transbordo, infeasível).

## Notas de Execução
- O caminho proporcional já foi eliminado na task_01 (dentro de `distribute`). Aqui, `_compute_offload_plan_with_lu` foi **mantido e refatorado** para delegar à cascata passando `priority_order` main-first — em vez de removido — pois é o adaptador `DistributionResult → OffloadPlan`. Desvio consciente do texto literal do TechSpec ("remover o método"), preservando o comportamento consolidado.
- O caminho `cpu_enabled=None` (modo manual com pesos explícitos do usuário) foi **preservado** intencionalmente: honra a distribuição manual e não é "legado proporcional". A cascata é a fonte da verdade do Auto-Balance (cpu_enabled True/False).
- `is_feasible` foi adicionado ao `OffloadPlan` (default `True`); o **bloqueio efetivo no `/start` + payload do alerta** pertence ao fluxo de Auto-Balance da task_03.
- **Correção pós-entrega (bug "tudo na CPU")**: `_compute_offload_plan_with_lu` lia a chave inexistente `vram_total_mb` de `get_metrics()` (que emite `mem_total`), zerando a VRAM e jogando 100% na CPU quando o `start()` recomputava a cascata. Corrigido para ler `mem_total` (com fallback). Regressão coberta por `test_offload_plan_reads_mem_total_key_not_all_cpu` e `test_offload_plan_maxes_gpus_then_spills_remainder_to_cpu`.

## Detalhes de Implementação
Editar `GPUManager.compute_offload_plan` e remover `_compute_offload_plan_with_lu` em `gpu_manager.py`. Reaproveitar `resolve_main_gpu_index`, `active_gpus_with_weight` e `compute_tensor_split`. Ver seções "Visão dos Componentes" e "Endpoints de API" do TechSpec para o comportamento de inviabilidade.

### Arquivos Relevantes
- `gpu_manager.py` — `compute_offload_plan` (~linhas 416–467) e `_compute_offload_plan_with_lu` (~469–532) a refatorar/remover.
- `load_distributor.py` — fornece a cascata (task_01).
- `auto_balance.py` — `estimate_model_vram_mb` para a estimativa de VRAM.
- `tests/unit/test_gpu_manager_core.py` — atualizar casos.

### Arquivos Dependentes
- `process_manager.py` — consome `OffloadPlan` e `resolve_main_gpu_index` ao montar os args; conferir compatibilidade.

### ADRs Relacionados
- [ADR-003: LoadDistributor como motor único de cascata por MB](../adrs/adr-003.md) — Define a delegação e a remoção do caminho proporcional.
- [ADR-002: Cálculo analítico determinístico da cascata](../adrs/adr-002.md) — Justifica o uso da estimativa de VRAM.

## Entregáveis
- `compute_offload_plan` delegando 100% à cascata, sem caminho proporcional/legado.
- `OffloadPlan` coerente (`n_gpu_layers`, `tensor_split`) nos cenários A–D.
- Sinalização de inviabilidade para o `/start`.
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**.
- Testes de integração do plano nos cenários A–D **(OBRIGATÓRIO)**.

## Testes
- Testes unitários:
  - [ ] `priority_order` coloca a GPU `is_main` em primeiro, depois por índice crescente.
  - [ ] Cenário A: `tensor_split` concentra 100% na principal; `n_gpu_layers` = total de camadas.
  - [ ] Cenário D: `cpu_pct>0` e `n_gpu_layers` < total (parte na CPU).
  - [ ] `cpu_enabled=False` e modelo não cabe → plano marca inviabilidade.
  - [ ] Remoção do legado: nenhuma chamada a `_compute_offload_plan_with_lu`.
- Testes de integração:
  - [ ] `compute_offload_plan` + `resolve_main_gpu_index` produzem `--tensor-split`/`--main-gpu` consistentes para 3090+2×P100 nos cenários A–D.
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Modo manual e Auto-Balance usam o mesmo motor de distribuição
- Nenhum ramo proporcional/legado remanescente em `gpu_manager.py`
