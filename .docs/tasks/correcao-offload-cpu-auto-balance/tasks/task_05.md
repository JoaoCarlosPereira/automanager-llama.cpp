---
status: pending
title: Atualizar process_manager.py - passar cpu_enabled para compute_offload_plan
type: backend
complexity: low
dependencies: [task_03, task_04]
---

# Task 05: Atualizar process_manager.py

## Visao Geral

Atualizar o metodo `ProcessManager.start()` para extrair o estado `cpu_enabled` das `gpu_weights` e passa-lo para `compute_offload_plan()`. Também remover a chamada condicional a `validate_weights()` (que contem o cap de 70%) e unificar para `validate_gpu_weights()`.

<critical>
- Ler o TechSpec secao 4.4 antes de implementar
- A logica de `start()` é critico - erros aqui quebram o funcionamento do sistema
- Manter toda a logica de build de `cmd`, env vars, e process management inalterada
- Apenas modificar as validacoes de pesos e a chamada a `compute_offload_plan()`
</critical>

<requirements>
1. Antes da chamada a `compute_offload_plan()`, extrair `cpu_enabled = any(w.active and w.device == "cpu" for w in gpu_weights)`
2. Passar `cpu_enabled` para `self.gpu_manager.compute_offload_plan(gpu_weights, total_layers, cpu_enabled=cpu_enabled)`
3. Substituir a chamada condicional `validate_weights`/`validate_gpu_weights` por apenas `validate_gpu_weights(gpu_weights)`
4. A logica de `validate_gpu_weights` (que valida soma ≈100% entre GPUs ativas) DEVE permanecer inalterada
5. Nenhuma outra alteracao nos metodos `start()`, `stop()`, `start_auto_balance()`, ou `OOMWatchdog`
</requirements>

## Subtarefas

- [ ] Adicionar `from schemas import StartRequest` se nao estiver importado (ja está, confirmar)
- [ ] Extrair `cpu_enabled = any(w.active and w.device == "cpu" for w in gpu_weights)` antes da chamada `compute_offload_plan`
- [ ] Atualizar chamada: `self.gpu_manager.compute_offload_plan(gpu_weights, total_layers, cpu_enabled=cpu_enabled)`
- [ ] Remover a logica condicional `if has_active_cpu: validate_weights else: validate_gpu_weights` (linhas 382-390)
- [ ] Substituir por: `ok, err = self.gpu_manager.validate_gpu_weights(gpu_weights)`
- [ ] Verificar que o bloco `if not ok: raise HTTPException` permanece inalterado
- [ ] Confirmar que `validate_gpu_weights` ainda valida a soma ≈100% entre GPUs ativas

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `process_manager.py` | Modificar validacao de pesos (linhas 382-390) e chamada a `compute_offload_plan` (linha 402) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `gpu_manager.py` (task_03) | `compute_offload_plan` agora aceita parametro `cpu_enabled` |
| `auto_balance.py` (task_04) | `validate_weights` cap removido, `validate_gpu_weights` permanece |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- `process_manager.start()` passando `cpu_enabled` para `compute_offload_plan()`
- Validacao unificada via `validate_gpu_weights()`
- Toda a logica de build de comando inalterada
- OOMWatchdog inalterado

## Testes

- [ ] `test_start_with_cpu_enabled` - verificar que `cpu_enabled=True` é passado para `compute_offload_plan`
- [ ] `test_start_with_cpu_disabled` - verificar que `cpu_enabled=False` é passado para `compute_offload_plan`
- [ ] `test_validate_gpu_weights_called` - confirmar que `validate_gpu_weights` é chamado independentemente de CPU estar ativo
- [ ] `test_start_command_build_unchanged` - verificar que a lista de `cmd` é construida corretamente
- [ ] `test_oom_watchdog_unchanged` - confirmar que OOMWatchdog não é afetado

## Critérios de Sucesso

- `cpu_enabled` passado corretamente para `compute_offload_plan`
- Validacao unificada via `validate_gpu_weights`
- Comandos de launch inalterados
- OOMWatchdog inalterado
