---
status: pending
title: Testes de integracao - cenarios reais de multi-GPU com CPU offload
type: test
complexity: high
dependencies: [task_03, task_04, task_05, task_06, task_08]
---

# Task 10: Testes de integracao - cenarios reais de multi-GPU com CPU offload

## Visao Geral

Criar testes de integracao que validam o comportamento do sistema completo em cenarios reais de multi-GPU com CPU offload. Os testes cobrem os fluxos manual e auto-balance, validando que a politica GPU-first, CPU minimo é aplicada consistentemente.

<critical>
- Ler o TechSpec secao 6.2 e PRD secoes de fluxos manual/auto-balance
- Os testes DEVEM simular cenarios end-to-end (UI -> API -> GPUManager -> llama-server)
- Usar mocks de nvidia-smi e subprocess.Popen para isolar os testes da hardware real
- Seguir o padrao de testes de integracao existente no projeto (tests/integration/)
- Testes devem ser deterministocos - sem dependencia de tempo real
</critical>

<requirements>
1. O arquivo DEVE estar em `tests/integration/test_offload_integration.py`
2. Devem ser implementados TODOS os 6 cenarios do TechSpec secao 6.2
3. Os testes DEVEM usar mocks de `nvidia-smi` e `subprocess.Popen`
4. O sistema DEVE ser testado em modo manual E auto-balance
5. Cada cenário DEVE validar o OffloadPlan retornado e os argumentos de llama-server
6. NENHUM teste DEVE iniciar um llama-server real
</requirements>

## Subtarefas

- [ ] Criar `tests/integration/test_offload_integration.py`
- [ ] Implementar fixtures: mock_gpus, mock_llama_server, mock_nvidia_smi
- [ ] Cenário 1: Manual, CPU desligado, modelo cabe - validar layers apenas em GPUs
- [ ] Cenário 2: Manual, CPU desligado, modelo nao cabe - validar erro de hardware incapaz
- [ ] Cenário 3: Manual, CPU ligado, spill-over - validar CPU recebe o que sobra
- [ ] Cenário 4: Auto-balance, GPUs suficientes - validar CPU=0
- [ ] Cenário 5: Auto-balance, OOM em todas, CPU usado - validar CPU escala sem cap 70%
- [ ] Cenário 6: Auto-balance, CPU desligado, OOM - validar falha com mensagem clara
- [ ] Rodar `pytest tests/integration/test_offload_integration.py -v` e confirmar todos passando

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `tests/integration/test_offload_integration.py` (NOVO) | Testes de integracao para cenarios multi-GPU + CPU |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `gpu_manager.py` (task_03) | `compute_offload_plan` integrado com LoadDistributor |
| `auto_balance.py` (task_04) | Escalonamento de CPU sem cap |
| `process_manager.py` (task_05) | Passa `cpu_enabled` para `compute_offload_plan` |
| `tests/integration/conftest.py` | Se existir fixtures de integracao, usar como base |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- Arquivo `tests/integration/test_offload_integration.py` com 6 cenarios
- Todos os testes de integracao passando via `pytest`
- NENHUM llama-server real iniciado durante testes
- Mocks de nvidia-smi e subprocess isolados

## Testes

### Cenários de Teste (6 cenarios)
- [ ] Manual, CPU desligado, modelo cabe - Valida que layers vão apenas para GPUs
- [ ] Manual, CPU desligado, modelo nao cabe - Valida erro claro de hardware incapaz
- [ ] Manual, CPU ligado, spill-over - Valida que CPU recebe o que sobra
- [ ] Auto-balance, GPUs suficientes - Valida que CPU=0
- [ ] Auto-balance, OOM em todas, CPU usado - Valida CPU escala sem cap 70%
- [ ] Auto-balance, CPU desligado, OOM - Valida falha com mensagem clara

## Critérios de Sucesso

- Arquivo `tests/integration/test_offload_integration.py` criado com 6 cenarios
- Todos os testes passando (`pytest` exit code 0)
- NENHUM llama-server real iniciado durante testes
- Testes executam em < 30 segundos
- Cobertura >= 80% das linhas modificadas em gpu_manager, auto_balance, process_manager
