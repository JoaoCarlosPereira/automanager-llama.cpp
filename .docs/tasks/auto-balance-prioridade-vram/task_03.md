---
status: completed
title: Aposentar probing empírico e lógica proporcional em auto_balance.py
type: refactor
complexity: high
dependencies:
  - task_01
  - task_02
---

# Aposentar probing empírico e lógica proporcional em auto_balance.py

## Visão Geral
Remover do caminho de decisão o probing empírico por OOM (`AutoBalanceProber`) e a matemática proporcional (`_split_pool_by_vram`, `compute_cpu_offload_weights`), preservando a estimativa de VRAM. O Auto-Balance passa a aplicar diretamente o plano determinístico da cascata.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O fluxo de Auto-Balance DEVE calcular a distribuição via cascata (task_01/02), sem iniciar o `llama-server` para sondar OOM.
- `estimate_model_vram_mb` e `model_weights_mb_from_disk` DEVEM ser preservados.
- A lógica proporcional (`_split_pool_by_vram`, `compute_cpu_offload_weights`) e o probing (`_find_feasible_split`, `_maximize_vram_per_gpu`, `_probe_start`) DEVEM ser removidos ou desconectados do caminho de decisão.
- Quando o resultado for inviável (CPU desligada e modelo não cabe), o fluxo DEVE persistir `hardware_incapable=true` + mensagem e retornar o payload de capacidade excedida.
- Os testes obsoletos do probing/proporcional DEVEM ser removidos ou reescritos.
</requirements>

## Subtarefas
- [x] 03.1 Reescrever `AutoBalanceProber.discover` para usar a cascata analítica e iniciar o modelo com os pesos resultantes.
- [x] 03.2 Aposentar o probing por start (renomeado para `_discover_empirical_deprecated`, fora do caminho de decisão).
- [x] 03.3 Preservar e reutilizar `estimate_model_vram_mb` / `model_weights_mb_from_disk`.
- [x] 03.4 Mapear inviabilidade para `build_hardware_capacity_failure` (code `hardware_capacity_exceeded`), consumido pelo fluxo existente.
- [x] 03.5 Reescrever os testes obsoletos de `discover` (3) e adicionar casos do fluxo analítico (fit/spill/infeasível).
- [x] 03.6 Reutilizar `planner.to_gpu_weights` para produzir a lista `GPUWeight` (incl. CPU) a partir do `DistributionResult`.

## Notas de Execução
- Decisão de **risco**: em vez de deletar ~800 linhas do prober (`_find_feasible_split`, `_maximize_vram_per_gpu`, `_probe_start`, `_split_pool_by_vram`, `compute_cpu_offload_weights`), elas foram **desconectadas do caminho de decisão** e mantidas como código morto/deprecado. Reduz risco e preserva os testes unitários de planner existentes. Coerente com ADR-002 ("aposentar do caminho principal"). Deleção física pode ser feita em limpeza futura.
- `discover()` agora **inicia o modelo** ao final (o probing antigo deixava o modelo rodando); o restante de `_run_auto_balance` (persistência + recovery_state) ficou inalterado.
- ⚠️ Verificação em hardware real (3090 + 2× P100) recomendada antes de produção, pois o `discover()` analítico não é exercido em GPUs reais pelos testes unitários.

## Detalhes de Implementação
Editar `auto_balance.py`: aposentar `AutoBalanceProber.discover` e métodos de probing; remover `_split_pool_by_vram` (~90–121) e `compute_cpu_offload_weights` (~651–703) do caminho de decisão. Manter `estimate_model_vram_mb` (~783–803) e `spill_order`. Ver seções "Visão dos Componentes" e "Análise de Impacto" do TechSpec.

### Arquivos Relevantes
- `auto_balance.py` — remover probing/proporcional; preservar estimativa e ordem de prioridade.
- `gpu_manager.py` — fornece o plano da cascata (task_02).
- `config_manager.py` — persistir `hardware_incapable`/mensagem.
- `tests/unit/test_auto_balance.py` — limpar/reescrever casos obsoletos.

### Arquivos Dependentes
- `process_manager.py` / rota `/start` — consome o resultado do Auto-Balance e o sinal de inviabilidade.
- `static/js/gpu.js` — `showAutoBalanceCapacityAlert` renderiza o payload de capacidade excedida.

### ADRs Relacionados
- [ADR-002: Cálculo analítico determinístico da cascata](../adrs/adr-002.md) — Justifica aposentar o probing.
- [ADR-001: Cascata estrita por prioridade como contrato único](../adrs/adr-001.md) — Define o bloqueio com alerta quando inviável.

## Entregáveis
- Auto-Balance determinístico baseado na cascata, sem starts de sondagem.
- Remoção do código proporcional/probing do caminho de decisão.
- Persistência de `hardware_incapable` + mensagem em caso inviável.
- Testes obsoletos removidos/reescritos.
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**.
- Testes de integração do fluxo Auto-Balance → plano **(OBRIGATÓRIO)**.

## Testes
- Testes unitários:
  - [ ] Auto-Balance com modelo que cabe → distribuição A/B sem chamar nenhum método de probing/start.
  - [ ] Auto-Balance com CPU ligada e modelo grande → cenário D com CPU>0 (sem proporcional).
  - [ ] Auto-Balance com CPU desligada e modelo não cabe → `hardware_incapable=true` + mensagem persistida.
  - [ ] `_to_gpu_weights` gera a lista `GPUWeight` correta (com entrada CPU quando aplicável).
  - [ ] Testes do probing/proporcional removidos não deixam referências quebradas.
- Testes de integração:
  - [ ] Fluxo `/start` com `auto_balance=true` aplica o plano da cascata e não inicia sondagem.
  - [ ] Caso inviável retorna o payload de capacidade excedida esperado pelo frontend.
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Nenhum start de `llama-server` para sondagem de OOM no caminho de decisão
- Lógica proporcional ausente em `auto_balance.py`
