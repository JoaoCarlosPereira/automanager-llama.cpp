---
status: completed
title: Implementar cascata estrita por MB em LoadDistributor.distribute
type: backend
complexity: medium
dependencies: []
---

# Implementar cascata estrita por MB em LoadDistributor.distribute

## Visão Geral
Reescrever `LoadDistributor.distribute` para alocar o modelo por uma cascata estrita por prioridade calculada em MB (preencher cada GPU até 98% antes de usar a próxima; CPU por último), tornando-a a fonte única da verdade da distribuição. É o núcleo da feature e base de todas as demais tarefas.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- A função DEVE percorrer as GPUs em ordem de prioridade (principal primeiro, depois por índice crescente) e alocar `min(VRAM_total × 0,98, restante_do_modelo)` em MB por GPU.
- A próxima GPU SÓ DEVE receber carga quando a anterior atingir o limite de 98%; NUNCA distribuir proporcionalmente.
- A CPU/RAM SÓ DEVE receber carga quando todas as GPUs habilitadas estiverem cheias.
- O limite de ocupação DEVE ser 98% (constante fixa, sem exposição na interface).
- Quando o modelo não couber e `cpu_enabled=False`, a função DEVE retornar `is_feasible=False`.
- A saída DEVE preservar o contrato `DistributionResult` (pesos em %), convertendo MB→% e reconciliando a soma em 100% no último dispositivo.
</requirements>

## Subtarefas
- [x] 01.1 Definir a nova assinatura recebendo `priority_order`, `gpu_vram` (MB) e `estimated_model_vram_mb`, com `vram_limit_pct=98.0`.
- [x] 01.2 Implementar o loop de preenchimento por prioridade em MB com corte em 98% por GPU.
- [x] 01.3 Tratar a sobra: alocar na CPU somente após todas as GPUs; marcar `is_feasible=False` quando CPU desligada e há sobra.
- [x] 01.4 Converter MB→% por dispositivo e reconciliar a soma em 100% (método largest-remainder `_mb_to_pct`).
- [x] 01.5 Remover o trecho de distribuição proporcional existente (Rule 3 & 4).
- [x] 01.6 Criar testes parametrizados dos cenários A–D e bordas.

## Detalhes de Implementação
Reescrever `LoadDistributor.distribute` em `load_distributor.py`, mantendo o dataclass `DistributionResult`. Ver seção "Interfaces Principais" e o pseudo-código da seção "Design de Implementação" do TechSpec. A estimativa de VRAM (em MB) é fornecida pelo chamador; esta função não faz I/O.

### Arquivos Relevantes
- `load_distributor.py` — contém `DistributionResult` e `LoadDistributor.distribute` a serem reescritos; remover o caminho proporcional.
- `tests/unit/test_load_distributor_cascade.py` — novo arquivo de testes da cascata.

### Arquivos Dependentes
- `gpu_manager.py` — `compute_offload_plan` consumirá a nova assinatura (tratado na task_02).

### ADRs Relacionados
- [ADR-001: Cascata estrita por prioridade como contrato único](../adrs/adr-001.md) — Define a regra de preenchimento e o limite de 98%.
- [ADR-003: LoadDistributor como motor único de cascata por MB](../adrs/adr-003.md) — Define onde a lógica vive e o contrato em %.

## Entregáveis
- `LoadDistributor.distribute` reescrito como cascata estrita por MB com `is_feasible`.
- Remoção da lógica proporcional do módulo.
- Novo `tests/unit/test_load_distributor_cascade.py` com cenários A–D e bordas.
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**.

## Testes
- Testes unitários:
  - [ ] Cenário A: modelo 20GB, 3090(24GB)+2×P100(16GB) → 3090=100%, P100s=0%, CPU=0%.
  - [ ] Cenário B: modelo 30GB → 3090 a 98%, P100#1 com a sobra, P100#2=0%, CPU=0%.
  - [ ] Cenário C: modelo 50GB → 3090 e P100#1 a 98%, P100#2 com a sobra, CPU=0%.
  - [ ] Cenário D: modelo 70GB → todas as GPUs a 98%, CPU com a sobra (>0%).
  - [ ] Borda: modelo cabe inteiro na principal → secundárias não ativadas (0%).
  - [ ] Borda: `cpu_enabled=False` e modelo não cabe → `is_feasible=False`.
  - [ ] Borda: GPU única → 100% na GPU.
  - [ ] Arredondamento: soma de `gpu_weights`+`cpu_weight` fecha em 100% com VRAM desigual.
  - [ ] Zero-offload: `cpu_weight==0` enquanto qualquer GPU tem folga abaixo de 98%.
- Testes de integração:
  - [ ] N/A nesta tarefa (função pura; integração coberta na task_02).
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Os quatro cenários A–D produzem a distribuição exata esperada
- Nenhum caminho proporcional permanece em `load_distributor.py`
