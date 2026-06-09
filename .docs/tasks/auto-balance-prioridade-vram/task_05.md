---
status: completed
title: Frontend limpa pins ao ativar o Auto-Balance
type: frontend
complexity: low
dependencies:
  - task_04
---

# Frontend limpa pins ao ativar o Auto-Balance

## Visão Geral
Ao ligar o `auto-balance-toggle`, a interface deve limpar/desabilitar visualmente os checkboxes `.gpu-pin` (e `.cpu-pin`), deixando claro que a cascata controla toda a distribuição e alinhando o comportamento visual ao do backend.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- Ao marcar o `auto-balance-toggle`, todos os `.gpu-pin` e `.cpu-pin` DEVEM ser desmarcados e seu indicador visual (anel âmbar) removido.
- Enquanto o Auto-Balance estiver ativo, os controles de pin DEVEM ficar desabilitados ou sem efeito.
- Ao desligar o Auto-Balance, os controles de pin DEVEM voltar a ficar disponíveis (estado limpo).
- O payload enviado ao `/start` DEVE refletir `pinned=false` quando o Auto-Balance estiver ativo.
</requirements>

## Subtarefas
- [x] 05.1 Adicionar `onAutoBalanceToggle`/`clearGpuPins` e ligar ao `auto-balance-toggle` em `bindGpuManualListeners`.
- [x] 05.2 Remover o indicador visual de pin (anel âmbar) ao limpar.
- [x] 05.3 Desabilitar os pins enquanto o toggle estiver ativo; reabilitar ao desligar.
- [x] 05.4 `collectDeviceWeightsFromUI` envia `pinned=false` (pins limpos) com Auto-Balance ligado.
- [x] 05.5 Adicionar testes de UI (jest) do comportamento do toggle; ajustar teste pré-existente conflitante.

## Detalhes de Implementação
Editar `static/js/gpu.js` (handlers de pin: `onGpuPinToggle` ~322–334; coleta: `collectDeviceWeightsFromUI` ~245–282) e `static/js/models.js` (leitura do `auto-balance-toggle` ~395). Ver seção "Visão dos Componentes" do TechSpec. Mudança puramente de frontend, alinhada ao backend da task_04.

### Arquivos Relevantes
- `static/js/gpu.js` — handlers de pin e coleta de pesos.
- `static/js/models.js` — leitura do `auto-balance-toggle` e envio ao `/start`.

### Arquivos Dependentes
- Backend `/start` (task_04) — já ignora `pinned` quando `auto_balance=true`; o frontend mantém coerência visual.

### ADRs Relacionados
- [ADR-001: Cascata estrita por prioridade como contrato único](../adrs/adr-001.md) — Decisão de Auto-Balance limpar pins.

## Entregáveis
- Handler do `auto-balance-toggle` que limpa e desabilita pins, com indicador visual removido.
- Coleta de pesos enviando `pinned=false` sob Auto-Balance.
- Testes de UI com cobertura >= 80% **(OBRIGATÓRIO)**.

## Testes
- Testes unitários (UI):
  - [ ] Ligar o `auto-balance-toggle` desmarca todos os `.gpu-pin`/`.cpu-pin`.
  - [ ] Indicador visual (anel âmbar) é removido ao limpar.
  - [ ] Com Auto-Balance ligado, `collectDeviceWeightsFromUI` retorna `pinned=false` para todos os dispositivos.
  - [ ] Desligar o Auto-Balance reabilita os controles de pin.
- Testes de integração:
  - [ ] Fluxo de UI: ligar Auto-Balance → iniciar → payload `/start` com `pinned=false`.
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Estado visual dos pins coerente com o comportamento do backend
- Nenhum pin marcado tem efeito enquanto o Auto-Balance está ativo
