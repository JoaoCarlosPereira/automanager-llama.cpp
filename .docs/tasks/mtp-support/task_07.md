---
status: completed
title: Atualizar gpu.js (badge, reset, listeners)
type: frontend
complexity: low
dependencies:
  - task_06
---

# Atualizar gpu.js (badge, reset, listeners)

## Visão Geral

Estende `gpu.js` com funções de badge MTP, reset de defaults e listener de change no toggle, seguindo o pattern de `updateThinkingBadge()` e `resetToDefaults()`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE implementar `updateMtpBadge(enabled)` espelhando `updateThinkingBadge()`
- DEVE resetar `mtp-toggle` para unchecked e `mtp-draft-tokens` para 3 em `resetToDefaults()`
- DEVE exportar `updateMtpBadge` para uso em `models.js` e `metrics.js`
- DEVE registrar listener `change` no toggle em `initDashboard()` (via `models.js`) ou documentar export para bind externo
- DEVE usar constantes de default alinhadas ao backend (`false`, `3`)
</requirements>

## Subtarefas

- [ ] 7.1 Implementar `updateMtpBadge()` com estilos ON/OFF
- [ ] 7.2 Estender `resetToDefaults()` para campos MTP
- [ ] 7.3 Exportar função e garantir bind do listener no init
- [ ] 7.4 Adicionar testes Jest em `static/js/gpu.test.js`

## Detalhes de Implementação

Referência: `updateThinkingBadge()` (~linha 439) e `resetToDefaults()` (~linha 397) em `static/js/gpu.js`. Ver TechSpec componente `gpu.js`.

### Arquivos Relevantes

- `static/js/gpu.js` — badge, reset, exports

### Arquivos Dependentes

- `static/js/models.js` — importa funções, bind no init (task_08)
- `static/js/metrics.js` — sync badge quando running (task_09)

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — reset mantém campos visíveis com defaults

## Entregáveis

- Funções MTP em `gpu.js`
- Testes Jest em `gpu.test.js`, cobertura >= 80% nas funções novas **(OBRIGATÓRIO)**

## Testes

- Testes unitários (Jest):
  - [ ] `updateMtpBadge(true)` define badge texto "ON" e classe violet
  - [ ] `updateMtpBadge(false)` define badge texto "OFF" e classe slate
  - [ ] `resetToDefaults()` define `mtp-toggle.checked=false` e `mtp-draft-tokens.value="3"`
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura Jest >= 80% em `updateMtpBadge` e paths MTP de `resetToDefaults`
- Badge MTP atualiza corretamente no toggle change
