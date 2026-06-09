---
status: completed
title: Injetar campos MTP no HTML do dashboard
type: frontend
complexity: low
dependencies:
  - task_05
---

# Injetar campos MTP no HTML do dashboard

## Visão Geral

Adiciona toggle MTP e input numérico de tokens de predição no painel de configuração do dashboard, sempre visíveis, seguindo o estilo visual de `thinking-toggle` e `parallel-slots`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE adicionar `id="mtp-toggle"` checkbox e `id="mtp-badge"` badge ON/OFF no painel de config
- DEVE adicionar `id="mtp-draft-tokens"` input numérico com min=1, max=6, default=3
- DEVE posicionar campos no bloco flex existente (~linha 919), após thinking-toggle
- DEVE incluir labels, ícones e tooltips no mesmo estilo dos campos atuais
- DEVE expor constantes de default no HTML ou `window.__constants` se necessário para JS
</requirements>

## Subtarefas

- [ ] 6.1 Injetar markup HTML do toggle MTP + badge
- [ ] 6.2 Injetar input numérico `mtp-draft-tokens`
- [ ] 6.3 Adicionar tooltips explicativos (MTP acelera inferência; tokens 2–3 típicos)
- [ ] 6.4 Estender `tests/unit/test_html_contract.py` com asserts dos novos ids

## Detalhes de Implementação

Referência visual: bloco `thinking-toggle` (~linha 958) e `parallel-slots` (~linha 936) em `llama_manager.py`. Ver seção **Experiência do Usuário** do PRD e ADR-001.

### Arquivos Relevantes

- `llama_manager.py` — template HTML do dashboard (`/` endpoint)

### Arquivos Dependentes

- `static/js/gpu.js` — bind e reset (task_07)
- `static/js/models.js` — leitura no start (task_08)
- `static/js/metrics.js` — sync running (task_09)

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — layout permanente no painel

## Entregáveis

- Markup HTML dos campos MTP em `llama_manager.py`
- Testes de contrato HTML em `test_html_contract.py`, cobertura dos novos ids **(OBRIGATÓRIO)**

## Testes

- Testes unitários (contrato HTML):
  - [ ] HTML contém `id="mtp-toggle"`
  - [ ] HTML contém `id="mtp-badge"`
  - [ ] HTML contém `id="mtp-draft-tokens"`
  - [ ] Input numérico possui `min="1"` e `max="6"`
- Meta de cobertura: >= 80% nos novos asserts
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Campos MTP visíveis no dashboard renderizado
- Contrato HTML validado por pytest
