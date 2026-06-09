---
status: completed
title: Markup do modal de update-notes no dashboard
type: frontend
complexity: low
dependencies: []
---

# Markup do modal de update-notes no dashboard

## Visão Geral

Adiciona o HTML do modal `#version-update-modal` em `_build_html()` dentro de `llama_manager.py`, seguindo a identidade visual glass/dark do dashboard e o padrão do `login-overlay`. O modal inicia oculto; o JS da tarefa 05 controlará exibição e conteúdo.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE adicionar modal com id `version-update-modal` oculto por default (`display: none` ou classe `hidden`)
- DEVE incluir cabeçalho com área para versão atual vs disponível (`#version-current-ref`, `#version-remote-ref`)
- DEVE incluir container rolável para lista de commits (`#version-commits-list`)
- DEVE incluir botão de dispensar/fechar (`#version-dismiss-btn`) e backdrop clicável
- DEVE usar classes Tailwind consistentes com o dashboard (glass, rounded, border-slate-800)
- DEVE incluir texto no rodapé orientando atualização manual no servidor
- DEVE usar z-index compatível com `login-overlay` (z-50)
- DEVE incluir atributos de acessibilidade: `role="dialog"`, `aria-modal="true"`, `aria-labelledby`
</requirements>

## Subtarefas

- [x] 4.1 Definir estrutura HTML do modal após o `login-overlay` em `_build_html()`
- [x] 4.2 Criar template de item de commit (placeholder ou `<template>`) para clonagem pelo JS
- [x] 4.3 Estilizar lista com altura máxima e overflow-y para muitos commits
- [x] 4.4 Adicionar asserções em `tests/unit/test_html_contract.py` para ids obrigatórios do modal
- [x] 4.5 Verificar renderização em viewport mobile (classes responsivas existentes)

## Detalhes de Implementação

Seguir padrão visual do `login-overlay` (linhas ~852–912 de `llama_manager.py`). Ver seções **Experiência do Usuário** do PRD e **F2 — Modal automático** do PRD. Testes de contrato seguem padrão de `test_html_contract.py` do projeto ui-unit-tests.

### Arquivos Relevantes

- `llama_manager.py` — função `_build_html()`, bloco `login-overlay` como referência de overlay fixo
- `tests/unit/test_html_contract.py` — testes de presença de elementos HTML obrigatórios

### Arquivos Dependentes

- `static/js/version.js` — manipulará os elementos por id (tarefa 05)
- `static/js/index.js` — não alterado nesta tarefa

### ADRs Relacionados

- [ADR-003: Frontend version.js com modal e sessionStorage](adrs/adr-003.md) — markup estático no template
- [ADR-001: Modal Automático na Abertura do Dashboard](adrs/adr-001.md) — conteúdo do modal (commits, versões)

## Entregáveis

- Markup completo do modal em `_build_html()`
- Testes de contrato HTML atualizados em `test_html_contract.py` **(OBRIGATÓRIO)**

## Testes

- Testes unitários (contrato HTML):
  - [ ] HTML gerado por `_build_html()` contém `id="version-update-modal"`
  - [ ] HTML contém `id="version-commits-list"` e `id="version-dismiss-btn"`
  - [ ] HTML contém `id="version-current-ref"` e `id="version-remote-ref"`
  - [ ] Modal possui `role="dialog"` e `aria-modal="true"`
- Meta de cobertura: >= 80% nos novos asserts de contrato
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Modal renderiza corretamente no HTML servido em `GET /` para usuário autenticado
- Elementos possuem ids estáveis documentados para o JS da tarefa 05
- Visual consistente com o tema dark/glass do dashboard
